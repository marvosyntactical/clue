# CLUED-IN: A Chat Agent That Actually Learns Across Sessions

**One-line pitch:** A chat assistant that fine-tunes itself on each
conversation, merges what it learned into persistent LoRA adapters via
CLUE, and demonstrably remembers what you taught it last week — without
RAG, without stuffing old transcripts into the prompt, without any
external memory store.

---

## 1. What the Demo Shows

The user has a multi-session conversation with an LLM. Each session is
a separate chat (fresh context window). Between sessions, the model
fine-tunes on the completed conversation and merges the update via CLUE.

**Session 1:** User teaches the model their code style preferences.
> "I always use snake_case, never camelCase. I prefer list comprehensions
> over map/filter. Always add type hints to function signatures."

**Session 2 (new context, no history in prompt):** User asks the model
to write a function.
> "Write a function that takes a list of temperatures in Fahrenheit and
> returns them in Celsius, filtering out any below freezing."

The model writes snake_case code with type hints and a list comprehension
— because it *learned* the preferences, not because they're in context.

**Session 3:** User teaches domain knowledge.
> "In our codebase, the config loader is at `core.config.load_settings()`.
> It returns a `Settings` dataclass with fields `db_url`, `api_key`, and
> `debug`."

**Session 4:** User asks for code that uses the config.
> "Write a function that connects to the database using our config."

The model correctly uses `core.config.load_settings()` and accesses
`settings.db_url` — knowledge from session 3 — while still respecting
the style preferences from session 1.

**The "wow" moment:** Open a "memory inspector" panel showing what the
model knows, and a comparison panel showing the same prompt answered
by the base model (no learning) side-by-side. The difference is visceral.

---

## 2. Why This Is Hard (and Why CLUE Solves It)

Naive fine-tuning on session 2 destroys session 1's knowledge
(catastrophic forgetting). This is the whole problem CLUE addresses:

- **Orthogonal init** (SLAO): each session's LoRA adapter starts in a
  subspace that doesn't interfere with previous sessions.
- **Asymmetric merge**: A (read directions) is replaced; B (write
  directions) is EMA-merged with λ = 1/√i.
- **Fisher-weighted merge**: parameters important to old sessions are
  protected during merging.

Each chat session = one "task" in CLUE's framework. The mapping is
exact.

---

## 3. Hardware & Model

### Target: single NVIDIA A40 (48 GB)

| Model | VRAM (bf16) | VRAM (4-bit + LoRA fp16) | Inference tok/s | FT time/session | Recommendation |
|-------|-------------|--------------------------|-----------------|-----------------|----------------|
| Qwen-2.5-7B-Instruct | 15 GB | 6 GB | ~40 | ~2-4 min | **Primary pick** |
| Llama-3.1-8B-Instruct | 16 GB | 7 GB | ~35 | ~2-4 min | Backup |
| Phi-3.5-mini (3.8B) | 8 GB | 4 GB | ~70 | ~1-2 min | Fast iteration |
| Qwen-2.5-3B-Instruct | 7 GB | 4 GB | ~80 | ~1 min | Fastest, weakest |

**Recommendation: Qwen-2.5-7B-Instruct, 4-bit quantized base + LoRA
in bf16.**

Rationale:
- 4-bit base + bf16 LoRA ≈ 10 GB inference, ≈ 20 GB fine-tuning.
  Fits on A40 with room for batch processing and KV cache.
- Qwen-2.5-7B has the best instruction-following in its class and
  strong code generation — critical for the demo tasks.
- 4-bit quantization (GPTQ or bitsandbytes NF4) is standard for
  LoRA fine-tuning (QLoRA) and doesn't measurably hurt LoRA quality.

**LoRA config:**

| Param | Value | Rationale |
|-------|-------|-----------|
| rank | 16 | Higher than our LLM CL experiments (r=8) because sessions carry more diverse information than classification tasks |
| alpha | 32 | 2× rank (standard) |
| target | all-linear | Our best CL results used all-modules; matches QLoRA default |
| dropout | 0.0 | No dropout — sessions are short, overfitting isn't the risk |

---

## 4. Serving Architecture

### Option A: HF Transformers + PEFT (recommended for v1)

```
┌─────────────────────────────────────────────────┐
│  Gradio UI                                       │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ Chat tab  │  │ Memory panel │  │ A/B panel │ │
│  └─────┬─────┘  └──────┬───────┘  └─────┬─────┘ │
│        │               │                │        │
│        └───────────────┼────────────────┘        │
│                        │                         │
└────────────────────────┼─────────────────────────┘
                         │ HTTP
┌────────────────────────┼─────────────────────────┐
│  Backend (FastAPI)     │                          │
│                        ▼                          │
│  ┌────────────────────────────────────────────┐  │
│  │  Model Server                              │  │
│  │  - Base model: 4-bit quantized, pinned     │  │
│  │  - Active LoRA: hot-swapped via PEFT       │  │
│  │  - generate(): standard HF pipeline        │  │
│  └────────────────────────────────────────────┘  │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  CLUE Engine (background thread)           │  │
│  │  - Queue: completed sessions               │  │
│  │  - Worker: fine-tune → Fisher → merge      │  │
│  │  - Output: updated LoRA adapter            │  │
│  └────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

**Pros:** Full control over LoRA loading, fine-tuning, and merging.
PEFT's `model.load_adapter()` / `model.set_adapter()` is instant.
Our existing CLUE code (methods/slao.py, models/lora.py) plugs in
directly.

**Cons:** Single-request inference (no continuous batching). Fine for
a demo with 1-5 concurrent users.

### Option B: vLLM with LoRA support

vLLM supports dynamic LoRA loading via `--enable-lora`. Inference is
much faster (continuous batching, PagedAttention), but:

- LoRA hot-swap requires reloading through the vLLM API, not instant
- Fine-tuning must happen in a separate process (vLLM holds the GPU)
- More moving parts, harder to debug

**Verdict:** Use Option A for the demo. If we need to scale to many
users, migrate to vLLM later — the LoRA adapters are the same files.

### Option C: SGLang

Similar trade-offs to vLLM. Better LoRA support in some benchmarks.
Worth considering if vLLM's LoRA loading proves too slow.

---

## 5. The Learning Loop

This is the core design question: when does fine-tuning happen?

### Recommended: fine-tune on session close

```
User opens Session i
  │
  ├── Chat normally (inference only, instant responses)
  │   Model uses merged LoRA from sessions 1..i-1
  │
  ├── User clicks "End Session" or closes tab
  │
  ├── Background: fine-tune on session i transcript
  │   ├── Format transcript as instruction-following pairs
  │   ├── LoRA fine-tune: ~1-3 epochs, ~1-4 min
  │   ├── Fisher estimation on session i data
  │   └── CLUE merge: orthogonal init → train → merge B
  │
  ├── Updated LoRA adapter saved to disk
  │
  User opens Session i+1
    └── Model loads merged LoRA, ready immediately
```

**Why this is the right boundary:**

1. **Natural task boundary.** CLUE needs discrete tasks. A session is
   a natural unit — coherent topic, clear start/end.
2. **No latency during chat.** Fine-tuning takes minutes. If we did it
   mid-conversation, the user would wait. At session end, it runs in
   the background.
3. **Clean training data.** A completed session is a curated
   (instruction, response) dataset. Mid-session, you'd be training on
   partial / possibly corrected interactions.
4. **Matches user mental model.** "The AI learns from our conversation
   after we're done" is intuitive.

### Alternative: TTT before each response (additive, not replacing)

Test-Time Training: before generating a response, do 1-5 gradient
steps on the current context (prompt + recent turns). This adapts the
model *within* a session, complementing CLUE's *cross-session* learning.

```
User sends message
  ├── Format recent turns as training data
  ├── 1-5 gradient steps on LoRA (TTT) — ~0.5-2 sec
  ├── Generate response with TTT-adapted LoRA
  └── Discard TTT updates (they're ephemeral)

Session ends
  └── CLUE merge (persistent, as above)
```

**Trade-off:** Adds 0.5-2s latency per response. Worth it if the demo
emphasizes within-session adaptation ("watch it learn in real time").
Not worth it if the demo emphasizes cross-session persistence.

**Recommendation:** Implement session-boundary learning first. Add TTT
as an optional toggle if time permits — it's a nice "level 2" demo
feature.

### What NOT to do

- **Fine-tune after every message.** Too slow (minutes), and single
  turns are too little data for meaningful gradient updates.
- **Fine-tune continuously in background during chat.** Race condition
  nightmare — inference reads LoRA weights while fine-tuning writes
  them. Solvable with double-buffering but adds complexity.
- **Accumulate all sessions and retrain from scratch.** Defeats the
  purpose. CLUE's whole point is O(1) memory, O(1) compute per task.

---

## 6. Training Data: How to Format Sessions

Each session transcript is converted to supervised fine-tuning (SFT)
pairs:

### Format 1: Direct conversation replay

Every (user message, assistant response) pair becomes a training
example. The model learns to reproduce its own responses given the
conversation history.

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "I prefer snake_case in all code."},
  {"role": "assistant", "content": "Got it! I'll use snake_case..."}
]}
```

**Pros:** Simple, captures style and preferences.
**Cons:** Trains on the model's own (possibly mediocre) responses.

### Format 2: Curated instruction pairs (recommended)

Extract the *teachings* from the session and format as explicit
instruction-response pairs. If the user corrects the model, use the
correction as the target.

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant. Use snake_case. Use type hints. Prefer list comprehensions."},
  {"role": "user", "content": "Write a function to convert Fahrenheit to Celsius."},
  {"role": "assistant", "content": "def convert_f_to_c(temps: list[float]) -> list[float]:\n    return [...]"}
]}
```

**Pros:** Higher-quality targets (corrections used, good responses
reinforced). System prompt captures learned preferences explicitly.
**Cons:** Requires a formatting step (can be LLM-assisted or rule-based).

### Format 3: Hybrid

Use Format 1 for the raw transcript, plus synthesize additional Format 2
examples by prompting the base model with the learned facts and
generating Q&A pairs. This augments the small session dataset.

**Recommendation:** Start with Format 1 (zero-effort), add Format 2
synthesis if results are weak. Format the transcript using the model's
chat template (Qwen's ChatML).

### Training hyperparameters

| Param | Value | Rationale |
|-------|-------|-----------|
| Epochs | 3 | Sessions are short (~10-50 turns). 3 epochs ensures learning. |
| LR | 2e-4 | Standard QLoRA rate |
| Optimizer | AdamW | Our CL results showed AdamW >> SGD |
| Batch size | 4 | Small dataset, don't need large batches |
| Grad accum | 2 | Effective batch 8 |
| Max length | 2048 | Full conversation context |

---

## 7. CLUE Configuration

```python
# methods/slao.py parameters
method = "slao"
a_init_method = "zca"          # ZCA whitening — preserves more structure than QR
fisher_merge_beta = 0.5        # Fisher-weighted B merge
fisher_lambda = 0.1            # Mild EWC penalty during training
fisher_gamma = 0.9             # Online EWC for long session sequences
fisher_samples = 128           # Sessions are short, 128 is enough
```

Each session is one "task." The lifecycle:

```
Session 1 (task 0):
  → Standard LoRA fine-tune on transcript
  → Store as merge_state (no merge needed for first task)
  → Estimate Fisher, snapshot reference params

Session 2 (task 1):
  → ZCA-whiten A from session 1's fine-tuned A
  → Copy session 1's fine-tuned B
  → Fine-tune on session 2 transcript (with EWC penalty)
  → Merge: A replaced, B EMA'd with Fisher weighting
  → Estimate Fisher, accumulate (online EWC)

Session N:
  → Same pattern. Merged adapter grows richer over time.
  → Memory cost is constant: one LoRA adapter, one Fisher accumulator.
```

---

## 8. Demo Interface (Gradio)

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  CLUED-IN: An Agent That Learns                    [?]  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────┐  ┌───────────────────────┐ │
│  │                         │  │  Session History       │ │
│  │     Chat Interface      │  │  ● Session 1: style    │ │
│  │                         │  │  ● Session 2: code     │ │
│  │  [user message...]      │  │  ● Session 3: domain   │ │
│  │  [assistant response..] │  │  ○ Session 4 (active)  │ │
│  │  [user message...]      │  │                       │ │
│  │  [assistant response..] │  │  Learning Status       │ │
│  │                         │  │  ✓ Sessions merged: 3  │ │
│  │                         │  │  ◌ Fine-tuning: idle   │ │
│  │                         │  │                       │ │
│  │  ┌───────────────────┐  │  │  Quick Teach           │ │
│  │  │ Type message...   │  │  │  [Remember: ...]       │ │
│  │  └───────────────────┘  │  │  [Forget: ...]         │ │
│  │  [Send] [End Session]   │  │                       │ │
│  └─────────────────────────┘  └───────────────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  A/B Comparison (toggle)                         │   │
│  │  ┌──────────────────┐  ┌──────────────────────┐  │   │
│  │  │ With learning    │  │ Without learning     │  │   │
│  │  │ (merged LoRA)    │  │ (base model only)    │  │   │
│  │  │                  │  │                      │  │   │
│  │  │ def convert_f_.. │  │ def convertFahr...   │  │   │
│  │  └──────────────────┘  └──────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Key UI elements

1. **Chat interface.** Standard chat with Send and End Session buttons.

2. **Session history sidebar.** Shows completed sessions with one-line
   summaries. Clicking a session shows what was learned.

3. **A/B comparison panel.** For any prompt, show the response from
   (a) the learned model and (b) the base model. This is the "proof"
   that learning happened. Toggle-able to avoid clutter.

4. **Learning status.** Shows how many sessions have been merged,
   whether fine-tuning is in progress, and ETA.

5. **Quick Teach.** A text box prefixed with "Remember:" that lets
   the user teach a fact without a full conversation. Internally, this
   generates a synthetic training pair and triggers a mini fine-tune.

### Tech stack

```
gradio >= 4.0          # UI
fastapi                # Backend API (if separating UI from model)
uvicorn                # ASGI server
transformers >= 4.40   # Model loading
peft >= 0.10           # LoRA
bitsandbytes           # 4-bit quantization
torch >= 2.2           # Training
```

For v1, a single Gradio app is simplest — no separate backend needed.
Gradio's `gr.ChatInterface` handles the chat, custom components handle
the sidebar and comparison panel.

---

## 9. Things That Make It More Impressive

### 9.1 Test-Time Training (TTT) — within-session adaptation

Before generating each response, do 1-5 gradient steps on the recent
conversation turns. This gives immediate in-context learning that's
visible to the user ("it got better mid-conversation"), complementing
CLUE's cross-session persistence.

**Implementation:** Clone the current LoRA weights, do a few gradient
steps, generate, then discard the TTT weights. The persistent LoRA is
untouched. At session end, fine-tune on the full transcript as usual.

**Cost:** 0.5-2s latency per response. Acceptable for a demo.

**When to show this:** Have a toggle "Instant Learning (TTT)" in the
UI. When enabled, the model visibly improves within the session. When
disabled, improvement only appears in the next session. The contrast
is powerful.

### 9.2 Live learning visualization

After each session merge, show:

- **Singular value spectrum** of the merged LoRA adapter (bar chart
  per layer). Audiences love watching this evolve across sessions.
- **Fisher heatmap** showing which parameters the model considers
  important. Bright spots = "this is what I learned."
- **Cosine similarity** between consecutive sessions' A matrices,
  showing how much the subspace rotates.

These are cheap to compute from quantities CLUE already has.

### 9.3 Side-by-side comparison with naive fine-tuning

Run a second model with naive sequential fine-tuning (no CLUE). After
5+ sessions, show both models answering a question from session 1.
The naive model will have forgotten; the CLUE model remembers. This
directly demonstrates the anti-forgetting property.

### 9.4 User-facing "what I've learned" summary

After each merge, prompt the base model: *"Given these LoRA weight
changes, what did the model likely learn?"* — this won't work. Instead:
store the session summaries (one-line each) in a list and display them.
Simpler and more honest.

### 9.5 Teachable moments

Pre-script a demo flow that hits the key beats:
1. Teach a preference → verify it sticks
2. Teach a fact → verify recall
3. Teach something contradictory → show graceful handling
4. Open 5th session → all 4 previous learnings intact
5. Show base model comparison → dramatic contrast

### 9.6 Stiefel variant (stretch)

If the Stiefelized CLUE method lands well on the O4dev benchmark,
use it for the demo with the Σ spectrum visualization. The singular
value evolution across sessions is a uniquely compelling visual that
no other CL method can show.

---

## 10. What Could Go Wrong

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Model doesn't learn from short sessions (10 turns) | Medium | High | Use 3 epochs, higher LR (2e-4), augment with synthetic pairs |
| Fine-tuning takes too long (>5 min) | Low | Medium | Use 3B model, reduce epochs to 2, or pre-tokenize |
| Forgetting still visible despite CLUE | Medium | Medium | Use Fisher merge β=0.5 + EWC λ=0.1; if still failing, increase both |
| Style learning works but fact learning doesn't | High | Medium | Focus demo on style/preference; facts are harder for LoRA |
| 4-bit quantization interacts badly with LoRA updates | Low | Low | Test early; fall back to 8-bit if needed |
| Gradio chat state management is fragile | Medium | Low | Keep state in a simple JSON file per user, not in Gradio state |

**Biggest risk:** fact learning. LoRA on attention layers is better at
learning *how* to respond (style, format, procedure) than *what* to say
(facts stored in MLP). The demo should lean into preferences and
procedures, not trivia recall.

---

## 11. Implementation Plan

### Phase 1: Core loop (2-3 days)

1. Load Qwen-2.5-7B-Instruct with 4-bit quantization + LoRA
2. Implement session transcript → SFT data formatting
3. Wire up CLUE's `before_task` / train / `after_task` cycle
4. Verify: session 1 teaches a style, session 2 retains it
5. Measure fine-tuning time per session

### Phase 2: Gradio UI (2 days)

6. Chat interface with End Session button
7. Session history sidebar
8. A/B comparison panel
9. Learning status indicator

### Phase 3: Polish (2 days)

10. TTT toggle (if time)
11. Singular value / Fisher visualization
12. Side-by-side naive-FT comparison
13. Pre-scripted demo flow with 5 sessions
14. Error handling, persistence across server restarts

### Phase 4: Demo prep (1 day)

15. Run through demo flow 3× end-to-end
16. Record backup video in case of live-demo failure
17. Prepare 2-minute "what you're seeing" script

---

## 12. File Structure

```
clue/
├── clued_in/
│   ├── app.py                 # Gradio application
│   ├── engine.py              # CLUE learning loop (fine-tune + merge)
│   ├── data_formatter.py      # Session transcript → SFT pairs
│   ├── model_server.py        # Model loading, inference, adapter swap
│   ├── ttt.py                 # Test-time training (optional)
│   ├── visualizations.py      # Spectrum, Fisher heatmaps
│   ├── sessions/              # Persisted session transcripts
│   │   ├── session_001.json
│   │   └── ...
│   ├── adapters/              # Saved LoRA checkpoints
│   │   ├── merged_after_001/
│   │   ├── merged_after_002/
│   │   └── current/           # Active merged adapter
│   └── config.yaml            # Model, LoRA, CLUE, UI settings
│
├── methods/slao.py            # Reused as-is
├── methods/fisher.py          # Reused as-is
├── models/lora.py             # Reused as-is
└── eval/metrics.py            # Reused for tracking learning curves
```

---

## 13. Decision: Which Model

Run this test before committing:

```bash
# 1. Load model with LoRA
# 2. Fine-tune on 20 synthetic preference-teaching turns (3 epochs)
# 3. Measure: (a) does it learn the preference? (b) wall-clock time?

python clued_in/benchmark_model.py --model Qwen/Qwen2.5-7B-Instruct --quant 4bit
python clued_in/benchmark_model.py --model Qwen/Qwen2.5-3B-Instruct --quant 4bit
python clued_in/benchmark_model.py --model microsoft/Phi-3.5-mini-instruct --quant 4bit
```

Pick the model where (a) is yes and (b) is under 3 minutes. If the 7B
model learns in <3 min, use it. If not, fall back to 3B.
