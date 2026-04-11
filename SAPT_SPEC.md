# SAPT: Shared Attention Framework for Parameter-Efficient Continual Learning

**Paper:** Zhao et al., "SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models" (ACL 2024)

**Code:** https://github.com/circle-hit/SAPT

---

## 1. Problem SAPT Solves

Standard LoRA-based CL methods face a tension: you either use one LoRA per
task (memory grows linearly, O(T)) or merge LoRAs somehow (information lost).
SAPT takes the **keep-all-LoRAs** approach but makes it work by learning to
dynamically combine them at inference time, without needing task IDs.

---

## 2. Architecture Overview

SAPT adds three small modules on top of a frozen LLM + per-task LoRAs:

```
                    ┌──────────────────────┐
                    │  Frozen LLM backbone  │
                    └───────┬──────────────┘
                            │ hidden states
                    ┌───────▼──────────────┐
                    │  Per-layer LoRA pool   │
                    │  [B₁A₁, B₂A₂, ..BₜAₜ]│◄── frozen (tasks 1..t-1)
                    │  + current BₜAₜ       │◄── trainable
                    └───────┬──────────────┘
                            │ weighted combination
                    ┌───────▼──────────────┐
    input ──────►   │  Shared Attention     │
    embeddings      │  (SALS module)        │   ──► attention weights a₁..aₜ
                    └───────┬──────────────┘
                            │
                    ┌───────▼──────────────┐
                    │  Attentive Reflection  │
                    │  (ARM module)          │   ──► KL loss on pseudo-samples
                    └──────────────────────┘
```

### What's frozen vs trainable at task t

| Component | Params | Trainable? |
|-----------|--------|------------|
| LLM backbone | θ_m | Frozen always |
| LoRA blocks 1..t-1 (q_proj, v_proj per layer) | θ_B1..θ_B(t-1) | Frozen |
| LoRA block t (current task) | θ_Bt | **Trainable** |
| Query projection network | θ_proj (W_down, W_up, LayerNorm) | **Trainable** (shared, updated each task) |
| Key vector for task t | k_t | **Trainable** |
| Key vectors for tasks 1..t-1 | k_1..k_(t-1) | Frozen |
| Generative replay LoRA for task t | B^ref_t | Trained separately |

---

## 3. The Shared Attention Mechanism (SALS)

This is the core innovation. For each input, SALS computes per-instance
attention weights over all task-specific LoRA blocks, then aggregates them
into a single effective LoRA.

### 3.1 Computing the query

Given input token embeddings `E_t ∈ R^(m × d)` (where m = sequence length,
d = hidden dim), first compress to a single vector:

```
e_t = MaxPool(E_t, dim=seq_len)     → R^d
```

Then project through a bottleneck MLP (the "query projection"):

```
h_down = W_down @ e_t               → R^d_p       (W_down ∈ R^(d_p × d))
h_up   = W_up @ SiLU(h_down)        → R^d         (W_up ∈ R^(d × d_p))
q_t    = LayerNorm(h_up)            → R^d
```

Where `d_p` is a bottleneck dimension (hyperparameter, typically 100).

**Rationale:** The bottleneck forces the query to extract a compressed
task-relevant representation from the input. MaxPool over sequence length
makes it length-invariant. The query projection is shared across all tasks
and updated continuously — this is critical because it learns to route
inputs to the right LoRA blocks as more blocks are added.

### 3.2 Computing attention weights

Each task i has a key vector `k_i ∈ R^d`. Attention weights are:

```
a_i = exp(q_t · k_i / T) / Σ_j exp(q_t · k_j / T)     for j = 1..t
```

This is a standard softmax attention over the key vectors. Temperature `T`
controls confidence:

- For SAPT-LoRA: `T = √d` (where d is hidden dim, e.g. √4096 = 64)
- Higher T → more uniform weights (less confident)
- Lower T → winner-take-all (more confident)

**Rationale:** The temperature prevents the softmax from collapsing to a
one-hot vector early in training, which would prevent knowledge transfer
between tasks. `√d` is the standard transformer scaling factor.

### 3.3 Aggregating LoRA blocks

The attention weights combine all LoRA outputs (not weights) per layer:

```python
# In each attention layer, for q_proj and v_proj:
def agg_lora_states(x, lora_current, lora_previous_list, attn_weights):
    # x: (batch, seq_len, d)
    # attn_weights: (batch, num_tasks, 1)
    current_out = lora_current(x)                    # (batch, seq_len, d)
    with torch.no_grad():
        prev_outs = [lora_i(x) for lora_i in lora_previous_list]  # frozen

    # Stack: (batch, num_tasks, seq_len * d)
    all_outs = torch.stack([current_out] + prev_outs, dim=1)
    all_outs = all_outs.reshape(batch, num_tasks, -1)

    # Weighted sum: (batch, 1, num_tasks) @ (batch, num_tasks, seq_len*d)
    aggregated = (attn_weights.transpose(1,2) @ all_outs).squeeze(1)
    return aggregated.reshape(batch, seq_len, d)
```

**Critical detail:** Previous LoRAs are run under `torch.no_grad()`. Only
the current task's LoRA gets gradients. The attention weights also get
gradients (through `q_t` and `k_t`), which is how the system learns to
route.

**Rationale for output aggregation (not weight aggregation):** LoRA outputs
are `B_i @ A_i @ x`. You can't just average the A and B matrices because
`(αB₁A₁ + βB₂A₂) @ x ≠ mean(B)mean(A) @ x` in general. So SAPT averages
the *outputs* instead, which is exact. This is more expensive (forward pass
through all LoRAs) but correct.

---

## 4. Attentive Reflection Module (ARM)

ARM prevents catastrophic forgetting of the attention routing. Without it,
as new tasks are learned, the query projection and key vectors drift, and
old tasks get routed to wrong LoRA blocks.

### 4.1 Pseudo-sample generation

After training task t, a separate generative replay block `B^ref_t` is
trained to reconstruct the input portions of task t's training data. This
is a separate LoRA adapter on the LLM, trained with standard LM loss to
generate text that looks like task t's inputs.

- Volume: 2% of the original training set (so for 1000 samples per class
  with 3 classes → ~60 pseudo-samples for MNLI)
- Trained with LoRA r=8, AdamW lr=0.001, 5k steps (for LLaMA)
- After training, pseudo-samples are generated and stored

**Rationale:** We can't store old task data (privacy, memory). But we need
old-task-like inputs to check that the attention routing still works. So
we train a small generator per task. 2% is enough because we only need the
inputs to match the distribution — we don't need exact replicas.

### 4.2 Storing reference attention weights

After training task t, compute the average attention weight vector across
the task's test/validation samples:

```
ā_t = (1/N) Σ_n attention_weights(x_n)      → R^t
```

This is a single vector per task representing "the typical routing for this
task." It is stored for future use.

### 4.3 KL divergence loss during training

When training task t+1, for each pseudo-sample from a previous task i:

1. Compute current attention weights: `â_i = softmax(q · [k_1..k_{t+1}] / T)`
   This has length t+1.

2. Pad the stored reference: `ā_i` has length i. Pad positions i+1..t+1 with 0.
   Then renormalize to sum to 1? **No** — from the code, the reference is used
   as-is with zero-padding, and KL divergence is computed on the raw logits
   (pre-softmax scores), not the probabilities:

```python
# From memory_replay in the code:
attn_scores = self.cal_attention(keys, pseudo_embeds, return_logits=True)
kl_loss = KLDivLoss(
    log_softmax(attn_scores.squeeze(), dim=1),    # current logits
    replay_labels                                   # stored reference dist
)
```

The stored `replay_labels` is the softmax attention distribution from when
the task was originally learned, zero-padded for new task positions.

**Rationale:** The KL loss says "for pseudo-samples from task i, the attention
distribution should still look like it did when task i was trained." This
prevents the query projection from drifting. The zero-padding for new task
positions means the constraint is only on the *old* tasks' attention weights —
the system is free to assign whatever weight it wants to the new task.

### 4.4 Combined training loss

```
L = L_task + λ · L_KL
```

where:
- `L_task` = standard cross-entropy on current task's data
- `L_KL` = KL divergence on pseudo-samples from all previous tasks
- `λ` = balancing coefficient (hyperparameter)

The replay is interleaved with task training: pseudo-sample batches are
processed at regular intervals (controlled by `data_replay_freq`), not
every step.

---

## 5. Training Lifecycle Per Task

### Task 1 (first task)
1. Initialize a fresh LoRA block (q_proj, v_proj) with kaiming A / zero B
2. Initialize key vector k_1 ~ Uniform(-1, 1) ∈ R^d
3. Initialize query projection (W_down, W_up, LayerNorm)
4. Train with standard LM loss (no replay, no KL)
5. After training: store average attention weights, train generative replay
   block, generate pseudo-samples

### Task t > 1
1. Freeze LoRA blocks 1..t-1 and keys k_1..k_{t-1}
2. Initialize new LoRA block for task t (kaiming A / zero B)
3. Initialize new key vector k_t ~ Uniform(-1, 1) ∈ R^d
4. Load query projection from previous task (it's shared and continues training)
5. Load all previous LoRA blocks into memory (frozen)
6. Train with combined loss: L_task + λ · L_KL
   - Task batches: forward through all LoRAs (frozen prev + trainable current),
     compute weighted output, standard CE loss
   - Replay batches (interleaved): forward pseudo-samples, compute KL between
     current and stored attention distributions
7. After training: store average attention weights, train generative replay
   block, generate pseudo-samples

### Inference (any task, no task ID needed)
1. Forward input through embedding layer
2. Compute query via query projection
3. Compute attention weights over all key vectors
4. Forward through all LoRA blocks, weighted-combine outputs
5. Decode

---

## 6. Hyperparameters for LLaMA-2-7B-chat (Long Sequence / Large Number of Tasks)

From the paper's Appendix C and the official code `gen_script_long_llama.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank r | 4 | Paper says 4 for LLaMA |
| LoRA alpha | 32 | From code (scaling = 32/4 = 8) |
| LoRA dropout | 0.0 | |
| LoRA target modules | q_proj, v_proj | From code: `lora_q` and `lora_v` in each attention layer |
| Learning rate | 5e-5 | From code |
| Attention LR (query proj) | 0.0 | Code sets `attn_lr=0.0` — meaning query proj uses same LR as LoRA |
| Epochs | 20 | From code, but some small-data tasks use max_steps instead |
| Per-device batch size | 2 | 4 GPUs × 2 = 8 total |
| Gradient accumulation | 4 | Effective batch = 8 × 4 = 32 |
| Effective batch size | 32 | |
| LR scheduler | constant | No warmup |
| λ (KL loss weight) | 2.0 | From code: `kl_ratio = 2` |
| Attention temperature | 1 | Code: `attn_temperature = 1`, which maps to `T = √d` |
| Query proj bottleneck d_p | 100 | `trans_hidden_dim` in code (inferred from paper) |
| Max source length | 1024 | From code |
| Max target length | 50 | |
| Precision | bf16 | |
| DeepSpeed | Stage 2 | |
| Replay after N epochs | 0 | Start replay from first epoch |
| Pseudo-sample volume | 2% of training set | |
| Replay generation | LoRA r=8, AdamW lr=0.001, 5k steps | Separate from main training |

### Per-task step overrides (from code)

For small-data tasks, training uses `max_steps` instead of `num_train_epochs`:

| Task | Training mode |
|------|--------------|
| cb | max_steps=100 |
| copa | max_steps=200 |
| boolq | max_steps=500 |
| imdb | max_steps=250 |
| dbpedia | max_steps=200 |
| multirc | max_steps=500 |
| All others | 20 epochs |

### State carried between tasks (from code)

The script chains tasks by passing:
- `--load_checkpoint_from .../trans_input.pt` — the query projection weights
- `--previous_lora_path lora_1,lora_2,...` — comma-separated paths to all frozen LoRAs
- `--previous_prompt_key_path .../prompts_keys_till_now.pt` — stacked key vectors

---

## 7. Memory and Compute Cost

### Memory (scales with T)
- T LoRA blocks: each is r × d × 2 modules (q, v) × L layers
  - For r=4, d=4096, L=32: 4 × 4096 × 2 × 2 × 32 = 2M params per task
  - 15 tasks → 30M params → ~60MB in bf16
- T key vectors: T × d = 15 × 4096 = 60K params (negligible)
- Query projection: d × d_p × 2 + d = ~820K params (shared, not per-task)
- T sets of pseudo-samples: ~2% × training data per task (stored on disk)

### Compute per training step
- Forward through **all T** LoRA blocks for each input (even though only
  the current one gets gradients). This makes training progressively slower
  as T grows.
- T=15 means 15× LoRA forward passes per sample vs 1× for SLAO.

### Comparison with SLAO

| Aspect | SLAO | SAPT |
|--------|------|------|
| Memory (LoRA storage) | O(1) — single merged LoRA | O(T) — one LoRA per task |
| Training compute per step | 1 LoRA forward | T LoRA forwards |
| Extra modules | None | Query proj + key vectors + replay LoRAs |
| Pseudo-sample generation | Not needed | Required (extra training per task) |
| Task ID at inference | Not needed | Not needed (attention routing) |
| Paper AA (O4) | 75.0 | 84.7 |
| Paper BWT (avg) | -3.5 | -2.9 |

---

## 8. Implementation Plan for Our Codebase

### 8.1 Why SAPT is harder to integrate than other methods

SAPT fundamentally changes the model architecture. It doesn't just change
how LoRA weights are initialized or merged — it:

1. Replaces the PEFT model with a custom model that holds multiple LoRA
   blocks per layer and routes between them
2. Adds a query projection network and key vectors
3. Requires pseudo-sample generation (a separate training pipeline)
4. Requires storing and loading per-task LoRA checkpoints

This means we **cannot** use HuggingFace PEFT's standard adapter API.
The official code literally forks the entire LLaMA model class.

### 8.2 Implementation approach

**Option A: Fork the model (like the official code)**

Rewrite `LlamaAttention` to hold a `ModuleList` of LoRA blocks per layer.
This is what the official code does. Maximally faithful but:
- Ties us to a specific model architecture
- 1500+ lines of boilerplate model code
- Hard to maintain across transformers versions

**Option B: Multi-adapter PEFT wrapper**

Use PEFT's multi-adapter support (`model.add_adapter()`) to load multiple
named LoRA adapters. Build the routing logic on top:
- Load adapter "task_0", "task_1", etc.
- For each forward pass, run input through each adapter, collect outputs,
  weight-combine them
- Only the current adapter gets gradients

This is cleaner but requires careful handling of PEFT's adapter switching
and may be slower (sequential adapter forward passes vs parallel).

**Option C: Manual LoRA pool (recommended)**

Implement a lightweight `LoRAPool` module that:
- Stores per-task A and B weight tensors (not full PEFT adapters)
- Computes all LoRA outputs in a single batched matmul
- Applies attention weighting

This gives us full control, minimal boilerplate, and is model-agnostic.

### 8.3 New files

```
clue/
├── methods/
│   └── sapt.py          # SAPT CL method (LoRAPool, QueryProjection,
│                         #   KeyStore, ARM, training lifecycle)
├── data/
│   └── replay.py        # Pseudo-sample generation and loading
```

### 8.4 New CLI flags

```
--method sapt
--sapt_bottleneck_dim 100     # Query projection bottleneck d_p
--sapt_kl_lambda 2.0          # λ for KL loss
--sapt_temperature 1          # Attention temperature mode
--sapt_replay_ratio 0.02      # Fraction of training data for replay
--sapt_replay_lr 0.001        # LR for replay generator training
--sapt_replay_steps 5000      # Steps for replay generator training
```

### 8.5 Integration with existing extensions

| Extension | Compatible with SAPT? |
|-----------|-----------------------|
| Riemannian | Yes — precondition current task's LoRA grads |
| Fisher | Partially — can regularize query projection, but per-task LoRAs are frozen anyway |
| LoRA+ | Yes — applies to current task's LoRA |
| GPM | Yes — project current task's A gradients |

---

## 9. Key Design Decisions

### 9.1 Why output aggregation, not weight averaging?

SLAO averages B matrices (EMA) and replaces A. SAPT averages LoRA
*outputs*. These are fundamentally different:

- Weight averaging: `(Σ αᵢBᵢ)(Σ αᵢAᵢ)x ≠ Σ αᵢ(BᵢAᵢx)` — the cross
  terms introduce interference
- Output averaging: `Σ αᵢ(BᵢAᵢx)` — exact, no cross terms

SAPT's approach is mathematically cleaner but requires storing all LoRAs
and running all of them during inference.

### 9.2 Why the query projection is shared and continuously updated

The query projection must learn to distinguish between ALL tasks seen so
far. If it were frozen after task 1, it would only know how to route task
1 inputs. By updating it on every task (with KL constraints to prevent
drift), it progressively learns a more refined routing function.

This is also the main fragility: if the KL constraints are too weak,
routing degrades for old tasks. If too strong, the projection can't adapt
to new tasks.

### 9.3 Why pseudo-samples instead of stored data

Privacy and memory. But note: SAPT's "data-free" claim is arguable — it
stores generated pseudo-samples that approximate the training distribution.
In practice, the generated text may leak information about the original data.

---

## 10. Results to Reproduce

The paper reports SAPT-LoRA results on the Long Sequence (15-task) benchmark.
The numbers from SLAO_SPEC.md Table:

| Method | O4 | O5 | O6 | avg |
|--------|-----|-----|-----|-----|
| SAPT-LoRA | 84.7 | 78.9 | 82.2 | 81.9 |

And BWT of -2.9 (averaged, from SLAO paper's comparison).

**Note:** These numbers come from the SLAO paper citing SAPT, not from
SAPT's own paper (which uses different benchmarks and metrics — Average
Performance and Forgetting Rate instead of AA and BWT). The SAPT paper
primarily reports on T5-Large and SuperNI, not LLaMA-2-7B-chat on O4/O5/O6.
