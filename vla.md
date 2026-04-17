# VLA Implementation Spec: CLUE on LIBERO via OpenVLA

**Goal:** Port CLUE (SLAO + Fisher-weighted B merging) from the LLM
continual learning setting to Vision-Language-Action models. Produce
a BWT result on LIBERO-Object that beats sequential fine-tuning and
EWC baselines.

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    clue/                                  │
│  (existing LLM CL code — reuse methods/, models/)        │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  vla/                        (NEW)               │    │
│  │  ├── train_vla.py            Main CL loop        │    │
│  │  ├── eval_vla.py             LIBERO rollouts     │    │
│  │  ├── data/                                       │    │
│  │  │   └── libero_loader.py    RLDS → per-task     │    │
│  │  └── configs/                                    │    │
│  │      └── libero_object.yaml  Task definitions    │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  methods/slao.py          ← reused as-is                 │
│  methods/fisher.py        ← reused as-is                 │
│  models/lora.py           ← reused (merge_B, etc.)       │
│  eval/metrics.py          ← reused (AccuracyMatrix)      │
└──────────────────────────────────────────────────────────┘
```

The key insight: CLUE's CL logic (orthogonal init, Fisher-weighted merge,
EWC penalty) operates on LoRA A/B weight tensors. It doesn't care whether
those tensors come from an LLM doing text classification or a VLA doing
action prediction. We reuse `methods/` and `models/` unchanged, and write
new training and evaluation loops that understand OpenVLA and LIBERO.

---

## 2. Base Model: OpenVLA-7B

**Model:** `openvla/openvla-7b` ([GitHub](https://github.com/openvla/openvla))

**Architecture:** Llama-2-7B backbone + DINOv2 vision encoder + action
tokenization head. Actions are discretized into 256 bins per dimension
(7 DoF), predicted as token sequences via cross-entropy loss.

**Why it works with CLUE:**
- Same Llama-2 backbone as our LLM experiments → LoRA attaches identically
- Cross-entropy loss on action tokens → Fisher estimation is unchanged
- PEFT LoRA with `target_modules="all-linear"` is the documented approach

**LoRA config (from OpenVLA docs):**

```python
LoraConfig(
    r=32,
    lora_alpha=16,         # min(32, 16) = 16, scaling = 16/32 = 0.5
    lora_dropout=0.0,
    target_modules="all-linear",  # OpenVLA default
    task_type=TaskType.CAUSAL_LM,
)
```

Note: OpenVLA uses `target_modules="all-linear"` by default — this applies
LoRA to every linear layer, equivalent to our `--lora_target_modules q_proj
k_proj v_proj o_proj gate_proj up_proj down_proj`. This is good — our
all-modules LLM results (BWT=-0.05%) were much better than q+v only.

---

## 3. Benchmark: LIBERO

### 3.1 Task suites

| Suite | # Tasks | Horizon | Variation | Use |
|-------|---------|---------|-----------|-----|
| LIBERO-Object | 10 | 300 steps | Same scene, different objects | Primary benchmark |
| LIBERO-Long | 10 | 600 steps | Long-horizon manipulation | Confirmation run |
| LIBERO-Spatial | 10 | 300 steps | Same objects, different locations | Post-demo |
| LIBERO-Goal | 10 | 300 steps | Same scene, different goals | Post-demo |

### 3.2 CL protocol

Tasks are presented sequentially. After training on task t, evaluate on
tasks 1..t (building the T×T accuracy matrix). "Accuracy" = success rate
over N rollout episodes.

There is no established CL ordering for LIBERO (unlike O4/O5/O6 for text).
We define our own and fix it across experiments. Use the default task
ordering from the LIBERO benchmark (task indices 0-9 in order).

### 3.3 Data

RLDS-formatted datasets, available on HuggingFace:
- `openvla/modified_libero_rlds` (with no-op filtering)
- Specific names: `libero_object_no_noops`, `libero_spatial_no_noops`, etc.

Each task has ~50 demonstrations. For CL, we train on all demonstrations
per task (no subsampling — small enough datasets that this is fast).

### 3.4 Evaluation

- **Rollouts per task per eval:** 20 (iteration), 50 (final numbers)
- **Max steps per episode:** 300 (Object/Spatial/Goal), 600 (Long)
- **Success:** LIBERO built-in success check (object in target region)
- **Parallelization:** 8 parallel MuJoCo envs per GPU
- **Metric:** Success rate ∈ [0, 1], used as "accuracy" in AccuracyMatrix

---

## 4. Implementation

### 4.1 `vla/data/libero_loader.py` — Per-task data loading

Wrap RLDS datasets into per-task PyTorch datasets compatible with OpenVLA's
collator.

```python
class LiberoTaskDataset:
    """Single LIBERO task's demonstrations, formatted for OpenVLA."""

    def __init__(self, suite_name: str, task_idx: int, data_root: str):
        # Load RLDS dataset for this suite
        # Filter to demonstrations for task_idx only
        # Each item: {"image": tensor, "instruction": str,
        #             "action_tokens": tensor, "proprio": tensor}
        ...

def get_libero_task_order(suite_name: str) -> list[str]:
    """Return ordered list of task names for a LIBERO suite."""
    ...
```

**Key detail:** OpenVLA's training uses `PaddedCollatorForActionPrediction`
from the OpenVLA codebase. We reuse it — import from the openvla package.

### 4.2 `vla/eval_vla.py` — LIBERO rollout evaluation

```python
@torch.no_grad()
def evaluate_libero_task(
    model,
    processor,
    suite_name: str,
    task_idx: int,
    n_rollouts: int = 20,
    max_steps: int = 300,
    n_parallel: int = 8,
) -> float:
    """Run rollouts for one LIBERO task, return success rate."""
    # 1. Create LIBERO env for this task
    # 2. Run n_rollouts episodes (parallelized in batches of n_parallel)
    # 3. For each step:
    #    a. Get observation (image + proprio)
    #    b. model.predict_action(image, instruction, proprio)
    #    c. env.step(action)
    #    d. Check success / max steps
    # 4. Return successes / n_rollouts
```

This replaces `eval/evaluate.py`'s text-generation-based evaluation.
The interface to AccuracyMatrix is identical — just pass the success
rate as "accuracy."

### 4.3 `vla/train_vla.py` — Main CL loop

This is the VLA analogue of `train.py`. Structure:

```python
def train_one_task_vla(model, dataset, method, args, fisher, ...):
    """One task's LoRA fine-tuning on OpenVLA."""
    # Same structure as train_one_task in train.py:
    # - DataLoader with OpenVLA's collator
    # - Forward pass: model(**batch) → loss (cross-entropy on action tokens)
    # - Fisher penalty if enabled
    # - GPM/Riemannian if enabled (probably skip for v1)
    # - Gradient accumulation, clipping, optimizer step
    ...

def main():
    # 1. Load OpenVLA-7B + apply LoRA via PEFT
    # 2. Instantiate SLAO method (reuse from methods/slao.py)
    # 3. Instantiate Fisher (reuse from methods/fisher.py)
    # 4. For each task in LIBERO suite:
    #    a. method.before_task(task_idx, task_name)
    #    b. Load task demonstrations
    #    c. train_one_task_vla(...)
    #    d. Estimate F_new (for Bayesian merge)
    #    e. method.after_task(task_idx)  ← Fisher-weighted merge happens here
    #    f. Accumulate Fisher
    #    g. Evaluate on tasks 0..task_idx via rollouts
    #    h. Update AccuracyMatrix
    # 5. Report AA, BWT
```

### 4.4 What we reuse unchanged from `clue/`

| Module | What it provides | Changes needed |
|--------|-----------------|---------------|
| `methods/slao.py` | Orthogonal init, asymmetric merge, Fisher-weighted merge | None |
| `methods/fisher.py` | Fisher estimation, EWC penalty, online EWC | None — `model(input_ids, labels).loss` works identically for OpenVLA |
| `models/lora.py` | `merge_B`, `merge_B_bayesian`, `extract_lora_state`, etc. | None |
| `eval/metrics.py` | `AccuracyMatrix`, AA, BWT computation | None — takes float accuracy values |

### 4.5 What's new

| Module | Purpose | Complexity |
|--------|---------|-----------|
| `vla/data/libero_loader.py` | Load RLDS data as per-task datasets | Medium — wrapping existing RLDS loading |
| `vla/eval_vla.py` | LIBERO rollout evaluation | Medium — wrapping LIBERO env API |
| `vla/train_vla.py` | CL training loop for VLA | Low — structural copy of `train.py` |

---

## 5. Training Hyperparameters

Based on OpenVLA defaults + our LLM CL findings:

| Parameter | Value | Source |
|-----------|-------|--------|
| LoRA rank | 32 | OpenVLA default |
| LoRA alpha | 16 | OpenVLA default |
| LoRA targets | all-linear | OpenVLA default (matches our best LLM config) |
| LR | 5e-4 | OpenVLA default |
| Optimizer | AdamW | Our finding: AdamW >> SGD for CL |
| LR scheduler | constant | OpenVLA default |
| Batch size | 16 | OpenVLA default |
| Grad accum | 4 | From Robot.md |
| Epochs per task | 3 | Robot.md (replaces max_steps) |
| Precision | bf16 | OpenVLA default |
| Image aug | True | OpenVLA default (random crop 90%) |
| Fisher merge beta | 0.5 | Best from LLM experiments |
| Fisher lambda | 0.1 | Conservative, from LLM experiments |
| Fisher gamma | 0.9 | Online EWC for 10-task sequences |
| Fisher samples | 256 | Same as LLM |

**Do NOT use:**
- `--riemannian` (hurt on LLM, no reason to expect different here)
- `--lora_plus_ratio` (no benefit at q+v, untested at all-linear)
- `--gpm_threshold` (adds complexity, save for ablation)

---

## 6. OpenVLA-Specific Considerations

### 6.1 Fisher on action tokens

OpenVLA predicts 7 action dimensions × 1 token each = 7 tokens per step,
via cross-entropy over 256 bins. The Fisher from these action tokens is
valid — it's the same `E[g²]` computation as for text tokens. The only
difference: action token distributions are much peakier (near-deterministic
optimal actions), which makes per-sample Fisher estimates noisier.

**Mitigation:** Use more Fisher samples (256 or 512) and online EWC
(γ=0.9) to smooth across tasks.

### 6.2 Vision encoder

OpenVLA's DINOv2 vision encoder is frozen by default. LoRA is applied
only to the LLM backbone. This means CLUE protects the LLM's action
prediction weights, not the visual representations.

If visual forgetting is an issue (tasks with very different visual
scenes), we may need to add LoRA to the vision encoder. But start
without — keep the scope minimal.

### 6.3 Action normalization

Each LIBERO suite has its own action statistics (mean, std). OpenVLA
uses per-dataset normalization keys (`unnorm_key`). In CL mode, each
task needs its own unnorm_key. Store these per-task at training time
and select the right one at eval time.

**Important:** This means we need task identity at eval time (to select
the normalization stats). This is fine for the CL benchmark setting
(we always know which task we're evaluating). For a task-agnostic
deployment, the normalization would need to be shared or predicted.

---

## 7. Dependencies

```
# Existing
torch>=2.2
transformers==4.40.1
peft>=0.10
datasets

# New for VLA
openvla                    # pip install -e . from openvla repo
libero                     # pip install -e . from LIBERO repo
robosuite                  # LIBERO's underlying sim (installed with LIBERO)
mujoco>=2.3               # Physics engine
tensorflow-datasets        # RLDS data loading
```

### Environment setup

```bash
# 1. Clone and install OpenVLA
git clone https://github.com/openvla/openvla.git
pip install -e openvla/

# 2. Clone and install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO/
pip install -r LIBERO/requirements.txt

# 3. Download RLDS datasets
git clone git@hf.co:datasets/openvla/modified_libero_rlds /data/libero_rlds

# 4. Verify: single-task fine-tune + eval
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    openvla/vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir /data/libero_rlds \
    --dataset_name libero_object_no_noops \
    --lora_rank 32 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --max_steps 5000 \
    --save_freq 5000
```

---

## 8. Implementation Order

### Phase 1: Infrastructure (Days 1-2)

1. Install OpenVLA + LIBERO + MuJoCo on GPU instance
2. Reproduce single-task OpenVLA LIBERO-Object result (~88% SR)
3. Write `libero_loader.py` — per-task dataset extraction from RLDS
4. Write `eval_vla.py` — rollout evaluation with parallel envs
5. Verify: train on task 0, evaluate, get ~88% success rate

### Phase 2: CL Loop (Days 3-4)

6. Write `train_vla.py` — structural copy of `train.py`, replacing:
   - Text tokenization → OpenVLA's image+action processing
   - Text generation eval → rollout eval
   - Dataset loading → LIBERO task datasets
7. Run sequential FT (no CL protection) on 10 tasks
   - This gives the forgetting floor
   - Verify AccuracyMatrix fills correctly
8. Run CLUE (Fisher merge β=0.5, EWC λ=0.1) on 10 tasks
   - If BWT > sequential FT, we're in business

### Phase 3: Baselines & Numbers (Days 5-7)

9. Run EWC-only baseline (fisher_lambda=0.1, no merge)
10. Run SLAO baseline (no Fisher at all)
11. Compare: CLUE vs SLAO vs EWC vs SeqFT
12. If CLUE wins: run 3 seeds for error bars
13. If not: debug Fisher estimation (check magnitudes, gradient flow)

---

## 9. Evaluation Protocol

### Iteration (Phases 1-2)

- 20 rollouts per task
- Evaluate after tasks {0, 4, 9} only (3 eval points vs 10)
- 1 seed

### Final numbers (Phase 3)

- 50 rollouts per task
- Evaluate after every task (full T×T matrix)
- 3 seeds
- Report AA ± std, BWT ± std

### Quick sanity checks

Before running full CL:
1. **Fisher sanity:** After training task 0, print Fisher stats per layer.
   Fisher should be non-zero, non-uniform, and concentrated on a subset
   of parameters. If Fisher is near-uniform, the estimate is noise.
2. **Single-task baseline:** Task 0 SR should be ~80-90%. If not, the
   LoRA config or training is wrong.
3. **Forgetting check:** After training task 1, eval task 0. If SeqFT
   drops <10%, forgetting is mild and our delta will be small. If it
   drops >30%, there's room for CLUE to shine.

---

## 10. Expected Results

Based on LLM results (scaled by VLA task difficulty):

| Method | Expected SR (avg) | Expected BWT | Rationale |
|--------|-------------------|-------------|-----------|
| Sequential FT | 40-55% | -25 to -15% | Strong forgetting on manipulation |
| EWC only | 60-70% | -8 to -4% | Penalty helps but not enough |
| SLAO | 65-75% | -5 to -2% | Orthogonal init helps |
| **CLUE** | **70-80%** | **-2 to +1%** | Fisher merge protects B merge |
| Multi-task oracle | 85-95% | N/A | Upper bound |

If CLUE achieves near-zero BWT on LIBERO while maintaining >70% SR,
that's a strong result — no existing VLA CL method achieves this.

---

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Fisher degenerate on peaked action dists | Medium | High | More samples (512), check stats |
| Forgetting is mild even for SeqFT (tasks too similar) | Medium | Medium | Use cross-suite protocol (40 tasks) |
| OpenVLA setup takes >2 days | Medium | High | Fall back to SmolVLA or LLM-only story |
| Rollout eval is too slow | Low | Medium | 8 parallel envs, 20 rollouts for iteration |
| LoRA rank 32 is too large for CL (subspace saturation) | Low | Medium | Try rank 8 or 16 as fallback |
