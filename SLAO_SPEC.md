# SLAO: Single LoRA Continual Learning with Orthogonal Initialization via Continual Merging

**Paper**: "Merge before Forget: A Single LoRA Continual Learning via Continual Merging" (Qiao & Mahdavi, 2025, arXiv:2512.23017)

---

## 1. Core Idea

SLAO maintains a **single merged LoRA** across all tasks with **O((m+n)r) constant memory** (no per-task LoRA storage). It achieves this through three mechanisms:

1. **Orthogonal initialization** of A via QR decomposition of the previous task's fine-tuned A
2. **Asymmetric merging**: A is directly replaced; B is merged with time-aware scaling
3. **Time-aware scaling** λ(i) = 1/√i for B merging

The key insight is that LoRA's A and B matrices behave asymmetrically during continual learning: B exhibits higher cosine similarity across tasks and its task vectors are approximately orthogonal, making it amenable to averaging-style merging. A changes more drastically between tasks, so it is initialized orthogonally and replaced directly.

---

## 2. Algorithm

### Pseudocode (Algorithm 1 from paper)

```
Input: T tasks with datasets D_1, ..., D_T; base model W_0; LoRA rank r
Output: Merged LoRA parameters (B_merge, A_merge)

# Task 1: standard LoRA fine-tuning
A_ft_1, B_ft_1 = finetune_lora(W_0, D_1)
B_merge = B_ft_1      # shape: (d, r)
A_merge = A_ft_1      # shape: (r, d)

for i = 2 to T:
    # Step 1: Orthogonal basis extraction via QR decomposition
    Q_i, R_i = QR(A_ft_{i-1}.T)           # A_ft_{i-1}.T is (d, r), Q is (d, r), R is (r, r)
    Q_i = Q_i * sign(diag(R_i)).T         # Sign correction for uniqueness

    # Step 2: Initialize new task LoRA
    A_init = Q_i.T                         # shape: (r, d), satisfies A_init @ A_init.T = I_r
    B_init = B_ft_{i-1}                    # shape: (d, r), from previous fine-tuned B

    # Step 3: Fine-tune on task i
    A_ft_i, B_ft_i = finetune_lora(W_0, D_i, A_init=A_init, B_init=B_init)

    # Step 4: Asymmetric merging
    A_merge = A_ft_i                                           # Direct replacement
    B_merge = B_merge + (1/sqrt(i)) * (B_ft_i - B_merge)      # Time-aware EMA

# Inference: W = W_0 + B_merge @ A_merge
```

### Key Properties

- **A_init has orthonormal rows**: `A_init @ A_init.T = I_r`
- **B merging is an EMA** with decreasing step size λ(i) = 1/√i
- **Only 3 matrices stored at any time**: B_merge, B_ft_{i-1} (or current B_ft_i), A_ft_i
- After task i completes, you can discard A_ft_{i-1} and B_ft_{i-1}

---

## 3. Important Implementation Functions

### 3.1 `orthogonal_init(A_prev: Tensor) -> Tensor`

```python
def orthogonal_init(A_prev: torch.Tensor) -> torch.Tensor:
    """
    Extract orthogonal basis from previous task's A matrix.
    
    Args:
        A_prev: (r, d) matrix from previous task's fine-tuned A
    Returns:
        A_new: (r, d) matrix with orthonormal rows (A_new @ A_new.T = I_r)
    """
    # QR decomposition of A_prev.T -> Q is (d, r), R is (r, r)
    Q, R = torch.linalg.qr(A_prev.T)
    # Sign correction: multiply columns of Q by sign of diagonal of R
    sign = torch.sign(torch.diag(R))
    Q = Q * sign.unsqueeze(0)  # broadcast (d, r) * (1, r)
    return Q.T  # (r, d)
```

### 3.2 `merge_B(B_merge: Tensor, B_ft: Tensor, task_idx: int) -> Tensor`

```python
def merge_B(B_merge: torch.Tensor, B_ft: torch.Tensor, task_idx: int) -> torch.Tensor:
    """
    Time-aware merging of B matrices.
    
    Args:
        B_merge: (d, r) current merged B
        B_ft: (d, r) fine-tuned B from current task
        task_idx: 1-indexed task number (>= 2)
    Returns:
        Updated B_merge
    """
    lam = 1.0 / math.sqrt(task_idx)
    return B_merge + lam * (B_ft - B_merge)
```

### 3.3 `finetune_lora(model, dataset, A_init, B_init, config) -> (A_ft, B_ft)`

Standard LoRA fine-tuning with custom initialization:
- Attach LoRA adapters to **query and value projection** matrices in all attention layers
- Initialize `lora_A` with `A_init` and `lora_B` with `B_init`
- Train with standard cross-entropy loss (no special regularization or orthogonal loss)
- Return the fine-tuned A and B weight dictionaries

### 3.4 `evaluate(model, B_merge, A_merge, eval_datasets) -> dict`

For each task's eval dataset:
- Load base model W_0
- Apply merged LoRA: `W = W_0 + B_merge @ A_merge` for each target layer
- Run inference with task-specific prompt/instruction
- Compute task-specific metric (accuracy or Rouge-L)

---

## 4. LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 8 |
| Target modules | Query (q_proj) and Value (v_proj) projections |
| Alpha | Not explicitly stated (likely 8 or 16; use default) |
| Dropout | Not stated (use 0.0) |

---

## 5. Training Hyperparameters (LLaMA-2-7B-chat)

### Standard CL & Large Number of Tasks benchmarks

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Batch size | 1 |
| Gradient accumulation steps | 8 |
| Effective batch size | 8 |
| Epochs | 1 |
| Optimizer | SGD (implied by paper's gradient update formulas; AdamW may work too) |
| Training samples per task | 1000 per class |
| Validation samples per task | 500 per class |

### SuperNI benchmark

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Epochs | 5 |
| Training instances per task | 1000 |
| Validation/test instances per task | 100 |

---

## 6. Target Evaluation: Large Number of Tasks with BWT (LLaMA-2-7B-chat)

This is the most informative evaluation because it:
- Tests 15 diverse tasks (longer task sequences stress forgetting more)
- Reports both Average Accuracy (AA) and Backward Transfer (BWT)
- Uses 3 different task orders to test robustness

### 6.1 Tasks (15 total)

| # | Dataset | Category | Task Type | Metric |
|---|---------|----------|-----------|--------|
| 1 | Yelp | CL Benchmark | Sentiment analysis | Accuracy |
| 2 | Amazon | CL Benchmark | Sentiment analysis | Accuracy |
| 3 | DBpedia | CL Benchmark | Topic classification | Accuracy |
| 4 | Yahoo | CL Benchmark | Topic classification | Accuracy |
| 5 | AG News | CL Benchmark | Topic classification | Accuracy |
| 6 | MNLI | GLUE | Natural language inference | Accuracy |
| 7 | QQP | GLUE | Paraphrase detection | Accuracy |
| 8 | RTE | GLUE | Natural language inference | Accuracy |
| 9 | SST-2 | GLUE | Sentiment analysis | Accuracy |
| 10 | WiC | SuperGLUE | Word sense disambiguation | Accuracy |
| 11 | CB | SuperGLUE | Natural language inference | Accuracy |
| 12 | COPA | SuperGLUE | Question answering | Accuracy |
| 13 | BoolQ | SuperGLUE | Boolean QA | Accuracy |
| 14 | MultiRC | SuperGLUE | Question answering | Accuracy |
| 15 | IMDB | - | Sentiment analysis | Accuracy |

### 6.2 Task Orders

| Order | Sequence |
|-------|----------|
| O4 | mnli → cb → wic → copa → qqp → boolqa → rte → imdb → yelp → amazon → sst-2 → dbpedia → ag → multirc → yahoo |
| O5 | multirc → boolqa → wic → mnli → cb → copa → qqp → rte → imdb → sst-2 → dbpedia → ag → yelp → amazon → yahoo |
| O6 | yelp → amazon → mnli → cb → copa → qqp → rte → imdb → sst-2 → dbpedia → ag → yahoo → multirc → boolqa → wic |

### 6.3 Data Preparation

Following O-LoRA protocol:
- 1000 randomly sampled training samples per class per task
- 500 validation samples per class per task
- Use Hugging Face datasets or equivalent

### 6.4 Metrics

**Average Accuracy (AA):**
```
AA = (1/T) * Σ_{i=1}^{T} a_{i,T}
```
where `a_{i,T}` is the accuracy on task i after training on all T tasks.

**Backward Transfer (BWT):**
```
BWT = (1/(T-1)) * Σ_{i=1}^{T-1} (a_{i,T} - a_{i,i})
```
where `a_{i,i}` is accuracy on task i immediately after training on task i, and `a_{i,T}` is accuracy on task i after training on the final task T.

BWT measures forgetting: negative means the model forgot; closer to 0 (or positive) is better.

**Implementation note**: To compute BWT, you must evaluate on ALL previous tasks after training each task (not just at the end). Store the full T×T accuracy matrix where entry (i, j) = accuracy on task i after training task j.

### 6.5 Results to Reproduce

**Average Accuracy (AA) on Large Number of Tasks (LLaMA-2-7B-chat):**

| Method | O4 | O5 | O6 | avg |
|--------|-----|-----|-----|-----|
| SeqLoRA | 69.1 | 66.0 | 71.1 | 68.7 |
| IncLoRA | 72.2 | 71.6 | 73.8 | 72.5 |
| O-LoRA | 74.0 | 72.0 | 74.6 | 73.5 |
| InfLoRA | 69.4 | 67.4 | 72.5 | 69.8 |
| SAPT-LoRA | 84.7 | 78.9 | 82.2 | 81.9 |
| CorDA | 73.4 | 72.7 | 74.0 | 73.4 |
| MagMax | 72.3 | 73.5 | 74.5 | 73.4 |
| KnOTS (zero init) | 61.5 | 60.1 | 58.0 | 59.9 |
| LoRA-LEGO | 58.8 | 58.7 | 53.2 | 56.9 |
| OPCM | 51.9 | 52.8 | 46.9 | 50.5 |
| **SLAO** | **75.0** | **74.4** | **75.1** | **74.8** |
| Multi-Task (upper bound) | - | - | - | 78.1 |

**Backward Transfer (BWT) on Large Number of Tasks (LLaMA-2-7B-chat):**

| Method | BWT |
|--------|-----|
| SeqLoRA | -17.2 |
| IncLoRA | -9.6 |
| O-LoRA | -4.0 |
| InfLoRA | -4.9 |
| SAPT-LoRA | -2.9 |
| CorDA | -4.5 |
| LoRM-BA | -6.7 |
| LoRM-AB | -4.1 |
| MagMax | -3.8 |
| OPCM | -3.9 |
| KnOTS (zero init) | -14.1 |
| LoRA-LEGO | -15.6 |
| **SLAO** | **-3.5** |

SLAO achieves BWT of **-3.5**, competitive with the best data-free methods (MagMax -3.8, O-LoRA -4.0), only behind SAPT-LoRA (-2.9) which requires generated pseudo-samples from previous tasks.

---

## 7. Baselines to Implement for Comparison

Priority baselines (simplest to implement, most informative comparison):

1. **SeqLoRA** (lower bound): Sequentially fine-tune a single LoRA on each task without any merging or protection. Same init each task.
2. **IncLoRA**: Add a new LoRA per task, freeze previous LoRAs. At inference, sum all LoRAs.
3. **Multi-Task** (upper bound): Train a single LoRA jointly on all tasks simultaneously.

---

## 8. Infrastructure Requirements

### Hardware
- NVIDIA A100 GPU (paper uses DeepSpeed)
- ~35 GB peak GPU memory for LLaMA-2-7B-chat with SLAO

### Software Stack
- PyTorch
- Hugging Face Transformers (for model loading)
- PEFT (for LoRA)
- DeepSpeed (optional, for memory efficiency)
- Datasets (Hugging Face) for loading benchmarks

### Training Cost Reference
- Standard CL (5 tasks): ~51 min GPU walltime
- Large Number of Tasks (15 tasks): ~1h 59min GPU walltime
- SuperNI (15 tasks): ~2h GPU walltime

---

## 9. Implementation Plan

### Directory Structure
```
clue/
├── configs/           # Experiment configs (model, tasks, hyperparams)
├── data/              # Data loading and preprocessing
│   ├── datasets.py    # Dataset loaders for all 15 tasks
│   └── prompts.py     # Task-specific prompt templates
├── methods/           # Continual learning methods
│   ├── base.py        # Base CL method interface
│   ├── slao.py        # SLAO implementation
│   ├── seq_lora.py    # Sequential LoRA baseline
│   └── inc_lora.py    # Incremental LoRA baseline
├── models/            # Model utilities
│   └── lora.py        # Custom LoRA with manual init support
├── eval/              # Evaluation
│   ├── metrics.py     # AA, BWT, accuracy matrix
│   └── evaluate.py    # Evaluation loop
├── train.py           # Main training script
└── run_experiment.py  # Experiment runner (handles task ordering)
```

### Key Design Decisions

1. **LoRA implementation**: Use HuggingFace PEFT but with custom initialization hooks. After creating LoRA adapter, manually overwrite `lora_A` and `lora_B` weights before training.

2. **State management**: After each task, save:
   - `A_ft_i` (per-layer fine-tuned A weights)
   - `B_ft_i` (per-layer fine-tuned B weights)  
   - `B_merge_i` (per-layer merged B weights)
   - Accuracy matrix row (eval all tasks seen so far)

3. **Per-layer operation**: The orthogonal init and merging happen **independently per layer**. Each attention layer has its own (A, B) pair for q_proj and v_proj.

4. **Evaluation protocol**: After training on each task i, evaluate on tasks 1..i to build the accuracy matrix. This is needed for BWT computation.
