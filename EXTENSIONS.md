# EXTENSIONS.md — Proposed Extensions to SLAO

This document specifies four extensions to the SLAO continual learning method.
Each is designed to be **composable**: they modify orthogonal parts of the
training pipeline and can be combined or used independently. Each is also
**method-agnostic** where possible — they should work with SeqLoRA and IncLoRA
too, not just SLAO.

---

## Overview

| # | Extension | What it changes | CLI flag |
|---|-----------|----------------|----------|
| 1 | Riemannian Preconditioning | Optimizer (gradient scaling) | `--riemannian` |
| 2 | EWC (Standard & Online) | Loss function (penalty term) | `--fisher_lambda <float>` |
| 3 | LoRA+ (Asymmetric LR) | Optimizer (per-group LR) | `--lora_plus_ratio <float>` |
| 4 | Gradient Projection Memory | Gradient update (projection) | `--gpm_threshold <float>` |
| 5 | Fisher-Weighted B Merging | Post-task merge (per-param rate) | `--fisher_merge_beta <float>` |

Composability matrix (all pairs are compatible):

|                    | Riemannian | Fisher | LoRA+ | GPM |
|--------------------|-----------|--------|-------|-----|
| **Riemannian**     | —         | yes    | yes*  | yes |
| **Fisher**         |           | —      | yes   | yes |
| **LoRA+**          |           |        | —     | yes |
| **GPM**            |           |        |       | —   |

\* Riemannian + LoRA+: Riemannian preconditioning already adapts effective
learning rates per-matrix. Combining both is valid (LoRA+ sets base LRs,
Riemannian rescales gradients on top), but may be redundant. We support it
but should ablate whether the combination helps.

---

## Extension 1: Riemannian Preconditioned LoRA

**Paper:** Prakash et al., "Riemannian Preconditioned LoRA for Fine-Tuning
Foundation Models" (arXiv:2402.02347)

### Core idea

Standard LoRA optimizes A and B with the same gradient rule, ignoring the
geometry of the low-rank manifold. Riemannian preconditioning applies r×r
correction matrices that account for the curvature:

```
grad_A' = (B^T B + δI)^{-1} @ grad_A
grad_B' = grad_B @ (A A^T + δI)^{-1}
```

This eliminates the need for different learning rates between A and B (though
it composes with LoRA+ if desired) and stabilizes training, especially early on.

### Interaction with SLAO

- At task start after orthogonal init, `A A^T = I_r`, so the B-preconditioner
  is `(I_r + δI)^{-1} ≈ I_r` — a clean identity start with no distortion.
- As A evolves during training, the preconditioner adapts automatically.
- Merging happens post-training and is completely unaffected.

### Implementation

**File:** `methods/riemannian.py`

```python
class RiemannianPreconditioner:
    """Wraps an optimizer to apply Riemannian preconditioning to LoRA grads."""

    def __init__(self, model, delta: float = 1e-6):
        self.delta = delta
        # Build map: lora_A param id -> (lora_A_module, lora_B_module)
        # and vice versa, so we can look up the partner matrix.
        self.a_to_b = {}  # id(A.weight) -> B_module
        self.b_to_a = {}  # id(B.weight) -> A_module
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                a_param = module.lora_A["default"].weight
                b_param = module.lora_B["default"].weight
                self.a_to_b[id(a_param)] = module.lora_B["default"]
                self.b_to_a[id(b_param)] = module.lora_A["default"]

    def precondition_grads(self):
        """Call after loss.backward(), before optimizer.step()."""
        for a_id, b_mod in self.a_to_b.items():
            a_param = self._find_param(a_id)
            B = b_mod.weight.data                    # (d, r)
            BtB = B.T @ B + self.delta * torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
            # grad_A' = BtB^{-1} @ grad_A
            a_param.grad.data = torch.linalg.solve(BtB, a_param.grad.data)

        for b_id, a_mod in self.b_to_a.items():
            b_param = self._find_param(b_id)
            A = a_mod.weight.data                    # (r, d)
            AAt = A @ A.T + self.delta * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            # grad_B' = grad_B @ AAt^{-1}
            b_param.grad.data = b_param.grad.data @ torch.linalg.inv(AAt)
```

### Integration point in train.py

Insert `preconditioner.precondition_grads()` after `loss.backward()` and
gradient accumulation, but before `optimizer.step()`:

```python
if step % args.grad_accum == 0:
    if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
    if preconditioner is not None:          # <-- new
        preconditioner.precondition_grads() # <-- new
    optimizer.step()
    optimizer.zero_grad()
```

### CLI

```
--riemannian              Enable Riemannian preconditioning
--riemannian_delta 1e-6   Regularization for matrix inversion
```

### Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `delta` | 1e-6 | Numerical stability only; not sensitive |

---

## Extension 2: EWC — Elastic Weight Consolidation (Standard & Online)

**References:**
- Standard EWC: Kirkpatrick et al., "Overcoming catastrophic forgetting in
  neural networks" (PNAS 2017)
- Online EWC: Schwarz et al., "Progress & Compress: A scalable framework for
  continual learning" (ICML 2018)

### Core idea

After training each task, estimate the diagonal of the empirical Fisher
information matrix over LoRA parameters. The Fisher tells you *which
parameters mattered* for this task — parameters with high Fisher values
had large gradient magnitudes during training, meaning small changes to
them cause large changes in the loss. During subsequent tasks, penalize
moving those parameters:

```
L_total = L_task + (λ/2) * Σ_j F_j * (θ_j − θ*_j)²
```

where `F_j` is the diagonal Fisher for parameter j and `θ*_j` is the
reference parameter value (the merged state for SLAO).

### Standard EWC vs Online EWC

**Standard EWC** sums the Fisher from each task:

```
F_accumulated = F_task1 + F_task2 + ... + F_taskT
```

Problem: the penalty magnitude grows with T. By task 15, the Fisher sum
is ~15× larger than after task 1. If λ is tuned for task 2, it's too
strong by task 10+, killing plasticity on later tasks.

**Online EWC** uses an exponential moving average with decay γ ∈ (0, 1]:

```
F_accumulated ← γ * F_accumulated + F_new_task
```

This keeps the Fisher magnitude bounded regardless of T:
- γ = 1.0 recovers standard EWC (pure sum)
- γ = 0.9 retains 90% of old Fisher, so after 10 tasks the contribution
  from task 1 is 0.9^9 ≈ 39% of its original value
- γ = 0.5 aggressively forgets old Fisher — only recent tasks matter

Online EWC also updates the reference point θ* after each task (we already
do this — it's the merged state).

**Recommendation:** Use online EWC (γ = 0.9) for sequences longer than 5
tasks. For O4dev (4 tasks), standard and online are nearly identical.

### Interaction with SLAO

SLAO's B merging is *post-hoc* forgetting mitigation. EWC is
*during-training* protection. These are complementary:

- After task i's after_task(): compute Fisher over LoRA params using the
  merged model (the model that will be evaluated).
- The reference point `theta*` for Fisher is the **merged state**, not the
  raw fine-tuned state. This is important: we want to penalize drift from
  the merged model, since that's what inference uses.
- For task i+1, SLAO reinitializes A orthogonally and sets B from the
  previous fine-tuned B. The Fisher penalty then acts on these new starting
  points, pulling them toward the merged state where needed.

### Memory cost

O(parameters), NOT O(T × parameters). The accumulated Fisher is a single
tensor per parameter, overwritten in-place via EMA. The reference params
are also a single snapshot. For 7 target modules at rank 8 on LLaMA-7B,
this is ~28MB total.

### Implementation

**File:** `methods/fisher.py`

```python
class DiagonalFisher:
    """Diagonal Fisher with standard (sum) or online (EMA) accumulation."""

    def __init__(self, model, gamma: float = 1.0):
        """
        Args:
            gamma: EMA decay. 1.0 = standard EWC, <1.0 = online EWC.
        """
        self.model = model
        self.gamma = gamma
        self.fisher: dict[str, torch.Tensor] = {}
        self.ref_params: dict[str, torch.Tensor] = {}

    def estimate(self, data_loader, n_samples=256):
        # Compute per-task Fisher F_new from empirical gradients
        # Accumulate: F ← γ * F + F_new
        ...

    def snapshot_ref_params(self):
        # Store current LoRA params as θ*
        ...

    def penalty(self) -> torch.Tensor:
        # Σ_j F_j * (θ_j − θ*_j)²
        ...
```

### CLI

```
--fisher_lambda <float>   Regularization strength (0 = disabled)
--fisher_gamma <float>    EMA decay: 1.0 = standard EWC, <1.0 = online EWC
--fisher_samples 256      Samples for Fisher estimation per task
```

### Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `fisher_lambda` | 0.0 (disabled) | 0.1 for gentle, 0.5–1.0 for strong. |
| `fisher_gamma` | 1.0 (standard EWC) | Use 0.9 for online EWC on long sequences. |
| `fisher_samples` | 256 | Diminishing returns above 256. |

---

## Extension 3: LoRA+ (Asymmetric Learning Rates)

**Paper:** Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models"
(arXiv:2402.12354)

### Core idea

Standard LoRA uses a single learning rate for both A and B. LoRA+ shows that
B should have a higher learning rate than A, by a factor `ratio`:

```
lr_A = lr
lr_B = lr * ratio
```

The theoretical justification is that with equal LRs, one matrix is
under-updated relative to the other. The paper recommends ratio in [2, 16].

### Interaction with SLAO

SLAO's A/B asymmetry is its defining insight. LoRA+ directly operationalizes
this: A (which gets orthogonally reinitialized and replaced) benefits from
slower, more careful updates, while B (which gets EMA-merged) benefits from
faster, more decisive updates that give the EMA better signal.

### Implementation

No new file needed — this is a change to `build_optimizer` in `train.py`:

```python
def build_optimizer(params_or_model, args) -> torch.optim.Optimizer:
    if args.lora_plus_ratio is not None and args.lora_plus_ratio != 1.0:
        ratio = args.lora_plus_ratio
        a_params, b_params = [], []
        for name, param in params_or_model:  # pass named_parameters
            if not param.requires_grad or "lora_" not in name:
                continue
            if "lora_A" in name:
                a_params.append(param)
            elif "lora_B" in name:
                b_params.append(param)
        param_groups = [
            {"params": a_params, "lr": args.lr},
            {"params": b_params, "lr": args.lr * ratio},
        ]
    else:
        param_groups = [{"params": params, "lr": args.lr}]

    cls = OPTIMIZER_REGISTRY[args.optimizer]
    kwargs = {"weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        kwargs["momentum"] = args.momentum
    return cls(param_groups, **kwargs)
```

### CLI

```
--lora_plus_ratio <float>   B_lr / A_lr ratio (default: 1.0 = standard LoRA)
```

### Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `lora_plus_ratio` | 1.0 (disabled) | Paper recommends 2–16. Start with 4. |

---

## Extension 4: Gradient Projection Memory (GPM)

**Paper:** Saha, Garg & Roy, "Gradient Projection Memory for Continual
Learning" (ICLR 2021, arXiv:2103.09762)

### Core idea

After each task, identify the subspace of input activations at each LoRA layer
that was important for that task. During subsequent tasks, project gradients
to be orthogonal to this accumulated subspace, preventing updates that would
interfere with previously learned representations.

This is the specific variant we implement:

### Algorithm

**Notation:** For a LoRA-injected linear layer, the forward pass is
`y = (W_0 + B A) x`. The input to the LoRA component is `x` (the hidden
state), and the LoRA gradient lies in `span(x)` (the input-gradient
relationship from GPM Section 4). We track the subspace of these inputs.

**Phase 1 — Subspace extraction (after training task t):**

For each LoRA-injected layer l:

1. Collect input activations from a reference subset of the current task's
   training data: `R_l = [x_1, x_2, ..., x_ns]`, shape `(d_in, ns)`.

2. Remove directions already in memory:
   `R_hat_l = R_l - M_l @ M_l^T @ R_l`

3. SVD on the residual: `R_hat_l = U Sigma V^T`.

4. Keep the smallest k such that:
   `||R_l_proj||_F^2 + ||(R_hat_l)_k||_F^2 >= eps * ||R_l||_F^2`
   where `R_l_proj = M_l @ M_l^T @ R_l` and `eps` is the threshold.

5. Update memory: `M_l = [M_l, u_1, ..., u_k]` (append new basis vectors).

**Phase 2 — Gradient projection (during training on task t+1):**

For each LoRA-injected layer l, after computing gradients:

```
grad_A' = grad_A - grad_A @ M_l @ M_l^T
grad_B' = grad_B - grad_B @ (A @ M_l) @ (A @ M_l)^T  ... (*)
```

The A projection is direct: `grad_A` operates on inputs `x`, so we project
away from the important input subspace `M_l`. This is equation (6) from GPM,
adapted for the `(r, d_in)` shape of `lora_A.weight`.

For B, the situation is different: `grad_B`'s rows are scaled by the output of
A (not the raw input), so the relevant subspace for B is `A @ M_l`. However,
since SLAO **replaces A entirely** at each task and B is **merged via EMA**,
directly projecting B gradients gives us less value and adds complexity. We
project **A gradients only** and rely on the Fisher regularization (Extension
2) or the B EMA merging to protect B.

**Simplified projection (what we implement):**

```
# For each LoRA layer's A parameter:
grad_A' = grad_A - grad_A @ M_l @ M_l^T
# B gradients are left unmodified.
```

This is clean, cheap, and theoretically grounded: A's gradient lies in
`span(input)`, so projecting away the important input subspace prevents A from
learning features that would interfere with old tasks' representations.

### Interaction with SLAO

- **Orthogonal init + GPM:** SLAO initializes A_new from QR(A_prev). GPM
  constrains A's *gradients* during training. These are complementary: init
  gives a good starting point, GPM keeps the trajectory in the right subspace.
- **B EMA merging:** Unaffected. GPM only modifies A's gradients.
- **Memory growth:** M_l grows by at most r columns per task (since LoRA's A
  is only rank r). After T tasks, M_l has at most T*r columns. For r=8 and
  T=15, that is 120 columns of dimension d_in=4096, storing 120 * 4096 floats
  = ~2MB per layer. With ~64 LoRA layers (32 layers * 2 projections), total
  GPM memory is ~128MB — trivial compared to the 14GB model.
- **Subspace saturation:** If the accumulated subspace fills d_in, no gradient
  can pass through and learning stops. At T*r=120 out of d_in=4096, we use
  only ~3% of the space. Not a concern for 15 tasks.

### Implementation

**File:** `methods/gpm.py`

```python
class GradientProjectionMemory:
    """Per-layer gradient projection memory for LoRA continual learning."""

    def __init__(self, model, threshold: float = 0.95):
        self.threshold = threshold
        self.n_reference_samples = 256
        # {layer_name: Tensor of shape (d_in, k_accumulated)}
        self.memory: dict[str, torch.Tensor] = {}
        self.model = model

    @torch.no_grad()
    def update_memory(self, data_loader, n_samples: int = 256):
        """Extract and store activation subspaces after training a task.

        Implements GPM Algorithm equations (5), (8), (9).
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # 1. Collect input activations at each LoRA layer via hooks
        activations = {}  # {layer_name: list of tensors}
        hooks = []
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                act_list = []
                activations[name] = act_list
                hook = module.register_forward_hook(
                    lambda mod, inp, out, store=act_list: store.append(
                        inp[0].detach().reshape(-1, inp[0].shape[-1])
                    )
                )
                hooks.append(hook)

        # Forward pass on reference samples
        count = 0
        for batch in data_loader:
            if count >= n_samples:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            self.model(input_ids=input_ids, attention_mask=attention_mask)
            count += input_ids.size(0)

        for h in hooks:
            h.remove()

        # 2. Per-layer: SVD on residual activations, update memory
        for layer_name, act_list in activations.items():
            if not act_list:
                continue
            R = torch.cat(act_list, dim=0).T  # (d_in, n_samples)
            R = R.float()  # SVD in float32 for stability

            if layer_name in self.memory:
                M = self.memory[layer_name]  # (d_in, k_prev)
                R_proj = M @ (M.T @ R)
                R_hat = R - R_proj
                proj_norm_sq = R_proj.norm() ** 2
            else:
                R_hat = R
                proj_norm_sq = 0.0

            U, S, _ = torch.linalg.svd(R_hat, full_matrices=False)
            R_norm_sq = R.norm() ** 2

            # Find smallest k satisfying the threshold criterion
            cumsum = torch.cumsum(S ** 2, dim=0)
            target = self.threshold * R_norm_sq - proj_norm_sq
            k = int((cumsum >= target).nonzero(as_tuple=True)[0][0].item()) + 1
            k = max(1, min(k, U.shape[1]))

            new_bases = U[:, :k]  # (d_in, k)

            if layer_name in self.memory:
                self.memory[layer_name] = torch.cat(
                    [self.memory[layer_name], new_bases], dim=1
                )
            else:
                self.memory[layer_name] = new_bases

    def project_grads(self):
        """Project lora_A gradients orthogonal to stored subspaces.

        Call after loss.backward(), before optimizer.step().
        Implements equation (6) from GPM, adapted for LoRA A shape (r, d_in):
            grad_A' = grad_A - grad_A @ M @ M^T
        """
        for name, module in self.model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if name not in self.memory:
                continue
            a_param = module.lora_A["default"].weight
            if a_param.grad is None:
                continue

            M = self.memory[name]  # (d_in, k)
            grad = a_param.grad.data  # (r, d_in)
            # Project away the protected subspace
            a_param.grad.data = grad - (grad @ M) @ M.T
```

### Integration point in train.py

```python
# After loss.backward() + grad accumulation, before optimizer.step():
if step % args.grad_accum == 0:
    if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
    if gpm is not None:                # <-- new
        gpm.project_grads()            # <-- new
    if preconditioner is not None:
        preconditioner.precondition_grads()
    optimizer.step()
    optimizer.zero_grad()
```

In the main CL loop, after `method.after_task(task_idx)`:
```python
if gpm is not None:
    gpm.update_memory(train_loader, n_samples=args.gpm_samples)
```

**Ordering note:** GPM projection must happen **before** Riemannian
preconditioning. Reason: GPM removes components that would harm old tasks,
then Riemannian rescales the surviving components for better optimization.
Reversing this would project away already-preconditioned directions, which
distorts the geometry that Riemannian preconditioning was trying to fix.

### CLI

```
--gpm_threshold <float>   Activation subspace threshold (0 = disabled, default: 0.95)
--gpm_samples 256         Reference samples for activation SVD
```

### Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `gpm_threshold` | 0.0 (disabled) | 0.90–0.99. Higher = more protection, less plasticity. |
| `gpm_samples` | 256 | More samples = better subspace estimate. 256 is sufficient. |

---

## Extension 5: Fisher-Weighted B Merging

**Background:** Builds on the diagonal Fisher from EWC (Kirkpatrick et al.,
"Overcoming catastrophic forgetting in neural networks", PNAS 2017) and
SLAO's asymmetric merging (Qiao & Mahdavi, "Merge before Forget", 2025).

### Motivation

SLAO merges B matrices with a uniform time-aware rate:

```
B_merge = B_merge + λ(i) * (B_ft - B_merge),   λ(i) = 1/√i
```

Every element of B gets the same merge rate λ(i), regardless of whether
that parameter is critical for old tasks or completely unused. This is
wasteful: parameters that are unimportant for old tasks could be merged
aggressively (fully absorbing new task information), while parameters that
are important should be merged conservatively (protecting old knowledge).

EWC's diagonal Fisher tells us exactly which parameters matter. After
training tasks 1..t-1, the accumulated Fisher F_j is large for parameters
whose gradients were large on old tasks — i.e., parameters the old tasks
relied on heavily.

**Key insight:** Instead of using Fisher only as an external regularization
penalty during training (Extension 2), we can embed importance information
directly into the merge operation itself. This makes the merge
importance-aware by construction, rather than relying on a penalty to
approximately prevent drift.

### Mathematical formulation

**Standard SLAO merge** (uniform rate):

```
B_merge_jk ← B_merge_jk + λ * (B_ft_jk - B_merge_jk)

where λ = 1/√i  (same for all j, k)
```

**Fisher-weighted merge** (per-parameter rate):

```
α_jk = λ / (1 + β · F̃_jk)

B_merge_jk ← B_merge_jk + α_jk · (B_ft_jk - B_merge_jk)
```

where:
- `F̃_jk = F_jk / mean(F)` is the **normalized** Fisher for that B matrix,
  scaled to have mean 1. This makes β interpretable independently of the
  raw Fisher magnitude (which varies across runs, layers, and tasks).
- `β ≥ 0` controls sensitivity to Fisher importance.
- `λ = 1/√i` is the base time-aware rate from SLAO.

**Properties:**

| Condition | α_jk | Behavior |
|-----------|------|----------|
| β = 0 | λ | Recovers standard SLAO (uniform merge) |
| F̃_jk = 0 (unimportant) | λ | Full merge rate — absorb new task freely |
| F̃_jk = 1 (average importance) | λ / (1 + β) | Reduced rate |
| F̃_jk >> 1 (critical) | ≈ λ / (β · F̃_jk) → 0 | Near-frozen — old knowledge protected |

With β = 1, a parameter of average importance is merged at half the base
rate. Parameters 10× more important than average are merged at 1/11 the
rate. This is a smooth, principled interpolation between "merge everything"
and "freeze everything."

### What about A?

SLAO replaces A directly (A_merge = A_ft) because A is orthogonally
reinitialized each task. Fisher-weighted merging applies **only to B**,
which is the matrix that accumulates knowledge across tasks via EMA.
This is consistent with SLAO's asymmetric treatment: A is task-specific,
B is shared/merged.

### Interaction with EWC regularization (Extension 2)

Fisher-weighted merging and EWC regularization use the same Fisher but
at different points in the pipeline:

- **EWC** (during training): penalizes the optimizer for moving away from
  the reference point. Prevents drift during gradient descent.
- **Fisher-weighted merging** (after training): controls how much of the
  new task's B is absorbed into the merged state.

These are complementary. EWC protects during training; Fisher-weighted
merging protects during merging. They can be used independently or
together. When used together, they share the same accumulated Fisher
(no extra computation).

### Requires Fisher estimation

Fisher-weighted merging needs the Fisher to be available at merge time.
This means `--fisher_lambda` does NOT need to be > 0 — you can use
Fisher-weighted merging without the EWC penalty during training. But the
Fisher estimation step must still run. Implementation: if
`--fisher_merge_beta > 0`, Fisher estimation is triggered regardless of
`--fisher_lambda`.

### Implementation

**Modified function in `models/lora.py`:**

```python
def merge_B_fisher(
    B_merge: torch.Tensor,
    B_ft: torch.Tensor,
    task_idx: int,
    fisher_B: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Fisher-weighted merging of B matrices.

    Args:
        B_merge: (d, r) current merged B.
        B_ft:    (d, r) fine-tuned B from current task.
        task_idx: 1-indexed task number (>= 2).
        fisher_B: (d, r) accumulated diagonal Fisher for this B matrix.
                  If None, falls back to uniform merge (standard SLAO).
        beta: sensitivity to Fisher importance (0 = uniform).
    """
    lam = 1.0 / math.sqrt(task_idx)

    if fisher_B is None or beta == 0:
        return B_merge + lam * (B_ft - B_merge)

    # Normalize Fisher to mean 1 for this layer
    f_mean = fisher_B.mean()
    if f_mean > 0:
        F_norm = fisher_B / f_mean
    else:
        F_norm = fisher_B

    # Per-parameter merge rate
    alpha = lam / (1.0 + beta * F_norm)

    return B_merge + alpha * (B_ft - B_merge)
```

**Integration in `methods/slao.py`** — `after_task()`:

```python
def after_task(self, task_idx: int) -> None:
    ...
    for layer_name in self.ft_state:
        A_ft = self.ft_state[layer_name]["A"]
        B_ft = self.ft_state[layer_name]["B"]
        B_prev_merge = self.merge_state[layer_name]["B"]

        # Look up Fisher for this layer's B, if available
        fisher_B = None
        if self.fisher is not None and self.fisher_merge_beta > 0:
            # Construct the parameter name for this layer's B
            b_param_name = f"{layer_name}.lora_B.default.weight"
            fisher_B = self.fisher.fisher.get(b_param_name)

        A_merged = A_ft.clone()
        B_merged = merge_B_fisher(
            B_prev_merge, B_ft, paper_i,
            fisher_B=fisher_B,
            beta=self.fisher_merge_beta,
        )
        new_merge[layer_name] = {"A": A_merged, "B": B_merged}
```

### CLI

```
--fisher_merge_beta <float>   Fisher-weighted merge sensitivity (0 = disabled, default)
```

When `--fisher_merge_beta > 0` and `--fisher_lambda == 0`, Fisher estimation
still runs (needed for the merge) but no EWC penalty is added to the loss.

### Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `fisher_merge_beta` | 0.0 (disabled) | Start with 1.0. Higher = more protection for important params. |

---

## Ordering of operations in the training step

When multiple extensions are active, the gradient modification order is:

```
loss.backward()                     # standard backprop
[+ fisher.penalty() in loss]        # Fisher modifies the loss, not gradients
gradient_accumulation_check()       # wait for grad_accum steps
clip_grad_norm()                    # safety: bound gradient magnitude
gpm.project_grads()                 # remove protected directions from A
riemannian.precondition_grads()     # rescale surviving gradients
optimizer.step()                    # AdamW (with LoRA+ param groups if active)
```

Rationale:
- Fisher penalty is part of the loss (autograd handles it).
- Gradient clipping happens first to prevent extreme gradients from
  corrupting the projection and preconditioning.
- GPM projection is a hard constraint (zero out protected directions).
  Must happen before preconditioning, which is a soft rescaling.
- Riemannian preconditioning adapts the effective learning rate per direction.
- LoRA+ is encoded in the optimizer's param groups, applied at step().

---

## Implementation plan

### File structure

```
clue/
├── methods/
│   ├── riemannian.py     # Extension 1: RiemannianPreconditioner class
│   ├── fisher.py         # Extension 2: DiagonalFisher class
│   └── gpm.py            # Extension 4: GradientProjectionMemory class
├── train.py              # Modified: CLI args, optimizer build, training loop
└── EXTENSIONS.md         # This file
```

Extension 3 (LoRA+) lives entirely in `train.py`'s `build_optimizer`.

### Implementation order

1. **LoRA+** — simplest (one function change), no new files, immediate test.
2. **Riemannian** — small standalone class, no state across tasks.
3. **Fisher** — needs cross-task state, reference param snapshots.
4. **GPM** — most complex (activation hooks, SVD, growing memory).

### Testing strategy

Each extension should be testable independently:

```bash
# Baseline
python train.py --method slao --task_order O4 ...

# LoRA+ only
python train.py --method slao --task_order O4 --lora_plus_ratio 4 ...

# Riemannian only
python train.py --method slao --task_order O4 --riemannian ...

# Fisher only
python train.py --method slao --task_order O4 --fisher_lambda 0.5 ...

# GPM only
python train.py --method slao --task_order O4 --gpm_threshold 0.95 ...

# All four combined
python train.py --method slao --task_order O4 \
    --lora_plus_ratio 4 --riemannian --fisher_lambda 0.5 --gpm_threshold 0.95 ...
```
