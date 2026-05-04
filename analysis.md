# CLUE: Method, Analysis, and Directions

**CLUE** — *Continually merged LoRAs Updating Elastically*.

A continual learning method for LoRA-adapted language models. CLUE
extends **SLAO** ("Merge before Forget" — Qiao & Mahdavi, 2025,
arXiv:2512.23017) by replacing SLAO's uniform B-EMA with an
*elastic* per-element merge weighted by the diagonal Fisher
information, drawing on the EWC line of work (Kirkpatrick et al.,
2017, PNAS; Schwarz et al., 2018, ICML). The "elastic" is literal —
parameters important to past tasks are stiffer (resist updates),
parameters that aren't are more elastic (update freely).

This document is for collaborators, future Claude Code instances,
and ourselves. It states (1) the *theoretical regime* CLUE inherits
from SLAO and what assumptions it relies on, (2) the method
precisely as implemented, (3) an analysis frame for understanding
why it works, (4) what we tried that didn't work, and (5) directions
to take this in.

---

## 1. Theoretical foundations

CLUE is built on top of three theoretical scaffoldings, all of which
come from prior work and which we extend rather than replace.

### 1.1 The NTK / linearization assumption

The central assumption SLAO and CLUE rely on is that fine-tuning a
small adapter on a large pretrained model proceeds in (or close to)
the **lazy regime** of Jacot, Gabriel & Hongler (2018,
arXiv:1806.07572) — sometimes called the "NTK regime" or
"linearization regime." Concretely:

For a network `f(x; θ)` with pretrained weights `θ_0`, a small
update `Δθ`, and the first-order Taylor expansion:

```
f(x; θ_0 + Δθ)  ≈  f(x; θ_0)  +  J(x; θ_0) · Δθ
```

where `J(x; θ_0) = ∂f/∂θ |_{θ_0}` is the Jacobian. The lazy regime
holds when `Δθ` is small enough that `J` doesn't change much over
training — i.e., the network is approximately a linear function of
the parameters around `θ_0`.

LoRA's parameterization is *designed* to keep `Δθ = B A` small (low
rank, small `lora_alpha/r` scaling) and only modifies a subset of
weights. Empirically, LoRA fine-tuning sits closer to the lazy regime
than full fine-tuning. The Aghajanyan et al. (2020,
arXiv:2012.13255) result that pretrained models have low intrinsic
dimensionality is essentially saying: useful task adaptation lives in
a low-dimensional subspace, and we don't need to leave the
linearization neighborhood to find it.

**What this gives us:**

1. **Task vectors are approximately additive.** Let `Δθ_t` be the
   fine-tuned update for task `t`. In the linearization,
   ```
   f(x; θ_0 + Σ_t Δθ_t)  ≈  f(x; θ_0)  +  Σ_t J(x) Δθ_t
                          =  f(x; θ_0)  +  J(x) Σ_t Δθ_t
   ```
   So summing or averaging task vectors gives an approximation of
   running all the tasks simultaneously. This is the foundation of
   **task arithmetic** (Ilharco et al. 2023, *Editing Models with
   Task Arithmetic*, arXiv:2212.04089) and of model-merging methods
   in general (Wortsman et al. 2022 *Model Soups*; Matena & Raffel
   2022 *Fisher-Weighted Averaging*).

2. **The loss is approximately quadratic.** For an MSE-like loss in
   the linearized model, the loss surface around `θ_0` is a
   paraboloid whose Hessian is the **Gauss-Newton matrix**, which
   coincides with the empirical Fisher in classification with
   cross-entropy (Martens 2014, arXiv:1412.1193). This justifies:
   - Diagonal Fisher as a meaningful local curvature.
   - EWC's Laplace-approximation argument (Kirkpatrick et al. 2017
     §2.1): the posterior near `θ*` is Gaussian with precision `F`.
   - Quadratic merge objectives: weight task contributions by their
     Fisher precisions, get the optimal posterior mean.

3. **Merging in parameter space ≈ merging in function space.** Two
   merging operations that produce the same `Δθ` produce
   approximately the same `f`. So we can search for good merge rules
   in parameter space without worrying that a "good" parameter
   merge is a bad function merge.

### 1.2 SLAO's specific use of linearization

SLAO's central claim — that the asymmetric "replace A, average B"
treatment is principled — uses two facts about LoRA in the lazy
regime:

**(i) B is the natural carrier of task delta.** LoRA initializes
`B = 0` so that `BA = 0` at training start. All task-induced change
flows through `B`. In the linearization, the task delta is
`Δθ_t = B_t A_t`, and SLAO observes that across tasks the `B_t`
matrices are approximately orthogonal in their column directions
(SLAO Fig. 2 and §3.2). This means averaging them doesn't cause
destructive interference — the orthogonal components add
constructively, which is the main mechanism behind task arithmetic.

**(ii) A is a basis selector, not a task delta.** A's update during
training mostly *aligns* the row space of `A` with input directions
the task cares about, rather than carrying task-specific magnitude.
SLAO observes high cosine similarity between `A_t` matrices across
tasks in the row space (the *directions* are stable; what changes is
the *coordinates*). So replacing A with `A_ft^t` after task `t`
doesn't lose much: the new A spans approximately the same row
subspace as the merged history of A, just re-coordinated.

The QR continuity in `before_task` (initialize `A_init = Q^T` from
QR of `A_ft^{t-1, T}`) operationalizes this: it picks a specific
orthonormal basis for the previous task's row space and uses that
as the starting point. The new task's training sits in that basis
unless it actively rotates out, which it usually doesn't because the
rotation costs gradient.

### 1.3 Empirical Fisher under linearization

The Fisher in CLUE comes from a specific operational definition that
matters for the analysis. We use the **empirical Fisher**:

```
F_j  =  E_{(x, y) ~ data}  [ (∂L(x, y; θ) / ∂θ_j)² ]
```

where `L` is the task cross-entropy on observed labels. This is what
[`DiagonalFisher._compute_fisher`](methods/fisher.py#L41) computes.

The empirical Fisher differs from the **true Fisher** (the same
thing but with `y` sampled from the model's own predictions) — the
difference is well-known and matters when the model's predictions
are very wrong (Kunstner et al. 2019, arXiv:1905.12558). For
pretrained models being adapted to a related task, the model's
predictions are usually reasonable, so empirical Fisher is a close
proxy for true Fisher.

Under the lazy regime, the empirical Fisher is the diagonal of the
Gauss-Newton matrix — which, for cross-entropy, equals the Hessian
of the loss. So the EWC penalty `Σ F_j (θ_j − θ*_j)²` is a diagonal
Laplace approximation to the negative log posterior around the
previous task's solution.

### 1.4 What CLUE adds to the SLAO analysis

CLUE extends SLAO's framework in one specific way: replacing SLAO's
uniform B-EMA `B_merge ← B_merge + λ(t)(B_ft − B_merge)` with a
per-element rate

```
α_jk  =  λ(t) / (1 + β · F̃_old, jk)
```

where `F̃_old` is the running Fisher of B normalized to mean 1.
The analysis-level argument:

- In SLAO, the merge weight is uniform: every B element is averaged
  at the same rate. This is optimal under a *uniform-precision*
  prior on which B elements matter to old tasks.
- In CLUE, the prior is *non-uniform*: B elements with high Fisher
  on old tasks are treated as having higher precision in their
  posterior, and we move them slowly. B elements with low Fisher
  have low precision — old tasks didn't care about them — so we
  move them at the full rate.
- This is *not* the closed-form Bayesian posterior mean (that would
  be `α = F_new / (F_old + F_new)`, which we tried as
  `merge_B_bayesian` and found marginally helpful). The CLUE form
  is heuristic but has the same qualitative shape: large F_old →
  small `α_jk` → preserve old-task knowledge.

The **"elastic"** terminology is borrowed from EWC: high-Fisher
parameters are *stiff* (small effective merge rate), low-Fisher
parameters are *elastic* (full merge rate). Compared to EWC's
training-time penalty, CLUE applies the elastic principle at *merge
time* on the B factor only, which is consistent with SLAO's claim
that B carries the task-specific information.

### 1.5 What's *not* assumed

To keep the theory honest about its scope:

- **No assumption** that the Fisher diagonal is a good approximation
  to the full Fisher matrix. We use the diagonal because it's
  cheap, but the off-diagonal correlations (between B elements at
  the same column, or between A and B) do matter and we ignore them.
- **No assumption** that all tasks lie in the same NTK ball. If
  later tasks are fundamentally different from earlier ones, the
  linearization breaks and additivity of task vectors becomes a
  poor approximation. This is a real failure mode (the SLAO paper's
  failure cases involve tasks from very different distributions).
- **No assumption** of strict orthogonality of B task vectors. SLAO
  observes *approximate* orthogonality empirically. Where this
  fails (e.g., two tasks that actually require the same B
  directions), the EMA merge will compromise between them, which is
  fine — it's exactly the "soft" averaging behavior we want.

### 1.6 Avenues to extend the SLAO analysis

These are concrete extensions of SLAO's theoretical work that would
strengthen CLUE specifically:

1. **Quantify the linearization error.** SLAO assumes lazy training
   throughout. For our settings (rank 8, 1 epoch, LR 1e-4), measure
   `‖f(x; θ_0 + ΔW) − f(x; θ_0) − J ΔW‖` empirically across tasks.
   If this error is small relative to the loss, the analysis is
   solid; if it's large, we're outside NTK and the merge formulas
   need re-derivation.
2. **Cross-task Jacobian alignment.** Linearity-of-merge requires
   that `J(x)` doesn't shift much across tasks. Measure
   `J_t(x) - J_0(x)` for representative inputs after each task.
   Should be small if SLAO's framing is correct.
3. **B-orthogonality formalized.** SLAO's Fig. 2 is qualitative.
   Compute pairwise cosine similarity of `B_t` matrices in our
   runs. Are they really near-orthogonal? Does CLUE's Fisher merge
   improve more on near-orthogonal pairs (low destructive
   interference) than on near-parallel ones (high)?
4. **Generalize beyond cross-entropy.** The Fisher = Gauss-Newton
   identity holds for cross-entropy with softmax outputs. For
   regression heads (e.g. OpenVLA's continuous action prediction)
   the Fisher is different. Re-derive the merge rule there. This
   is a real risk for the VLA work.

---

## 2. The method

### 2.1 Setup and notation

Let `W_0 ∈ R^{d_out × d_in}` be a frozen pretrained linear layer.
LoRA adds a low-rank update `ΔW = B A` with `A ∈ R^{r × d_in}` and
`B ∈ R^{d_out × r}`. We do continual learning on tasks `t = 1, ..., T`.

Per-task notation:
- `(A_ft^t, B_ft^t)`: fine-tuned LoRA weights *after* training on
  task `t`.
- `(A_merge^t, B_merge^t)`: the merged LoRA weights *carried forward*
  from tasks `1..t`. These are what's used for inference on past
  tasks at task time `t`.

CLUE keeps **only the merged state** between tasks. There is no
per-task adapter to swap in at inference. Memory is `O((m+n)r)`
regardless of `T`.

### 2.2 Per-task lifecycle

The implementation lives in [methods/slao.py](methods/slao.py)
(`SLAO.before_task`, `SLAO.after_task`) plus the loop wiring in
[train.py](train.py) around `train_one_task`.

**Phase A — Initialization (`before_task`):**

For task `t = 1`, use PEFT default LoRA init: A is Kaiming-uniform,
B is zero. So `B A = 0` initially. Train normally.

For task `t > 1`, initialize from the previous task's fine-tuned
state via:

```
A_init  =  Q^T,  where  A_ft^{t-1, T} = Q R   (QR with sign correction)
B_init  =  B_ft^{t-1}                          (copy as-is)
```

Implemented in `orthogonal_init_A` at
[models/lora.py:77](models/lora.py#L77):

```python
def orthogonal_init_A(A_prev: torch.Tensor) -> torch.Tensor:
    """A_init has orthonormal rows: A_init A_init^T = I_r."""
    Q, R = torch.linalg.qr(A_prev.T)            # (d, r), (r, r)
    sign = torch.sign(torch.diag(R))
    sign[sign == 0] = 1.0
    Q = Q * sign.unsqueeze(0)
    return Q.T                                  # (r, d)
```

The QR-of-`A_ft^{t-1, T}` extracts an orthonormal basis for the row
space of the previous A. We start the new task in **the same row
subspace** as the old A but with rows orthogonalized so that updates
are well-conditioned.

**Phase B — Training:**

Standard LoRA fine-tuning. Optimizer is AdamW with `lr = 1e-4`,
`batch_size = 8`, `epochs = 1` for the LLM CL benchmark. No special
loss term unless EWC is active (see §2.4).

Trainable parameters returned by
[`SLAO.get_trainable_params`](methods/slao.py#L76) are exactly the
LoRA `A` and `B` parameters; the base model is frozen.

**Phase C — Merge (`after_task`):**

Snapshot `(A_ft^t, B_ft^t)`. Then update the merged state:

```
A_merge^t  =  A_ft^t                                   (direct replacement)
B_merge^t  =  B_merge^{t-1}  +  α_t ⊙ (B_ft^t − B_merge^{t-1})
```

The **A side is replaced wholesale** — justified by §1.2(ii) above.
The **B side is averaged** elementwise toward `B_ft^t` at per-element
rate `α_t`, justified by §1.2(i) and §1.4.

In **vanilla SLAO**, `α_t` is a uniform scalar `λ(t) = 1/√t` — a
time-aware EMA whose decay schedule is borrowed from non-stationary
stochastic-approximation theory (the natural step size for tracking
a slowly drifting target).

In **CLUE**, `α_t` becomes per-element:

```
α_t,jk  =  λ(t) / (1 + β · F̃_jk)
```

with `F̃ = F_old / mean(F_old)` (per-layer normalization to make
`β` scale-free across layers and runs). This is `merge_B` at
[models/lora.py:116](models/lora.py#L116):

```python
def merge_B(B_merge, B_ft, task_idx, fisher_B=None, beta=0.0):
    lam = 1.0 / math.sqrt(task_idx)
    if fisher_B is None or beta == 0:
        return B_merge + lam * (B_ft - B_merge)
    f_mean = fisher_B.mean()
    F_norm = fisher_B / f_mean if f_mean > 0 else fisher_B
    alpha = lam / (1.0 + beta * F_norm)
    return B_merge + alpha * (B_ft - B_merge)
```

### 2.3 Asymmetric A and B (the SLAO insight, restated)

A is replaced; B is merged. The justification is in §1.2:

1. B-zero init means B carries the task delta in the linearization.
2. A is a basis selector — its row space transfers across tasks
   even when its specific coordinates do not.
3. LoRA+ (Hayou et al. 2024) shows B should have a higher learning
   rate than A, consistent with B carrying the bulk of task-specific
   information.

So the asymmetric treatment is not arbitrary. It's the merge
operation that respects the role of each factor.

### 2.4 EWC: where it lives

Optionally, CLUE adds an EWC penalty *during training* (Kirkpatrick
et al., 2017). Implemented in [methods/fisher.py](methods/fisher.py).

- `DiagonalFisher.estimate(loader, n_samples)` computes the
  empirical diagonal Fisher `F_j = E[(∂L/∂θ_j)²]` over `n_samples`
  examples. This is `_compute_fisher` at
  [methods/fisher.py:41](methods/fisher.py#L41). See §1.3 for what
  this is and what assumptions it requires.
- `accumulate(F_new)` does `F ← γ·F + F_new`. With `γ = 1.0` we get
  vanilla EWC (sum of per-task Fishers). With `γ < 1.0` we get
  Online EWC (Schwarz et al. 2018) — Fisher mass stays bounded as
  `T` grows.
- `snapshot_ref_params()` saves the current LoRA weights as the
  reference point `θ*` for the next task's penalty. We snapshot
  *after* the merge so the reference is the merged state.
- `penalty()` returns `Σ_j F_j (θ_j − θ*_j)²`. The training loop in
  [train.py:236](train.py#L236) adds `(λ_F / 2) · penalty()` to the
  loss.

**Important detail:** `penalty()` iterates over *all* LoRA
parameters (both A and B) without filtering. So **EWC operates on A
and B symmetrically**, while the **Fisher-weighted merge operates on
B only**. These are two distinct uses of the same Fisher quantity at
different points in the pipeline.

In the headline `bench_fmerge05` config, **EWC is off** (`λ_F = 0`).
Only the Fisher-weighted merge is active. The reasoning, supported
by ablations (§4): EWC during training penalizes drift, but SLAO's
merge already does drift control via the B-EMA. The merge operates
in the right regime (post-hoc, on the right factor); EWC penalties
are redundant on top.

### 2.5 Why CLUE beats SLAO

The named contribution of CLUE over vanilla SLAO is the single
hyperparameter `fisher_merge_beta` in [merge_B](models/lora.py#L116),
acting through

```
α_t,jk  =  λ(t) / (1 + β F̃_jk)
```

This turns SLAO's *uniform* B-EMA into an *importance-weighted* B-EMA.
The asymmetric A-B treatment is preserved; only the per-element
scaling of B is added.

Headline numbers (averaged across 9 runs: 3 task orders × 3 seeds,
LLM CL benchmark, `fisher_merge_beta = 0.5`,
`lora_target_modules = q_proj v_proj`, `lora_rank = 8`,
`batch_size = 8`, `lr = 1e-4`, 1 epoch per task):

| Metric | Mean | Notes |
|---|---|---|
| Average Accuracy (AA) | **80.3%** | vs SLAO ~74-76% on same protocol |
| Backward Transfer (BWT) | **+1.0%** | vs SLAO ~−3 to −5%; positive BWT is rare |

Per-order results in [outputs/bench_fmerge05_*/results.json](outputs/).

---

## 3. Analysis frame

This section is what you'd want to expand for a paper — analyses
that would *explain* the empirical advantage, with pointers to the
relevant literature.

### 3.1 Spectral view of `B A`

The product `ΔW = B A` is rank-`r` with thin SVD
`ΔW = U Σ V^T`, where `U ∈ R^{d_out × r}` and `V ∈ R^{d_in × r}` lie
on Stiefel manifolds and `Σ ∈ R^r` is the singular-value vector. The
parameterization `(A, B)` is *gauge-redundant*: for any
`G ∈ GL(r)`, `(A, B) ↦ (G A, B G^{-1})` leaves the product unchanged.
There are `r²` redundant degrees of freedom.

This has two consequences:

1. **The Fisher on raw `B` is gauge-dependent.** If task `t` and
   task `t+1` happen to converge to representations of the *same*
   ΔW under different gauges, their B Fishers differ by a
   `G`-dependent factor. SLAO's QR initialization keeps the gauge
   approximately fixed across tasks (§1.2 above), but only
   approximately. This is one source of variance in the Fisher merge.

2. **Σ is the gauge-invariant content.** The `r` singular values
   are a coordinate-free description of ΔW's "size" along each
   principal direction. Merging in `(A, B)`-space approximately
   equals merging in `(U, Σ, V)`-space, but with a less clean link
   to the actual update geometry.

The Stiefelized CLUE variant in
[methods/stiefel_clue.py](methods/stiefel_clue.py) operationalizes
(2): decomposes the merged ΔW into SVD at task boundaries and
merges Σ with Fisher weighting (Fisher on Σ has only `r` scalars
per layer, ~500× lower per-element variance than Fisher on B, which
has `d_out × r`). It empirically did *not* beat plain CLUE in our
pilot runs — see §4 for the detailed analysis.

**Analyses to do here:**

- **Singular-value spectrum across tasks.** Plot
  `σ_1(ΔW^t), ..., σ_r(ΔW^t)` per task per layer. Does the
  spectrum evolve smoothly under SLAO/CLUE merging? Does merging
  preferentially shrink small σ's (good — pruning irrelevant
  directions) or large ones (bad — destroying old-task capacity)?
- **Rank saturation.** When does `σ_r/σ_1` collapse below noise?
  That's when the rank-`r` budget is exhausted and the method needs
  rank growth.
- **Subspace alignment** via principal angles
  `Θ(U^t, U^{t+1})`. Tells us whether successive merged column
  spaces drift smoothly (SLAO's prediction) or jump.

### 3.2 Why QR init keeps the gauge close (and ZCA gets closer)

QR with sign correction places A in the same row-subspace as the
previous task's A, with orthonormal rows so training is
well-conditioned. With B inherited unchanged and A re-orthogonalized,
the gauge `G` between task `t` and `t-1` is *close to identity* —
small enough that Fisher accumulation across tasks is meaningful.

ZCA whitening (`zca_whiten_A` at
[models/lora.py:94](models/lora.py#L94)) is the orthogonal Procrustes
solution: the closest orthonormal-row matrix to `A_ft^{t-1}` in
Frobenius norm. ZCA is gauge-closer than QR (no Gram-Schmidt-induced
column ordering), which should improve the Fisher-merge
approximation. The O4dev pilot showed ZCA matches QR but doesn't
strictly improve — see §4.

References:
- Higham, *Functions of Matrices* (2008), Ch. 8 — polar decomposition
  and orthogonal Procrustes.
- Vandereycken (SIOPT 2013) — fixed-rank manifold geometry.

### 3.3 The Fisher merge as approximate Bayesian posterior

Treat each task as observing a Gaussian likelihood on B with mean
`B_ft^t` and precision `F_t`. Stacking tasks 1..T gives a posterior
with precision `Σ_t F_t` and mean
`(Σ_t F_t)^{-1} Σ_t F_t B_ft^t`. The standard Bayesian update at
task `t`:

```
α_t  =  F_t / (F_{1..t-1} + F_t)
B_t  =  B_{t-1} + α_t · (B_ft^t − B_{t-1})
```

CLUE's `α = λ(t) / (1 + β F̃_old)` is *not* this exact Bayesian
update, but has the same qualitative shape: small `α` where old
Fisher is large. The differences:

1. CLUE uses only `F_old` in the denominator. The Bayesian variant
   `merge_B_bayesian` ([models/lora.py:157](models/lora.py#L157))
   uses both, and on O4dev it matched but didn't exceed CLUE —
   likely because per-task `F_new` is much noisier than accumulated
   `F_old` (variance reduction by averaging) and using it adds noise
   to `α`.
2. CLUE adds a tunable `β` and a `λ(t) = 1/√t` time factor that the
   strict Bayesian update doesn't. The time factor is the
   non-stationary stochastic-approximation step size — it keeps the
   merge rate bounded as `T` grows so that late-task knowledge
   doesn't over-dominate.

This connects CLUE to:
- **Online Bayesian deep learning / Laplace approximation**
  (MacKay 1992; Ritter, Botev, Barber, *Online Structured Laplace
  Approximations for Overcoming Catastrophic Forgetting*, NeurIPS
  2018, arXiv:1805.07810).
- **Fisher-weighted model averaging** (Matena & Raffel, *Merging
  Models with Fisher-Weighted Averaging*, NeurIPS 2022,
  arXiv:2111.09832) — the closest direct comparison; their
  averaging is the exact precision-weighted mean we approximate.
- **EWC as Laplace approximation** (Kirkpatrick et al. 2017 §2.1).
  Same Fisher, different application: theirs adds a quadratic
  penalty during training; ours weights the merge step.

### 3.4 Statistical-learning view

CLUE's merge can be derived as the closed-form solution to a
weighted least-squares problem at task `t`:

```
B_merge^t  =  argmin_B   F_old · ‖B − B_merge^{t-1}‖²
                       + λ(t) · ‖B − B_ft^t‖²       (uniform on second term)
```

The first-order optimum is a per-element convex combination
weighted by `F_old` and `λ(t)`. The CLUE form
`α = λ(t) / (1 + β F̃_old)` arises when you normalize the Fisher
contribution to make `β` scale-free.

To formalize this further, two routes:

- **Generalization analysis under task drift.** Show that the
  merged solution satisfies a generalization bound on the union of
  task distributions, with the bound improving as the
  Fisher-weighted merge tightens the constraint on important
  parameters. The Matena & Raffel paper's analysis is a good
  starting point — they bound the deviation between the
  precision-weighted mean and the multi-task optimum.
- **PAC-Bayes view.** Treat the merged adapter as a posterior over
  weights; the KL term against a reasonable prior is bounded by
  the EWC-like quadratic. PAC-Bayes bounds (Catoni 2007;
  Dziugaite & Roy 2017, arXiv:1703.11008) give risk bounds in
  terms of this KL.

### 3.5 Empirical analyses we should run

The numbers are good (AA 80.3%, BWT +1.0%) but we don't yet have a
*mechanistic* story. Concrete analyses (none requiring new training
runs — all are post-hoc on saved adapter states with `--save_adapters`):

1. **Fisher histograms per layer** at each task boundary. Is Fisher
   concentrated (a few high-Fisher columns of B carrying most of
   the load) or diffuse? Concentrated is good for CLUE — it means
   `α_jk` varies a lot across `(j,k)` and the per-element merge
   actually does different things in different places.
2. **Effective merge rate `mean(α_jk)` per task.** With β=0.5 and
   normalized Fisher, mean(α) should still be ≈ λ(t). Verify.
3. **Per-task forgetting attribution.** For each old task, plot
   loss change after task `t` vs. magnitude of B drift on layers
   important for that task. If CLUE works because of the B Fisher
   merge, drift should be small on important layers.
4. **Cosine similarity of A across tasks.** SLAO Fig. 2 claims high
   cosine similarity. Verify on our runs. If not high, the QR
   continuity assumption is broken and our gauge argument needs
   re-examining.
5. **Decompose the BWT delta.** Can we attribute the −3-5% → +1%
   BWT improvement to specific layers? Specific directions in
   B-space?
6. **NTK validation (§1.6 item 1).** Measure the linearization
   error empirically. Should be small if the analysis is sound.

---

## 4. What we tried, and what worked

This section is the institutional memory. Each variant: what it
does, the empirical result on the O4dev pilot (4 tasks: mnli → qqp
→ wic → copa, 1 seed), why it does or doesn't help.

All numbers below are from the [outputs/](outputs/) directory.
Output subdirectories listed for each row.

### Baselines and merge variants (q+v target)

| Config | AA | BWT | Output dir |
|---|---|---|---|
| SLAO baseline (β=0, λ=0) | 0.8115 | −0.0175 | [`slao_O4dev`](outputs/slao_O4dev) |
| Fisher merge β=0.5 (CLUE pick) | 0.8105 | **+0.0045** | [`slao_O4dev_qv_fmerge05`](outputs/slao_O4dev_qv_fmerge05) |
| Fisher merge β=1.0 | 0.7930 | +0.0015 | [`slao_O4dev_qv_fmerge1`](outputs/slao_O4dev_qv_fmerge1) |

The CLUE win on O4dev is BWT going from −0.018 to +0.005 at the
cost of <0.001 AA. Higher β protects more aggressively but hurts AA
substantially.

### Initialization variants

| Config | AA | BWT | Output dir |
|---|---|---|---|
| QR init (default, q+v) | 0.8115 | −0.0175 | [`slao_O4dev`](outputs/slao_O4dev) |
| Bayesian merge with QR init | 0.8161 | −0.0120 | [`slao_O4dev_bayesian`](outputs/slao_O4dev_bayesian) |
| Bayesian merge with ZCA init | 0.8124 | −0.0141 | [`slao_O4dev_bay_zca`](outputs/slao_O4dev_bay_zca) |

We don't have a clean ZCA-only ablation (without Bayesian merge); the
two ZCA runs we have both pair it with Bayesian. Comparing
`bayesian` vs `bay_zca`: ZCA gives slightly worse AA and slightly
worse BWT here, suggesting QR is fine on this short sequence. Worth
re-running with longer sequences before drawing conclusions.

### EWC + merge combinations (q+v target)

| Config | AA | BWT | Output dir |
|---|---|---|---|
| EWC λ=0.1 only (no Fisher merge) | 0.8091 | −0.0155 | [`slao_O4dev_qv_f01`](outputs/slao_O4dev_qv_f01) |
| EWC λ=0.1 + Fisher merge β=0.5 | 0.8126 | +0.0038 | [`slao_O4dev_qv_fmerge05_f01`](outputs/slao_O4dev_qv_fmerge05_f01) |
| Strong EWC λ=0.4 (all-mods, separate config) | 0.7915 | −0.0459 | [`slao_O4dev8_allmods_f03_plus2`](outputs/slao_O4dev8_allmods_f03_plus2) |

EWC alone barely improves on baseline (−0.0155 vs −0.0175 BWT).
EWC + Fisher merge gives essentially the same numbers as Fisher
merge alone (compare 0.8126/+0.0038 vs 0.8105/+0.0045) — the EWC
contribution is within noise. Strong EWC hurts AA without giving a
proportional BWT gain. **Conclusion: Fisher merge does the work,
EWC is redundant on top.**

### All-modules variants (LoRA on every linear layer)

| Config | AA | BWT | Output dir |
|---|---|---|---|
| All-mods, no Fisher anything | 0.8186 | −0.0026 | [`slao_O4dev_allmods`](outputs/slao_O4dev_allmods) |
| All-mods + EWC λ=0.1 (no Fisher merge) | 0.8205 | −0.0005 | [`slao_O4dev_allmods_f01`](outputs/slao_O4dev_allmods_f01) |
| All-mods + EWC λ=0.1 + LoRA+ ratio 4 | 0.7656 | −0.0538 | [`slao_O4dev_allmods_f01_lp4`](outputs/slao_O4dev_allmods_f01_lp4) |

We don't have an all-modules + Fisher merge run. On the
all-modules base, EWC alone already gets BWT close to zero (−0.0005)
without needing the Fisher merge — but adding LoRA+ (ratio 4) breaks
this. **Open question:** would all-modules + Fisher merge β=0.5
(without EWC) match or beat all-modules + EWC? Worth running.

### Other extensions (q+v unless noted)

| Config | AA | BWT | Output dir |
|---|---|---|---|
| Riemannian preconditioning | 0.7542 | −0.0748 | [`slao_O4dev_riem`](outputs/slao_O4dev_riem) — actively hurts |
| GPM threshold 0.95 + EWC λ=0.1 | 0.8010 | −0.0215 | [`slao_O4dev_qv_gpm95_f01`](outputs/slao_O4dev_qv_gpm95_f01) |
| Fisher merge β=0.5 + LoRA+ ratio 2 | 0.8093 | +0.0011 | [`slao_O4dev_qv_fmerge05_lp2`](outputs/slao_O4dev_qv_fmerge05_lp2) |
| Rank 16 (vs 8), all-mods | 0.7968 | −0.0381 | [`slao_O4dev_allmods_rank16`](outputs/slao_O4dev_allmods_rank16) |

Riemannian preconditioning was a clear loss. GPM and LoRA+ added
nothing useful. Rank 16 with no compensating tuning was worse —
expected, since each task converges to a higher-rank solution that's
harder to merge.

### Stiefelized variants

The Stiefelized CLUE in [methods/stiefel_clue.py](methods/stiefel_clue.py)
reparameterizes the update as thin SVD `(U, Σ, V)` and merges in
tangent space. Motivation in [STIEFEL.md](STIEFEL.md): Fisher on Σ
has only `r` scalars per layer (vs `d_out × r` on B), giving ~500×
variance reduction.

| Config | AA | BWT | Output dir |
|---|---|---|---|
| Stiefel + Bayesian merge (initial) | NaN | NaN | [`slao_O4dev_stiefel_bay`](outputs/slao_O4dev_stiefel_bay) — failed |
| Stiefel + Bayesian merge (after fix) | 0.8084 | −0.0166 | [`slao_O4dev_stiefel_bay_fix2`](outputs/slao_O4dev_stiefel_bay_fix2) |
| Stiefel + Bayesian merge (final fix) | 0.8064 | −0.0198 | [`slao_O4dev_stiefel_bay_fix3`](outputs/slao_O4dev_stiefel_bay_fix3) |

The first two attempts (`stiefel_bay`, `stiefel_bay_fix`) NaN'd due
to retraction bugs (eigh on degenerate matrices, SVD gauge drift —
documented in commit history). The fixes recovered training but
matched, not beat, plain CLUE. The gauge ambiguity Stiefel
eliminates is too small to matter on QR-initialized 4-task runs.

### Bench averages (the headline number)

`bench_fmerge05_*` runs are q+v with `fisher_merge_beta=0.5`,
3 task orders × 3 seeds. Verified:

| Set | N | AA mean ± std | BWT mean ± std |
|---|---|---|---|
| All 9 runs | 9 | **0.8028 ± 0.0095** | **+0.0103 ± 0.0145** |
| O4 only | 3 | 0.7978 | +0.0043 |
| O5 only | 3 | 0.8069 | +0.0201 |
| O6 only | 3 | 0.8036 | +0.0066 |

Per-run files: [`outputs/bench_fmerge05_*/results.json`](outputs/).

### What the empirical search converged to

Takeaway: **the Fisher merge does the work, EWC adds little**. Our
ablation is consistent with the SLAO paper's claim that asymmetric
merge alone suffices. CLUE extends this by importance-weighting B
without needing the EWC penalty during training.

### What the empirical search converged to

```
method                  = slao
lora_rank               = 8
lora_alpha              = 16
lora_target_modules     = q_proj v_proj   (or all-linear for higher AA)
optimizer               = adamw
lr                      = 1e-4
batch_size              = 8
grad_accum              = 1
epochs                  = 1
fisher_merge_beta       = 0.5             ← the one CLUE knob
fisher_lambda           = 0.0             ← EWC off
fisher_gamma            = 1.0             ← unused at λ=0
fisher_samples          = 256
a_init_method           = qr              ← SLAO default
bayesian_merge          = false
samples_per_class_train = 1000
```

The headline run (`bench_fmerge05_O4_sXX`) used q+v targeting; the
all-modules variant gives ~+1% AA but is a separate ablation.

---

## 5. Directions

In roughly decreasing payoff and decreasing effort:

### 5.1 Vision-Language-Action (VLA) generalization

Clearest test of CLUE's generality. Pipeline scaffolded in
[vla/](vla/) — see [vla_howto.md](vla_howto.md) for the next pod.
Stakes:

- If CLUE's BWT gains transfer to OpenVLA on LIBERO, that's a
  Demo-Day-quality result (no published method clearly beats SeqFT
  on LIBERO CL).
- If they don't, that's also informative: the Fisher-merge formula
  may depend on properties specific to text classification heads,
  and we'd need to re-derive for action heads (§1.6 item 4).

### 5.2 Mechanistic analysis (§3.5)

Run Fisher histograms, spectrum plots, per-layer drift attribution,
and NTK validation on the existing 9-run benchmark. None require
new training runs. Goal: turn empirical AA/BWT into a paper-worthy
story about *why* CLUE works.

### 5.3 Rank growth diagnostic

Stiefel parameterization gives a principled saturation signal:
when `σ_r / σ_1` is below noise after merge, rank-`r` budget is
exhausted. Implementing rank growth (start `r=4`, grow to `r=16`):

1. Track `σ_min / σ_max` per layer per task.
2. When it drops below threshold, append a column to B and a row
   to A, both Kaiming-initialized at small magnitude.
3. Continue training; the new direction takes up slack.

A few hundred lines in [methods/stiefel_clue.py](methods/stiefel_clue.py)
and the [models/lora.py](models/lora.py) merge path.

### 5.4 Theoretical analysis

Two specific items that would strengthen the paper:

1. **Generalization bound.** Use Matena & Raffel's Fisher-weighted
   averaging as comparison. Show the CLUE form
   `α = λ(t) / (1 + β F̃)` gives a competitive bound under
   reasonable assumptions.
2. **Why `λ(t) = 1/√t`?** SLAO uses it without strong justification.
   Compare to optimal step size in stochastic approximation
   (Robbins-Monro: `1/t` for stationary; `1/√t` for non-stationary
   bandits / online convex optimization). Likely best step under
   our drifting-Fisher model.

### 5.5 Bayesian merge done right (✓ implemented)

The previous Bayesian merge failed on the first task because there
was no `F_old` to anchor the rate, and the code set α=1 (clamped to
α_max=0.95) which over-merged toward the new task. **Fixed** in
[`merge_B_bayesian`](models/lora.py#L157): when `fisher_old is None`,
use a uniform prior `F_old ≡ prior_strength` so

```
α_jk = F̃_new_jk / (prior_strength + F̃_new_jk + ε)
```

With `prior_strength=1.0` (default), the average α equals 0.5, close
to SLAO's `λ(2) ≈ 0.71`. High-Fisher elements still merge faster
than low-Fisher ones from the first task, which is the whole point.
Configurable via `--bayesian_prior_strength` in train.py. Ablation
remains to be done.

### 5.6 Compositional task structure

If task `t+1` is "close" to a previous task, Fisher should reflect
that and the merge should preserve old knowledge essentially
perfectly. If far, the merge should accept controlled forgetting.
We don't have a per-task-pair similarity analysis. Adding one
(cosine similarity of fine-tuned B_ft across tasks) would let us
predict per-pair forgetting and verify CLUE's behavior matches.

### 5.7 Better Fisher

Switch from empirical Fisher to true Fisher:

```python
loss = -model(input_ids).log_softmax(-1).gather(-1, sampled_y).mean()
loss.backward()
```

Theoretically cleaner (§1.3) and is what EWC originally used.
Whether it improves results is empirical. Worth a half-day run.

### 5.8 NTK validation

Test the assumption from §1 directly:

1. Pick a representative input `x`.
2. Compute `f(x; θ_0)`, `f(x; θ_0 + ΔW^t)`, and
   `f(x; θ_0) + J(x; θ_0) ΔW^t` for each task.
3. Plot `‖f − (f_0 + J ΔW)‖ / ‖f_0‖` per task.

If small (≪ 1), our analysis is valid. If large, we're in feature
learning territory and merge formulas need new derivation.
Cheap to run on existing checkpoints.

---

## 6. References

- **SLAO**: Qiao & Mahdavi, *Merge before Forget: A Single LoRA
  Continual Learning via Continual Merging*, 2025.
  [arXiv:2512.23017](https://arxiv.org/abs/2512.23017).
- **EWC**: Kirkpatrick, Pascanu, Rabinowitz et al., *Overcoming
  catastrophic forgetting in neural networks*, PNAS 2017.
  [arXiv:1612.00796](https://arxiv.org/abs/1612.00796).
- **Online EWC**: Schwarz, Czarnecki, Luketina et al., *Progress &
  Compress: A scalable framework for continual learning*, ICML 2018.
  [arXiv:1805.06370](https://arxiv.org/abs/1805.06370).
- **NTK / lazy training**: Jacot, Gabriel, Hongler, *Neural Tangent
  Kernel: Convergence and Generalization in Neural Networks*,
  NeurIPS 2018. [arXiv:1806.07572](https://arxiv.org/abs/1806.07572).
  Lee et al., *Wide Neural Networks of Any Depth Evolve as Linear
  Models Under Gradient Descent*, NeurIPS 2019.
  [arXiv:1902.06720](https://arxiv.org/abs/1902.06720).
- **Intrinsic dimensionality of fine-tuning**: Aghajanyan,
  Zettlemoyer, Gupta, *Intrinsic Dimensionality Explains the
  Effectiveness of Language Model Fine-Tuning*, 2020.
  [arXiv:2012.13255](https://arxiv.org/abs/2012.13255). Justifies
  why LoRA works at all and why low rank is enough.
- **Task arithmetic / model merging**: Ilharco et al., *Editing
  Models with Task Arithmetic*, ICLR 2023.
  [arXiv:2212.04089](https://arxiv.org/abs/2212.04089).
  Wortsman et al., *Model Soups*, ICML 2022,
  [arXiv:2203.05482](https://arxiv.org/abs/2203.05482).
- **LoRA**: Hu, Shen, Wallis et al., *LoRA: Low-Rank Adaptation of
  Large Language Models*, 2021.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). The original.
- **LoRA+**: Hayou, Ghosh, Yu, *LoRA+: Efficient Low Rank Adaptation
  of Large Models*, 2024.
  [arXiv:2402.12354](https://arxiv.org/abs/2402.12354). Justifies
  asymmetric A/B treatment via different optimal LRs.
- **Fisher-Weighted Averaging**: Matena & Raffel, *Merging Models
  with Fisher-Weighted Averaging*, NeurIPS 2022.
  [arXiv:2111.09832](https://arxiv.org/abs/2111.09832). Closest
  comparison to CLUE's merge formula.
- **Empirical vs true Fisher**: Kunstner, Balles, Hennig, *Limitations
  of the Empirical Fisher Approximation for Natural Gradient
  Descent*, NeurIPS 2019.
  [arXiv:1905.12558](https://arxiv.org/abs/1905.12558).
- **Gauss-Newton ≡ Fisher for cross-entropy**: Martens, *New
  insights and perspectives on the natural gradient method*, 2014.
  [arXiv:1412.1193](https://arxiv.org/abs/1412.1193).
- **Online Bayesian / Laplace deep learning**: Ritter, Botev, Barber,
  *Online Structured Laplace Approximations for Overcoming
  Catastrophic Forgetting*, NeurIPS 2018.
  [arXiv:1805.07810](https://arxiv.org/abs/1805.07810).
- **Riemannian preconditioning for LoRA**: Prakash, Zhao et al.,
  *Riemannian Preconditioned LoRA for Fine-Tuning Foundation
  Models*, 2024.
  [arXiv:2402.02347](https://arxiv.org/abs/2402.02347).
- **Riemannion (manifold-aware optimizer)**: Bogachev et al., 2025.
  [arXiv:2507.12142](https://arxiv.org/abs/2507.12142). Used in
  Stiefelized variant.
- **Fixed-rank manifold geometry**: Vandereycken, *Low-rank matrix
  completion by Riemannian optimization*, SIOPT 2013.
- **Polar decomposition / orthogonal Procrustes**: Higham,
  *Functions of Matrices*, SIAM 2008, Ch. 8. ZCA whitening
  background.
- **PAC-Bayes**: Catoni, *PAC-Bayesian Supervised Classification*,
  IMS 2007. Dziugaite & Roy, *Computing Nonvacuous Generalization
  Bounds*, UAI 2017.
  [arXiv:1703.11008](https://arxiv.org/abs/1703.11008).
- **O-LoRA / orthogonal subspace CL**: Wang, Liu et al.,
  *Orthogonal Subspace Learning for Language Model Continual
  Learning*, 2023.
  [arXiv:2310.14152](https://arxiv.org/abs/2310.14152).
- **GPM**: Saha, Garg, Roy, *Gradient Projection Memory for
  Continual Learning*, ICLR 2021.
  [arXiv:2103.09762](https://arxiv.org/abs/2103.09762).
