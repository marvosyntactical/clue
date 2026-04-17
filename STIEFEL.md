# Stiefelized CLUE

Reparameterize the LoRA update on the fixed-rank matrix manifold $\mathcal{M}_r$,
train with Riemannion (Bogachev et al., 2025, arXiv:2507.12142), and perform
CL merging in the tangent space at the previous task's merged point. This
eliminates the (A, B) gauge ambiguity that CLUE currently papers over, and
— more importantly — changes the quantities that matter (singular directions
and singular values) into objects where Fisher-weighted merging has *much*
lower estimation variance than on raw B.

---

## 1. Motivation

CLUE currently operates on (A, B) with two known problems:

1. **Gauge ambiguity.** $(A, B) \sim (GA, BG^{-1})$ for any $G \in GL(r)$ — so
   the Fisher on B is computed in whatever gauge the previous task's
   optimization happened to land in. F_old, accumulated across many tasks,
   mixes gauges. Fisher-weighted merging is approximately correct because
   QR-continuity keeps gauges close, but this is a fragile approximation.

2. **B_init inheritance.** Setting $B_{\text{init}} = B_{\text{ft}, i-1}$
   means $B_{\text{ft}, i}$ inherits old structure, so $F_{\text{new}}$
   correlates with $F_{\text{old}}$, collapsing the Bayesian merge rate
   toward uniform. This is almost certainly why Bayesian merging on
   (A, B) gave only marginal gains.

The fixed-rank manifold parameterization addresses both: there is no gauge
freedom, and the natural tangent-space parameterization starts each new task
at $\xi = 0$ with no inherited structure.

---

## 2. Setup

Represent the update $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
as a point on

$$\mathcal{M}_r = \{\Delta W : \text{rank}(\Delta W) = r\}$$

via the thin SVD parameterization

$$\Delta W = U \Sigma V^T, \quad U \in \text{St}(d_{\text{out}}, r), \; V \in \text{St}(d_{\text{in}}, r), \; \Sigma \in \mathbb{R}^r_{>0}$$

Residual gauge: sign flips on $(u_k, v_k)$ pairs when all $\sigma_k$ distinct.
Generically discrete — essentially gauge-free.

**Training**: use Riemannion (Muon on $\mathcal{M}_r$) — Euclidean gradients
projected onto the tangent space at the current point, orthogonalized via
Newton-Schulz or polar retraction. Per Bogachev et al., per-iteration cost
is $O((d_{\text{in}} + d_{\text{out}}) r^2 + r^3)$ with 2.5%–32.5% wall-clock
overhead over Adam on (A, B), scaling with $r$.

---

## 3. Continual learning: tangent-space merging

After task $i$, store the merged point $P_i = (U_i, \Sigma_i, V_i) \in \mathcal{M}_r$.

**Init for task $i+1$**: parameterize the new task's update as a perturbation
of $P_i$ in the tangent space:

$$\Delta W_{i+1}(\xi) = \text{Retr}_{P_i}(\xi), \quad \xi \in T_{P_i} \mathcal{M}_r$$

where $\xi$ is the trainable parameter initialized at $\xi = 0$. Train with
Riemannion on $\xi$. The tangent space decomposes as

$$\xi = (\dot{U}, \dot{\Sigma}, \dot{V}), \quad U_i^T \dot{U} + \dot{U}^T U_i = 0, \; V_i^T \dot{V} + \dot{V}^T V_i = 0$$

so $\dot{U}, \dot{V}$ live in Stiefel tangent spaces (skew + horizontal
components; no manifold constraint on $\xi$ itself — just linear
constraints). $\dot{\Sigma} \in \mathbb{R}^r$ is unconstrained.

**Merging** is now an operation on a vector space (the tangent space at
$P_i$), which is what makes this clean:

1. Train to obtain $\xi_{i+1}^\star = (\dot{U}_{\text{ft}}, \dot{\Sigma}_{\text{ft}}, \dot{V}_{\text{ft}})$.
2. Estimate Fisher on $\xi$ components on task $i+1$ data: $F_{\text{new}}$
   decomposes as $F^U_{\text{new}} \in \mathbb{R}^{d_{\text{out}} \times r}$,
   $F^\Sigma_{\text{new}} \in \mathbb{R}^r$, $F^V_{\text{new}} \in \mathbb{R}^{d_{\text{in}} \times r}$.
3. Apply Bayesian merge **component-wise** in tangent space:

$$\xi_{\text{merge}} = \frac{F_{\text{new}}}{F_{\text{old}} + F_{\text{new}} + \epsilon} \odot \xi_{i+1}^\star$$

(with standard normalization and clipping from the Bayesian merge spec).
Since $\xi_{\text{prev}}$ in this parameterization is $0$ by construction,
the merge simplifies to a scaling.

4. Retract: $P_{i+1} = \text{Retr}_{P_i}(\xi_{\text{merge}})$.
5. Re-anchor tangent space at $P_{i+1}$ for the next task.

**No alignment needed.** Column $k$ of $\dot{V}$ perturbs column $k$ of $V_i$
by construction. Correspondence is built into the parameterization.

---

## 4. Why this should give *better* numbers, not just cleaner theory

Theoretical cleanness is a weak selling point. Stronger claims:

### 4.1 Fisher estimation variance drops ~500×

In CLUE, Fisher is estimated per element of $B$: ~$d_{\text{out}} \cdot r \approx 4096 \cdot 8 = 32{,}768$
scalars per layer, from 256 samples. That's ~0.008 samples per parameter — an
extremely noisy estimate, fundamentally limited by sample count.

In the Stiefelized version, the most important Fisher is on $\Sigma$: **$r = 8$
scalars per layer**, from the same 256 samples. That's 32 samples per
parameter — a ~500× reduction in per-scalar variance. Fisher-weighted
merging on $\Sigma$ is in a *qualitatively different regime* from Fisher on B.

This matters because your current Bayesian merge gave only marginal gains
— and the most plausible reason is Fisher estimation noise swamping the
signal. The Stiefelized parameterization isolates the load-bearing quantity
($\Sigma$ per direction) from the high-dimensional noise ($B$ per element).

**Empirical prediction.** Fisher-weighted merging on $\Sigma$ should outperform
CLUE's Bayesian merge on B by a larger margin than CLUE beats uniform.

### 4.2 No B_init inheritance problem

$\xi$ starts at 0 at the start of each task. There is no inherited structure
from the previous task's Fisher — $F_{\text{new}}$ and $F_{\text{old}}$
measure genuinely different things (tangent-space importance for the new
task vs. accumulated importance for past tasks). The Bayesian merge
finally has signal to work with.

### 4.3 Operational simplification vs. SLAO

SLAO requires, per task boundary:
- QR of $A_{\text{ft}, i-1}^T$ with sign correction.
- Reinit A to $Q^T$, B to $B_{\text{ft}, i-1}$ (or corrected $B_{\text{ft}, i-1} R^T$).
- Asymmetric merge: A replaced, B EMA'd.

Stiefelized CLUE, per task boundary:
- Compute $\xi_{\text{merge}}$ from Fisher-weighted tangent vector.
- Retract to get new base point.
- Reset $\xi = 0$ for next task.

The latter is fewer moving parts, and crucially **no asymmetric treatment
of read vs. write factors** — U and V are handled symmetrically by the
geometry, which is structurally honest (the asymmetry in SLAO was a
workaround for the gauge problem, not a feature).

### 4.4 Principled rank growth (future work but defensible today)

The accumulated tangent vector norm $\|\xi_{\text{merge}}\|$ and the ratio
$\sigma_r / \sigma_1$ of the merged point give a **quantitative saturation
diagnostic**: when $\sigma_r$ is no longer meaningfully smaller than the
truncated residual of the combined update, the rank-$r$ budget is exhausted.
This gives a principled trigger for rank growth ($r \to r+1$), which SLAO
has no natural equivalent of.

For Demo Day this is a "future work" talking point — but it's a *real*
one, not hand-waving, because the diagnostic is well-defined.

### 4.5 Inherits Riemannion's single-task gains

Bogachev et al. report consistent per-task improvements over standard LoRA
and Muon-on-(A,B). If those gains persist under CL (no obvious reason they
wouldn't for task 1), they **compound** with the merging improvements.
Each task's $\xi_{\text{ft}}^\star$ is a better fit to that task's data,
so the merged model is built from higher-quality components.

---

## 5. Cons vs. SLAO (honest)

### 5.1 Implementation complexity

Riemannion is real engineering. Tangent-space projections, retractions,
Newton-Schulz orthogonalization, vector transport between task base points.
Bogachev et al. have a reference implementation; adapting it for CL
(re-anchoring, tangent-space Fisher, merged retractions) is a week of work
minimum. SLAO is ~200 lines of Python.

### 5.2 Runtime overhead

2.5%–32.5% per Riemannion paper at ranks 4–16. At $r = 8$ (your setting),
expect ~10–15%. Absolute cost: ~15–20 extra minutes on a 2h run.
Not prohibitive, not free.

### 5.3 Singular value crossings

When $\sigma_j \to \sigma_k$, the corresponding $(u, v)$ columns become
rotation-ambiguous and continuity of the parameterization breaks locally.
Generic-position assumption is fine for small-LR training but can fail
under aggressive schedules or for certain task pairs. Mitigations
(repulsive barrier on $\sigma$ gap, crossing detection) are possible
but ugly. SLAO has no analog because (A, B) has no ordered structure.

### 5.4 Stability near $\Sigma \approx 0$

At initialization, if $\Sigma$ is small, the Stiefel factors $U, V$ are
ill-conditioned — their gradients can be large and poorly scaled. Need
careful warm-up, or initialize $\Sigma$ at moderate positive values (e.g.,
from a small LoRA pre-training step) rather than zero.

### 5.5 Empirical result uncertain

The strongest argument (§4.1, Fisher variance reduction) is a prediction,
not a measurement. If the gain from reduced Fisher variance is small in
practice — e.g., because the Fisher on B was already good enough — the
Stiefelized method's primary advantage disappears and we're left with a
more complex pipeline for similar numbers.

---

## 6. Scope

**Demo Day (~3 weeks):** Too much scope for a from-scratch implementation.
The cleanest Demo-Day-compatible version is a **pilot experiment** on a
single task order (O4) with one seed: implement the Stiefelized
parameterization and tangent-space Bayesian merge, train on 15 tasks,
compare AA/BWT to CLUE. Even a negative result is publishable as an
honest comparison.

**Post-Demo Day paper:** Full implementation with Riemannion, all three
task orders, 3 seeds each, rank growth diagnostic, comparison to CLUE
on same hardware. This is a genuine paper-sized contribution — the
combination (SVD parameterization + Riemannion training + tangent-space
Fisher-weighted merging for CL) is not in the literature as of April 2026.

---

## 7. Experiments

Priority order:

**S1. Pilot (O4, 1 seed, ~3 days to implement + 1 day train).**
Does the full pipeline run and give reasonable AA on task 1 alone?
Sanity check: task-1-only AA should match or beat CLUE task-1-only AA.
If it's worse, something is broken.

**S2. Full CL run (O4, 3 seeds).** Compare to CLUE. Report AA, BWT, and
— critically — **Fisher estimation variance on $\Sigma$ vs. on B** as a
mechanistic check on §4.1's claim.

**S3. All orders (O4, O5, O6, 3 seeds each).** The full 9-run protocol.
Ready for Demo Day or paper.

**S4. Rank growth ablation.** Start at $r = 4$, allow growth; compare to
fixed $r = 8$. Tests the §4.4 claim that growth is principled and helpful.

**S5. Diagnostic: spectrum evolution.** Plot $\sigma_1, \ldots, \sigma_r$
across tasks for a representative layer. Clean spectral structure
(few dominant directions, visible reuse) confirms the parameterization
is exploiting real low-dim structure. Flat spectrum means the manifold
structure isn't helping.

---

## 8. Relation to prior work

- **Riemannion / LoRA meets Riemannion** (Bogachev et al., 2025, arXiv:2507.12142):
  provides the training optimizer. Single-task only; no CL.
- **SLAO** (Qiao & Mahdavi, 2025, arXiv:2512.23017): CL in (A, B) with QR-init
  continuity and asymmetric B merge. The method we're superseding.
- **CLUE** (ours, current): Fisher-weighted B merge on top of SLAO.
  AA 80.3, BWT +1.0 on Large Number of Tasks.
- **Riemannian Preconditioned LoRA** (Prakash et al., 2024): preconditions
  (A, B) gradients; single-task; doesn't change parameterization.
- **GPM / O-LoRA / InfLoRA**: subspace tracking in Euclidean ambient space;
  none reparameterize onto $\mathcal{M}_r$.

**Claimed contribution:** first CL method on the fixed-rank manifold with
tangent-space Fisher-weighted merging. Prior CL-LoRA work operates in
gauge-dependent coordinates; prior manifold-LoRA work (Riemannion) does
single-task only. The synthesis is the contribution.

---

## 9. Failure criteria (kill-the-project triggers)

- **S1 shows task-1 AA below CLUE task-1 AA by more than 2 points.** Implies
  the parameterization is actively harmful for single-task training, not
  just expensive. Abandon.
- **S2 shows CLUE beats Stiefelized CLUE on both AA and BWT.** Implies
  Fisher-variance-reduction argument is wrong in practice. Either adjust
  the story (is it diagnostic-only?) or abandon.
- **Runtime overhead exceeds 50%.** Implies numerical issues requiring many
  retractions per step. Implementation problem, not conceptual — but if
  unfixable in available time, defer.

If S2 is a clean win (e.g., AA ≥ 81, BWT ≥ +1), this supersedes CLUE
as the headline method.