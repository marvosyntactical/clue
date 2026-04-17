# Bayesian Fisher-Weighted B Merging

Extension to CLUE's Fisher-weighted merging (EXTENSIONS.md §5). Replaces the
ad-hoc rate `α = λ / (1 + β F̃_old)` with a posterior-style update that uses
both the **old** accumulated Fisher and the **new** task's Fisher.

---

## Motivation

Current CLUE merge rate uses only F_old:

```
α_jk = λ / (1 + β · F̃_old_jk)
```

This protects parameters old tasks cared about, but ignores how much the
new task actually relies on each parameter. A parameter the new task barely
uses still gets merged at the base rate λ on dimensions where F_old is
small — dragging B_merge toward whatever noisy value SGD left in B_ft_i.

The Bayesian form addresses this by letting **both** Fishers vote.

---

## Derivation

Treat each merge as a Laplace-approximate posterior update. Around the
previous merged point B_merge, model the loss for old tasks as a quadratic
with diagonal precision F_old. After training task i, the loss for task i
around B_ft_i is locally quadratic with precision F_new.

Combining the two diagonal Gaussians (in the natural parameterization)
gives the precision-weighted mean:

```
B_posterior_jk = (F_old_jk · B_merge_jk + F_new_jk · B_ft_jk) / (F_old_jk + F_new_jk)
```

Rewriting in CLUE's update form `B_merge ← B_merge + α · (B_ft − B_merge)`:

```
α_jk = F_new_jk / (F_old_jk + F_new_jk + ε)
```

where ε is a small constant for numerical stability when both Fishers vanish.

---

## Properties

| Condition | α_jk | Behavior |
|-----------|------|----------|
| F_new = F_old | 0.5 | Balanced merge |
| F_new >> F_old | → 1 | Absorb new task's value (it cares more) |
| F_old >> F_new | → 0 | Preserve merged value (old tasks care more) |
| F_new = F_old = 0 | 0 (via ε) | No update on parameters neither task uses |

Notable difference from CLUE's current form: **no β hyperparameter**. The
relative magnitudes of F_new and F_old set the rate directly. This is a
feature — one fewer thing to tune — but means the formula is sensitive to
Fisher estimation noise. See "Stabilization" below.

Also notable: **no λ = 1/√i factor**. The posterior derivation gives the
optimal combination weights directly; λ was a heuristic damping in the
original SLAO/CLUE formulation. Whether to retain λ as an extra dampener
is an empirical question (see ablations).

---

## Stabilization

Three issues with the raw form:

1. **Fisher estimation noise.** With ~256 samples, per-parameter Fisher
   estimates are noisy. Two parameters with true F_new = F_old can have
   estimated α ranging from ~0.2 to ~0.8.

2. **Magnitude drift.** F_new and F_old have different scales (F_old is
   accumulated over i−1 tasks; F_new is one task). Without normalization,
   α is biased toward 0 for late tasks.

3. **Dead parameters.** When both Fishers are tiny but nonzero, the ratio
   is dominated by noise.

**Fixes:**

- Normalize each Fisher per-layer to mean 1 before combining:
  `F̃ = F / mean(F)`. This removes the magnitude drift from accumulation.
- Add a damping floor: `α_jk = max(α_min, F_new / (F_old + F_new + ε))`
  with `α_min ≈ 0.01` to ensure parameters always get *some* update.
- Optional: clip α ∈ [α_min, α_max] with `α_max ≈ 0.95` to prevent
  full replacement on noisy spikes.

Final form:

```python
F_new_norm = F_new / F_new.mean().clamp(min=1e-12)
F_old_norm = F_old / F_old.mean().clamp(min=1e-12)
alpha = F_new_norm / (F_old_norm + F_new_norm + eps)
alpha = alpha.clamp(min=alpha_min, max=alpha_max)
B_merge_new = B_merge + alpha * (B_ft - B_merge)
```

---

## Implementation

**File:** `models/lora.py` — add alongside `merge_B_fisher`.

```python
def merge_B_bayesian(
    B_merge: torch.Tensor,
    B_ft: torch.Tensor,
    fisher_old: torch.Tensor,
    fisher_new: torch.Tensor,
    eps: float = 1e-8,
    alpha_min: float = 0.01,
    alpha_max: float = 0.95,
    use_lambda_damping: bool = False,
    task_idx: int | None = None,
) -> torch.Tensor:
    """Posterior-style merge of B using both old and new Fisher.

    Args:
        B_merge: (d, r) current merged B.
        B_ft:    (d, r) fine-tuned B from current task.
        fisher_old: (d, r) accumulated Fisher from tasks 1..i-1.
        fisher_new: (d, r) Fisher estimated on current task i.
        eps: numerical floor.
        alpha_min, alpha_max: clip range for α.
        use_lambda_damping: if True, multiply α by 1/√i (ablation).
        task_idx: required if use_lambda_damping is True.
    """
    f_new_mean = fisher_new.mean().clamp(min=1e-12)
    f_old_mean = fisher_old.mean().clamp(min=1e-12)
    f_new_n = fisher_new / f_new_mean
    f_old_n = fisher_old / f_old_mean

    alpha = f_new_n / (f_old_n + f_new_n + eps)
    alpha = alpha.clamp(min=alpha_min, max=alpha_max)

    if use_lambda_damping:
        assert task_idx is not None
        alpha = alpha * (1.0 / math.sqrt(task_idx))

    return B_merge + alpha * (B_ft - B_merge)
```

**Modified `methods/slao.py` — `after_task()`:**

The flow becomes:

1. Estimate **F_new** on current task i's training data, *before* merging.
   (One forward+backward pass over ~256 samples per task. ~10s on A100.)
2. Look up **F_old** = accumulated Fisher from previous tasks.
3. Call `merge_B_bayesian(B_merge, B_ft, fisher_old, fisher_new, ...)`.
4. Update accumulated Fisher: `F_old ← γ · F_old + F_new` (online EWC EMA).

**Crucial:** F_new is computed *after* training task i but *before* the
Fisher accumulator absorbs it. Otherwise F_old already contains F_new and
the formula degenerates.

---

## CLI

```
--bayesian_merge              Use Bayesian merge instead of fisher_merge_beta form
--bayesian_alpha_min 0.01     Floor on per-param merge rate
--bayesian_alpha_max 0.95     Cap on per-param merge rate
--bayesian_lambda_damping     Apply 1/√i damping on top (ablation)
```

When `--bayesian_merge` is set, `--fisher_merge_beta` is ignored.
Fisher estimation runs unconditionally (needed for both F_new and F_old).

---

## Ablations to run

Across all three task orders, 3 seeds (matches CLUE's 9-run protocol):

| Run | Description | Hypothesis |
|-----|-------------|------------|
| B0 | CLUE baseline (β=0.5) | AA = 80.3, BWT = +1.0 (current) |
| B1 | Bayesian, no λ damping, α ∈ [0.01, 0.95] | Primary test. Expect AA ↑, BWT roughly preserved. |
| B2 | Bayesian + λ damping | Tests whether 1/√i still helps once posterior weighting is in. Expect ≈ B1 or slightly worse (over-damping). |
| B3 | Bayesian, no clipping (α_min=0, α_max=1) | Tests whether clipping matters. Expect slightly worse if Fisher estimates are noisy. |
| B4 | Bayesian + EWC penalty (fisher_lambda=0.5) | Tests double-counting hypothesis. Expect ≈ B1 or worse, confirming EWC penalty redundant. |

---

## Predictions

If the mechanism story from chat holds (Fisher merging works because it
prevents noisy contamination of unused parameters):

- **B1 > B0 on AA**, by 1–2 points. The new-task Fisher tells us to
  *not* update parameters the new task didn't use, regardless of F_old.
- **B1 ≈ B0 on BWT.** Both protect old-important parameters; B1 just
  also prevents spurious updates elsewhere.
- **Gap largest on O5** (most heterogeneous task transitions early).
- **B4 ≤ B1.** Fisher-weighted merging already does what EWC penalty
  attempts; doing both is redundant or counterproductive.

If B1 ≤ B0, the mechanism story is wrong and CLUE's protection of
F_old-important parameters is doing more work than the noise-suppression
on F_new-unimportant parameters. Worth knowing either way.

---

## Cost

- One additional Fisher pass per task (~10s on A100 for 256 samples).
- No new memory beyond a transient F_new tensor (same size as F_old,
  freed after merge).
- Total runtime overhead: negligible (~2.5 min added to a 2h run).