"""Per-task analysis utilities for CLUE.

Implements the analyses listed in `analysis.md` §3.5 and the related
extensions in §1.6:

  1. Fisher histograms per layer at each task boundary.
  2. Effective merge rate `mean(α_jk)` per task.
  3. Per-layer SVD spectrum of B@A (singular values, condition number).
  4. Cosine similarity of A and B with previous task.
  5. Cross-task A and B similarity matrices (computed at end of run).
  6. Frobenius norms of A, B, and B@A.

These are computed from in-memory state during training (no
re-running needed) and dumped to JSON per task. A summary is also
written at the end of the run.

This module deliberately avoids extra GPU work during the main
training loop — all analyses use tensors already computed for the
merge step.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import torch


def _tensor_stats(t: torch.Tensor, n_bins: int = 20) -> dict:
    """Summary stats for a 1D or 2D tensor: mean, std, percentiles, histogram."""
    flat = t.flatten().float()
    n = flat.numel()
    if n == 0:
        return {"n": 0}
    sorted_vals, _ = flat.sort()
    pct = lambda p: sorted_vals[min(int(p * n), n - 1)].item()

    # Build histogram on log scale if range is wide enough
    mn, mx = flat.min().item(), flat.max().item()
    if mn > 0 and mx > 0 and mx / max(mn, 1e-30) > 1e3:
        # Log-spaced bins
        log_min = math.log10(max(mn, 1e-30))
        log_max = math.log10(max(mx, 1e-30))
        bin_edges = torch.logspace(log_min, log_max, n_bins + 1)
        hist = torch.histc(flat.clamp(min=mn), bins=n_bins, min=mn, max=mx)
        bins_log = True
    else:
        hist = torch.histc(flat, bins=n_bins, min=mn, max=mx)
        bin_edges = torch.linspace(mn, mx, n_bins + 1)
        bins_log = False

    return {
        "n": int(n),
        "mean": flat.mean().item(),
        "std": flat.std().item() if n > 1 else 0.0,
        "min": mn,
        "max": mx,
        "p01": pct(0.01),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p99": pct(0.99),
        "histogram_counts": hist.tolist(),
        "histogram_bin_edges": bin_edges.tolist(),
        "histogram_log_scale": bins_log,
    }


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened tensors."""
    a, b = a.flatten().float(), b.flatten().float()
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return (a @ b / (na * nb)).item()


def _row_cosine_sims(A: torch.Tensor, B: torch.Tensor) -> list[float]:
    """Per-row cosine similarity. A and B must have the same shape (r, d)."""
    if A.shape != B.shape:
        return []
    A_norm = A / A.norm(dim=1, keepdim=True).clamp(min=1e-12)
    B_norm = B / B.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (A_norm * B_norm).sum(dim=1).tolist()


def _col_cosine_sims(A: torch.Tensor, B: torch.Tensor) -> list[float]:
    """Per-column cosine similarity. A and B must have the same shape (d, r)."""
    if A.shape != B.shape:
        return []
    A_norm = A / A.norm(dim=0, keepdim=True).clamp(min=1e-12)
    B_norm = B / B.norm(dim=0, keepdim=True).clamp(min=1e-12)
    return (A_norm * B_norm).sum(dim=0).tolist()


def _svd_summary_from_factors(
    A: torch.Tensor, B: torch.Tensor, max_top: int = 16
) -> dict:
    """SVD summary of B @ A using the rank-r factors directly.

    B is (d_out, r), A is (r, d_in). The product B@A has rank ≤ r, so its
    singular values are the square roots of the eigenvalues of
    (A A^T)(B^T B), an (r, r) matrix. Far cheaper than full SVD of the
    (d_out × d_in) product matrix.
    """
    # Use float32 on GPU for stability + speed
    A_f = A.float()
    B_f = B.float()
    C = A_f @ A_f.T          # (r, r)
    G = B_f.T @ B_f          # (r, r)
    M = G @ C                # (r, r), real eigenvalues
    eigvals = torch.linalg.eigvals(M).real
    sigma_sq = eigvals.clamp(min=0)
    sigma, _ = sigma_sq.sqrt().sort(descending=True)
    sigma = sigma.cpu()

    s0 = sigma[0].item() if len(sigma) > 0 else 0.0
    s_last = sigma[-1].item() if len(sigma) > 0 else 0.0
    return {
        "singular_values": sigma[:max_top].tolist(),
        "rank_effective": int((sigma > 1e-6 * s0).sum().item()) if s0 > 0 else 0,
        "frobenius_norm": sigma.norm().item(),
        "spectral_norm": s0,
        "condition_number": (s0 / max(s_last, 1e-12)) if s0 > 0 else 0.0,
        "rank": int(A.shape[0]),
    }


def _svd_summary(BA: torch.Tensor, max_top: int = 16) -> dict:
    """SVD summary of an arbitrary matrix (kept for backward compat).

    Prefer `_svd_summary_from_factors(A, B)` when you have the LoRA
    factors — this avoids materializing the full (d_out, d_in) product.
    """
    s = torch.linalg.svdvals(BA.float())
    s = s.cpu()
    return {
        "singular_values": s[:max_top].tolist(),
        "rank_effective": int((s > 1e-6 * s[0]).sum().item()) if s[0] > 0 else 0,
        "frobenius_norm": s.norm().item(),
        "spectral_norm": s[0].item() if len(s) > 0 else 0.0,
        "condition_number": (s[0] / s[-1].clamp(min=1e-12)).item() if len(s) > 0 else 0.0,
        "rank": int(min(BA.shape)),
    }


def _effective_alpha_stats(
    fisher_B: Optional[torch.Tensor],
    lam: float,
    beta: float,
    prior_strength: float = 1.0,
    fisher_new: Optional[torch.Tensor] = None,
    bayesian: bool = False,
) -> dict:
    """Compute the effective per-element merge rate α_jk and summary stats.

    For SLAO + Fisher merge:  α_jk = λ / (1 + β · F̃_old_jk)
    For Bayesian merge:       α_jk = F̃_new_jk / (F̃_old_jk + F̃_new_jk + ε)
    """
    if bayesian and fisher_new is not None:
        f_new_norm = fisher_new / fisher_new.mean().clamp(min=1e-12)
        if fisher_B is None:
            f_old_norm = torch.full_like(f_new_norm, prior_strength)
        else:
            f_old_norm = fisher_B / fisher_B.mean().clamp(min=1e-12)
        alpha = f_new_norm / (f_old_norm + f_new_norm + 1e-8)
    elif fisher_B is not None and beta > 0:
        f_norm = fisher_B / fisher_B.mean().clamp(min=1e-12)
        alpha = lam / (1.0 + beta * f_norm)
    else:
        # Uniform SLAO
        return {
            "uniform": True,
            "alpha": lam,
            "fraction_below_lam": 0.0,
        }

    alpha = alpha.flatten().float().cpu()
    return {
        "uniform": False,
        "lam_baseline": lam,
        "alpha_mean": alpha.mean().item(),
        "alpha_std": alpha.std().item(),
        "alpha_min": alpha.min().item(),
        "alpha_max": alpha.max().item(),
        "alpha_p01": alpha.quantile(0.01).item(),
        "alpha_p50": alpha.quantile(0.50).item(),
        "alpha_p99": alpha.quantile(0.99).item(),
        "fraction_below_lam": (alpha < lam).float().mean().item(),
    }


@torch.no_grad()
def collect_task_analysis(
    task_idx: int,
    task_name: str,
    paper_i: int,
    ft_state: dict,
    merge_state_pre: Optional[dict],
    merge_state_post: dict,
    fisher_old: Optional[Dict[str, torch.Tensor]],
    fisher_new: Optional[Dict[str, torch.Tensor]],
    args,
) -> dict:
    """Collect per-layer analysis at a task boundary.

    Args:
        task_idx: 0-indexed task number.
        task_name: name of the task.
        paper_i: 1-indexed (= task_idx + 1).
        ft_state: {layer_name: {"A", "B"}} fine-tuned state for this task.
        merge_state_pre: merged state BEFORE this task's merge step
                         (None for task 0).
        merge_state_post: merged state AFTER this task's merge step.
        fisher_old: full Fisher dict (keyed by full param name) — the
                    accumulated Fisher BEFORE this task's update.
        fisher_new: per-task Fisher dict — the new Fisher just estimated.
        args: the argparse Namespace (for hyperparameters).

    Returns:
        Dict suitable for JSON serialization.
    """
    lam = 1.0 / math.sqrt(paper_i) if paper_i >= 1 else 1.0
    out = {
        "task_idx": task_idx,
        "task_name": task_name,
        "paper_i": paper_i,
        "lambda_t": lam,
        "fisher_merge_beta": getattr(args, "fisher_merge_beta", 0.0),
        "bayesian_merge": getattr(args, "bayesian_merge", False),
        "bayesian_prior_strength": getattr(args, "bayesian_prior_strength", 1.0),
        "layers": {},
    }

    for layer_name, ft in ft_state.items():
        A_ft = ft["A"]                # (r, d_in)
        B_ft = ft["B"]                # (d_out, r)
        layer = {}

        # SVD summary of fine-tuned ΔW (fast path: factor-based, never
        # materializes the full d_out × d_in product)
        layer["svd_ft"] = _svd_summary_from_factors(A_ft, B_ft)

        # SVD summary of merged ΔW (post)
        post = merge_state_post[layer_name]
        layer["svd_merged"] = _svd_summary_from_factors(post["A"], post["B"])

        # Frobenius norms (use trace identity: ‖BA‖_F² = trace(A^T B^T B A)
        # = trace((A A^T)(B^T B)). The Frobenius norm of B@A equals the
        # Frobenius norm of the singular value vector — already in svd_ft).
        layer["norms"] = {
            "A_ft": A_ft.norm().item(),
            "B_ft": B_ft.norm().item(),
            "BA_ft": layer["svd_ft"]["frobenius_norm"],
            "A_merged": post["A"].norm().item(),
            "B_merged": post["B"].norm().item(),
            "BA_merged": layer["svd_merged"]["frobenius_norm"],
        }

        # Fisher stats on B
        b_key = f"{layer_name}.lora_B.default.weight"
        a_key = f"{layer_name}.lora_A.default.weight"
        f_old_B = fisher_old.get(b_key) if fisher_old else None
        f_new_B = fisher_new.get(b_key) if fisher_new else None
        f_old_A = fisher_old.get(a_key) if fisher_old else None
        f_new_A = fisher_new.get(a_key) if fisher_new else None

        if f_old_B is not None:
            layer["fisher_old_B"] = _tensor_stats(f_old_B)
        if f_new_B is not None:
            layer["fisher_new_B"] = _tensor_stats(f_new_B)
        if f_old_A is not None:
            layer["fisher_old_A"] = _tensor_stats(f_old_A)
        if f_new_A is not None:
            layer["fisher_new_A"] = _tensor_stats(f_new_A)

        # Effective α stats — use fisher_old (what the merge actually used)
        layer["effective_alpha"] = _effective_alpha_stats(
            fisher_B=f_old_B,
            lam=lam,
            beta=getattr(args, "fisher_merge_beta", 0.0),
            prior_strength=getattr(args, "bayesian_prior_strength", 1.0),
            fisher_new=f_new_B,
            bayesian=getattr(args, "bayesian_merge", False),
        )

        # Cosine similarity to previous task (if available)
        if merge_state_pre is not None and layer_name in merge_state_pre:
            prev = merge_state_pre[layer_name]
            layer["cos_sim_prev"] = {
                "A_overall": _cosine_sim(post["A"], prev["A"]),
                "B_overall": _cosine_sim(post["B"], prev["B"]),
                "A_per_row": _row_cosine_sims(post["A"], prev["A"]),
                "B_per_col": _col_cosine_sims(post["B"], prev["B"]),
            }
            # Compare ft vs prev merge directly (drift before any merging)
            layer["drift_ft_vs_prev_merge"] = {
                "A_overall": _cosine_sim(A_ft, prev["A"]),
                "B_overall": _cosine_sim(B_ft, prev["B"]),
            }

        out["layers"][layer_name] = layer

    return out


def write_task_analysis(
    output_path: str | Path,
    analysis: dict,
) -> None:
    """Save analysis dict to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)


def write_run_summary(
    output_dir: str | Path,
    task_analyses: list[dict],
) -> None:
    """Compute and save a run-level summary aggregating all task analyses.

    Includes:
      - Cross-task A and B cosine similarity matrices (per-layer averaged).
      - Spectrum evolution: σ_1, σ_min/σ_max trajectory per layer.
      - Effective α evolution per task.
    """
    output_path = Path(output_dir) / "analysis_summary.json"
    if not task_analyses:
        return

    layer_names = list(task_analyses[0]["layers"].keys())

    # Spectrum evolution: for each layer, list of [σ_1, σ_min, condition, frob_norm]
    # per task, indexed by task_idx
    spectrum = {}
    for ln in layer_names:
        per_task = []
        for ta in task_analyses:
            svd = ta["layers"][ln].get("svd_merged", {})
            per_task.append({
                "task_idx": ta["task_idx"],
                "task_name": ta["task_name"],
                "spectral_norm": svd.get("spectral_norm"),
                "condition_number": svd.get("condition_number"),
                "frobenius_norm": svd.get("frobenius_norm"),
                "singular_values": svd.get("singular_values", [])[:8],
            })
        spectrum[ln] = per_task

    # Alpha evolution
    alpha_evol = {}
    for ln in layer_names:
        per_task = []
        for ta in task_analyses:
            ea = ta["layers"][ln].get("effective_alpha", {})
            per_task.append({
                "task_idx": ta["task_idx"],
                "task_name": ta["task_name"],
                "alpha_mean": ea.get("alpha_mean"),
                "lam_baseline": ea.get("lam_baseline"),
                "fraction_below_lam": ea.get("fraction_below_lam"),
            })
        alpha_evol[ln] = per_task

    # Cosine similarity vs prev (per task, per layer, averaged across rows/cols)
    cos_evol = {}
    for ln in layer_names:
        per_task = []
        for ta in task_analyses:
            cs = ta["layers"][ln].get("cos_sim_prev")
            if cs is None:
                per_task.append(None)
            else:
                per_task.append({
                    "task_idx": ta["task_idx"],
                    "A_overall": cs.get("A_overall"),
                    "B_overall": cs.get("B_overall"),
                })
        cos_evol[ln] = per_task

    # Run-level aggregates: average alpha_mean across layers per task
    # and average cosine similarity across layers per task
    n_tasks = len(task_analyses)
    summary_per_task = []
    for ta in task_analyses:
        alphas = [
            ta["layers"][ln].get("effective_alpha", {}).get("alpha_mean")
            for ln in layer_names
        ]
        alphas = [a for a in alphas if a is not None]
        cos_A = []
        cos_B = []
        for ln in layer_names:
            cs = ta["layers"][ln].get("cos_sim_prev")
            if cs:
                cos_A.append(cs["A_overall"])
                cos_B.append(cs["B_overall"])
        summary_per_task.append({
            "task_idx": ta["task_idx"],
            "task_name": ta["task_name"],
            "lambda_t": ta["lambda_t"],
            "mean_alpha_across_layers": (sum(alphas) / len(alphas)) if alphas else None,
            "mean_A_cos_sim_to_prev_across_layers": (sum(cos_A) / len(cos_A)) if cos_A else None,
            "mean_B_cos_sim_to_prev_across_layers": (sum(cos_B) / len(cos_B)) if cos_B else None,
        })

    out = {
        "n_tasks": n_tasks,
        "layer_names": layer_names,
        "summary_per_task": summary_per_task,
        "spectrum_per_layer": spectrum,
        "alpha_per_layer": alpha_evol,
        "cos_sim_to_prev_per_layer": cos_evol,
    }

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
