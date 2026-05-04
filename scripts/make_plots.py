#!/usr/bin/env python3
"""Generate analysis plots for the README from a CLUE run.

Usage:
    python scripts/make_plots.py outputs/<run_dir> figures/

Reads analysis_summary.json + results.json from <run_dir> and writes:
    - accuracy_matrix.png       T×T heatmap
    - alpha_evolution.png       per-task α and λ(t) baseline
    - spectrum_evolution.png    σ_1, σ_min/σ_max per task
    - cosine_evolution.png      cos(A_merge, A_prev) etc per task
    - fisher_distribution.png   histogram of Fisher on B at task 3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Accrue palette
GREEN = "#3a6b4c"
GREEN_LIGHT = "#4a8b62"
GREEN_PALE = "#e8f0eb"
TEXT = "#1a1a1a"
MUTED = "#6b7280"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": TEXT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 130,
})


def plot_accuracy_matrix(results: dict, task_names: list[str], out_path: Path):
    M = np.array(results["matrix"])
    T = M.shape[0]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = plt.cm.YlGn
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap=cmap, vmin=0.5, vmax=1.0, aspect="equal")
    for i in range(T):
        for j in range(T):
            if not np.isnan(M[i, j]):
                color = "white" if M[i, j] > 0.78 else TEXT
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color=color, fontsize=10)
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"after t={t}" for t in task_names], rotation=30, ha="right")
    ax.set_yticklabels([f"eval {t}" for t in task_names])
    ax.set_xlabel("Trained through")
    ax.set_ylabel("Eval task")
    ax.set_title(f"Accuracy matrix — AA={results['average_accuracy']:.4f}, BWT={results['backward_transfer']:+.4f}",
                 color=GREEN, weight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy", color=MUTED)
    fig.savefig(out_path)
    plt.close(fig)


def plot_alpha_evolution(summary: dict, out_path: Path):
    """Plot α evolution: average α across layers vs λ(t) baseline."""
    fig, ax = plt.subplots(figsize=(7, 4))
    n_tasks = summary["n_tasks"]
    layers = summary["layer_names"]

    # Per-layer alpha curves (light)
    for layer_name in layers:
        alpha_per_t = summary["alpha_per_layer"][layer_name]
        ts = [e["task_idx"] for e in alpha_per_t if e["alpha_mean"] is not None]
        alphas = [e["alpha_mean"] for e in alpha_per_t if e["alpha_mean"] is not None]
        if ts:
            ax.plot(ts, alphas, color=GREEN_LIGHT, alpha=0.05, linewidth=0.5)

    # Mean across layers (bold)
    summary_per_task = summary["summary_per_task"]
    ts = [e["task_idx"] for e in summary_per_task if e["mean_alpha_across_layers"] is not None]
    means = [e["mean_alpha_across_layers"] for e in summary_per_task if e["mean_alpha_across_layers"] is not None]
    ax.plot(ts, means, color=GREEN, linewidth=2.5, marker="o", markersize=8,
            label="mean α (across layers)")

    # λ(t) = 1/√(t+1) baseline
    lambdas = [e["lambda_t"] for e in summary_per_task]
    ts_all = [e["task_idx"] for e in summary_per_task]
    ax.plot(ts_all, lambdas, color=MUTED, linewidth=1.5, linestyle="--",
            marker="s", markersize=6, label="λ(t) = 1/√(t+1) (uniform SLAO baseline)")

    ax.set_xlabel("Task index")
    ax.set_ylabel("Effective merge rate α")
    ax.set_title("Effective merge rate α per task\n"
                 "(α below the dashed line = old knowledge protected)",
                 color=GREEN, weight="bold")
    ax.set_xticks(ts_all)
    ax.set_xticklabels([f"{e['task_idx']}: {e['task_name']}" for e in summary_per_task])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, max(1.0, max(lambdas) * 1.1))
    fig.savefig(out_path)
    plt.close(fig)


def plot_spectrum_evolution(summary: dict, out_path: Path):
    """Plot top-r singular values per task (mean across layers)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n_tasks = summary["n_tasks"]
    layers = summary["layer_names"]

    # Get rank from first layer's first task entry
    first_sv = summary["spectrum_per_layer"][layers[0]][0]["singular_values"]
    r = len(first_sv)

    # Average across layers per task per rank
    sv_matrix = np.zeros((n_tasks, r))
    for layer_name in layers:
        per_task = summary["spectrum_per_layer"][layer_name]
        for entry in per_task:
            sv = entry["singular_values"]
            sv_matrix[entry["task_idx"], :len(sv)] += np.array(sv)
    sv_matrix /= len(layers)

    # Left: singular values per task, by rank
    ax = axes[0]
    cmap = plt.cm.viridis
    for k in range(r):
        ax.plot(range(n_tasks), sv_matrix[:, k], marker="o",
                color=cmap(k / max(r-1, 1)), label=f"σ_{k+1}", linewidth=2)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Singular value (mean across layers)")
    ax.set_title("Singular value spectrum of merged ΔW = B@A\n"
                 "(layer-averaged, per task)",
                 color=GREEN, weight="bold")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.2)

    # Right: condition number σ_1/σ_r per task per layer
    ax = axes[1]
    for layer_name in layers:
        per_task = summary["spectrum_per_layer"][layer_name]
        ts = [e["task_idx"] for e in per_task]
        cond = [e["condition_number"] for e in per_task]
        ax.plot(ts, cond, color=GREEN_LIGHT, alpha=0.1, linewidth=0.5)

    # Mean condition number
    cond_mean = np.zeros(n_tasks)
    for layer_name in layers:
        per_task = summary["spectrum_per_layer"][layer_name]
        for entry in per_task:
            cond_mean[entry["task_idx"]] += entry["condition_number"]
    cond_mean /= len(layers)
    ax.plot(range(n_tasks), cond_mean, color=GREEN, linewidth=2.5,
            marker="o", markersize=8, label="mean (across layers)")
    ax.set_xlabel("Task index")
    ax.set_ylabel("Condition number σ_1 / σ_r")
    ax.set_title("Spectral conditioning per task\n"
                 "(low = full rank used; high = near-saturation)",
                 color=GREEN, weight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.savefig(out_path)
    plt.close(fig)


def plot_cosine_evolution(summary: dict, out_path: Path):
    """Plot cosine similarity to previous merged state per task per layer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    layers = summary["layer_names"]
    n_tasks = summary["n_tasks"]

    # Mean A and B cos sim across layers per task
    cos_A_mean = []
    cos_B_mean = []
    ts = []
    for entry in summary["summary_per_task"]:
        if entry["mean_A_cos_sim_to_prev_across_layers"] is not None:
            cos_A_mean.append(entry["mean_A_cos_sim_to_prev_across_layers"])
            cos_B_mean.append(entry["mean_B_cos_sim_to_prev_across_layers"])
            ts.append(entry["task_idx"])

    # Per-layer A cos sim
    ax = axes[0]
    for layer_name in layers:
        per_task = summary["cos_sim_to_prev_per_layer"][layer_name]
        ys = [e["A_overall"] if e else None for e in per_task]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ys = [y for y in ys if y is not None]
        if xs:
            ax.plot(xs, ys, color=GREEN_LIGHT, alpha=0.1, linewidth=0.5)
    if ts:
        ax.plot(ts, cos_A_mean, color=GREEN, linewidth=2.5, marker="o", markersize=8,
                label="mean (across layers)")
    ax.set_xlabel("Task index")
    ax.set_ylabel("cos(A_merged, A_merged_prev)")
    ax.set_title("A drift per task\n(low = A rotates; high = A stays)",
                 color=GREEN, weight="bold")
    ax.set_xticks(range(n_tasks))
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Per-layer B cos sim
    ax = axes[1]
    for layer_name in layers:
        per_task = summary["cos_sim_to_prev_per_layer"][layer_name]
        ys = [e["B_overall"] if e else None for e in per_task]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ys = [y for y in ys if y is not None]
        if xs:
            ax.plot(xs, ys, color=GREEN_LIGHT, alpha=0.1, linewidth=0.5)
    if ts:
        ax.plot(ts, cos_B_mean, color=GREEN, linewidth=2.5, marker="o", markersize=8,
                label="mean (across layers)")
    ax.set_xlabel("Task index")
    ax.set_ylabel("cos(B_merged, B_merged_prev)")
    ax.set_title("B drift per task\n(low = B churns; high = B stable)",
                 color=GREEN, weight="bold")
    ax.set_xticks(range(n_tasks))
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.savefig(out_path)
    plt.close(fig)


def plot_fisher_distribution(run_dir: Path, out_path: Path):
    """Plot Fisher percentile spread per task across all layers.

    Box-and-whisker style: shows how the Fisher mass concentrates over
    tasks. A widening gap between p99 and mean means a heavier tail —
    a few B elements carrying disproportionate old-task importance.
    """
    # Gather percentile stats across tasks
    rows = []
    for task_dir in sorted(run_dir.glob("task_*")):
        analysis_path = task_dir / "analysis.json"
        if not analysis_path.exists():
            continue
        a = json.loads(analysis_path.read_text())
        task_idx = a["task_idx"]
        task_name = a["task_name"]
        # Aggregate Fisher_old_B across all layers (median, p99, max)
        means = []
        p50s = []
        p99s = []
        maxes = []
        for ln, layer in a["layers"].items():
            f = layer.get("fisher_old_B")
            if f is None:
                continue
            means.append(f["mean"])
            p50s.append(f["p50"])
            p99s.append(f["p99"])
            maxes.append(f["max"])
        if means:
            rows.append({
                "task_idx": task_idx,
                "task_name": task_name,
                "mean_of_means": np.mean(means),
                "median_of_p50": np.median(p50s),
                "median_of_p99": np.median(p99s),
                "max_max": np.max(maxes),
                "tail_ratio_p99_over_mean": np.median([
                    p / m if m > 0 else 0 for p, m in zip(p99s, means)
                ]),
            })

    if not rows:
        print("No Fisher data, skipping fisher plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ts = [r["task_idx"] for r in rows]

    ax.plot(ts, [r["mean_of_means"] for r in rows],
            marker="o", color=GREEN, linewidth=2, markersize=8, label="mean")
    ax.plot(ts, [r["median_of_p50"] for r in rows],
            marker="s", color=GREEN_LIGHT, linewidth=1.5, markersize=6, label="median (p50)")
    ax.plot(ts, [r["median_of_p99"] for r in rows],
            marker="^", color="#c0392b", linewidth=1.5, markersize=6, label="p99 (heavy-tail edge)")
    ax.plot(ts, [r["max_max"] for r in rows],
            marker="x", color=MUTED, linewidth=1, markersize=6, label="max")

    ax.set_yscale("log")
    ax.set_xlabel("Task index")
    ax.set_ylabel("Fisher diagonal value on B (log scale)")
    ax.set_xticks(ts)
    ax.set_xticklabels([f"{r['task_idx']}: {r['task_name']}" for r in rows])
    ax.set_title("Fisher spread on B across tasks\n"
                 "(p99 >> mean = heavy tail; some B elements concentrate task importance)",
                 color=GREEN, weight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2, which="both")
    fig.savefig(out_path)
    plt.close(fig)


def main():
    if len(sys.argv) < 3:
        print("Usage: make_plots.py <run_dir> <figures_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    fig_dir = Path(sys.argv[2])
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_dir / "analysis_summary.json").read_text())
    results = json.loads((run_dir / "results.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    task_names = config.get("task_order_resolved", [])

    print(f"Generating plots for {run_dir.name} ({len(task_names)} tasks)...")
    plot_accuracy_matrix(results, task_names, fig_dir / "accuracy_matrix.png")
    print("  ✓ accuracy_matrix.png")
    plot_alpha_evolution(summary, fig_dir / "alpha_evolution.png")
    print("  ✓ alpha_evolution.png")
    plot_spectrum_evolution(summary, fig_dir / "spectrum_evolution.png")
    print("  ✓ spectrum_evolution.png")
    plot_cosine_evolution(summary, fig_dir / "cosine_evolution.png")
    print("  ✓ cosine_evolution.png")
    plot_fisher_distribution(run_dir, fig_dir / "fisher_spread.png")
    print("  ✓ fisher_distribution.png")
    print(f"Plots written to {fig_dir}/")


if __name__ == "__main__":
    main()
