"""Continual learning metrics: Average Accuracy (AA) and Backward Transfer (BWT)."""

import json
from pathlib import Path

import numpy as np


class AccuracyMatrix:
    """Tracks the T x T accuracy matrix for continual learning evaluation.

    Entry (i, j) = accuracy on task i after training on task j.
    Only entries where j >= i are filled (can't evaluate before training).
    """

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.matrix = np.full((num_tasks, num_tasks), np.nan)

    def update(self, eval_task_idx: int, trained_up_to: int, accuracy: float) -> None:
        """Record accuracy on eval_task_idx after training through trained_up_to."""
        self.matrix[eval_task_idx, trained_up_to] = accuracy

    def average_accuracy(self) -> float:
        """AA = (1/T) * sum_{i=1}^{T} a_{i,T}"""
        final_col = self.matrix[:, self.num_tasks - 1]
        return float(np.nanmean(final_col))

    def backward_transfer(self) -> float:
        """BWT = (1/(T-1)) * sum_{i=1}^{T-1} (a_{i,T} - a_{i,i})"""
        if self.num_tasks < 2:
            return 0.0
        bwt_sum = 0.0
        count = 0
        for i in range(self.num_tasks - 1):
            a_i_T = self.matrix[i, self.num_tasks - 1]
            a_i_i = self.matrix[i, i]
            if not (np.isnan(a_i_T) or np.isnan(a_i_i)):
                bwt_sum += a_i_T - a_i_i
                count += 1
        return float(bwt_sum / count) if count > 0 else 0.0

    def current_average_accuracy(self, trained_up_to: int) -> float:
        """AA after training through task `trained_up_to` (0-indexed).

        Returns mean of column `trained_up_to` for rows 0..trained_up_to.
        """
        col = self.matrix[: trained_up_to + 1, trained_up_to]
        return float(np.nanmean(col))

    def current_backward_transfer(self, trained_up_to: int) -> float:
        """BWT after training through task `trained_up_to` (0-indexed)."""
        if trained_up_to < 1:
            return 0.0
        bwt_sum = 0.0
        count = 0
        for i in range(trained_up_to):
            a_i_cur = self.matrix[i, trained_up_to]
            a_i_i = self.matrix[i, i]
            if not (np.isnan(a_i_cur) or np.isnan(a_i_i)):
                bwt_sum += a_i_cur - a_i_i
                count += 1
        return float(bwt_sum / count) if count > 0 else 0.0

    def per_task_accuracies(self, trained_up_to: int) -> dict:
        """Return {task_idx: accuracy} for all tasks evaluated after trained_up_to."""
        accs = {}
        for i in range(trained_up_to + 1):
            v = self.matrix[i, trained_up_to]
            if not np.isnan(v):
                accs[i] = float(v)
        return accs

    def save(self, path: str) -> None:
        """Save the accuracy matrix and derived metrics to disk."""
        out = {
            "matrix": self.matrix.tolist(),
            "average_accuracy": self.average_accuracy(),
            "backward_transfer": self.backward_transfer(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def save_task_results(
        self, path: str, task_idx: int, task_names: list
    ) -> None:
        """Save per-task results after training through task_idx."""
        accs = self.per_task_accuracies(task_idx)
        out = {
            "trained_up_to": task_idx,
            "trained_up_to_task": task_names[task_idx],
            "current_AA": self.current_average_accuracy(task_idx),
            "current_BWT": self.current_backward_transfer(task_idx),
            "per_task_accuracy": {
                task_names[i]: acc for i, acc in accs.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def __repr__(self) -> str:
        aa = self.average_accuracy()
        bwt = self.backward_transfer()
        return f"AccuracyMatrix(AA={aa:.4f}, BWT={bwt:.4f})"
