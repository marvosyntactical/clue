"""Diagonal Fisher Regularization for LoRA Continual Learning (Extension 2).

Supports two modes:

  **Standard EWC** (Kirkpatrick et al., 2017):
    Accumulates Fisher by summing across tasks. The penalty grows with T.
    L_total = L_task + (λ/2) * Σ_j F_j * (θ_j − θ*_j)²

  **Online EWC** (Schwarz et al., 2018):
    Uses an EMA of the Fisher with decay γ. The penalty stays bounded.
    F_j ← γ * F_old_j + F_new_j
    θ* is updated each task to the current merged state.

Online EWC is preferred for longer task sequences (T > 5) because standard
EWC's summed Fisher becomes increasingly stiff, killing plasticity.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DiagonalFisher:
    """Accumulates diagonal Fisher information across tasks."""

    def __init__(self, model: nn.Module, gamma: float = 1.0):
        """
        Args:
            model: The PEFT model.
            gamma: EMA decay for online EWC. 1.0 = standard EWC (sum).
                   < 1.0 = online EWC (e.g. 0.9 retains 90% of old Fisher).
        """
        self.model = model
        self.gamma = gamma
        # Accumulated Fisher: {param_name: Tensor}
        self.fisher: dict[str, torch.Tensor] = {}
        # Reference params: {param_name: Tensor}
        self.ref_params: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _compute_fisher(self, data_loader: DataLoader, n_samples: int = 256) -> dict[str, torch.Tensor]:
        """Compute per-task diagonal Fisher (not accumulated)."""
        self.model.eval()
        device = next(self.model.parameters()).device

        fisher_new: dict[str, torch.Tensor] = {}
        count = 0

        for batch in data_loader:
            if count >= n_samples:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            self.model.zero_grad()
            with torch.enable_grad():
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
                loss.backward()

            for name, param in self.model.named_parameters():
                if "lora_" in name and param.grad is not None:
                    if name not in fisher_new:
                        fisher_new[name] = torch.zeros_like(param.data)
                    fisher_new[name].add_(param.grad.data ** 2)

            count += input_ids.size(0)

        self.model.zero_grad()

        for name in fisher_new:
            fisher_new[name].div_(max(count, 1))

        return fisher_new

    def estimate_new(self, data_loader: DataLoader, n_samples: int = 256) -> dict[str, torch.Tensor]:
        """Estimate Fisher for the current task WITHOUT accumulating.

        Returns the raw per-task Fisher dict. Use for Bayesian merge
        (need F_new before it gets folded into F_old).
        """
        return self._compute_fisher(data_loader, n_samples)

    def estimate(self, data_loader: DataLoader, n_samples: int = 256):
        """Estimate diagonal Fisher and accumulate into running total."""
        fisher_new = self._compute_fisher(data_loader, n_samples)
        self.accumulate(fisher_new)

    def accumulate(self, fisher_new: dict[str, torch.Tensor]):
        """Fold a per-task Fisher into the running accumulator."""
        for name, f_new in fisher_new.items():
            if name in self.fisher:
                self.fisher[name] = self.gamma * self.fisher[name] + f_new
            else:
                self.fisher[name] = f_new

    def snapshot_ref_params(self):
        """Save current LoRA params as the reference point θ*.

        For SLAO, call this after after_task() applies the merged state,
        so the reference is the merged model.
        """
        self.ref_params = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                self.ref_params[name] = param.data.clone()

    def penalty(self) -> torch.Tensor:
        """Compute Σ_j F_j * (θ_j − θ*_j)².

        Returns a scalar tensor (on the correct device, with grad).
        The caller multiplies by (λ/2).
        """
        total = None
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.ref_params:
                term = (self.fisher[name] * (param - self.ref_params[name]) ** 2).sum()
                total = term if total is None else total + term
        if total is None:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)
        return total
