"""Stiefelized CLUE: CL with tangent-space Fisher-weighted merge on Σ.

Trains each task with standard LoRA (no manifold constraint during training).
At task boundaries, decomposes ΔW = B @ A into thin SVD (U, Σ, V) and merges
using SLAO's proven asymmetric strategy for U/V with Fisher-weighted merging
on Σ where the variance reduction actually matters.

The key insight from STIEFEL.md: Fisher on Σ has r scalars estimated from 256
samples (~32 samples/scalar), vs. d×r scalars for Fisher on B (~0.008
samples/scalar). The ~500× variance reduction makes Fisher-weighted merging
on Σ qualitatively better than on B. But Fisher on U and V components is
still high-dimensional and noisy — so we use SLAO's time-tested approach
for those.

Merge strategy (hybrid):
  - V (= A's row space): fully replaced, like SLAO replaces A
  - U (= B's column space): time-aware EMA, like SLAO merges B
  - Σ (singular values): Fisher-weighted Bayesian merge (the Stiefel innovation)

Lifecycle per task i (0-indexed):
  before_task: decompose merged point into SVD, set A = V^T, B = U·diag(S)
  [standard LoRA training]
  after_task:  SVD of fine-tuned state, align with base, hybrid merge
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn

from methods.base import ContinualMethod
from models.lora import (
    LoRAState,
    SVDState,
    extract_lora_state,
    lora_to_svd,
    merge_tangent_bayesian,
    merge_tangent_uniform,
    retract_tangent,
    set_lora_state,
    svd_to_lora,
    _polar_retract,
)
from utils import get_logger

logger = get_logger(__name__)


def _align_svd(
    base: Dict[str, torch.Tensor],
    ft: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Align fine-tuned SVD factors with base SVD to resolve sign ambiguity.

    SVD has per-column sign freedom: (u_k, v_k) and (-u_k, -v_k) give the
    same rank-1 component. We flip columns of U_ft and V_ft so that each
    column has positive inner product with the corresponding base column.

    Also handles singular-value reordering: if singular values crossed during
    training, column k of ft may correspond to a different column of base.
    We match by maximum absolute cosine similarity.

    Args:
        base: {"U": (d_out, r), "S": (r,), "V": (d_in, r)}
        ft:   {"U": (d_out, r), "S": (r,), "V": (d_in, r)}
    Returns:
        Aligned copy of ft (same shapes, reordered and sign-corrected).
    """
    U_b, V_b = base["U"], base["V"]
    U_f, S_f, V_f = ft["U"].clone(), ft["S"].clone(), ft["V"].clone()
    r = U_b.shape[1]

    # Cosine similarity matrix between base and ft U columns
    cos_U = U_b.T @ U_f  # (r, r)

    # Greedy assignment: for each base column, pick best ft column
    used = set()
    perm = []
    signs = []
    for i in range(r):
        similarities = cos_U[i].abs()
        for j in used:
            similarities[j] = -1.0
        j = similarities.argmax().item()
        perm.append(j)
        used.add(j)
        signs.append(1.0 if cos_U[i, j] >= 0 else -1.0)

    perm_t = torch.tensor(perm, device=U_f.device)
    signs_t = torch.tensor(signs, device=U_f.device, dtype=U_f.dtype)

    U_aligned = U_f[:, perm_t] * signs_t.unsqueeze(0)
    V_aligned = V_f[:, perm_t] * signs_t.unsqueeze(0)
    S_aligned = S_f[perm_t]

    return {"U": U_aligned, "S": S_aligned, "V": V_aligned}


class StiefelizedCLUE(ContinualMethod):
    """Stiefelized CLUE: SLAO-style asymmetric merge with Fisher on Σ."""

    def __init__(self, model: nn.Module, args):
        super().__init__(model, args)
        self.base_svd: SVDState | None = None
        self.merge_state: LoRAState | None = None
        # Fisher (set externally by train.py)
        self.fisher = None
        self.fisher_new: Dict[str, torch.Tensor] | None = None
        # Config
        self.bayesian_merge = getattr(args, "bayesian_merge", False)
        self.bayesian_alpha_min = getattr(args, "bayesian_alpha_min", 0.01)
        self.bayesian_alpha_max = getattr(args, "bayesian_alpha_max", 0.95)
        self.fisher_merge_beta = getattr(args, "fisher_merge_beta", 0.0)
        # Accumulated tangent-space Fisher on Σ only (the low-variance quantity)
        self.fisher_sigma_old: Dict[str, torch.Tensor] = {}

    def before_task(self, task_idx: int, task_name: str) -> None:
        """Initialize LoRA for the new task.

        Task 0: use default PEFT initialization.
        Task i>0: decompose merged state into SVD, convert to (A, B) with
                  A = V^T (orthonormal rows), B = U @ diag(S).
        """
        if task_idx == 0:
            logger.info("Task 0: using default LoRA initialization")
            return

        assert self.merge_state is not None

        self.base_svd = lora_to_svd(self.merge_state)
        base_lora = svd_to_lora(self.base_svd)
        set_lora_state(self.model, base_lora)

        logger.info(
            f"Task {task_idx} ({task_name}): "
            f"Stiefel base point set, A=V^T (orthonormal), B=U·diag(S)"
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """All LoRA A and B parameters (standard training)."""
        params = []
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                params.append(param)
        return params

    def after_task(self, task_idx: int) -> None:
        """Hybrid asymmetric merge: SLAO-style for U/V, Fisher on Σ.

        Task 0: store as initial merged state.
        Task i>0:
          1. SVD of fine-tuned B_ft @ A_ft, aligned with base
          2. V_merged = V_ft (direct replacement, like SLAO replaces A)
          3. Σ_merged = Bayesian or uniform merge of Σ
          4. U_merged = time-aware EMA (like SLAO merges B), then re-orthogonalize
        """
        ft_state = extract_lora_state(self.model)

        if task_idx == 0:
            self.merge_state = ft_state
            self.base_svd = lora_to_svd(ft_state)
            logger.info("Task 0: stored initial merged state")
            return

        assert self.base_svd is not None

        paper_i = task_idx + 1  # 1-indexed for λ(i)
        lam = 1.0 / math.sqrt(paper_i)

        ft_svd = lora_to_svd(ft_state)

        use_bayesian_sigma = self.bayesian_merge and self.fisher_new is not None

        new_svd: SVDState = {}
        for layer_name in self.base_svd:
            base = self.base_svd[layer_name]
            ft_aligned = _align_svd(base, ft_svd[layer_name])

            # --- V: direct replacement (like SLAO's A_merge = A_ft) ---
            V_merged = ft_aligned["V"]

            # --- Σ: Fisher-weighted Bayesian merge (the Stiefel innovation) ---
            xi_S = ft_aligned["S"] - base["S"]

            if use_bayesian_sigma:
                f_S_new = self._estimate_sigma_fisher(layer_name)
                f_S_old = self.fisher_sigma_old.get(layer_name)

                merged_xi_S = merge_tangent_bayesian(
                    xi_S,
                    f_S_old,
                    f_S_new,
                    alpha_min=self.bayesian_alpha_min,
                    alpha_max=self.bayesian_alpha_max,
                )
                S_merged = base["S"] + merged_xi_S

                # Accumulate Σ Fisher
                gamma = getattr(self.args, "fisher_gamma", 0.9)
                if layer_name in self.fisher_sigma_old:
                    self.fisher_sigma_old[layer_name] = (
                        gamma * self.fisher_sigma_old[layer_name] + f_S_new
                    )
                else:
                    self.fisher_sigma_old[layer_name] = f_S_new
            else:
                # Uniform merge on Σ
                S_merged = base["S"] + lam * xi_S

            S_merged = S_merged.clamp(min=1e-8)

            # --- U: time-aware EMA (like SLAO's B merge), then re-orthogonalize ---
            # U carries the "write" directions, analogous to B in SLAO
            U_merged_raw = base["U"] + lam * (ft_aligned["U"] - base["U"])
            U_merged = _polar_retract(U_merged_raw)

            new_svd[layer_name] = {
                "U": U_merged, "S": S_merged, "V": V_merged
            }

        self.base_svd = new_svd
        self.merge_state = svd_to_lora(new_svd)
        set_lora_state(self.model, self.merge_state)

        logger.info(
            f"Task {task_idx}: Stiefel hybrid merge "
            f"(V=replace, U=EMA λ={lam:.3f}, "
            f"Σ={'Bayesian' if use_bayesian_sigma else f'uniform λ={lam:.3f}'})"
        )

    def _estimate_sigma_fisher(self, layer_name: str) -> torch.Tensor:
        """Estimate Fisher on Σ from the standard (A, B) Fisher.

        F^Σ_k = Σ_j U_{jk}² · F^B_{jk} + Σ_j V_{jk}² · F^A_{kj}

        This projects the diagonal (A, B) Fisher along the singular
        directions, yielding r scalars with ~500× less variance than
        the raw d×r Fisher on B.
        """
        base = self.base_svd[layer_name]
        U, S, V = base["U"], base["S"], base["V"]
        r = S.shape[0]

        b_key = f"{layer_name}.lora_B.default.weight"
        a_key = f"{layer_name}.lora_A.default.weight"
        f_B = self.fisher_new.get(b_key)  # (d_out, r) or None
        f_A = self.fisher_new.get(a_key)  # (r, d_in) or None

        f_S = torch.zeros(r, device=S.device, dtype=S.dtype)
        if f_B is not None:
            # (U²)^T @ f_B → (r, r), take diag
            f_S = f_S + ((U ** 2).T @ f_B).diag()
        if f_A is not None:
            # f_A @ V² → (r, r), take diag
            f_S = f_S + (f_A @ (V ** 2)).diag()

        return f_S

    def get_model(self) -> nn.Module:
        """Return model with merged LoRA weights applied."""
        if self.merge_state is not None:
            set_lora_state(self.model, self.merge_state)
        return self.model
