"""LoRA weight management utilities for continual learning.

Provides functions to extract, set, and manipulate LoRA A/B weights
from a PEFT model, independent of PEFT internals.
"""

import math
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

LoRAState = Dict[str, Dict[str, torch.Tensor]]  # {layer_name: {"A": ..., "B": ...}}


# ---------------------------------------------------------------------------
# Iterating over LoRA layers
# ---------------------------------------------------------------------------

def _iter_lora_layers(model: nn.Module):
    """Yield (name, module) for every LoRA-injected linear layer."""
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield name, module


# ---------------------------------------------------------------------------
# Weight extraction / injection
# ---------------------------------------------------------------------------

def extract_lora_state(model: nn.Module, adapter_name: str = "default") -> LoRAState:
    """Extract all LoRA A and B weights from a PEFT model.

    Returns dict: {layer_name: {"A": Tensor(r, d), "B": Tensor(d, r)}}
    """
    state: LoRAState = OrderedDict()
    for name, module in _iter_lora_layers(model):
        A = module.lora_A[adapter_name].weight.data.clone()  # (r, in_features)
        B = module.lora_B[adapter_name].weight.data.clone()  # (out_features, r)
        state[name] = {"A": A, "B": B}
    return state


def set_lora_state(
    model: nn.Module, state: LoRAState, adapter_name: str = "default"
) -> None:
    """Overwrite LoRA A and B weights in a PEFT model from a state dict."""
    for name, module in _iter_lora_layers(model):
        if name in state:
            module.lora_A[adapter_name].weight.data.copy_(state[name]["A"])
            module.lora_B[adapter_name].weight.data.copy_(state[name]["B"])


def set_lora_A(
    model: nn.Module, a_state: Dict[str, torch.Tensor], adapter_name: str = "default"
) -> None:
    """Set only the A weights.  a_state: {layer_name: Tensor(r, d)}."""
    for name, module in _iter_lora_layers(model):
        if name in a_state:
            module.lora_A[adapter_name].weight.data.copy_(a_state[name])


def set_lora_B(
    model: nn.Module, b_state: Dict[str, torch.Tensor], adapter_name: str = "default"
) -> None:
    """Set only the B weights.  b_state: {layer_name: Tensor(d, r)}."""
    for name, module in _iter_lora_layers(model):
        if name in b_state:
            module.lora_B[adapter_name].weight.data.copy_(b_state[name])


# ---------------------------------------------------------------------------
# LoRA-specific operations for SLAO
# ---------------------------------------------------------------------------

def orthogonal_init_A(A_prev: torch.Tensor) -> torch.Tensor:
    """Compute orthogonal initialization for A from previous task's A.

    Uses QR decomposition with sign correction so that A_new has orthonormal rows.

    Args:
        A_prev: (r, d) — fine-tuned A from the previous task.
    Returns:
        A_new: (r, d) with A_new @ A_new.T ≈ I_r.
    """
    Q, R = torch.linalg.qr(A_prev.T)  # (d, r), (r, r)
    sign = torch.sign(torch.diag(R))
    sign[sign == 0] = 1.0
    Q = Q * sign.unsqueeze(0)
    return Q.T  # (r, d)


def zca_whiten_A(A_prev: torch.Tensor) -> torch.Tensor:
    """Compute ZCA-whitened initialization for A from previous task's A.

    Returns (A A^T)^{-1/2} A, which satisfies A_new @ A_new.T = I_r
    and is the closest orthonormal-row matrix to A_prev in Frobenius norm.

    Args:
        A_prev: (r, d) — fine-tuned A from the previous task.
    Returns:
        A_new: (r, d) with A_new @ A_new.T ≈ I_r.
    """
    # A A^T is (r, r) — small, so eigendecomposition is cheap
    G = A_prev @ A_prev.T  # (r, r)
    # Symmetric eigendecomposition: G = V diag(eigvals) V^T
    eigvals, V = torch.linalg.eigh(G)
    # Clamp for numerical safety (all eigenvalues should be positive)
    eigvals = eigvals.clamp(min=1e-12)
    # G^{-1/2} = V diag(1/sqrt(eigvals)) V^T
    inv_sqrt = V @ torch.diag(eigvals.rsqrt()) @ V.T
    return inv_sqrt @ A_prev  # (r, d)


def merge_B(
    B_merge: torch.Tensor,
    B_ft: torch.Tensor,
    task_idx: int,
    fisher_B: torch.Tensor | None = None,
    beta: float = 0.0,
) -> torch.Tensor:
    """Time-aware merging of B matrices, optionally Fisher-weighted.

    Standard (beta=0):
        B_new = B_merge + λ * (B_ft - B_merge),  λ = 1/√i

    Fisher-weighted (beta>0):
        α_jk = λ / (1 + β · F̃_jk)
        B_new_jk = B_merge_jk + α_jk · (B_ft_jk - B_merge_jk)

    where F̃ = F / mean(F) is the normalized Fisher for this B matrix.

    Args:
        B_merge:  (d, r) current merged B.
        B_ft:     (d, r) fine-tuned B from current task.
        task_idx: 1-indexed task number (>= 2).
        fisher_B: (d, r) accumulated diagonal Fisher for this B. None = uniform.
        beta:     sensitivity to Fisher importance (0 = standard SLAO).
    """
    lam = 1.0 / math.sqrt(task_idx)

    if fisher_B is None or beta == 0:
        return B_merge + lam * (B_ft - B_merge)

    # Normalize Fisher to mean 1 so beta is scale-invariant
    f_mean = fisher_B.mean()
    if f_mean > 0:
        F_norm = fisher_B / f_mean
    else:
        F_norm = fisher_B

    alpha = lam / (1.0 + beta * F_norm)
    return B_merge + alpha * (B_ft - B_merge)


def merge_B_bayesian(
    B_merge: torch.Tensor,
    B_ft: torch.Tensor,
    fisher_old: torch.Tensor | None,
    fisher_new: torch.Tensor,
    eps: float = 1e-8,
    alpha_min: float = 0.01,
    alpha_max: float = 0.95,
    use_lambda_damping: bool = False,
    task_idx: int | None = None,
) -> torch.Tensor:
    """Bayesian posterior-style merge using both old and new Fisher.

    Derives per-parameter merge rate from the precision-weighted mean of
    two diagonal Gaussians:
        α_jk = F̃_new_jk / (F̃_old_jk + F̃_new_jk + ε)

    Both Fishers are normalized per-layer to mean 1 before combining,
    removing scale differences from accumulation.

    Args:
        B_merge:   (d, r) current merged B.
        B_ft:      (d, r) fine-tuned B from current task.
        fisher_old: (d, r) accumulated Fisher from tasks 1..i-1. None for task 1.
        fisher_new: (d, r) Fisher estimated on current task i.
        eps:       numerical floor for division.
        alpha_min: floor on per-param merge rate.
        alpha_max: cap on per-param merge rate.
        use_lambda_damping: multiply α by 1/√i (ablation).
        task_idx:  1-indexed, required if use_lambda_damping is True.
    """
    f_new_norm = fisher_new / fisher_new.mean().clamp(min=1e-12)

    if fisher_old is None:
        # First merge (task 1 → task 2): no old Fisher, use full new signal
        alpha = torch.ones_like(B_merge)
    else:
        f_old_norm = fisher_old / fisher_old.mean().clamp(min=1e-12)
        alpha = f_new_norm / (f_old_norm + f_new_norm + eps)

    alpha = alpha.clamp(min=alpha_min, max=alpha_max)

    if use_lambda_damping:
        assert task_idx is not None
        alpha = alpha * (1.0 / math.sqrt(task_idx))

    return B_merge + alpha * (B_ft - B_merge)


def clone_state(state: LoRAState) -> LoRAState:
    """Deep-clone a LoRA state dict."""
    return OrderedDict(
        (k, {"A": v["A"].clone(), "B": v["B"].clone()}) for k, v in state.items()
    )


# ---------------------------------------------------------------------------
# SVD parameterization utilities (for Stiefelized CLUE)
# ---------------------------------------------------------------------------

# Type for SVD state: {layer_name: {"U": (d_out, r), "S": (r,), "V": (d_in, r)}}
SVDState = Dict[str, Dict[str, torch.Tensor]]


def lora_to_svd(state: LoRAState) -> SVDState:
    """Decompose LoRA (A, B) into thin SVD: ΔW = B @ A = U @ diag(S) @ V^T.

    Args:
        state: {layer_name: {"A": (r, d_in), "B": (d_out, r)}}
    Returns:
        {layer_name: {"U": (d_out, r), "S": (r,), "V": (d_in, r)}}
    """
    svd_state: SVDState = OrderedDict()
    for name, tensors in state.items():
        A = tensors["A"]  # (r, d_in)
        B = tensors["B"]  # (d_out, r)
        dW = B @ A        # (d_out, d_in)
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        # Keep only rank r
        r = A.shape[0]
        svd_state[name] = {
            "U": U[:, :r].clone(),    # (d_out, r)
            "S": S[:r].clone(),       # (r,)
            "V": Vh[:r, :].T.clone(), # (d_in, r)
        }
    return svd_state


def svd_to_lora(svd_state: SVDState) -> LoRAState:
    """Convert SVD (U, S, V) back to LoRA (A, B).

    Convention: A = V^T (orthonormal rows), B = U @ diag(S).
    This puts all singular-value magnitude into B, keeping A with
    unit-norm rows — matching SLAO's orthogonal-init convention and
    preventing gradient amplification from large singular values.

    Args:
        svd_state: {layer_name: {"U": (d_out, r), "S": (r,), "V": (d_in, r)}}
    Returns:
        {layer_name: {"A": (r, d_in), "B": (d_out, r)}}
    """
    lora_state: LoRAState = OrderedDict()
    for name, tensors in svd_state.items():
        U = tensors["U"]  # (d_out, r)
        S = tensors["S"]  # (r,)
        V = tensors["V"]  # (d_in, r)
        A = V.T                            # (r, d_in) — orthonormal rows
        B = U @ torch.diag(S)             # (d_out, r)
        lora_state[name] = {"A": A, "B": B}
    return lora_state


def retract_tangent(
    base: SVDState,
    xi_U: Dict[str, torch.Tensor],
    xi_S: Dict[str, torch.Tensor],
    xi_V: Dict[str, torch.Tensor],
) -> SVDState:
    """Retract a tangent vector at base point to a new point on M_r.

    Given base (U, S, V) and tangent vector (ξ_U, ξ_S, ξ_V), compute:
        U_new = polar(U + ξ_U)  (project back to Stiefel manifold)
        S_new = S + ξ_S         (unconstrained)
        V_new = polar(V + ξ_V)  (project back to Stiefel manifold)

    Uses polar retraction: X → X (X^T X)^{-1/2}.

    Args:
        base: SVD state at the base point.
        xi_U: {layer: (d_out, r)} tangent perturbation for U.
        xi_S: {layer: (r,)} tangent perturbation for S.
        xi_V: {layer: (d_in, r)} tangent perturbation for V.
    Returns:
        New SVD state after retraction.
    """
    new_svd: SVDState = OrderedDict()
    for name in base:
        U = base[name]["U"]
        S = base[name]["S"]
        V = base[name]["V"]

        U_new = _polar_retract(U + xi_U[name])
        S_new = (S + xi_S[name]).clamp(min=1e-8)  # keep positive
        V_new = _polar_retract(V + xi_V[name])

        new_svd[name] = {"U": U_new, "S": S_new, "V": V_new}
    return new_svd


def _polar_retract(X: torch.Tensor) -> torch.Tensor:
    """Polar retraction: X → X (X^T X)^{-1/2}.

    Projects a (d, r) matrix onto the Stiefel manifold St(d, r).
    Uses eigendecomposition of the small (r, r) Gram matrix.
    """
    G = X.T @ X  # (r, r)
    eigvals, eigvecs = torch.linalg.eigh(G)
    eigvals = eigvals.clamp(min=1e-12)
    inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    return X @ inv_sqrt


def project_euclidean_to_tangent(
    grad_A: torch.Tensor,
    grad_B: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Euclidean gradients on (A, B) to tangent-space gradients (ξ_U, ξ_S, ξ_V).

    The LoRA product is ΔW = B @ A = U diag(S) V^T.
    The Euclidean gradient of the loss w.r.t. ΔW is:
        grad_W = grad_B @ A + B @ grad_A  (by chain rule on B @ A)

    Actually, more precisely, we use the fact that:
        d(loss)/d(ΔW) can be decomposed in the tangent space of M_r at (U, S, V).

    The tangent-space components are (see e.g. Vandereycken 2013):
        grad_S = diag(U^T grad_W V)                    — (r,)
        grad_U = (I - U U^T) grad_W V diag(S)^{-1}    — (d_out, r), horizontal
        grad_V = (I - V V^T) grad_W^T U diag(S)^{-1}  — (d_in, r), horizontal

    But we receive grad_A and grad_B, not grad_W. Reconstruct:
        grad_W = grad_B @ A.T   ... no, that's not right either.

    The chain rule for L(B @ A) gives:
        dL/dA = B^T @ dL/d(ΔW)   →   grad_W = B^{-T} @ grad_A  (underdetermined)
        dL/dB = dL/d(ΔW) @ A^T   →   grad_W = grad_B @ A^{-T}  (underdetermined)

    Simplest correct approach: reconstruct grad_W from the two gradients.
    Since A = diag(sqrt(S)) V^T and B = U diag(sqrt(S)):
        grad_W = grad_B @ (diag(sqrt(S)) V^T)^{-T}  ... still ambiguous.

    Instead, use the direct formula. For B @ A with B = (d_out, r), A = (r, d_in):
        dL/d(B@A) contracted with tangent of (U, S, V):

    Actually the cleanest approach:
        grad_W_approx = grad_B @ A     (this is dL/d(ΔW) @ A^T @ A ... no)

    Let's use the correct approach. grad_B and grad_A are the partial derivatives:
        grad_B = dL/dW @ A^T   where W = B@A
        grad_A = B^T @ dL/dW

    So: dL/dW = grad_B @ (A^T)^{+} = grad_B @ A^T @ (A @ A^T)^{-1}
    Or: dL/dW = (B^T)^{+} @ grad_A = (B @ B^T)^{-1} @ B @ grad_A

    With our SVD parameterization A = diag(√S) V^T, B = U diag(√S):
        B^T = diag(√S) U^T
        (B^T)^+ = U diag(1/√S)
        dL/dW = U diag(1/√S) @ grad_A

    Then decompose dL/dW in the tangent space:
        grad_S_k = U_k^T (dL/dW) V_k
        grad_U = (I - UU^T) dL/dW V S^{-1}
        grad_V = (I - VV^T) dL/dW^T U S^{-1}

    Args:
        grad_A: (r, d_in) gradient of loss w.r.t. A.
        grad_B: (d_out, r) gradient of loss w.r.t. B.
        U: (d_out, r) current Stiefel factor.
        S: (r,) current singular values.
        V: (d_in, r) current Stiefel factor.
    Returns:
        (grad_xi_U, grad_xi_S, grad_xi_V) tangent-space gradients.
    """
    sqrt_S = S.clamp(min=1e-8).sqrt()
    inv_sqrt_S = 1.0 / sqrt_S

    # Reconstruct grad_W = U diag(1/√S) @ grad_A
    grad_W = U @ (torch.diag(inv_sqrt_S) @ grad_A)  # (d_out, d_in)

    # Tangent-space decomposition
    # Σ component: diag of U^T grad_W V
    UtgW = U.T @ grad_W   # (r, d_in)
    grad_xi_S = (UtgW @ V).diag()  # (r,)

    # U component (horizontal): (I - U U^T) grad_W V / S
    inv_S = 1.0 / S.clamp(min=1e-8)
    grad_xi_U = (grad_W @ V - U @ torch.diag(grad_xi_S)) @ torch.diag(inv_S)  # (d_out, r)

    # V component (horizontal): (I - V V^T) grad_W^T U / S
    gWtU = grad_W.T @ U   # (d_in, r)
    VtgWtU = V.T @ gWtU   # (r, r) — this equals UtgW @ V = has grad_xi_S on diag
    grad_xi_V = (gWtU - V @ torch.diag(grad_xi_S)) @ torch.diag(inv_S)  # (d_in, r)

    return grad_xi_U, grad_xi_S, grad_xi_V


def merge_tangent_bayesian(
    xi: torch.Tensor,
    fisher_old: torch.Tensor | None,
    fisher_new: torch.Tensor,
    eps: float = 1e-8,
    alpha_min: float = 0.01,
    alpha_max: float = 0.95,
) -> torch.Tensor:
    """Bayesian merge of a tangent-space component.

    Since the previous task's tangent vector is 0 by construction
    (we re-anchor at the merged point), the merge simplifies to scaling:
        ξ_merge = α ⊙ ξ_ft

    where α = F_new / (F_old + F_new + ε), both normalized to mean 1.

    Args:
        xi: tangent vector component (any shape).
        fisher_old: accumulated Fisher for this component. None for first merge.
        fisher_new: Fisher for current task.
        eps, alpha_min, alpha_max: clipping parameters.
    Returns:
        Merged tangent vector.
    """
    f_new_norm = fisher_new / fisher_new.mean().clamp(min=1e-12)

    if fisher_old is None:
        alpha = torch.ones_like(xi)
    else:
        f_old_norm = fisher_old / fisher_old.mean().clamp(min=1e-12)
        alpha = f_new_norm / (f_old_norm + f_new_norm + eps)

    alpha = alpha.clamp(min=alpha_min, max=alpha_max)
    return alpha * xi


def merge_tangent_uniform(
    xi: torch.Tensor,
    task_idx: int,
) -> torch.Tensor:
    """Uniform time-aware merge of a tangent vector.

    Since ξ_prev = 0:  ξ_merge = λ(i) * ξ_ft,  λ(i) = 1/√i.
    """
    lam = 1.0 / math.sqrt(task_idx)
    return lam * xi
