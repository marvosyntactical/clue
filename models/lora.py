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


def merge_B(
    B_merge: torch.Tensor, B_ft: torch.Tensor, task_idx: int
) -> torch.Tensor:
    """Time-aware merging of B matrices.

    B_merge_new = B_merge + λ(i) * (B_ft - B_merge), λ(i) = 1/√i.

    Args:
        B_merge: (d, r) current merged B.
        B_ft:    (d, r) fine-tuned B from current task.
        task_idx: 1-indexed task number (>= 2).
    """
    lam = 1.0 / math.sqrt(task_idx)
    return B_merge + lam * (B_ft - B_merge)


def clone_state(state: LoRAState) -> LoRAState:
    """Deep-clone a LoRA state dict."""
    return OrderedDict(
        (k, {"A": v["A"].clone(), "B": v["B"].clone()}) for k, v in state.items()
    )
