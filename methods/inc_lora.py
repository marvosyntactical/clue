"""IncLoRA baseline: incrementally add a new LoRA per task, freeze previous.

At inference, apply the sum of all per-task LoRA deltas (B_i @ A_i) on top
of the base weights.  This preserves old knowledge by freezing, but memory
grows linearly with the number of tasks.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from methods.base import ContinualMethod
from models.lora import LoRAState, extract_lora_state, set_lora_state, clone_state
from utils import get_logger

logger = get_logger(__name__)


class IncLoRA(ContinualMethod):
    """Incremental LoRA — one frozen adapter per task, summed at inference."""

    def __init__(self, model: nn.Module, args):
        super().__init__(model, args)
        self.saved_states: List[LoRAState] = []  # one per completed task

    def before_task(self, task_idx: int, task_name: str) -> None:
        """Re-initialize LoRA weights to default (small random A, zero B)."""
        logger.info(f"IncLoRA: re-initializing LoRA for task {task_idx} ({task_name})")
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # Kaiming init for A, zero for B (PEFT default)
                nn.init.kaiming_uniform_(module.lora_A["default"].weight)
                nn.init.zeros_(module.lora_B["default"].weight)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = []
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                params.append(param)
        return params

    def after_task(self, task_idx: int) -> None:
        """Save the current LoRA state and accumulate into merged weights."""
        state = extract_lora_state(self.model)
        self.saved_states.append(clone_state(state))
        logger.info(
            f"IncLoRA: saved adapter {task_idx}, "
            f"total adapters = {len(self.saved_states)}"
        )
        # Apply the summed delta for evaluation
        self._apply_summed_state()

    def _apply_summed_state(self) -> None:
        """Sum all saved LoRA deltas and set them on the model."""
        if not self.saved_states:
            return

        # Use the first state as a template for layer names
        layer_names = list(self.saved_states[0].keys())
        summed: LoRAState = {}

        for ln in layer_names:
            # Sum the low-rank products: sum_i (B_i @ A_i)
            # Then factor back into a single (B, A) via SVD for rank-r approx
            # Simpler approach: since all have same rank, sum B and average A
            # Actually, the correct approach is to sum the deltas directly:
            # delta_total = sum_i (B_i @ A_i)
            # We store this as B_sum, A_sum via truncated SVD of the sum.
            first = self.saved_states[0][ln]
            r = first["A"].shape[0]
            d_in = first["A"].shape[1]
            d_out = first["B"].shape[0]
            device = first["A"].device
            dtype = first["A"].dtype

            delta_sum = torch.zeros(d_out, d_in, device=device, dtype=dtype)
            for st in self.saved_states:
                delta_sum += st[ln]["B"] @ st[ln]["A"]

            # Truncated SVD to fit back into rank r
            U, S, Vt = torch.linalg.svd(delta_sum, full_matrices=False)
            # Keep top-r components
            B_approx = U[:, :r] * S[:r].unsqueeze(0)  # (d_out, r)
            A_approx = Vt[:r, :]                        # (r, d_in)
            summed[ln] = {"A": A_approx, "B": B_approx}

        set_lora_state(self.model, summed)

    def get_model(self) -> nn.Module:
        self._apply_summed_state()
        return self.model