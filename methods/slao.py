"""SLAO: Single LoRA continual learning with Orthogonal initialization.

Implements the method from "Merge before Forget" (Qiao & Mahdavi, 2025).

Lifecycle per task i (0-indexed):
  before_task: orthogonal-init A from previous task, copy previous B
  [training]
  after_task:  merge B with time-aware scaling, replace A directly
"""

from typing import List

import torch.nn as nn

from methods.base import ContinualMethod
from models.lora import (
    LoRAState,
    clone_state,
    extract_lora_state,
    merge_B,
    orthogonal_init_A,
    set_lora_state,
)
from utils import get_logger

logger = get_logger(__name__)


class SLAO(ContinualMethod):
    """SLAO continual learning method."""

    def __init__(self, model: nn.Module, args):
        super().__init__(model, args)
        # State across tasks
        self.ft_state: LoRAState | None = None     # last fine-tuned state
        self.merge_state: LoRAState | None = None   # running merged state

    def before_task(self, task_idx: int, task_name: str) -> None:
        """Initialize LoRA for the new task.

        Task 0: use default PEFT initialization (handled externally).
        Task i>0: orthogonal-init A from previous ft A, copy previous ft B.
        """
        if task_idx == 0:
            logger.info("Task 0: using default LoRA initialization")
            return

        assert self.ft_state is not None, "ft_state must exist after task 0"

        logger.info(
            f"Task {task_idx} ({task_name}): "
            "orthogonal A init + previous B init"
        )
        new_state = {}
        for layer_name, prev in self.ft_state.items():
            A_new = orthogonal_init_A(prev["A"])  # (r, d)
            B_new = prev["B"].clone()             # (d, r)
            new_state[layer_name] = {"A": A_new, "B": B_new}

        set_lora_state(self.model, new_state)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """All LoRA A and B parameters."""
        params = []
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                params.append(param)
        return params

    def after_task(self, task_idx: int) -> None:
        """Snapshot fine-tuned state and merge B.

        Task 0: just store as both ft_state and merge_state.
        Task i>0: A_merge = A_ft_i (direct), B_merge via time-aware scaling.
        """
        self.ft_state = extract_lora_state(self.model)

        if task_idx == 0:
            self.merge_state = clone_state(self.ft_state)
            logger.info("Task 0: stored initial merged state")
            return

        # Paper uses 1-indexed task_idx for lambda; task_idx=1 -> i=2 in paper
        paper_i = task_idx + 1  # convert 0-indexed to 1-indexed

        new_merge: LoRAState = {}
        for layer_name in self.ft_state:
            A_ft = self.ft_state[layer_name]["A"]
            B_ft = self.ft_state[layer_name]["B"]
            B_prev_merge = self.merge_state[layer_name]["B"]

            # Asymmetric merge
            A_merged = A_ft.clone()  # direct replacement
            B_merged = merge_B(B_prev_merge, B_ft, paper_i)

            new_merge[layer_name] = {"A": A_merged, "B": B_merged}

        self.merge_state = new_merge
        logger.info(
            f"Task {task_idx}: merged (λ={1.0 / (paper_i ** 0.5):.4f})"
        )

        # Apply merged state to model for evaluation
        set_lora_state(self.model, self.merge_state)

    def get_model(self) -> nn.Module:
        """Return model with merged LoRA weights applied."""
        if self.merge_state is not None:
            set_lora_state(self.model, self.merge_state)
        return self.model