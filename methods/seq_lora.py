"""SeqLoRA baseline: sequentially fine-tune a single LoRA with no protection.

This is the simplest baseline — train LoRA on task 1, then continue training
the same LoRA on task 2, etc.  No merging, no orthogonal init, no freezing.
Severe forgetting is expected.
"""

from typing import List

import torch.nn as nn

from methods.base import ContinualMethod
from utils import get_logger

logger = get_logger(__name__)


class SeqLoRA(ContinualMethod):
    """Sequential LoRA — train the same adapter on each task in sequence."""

    def before_task(self, task_idx: int, task_name: str) -> None:
        logger.info(f"SeqLoRA: starting task {task_idx} ({task_name}), no special init")

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = []
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                params.append(param)
        return params

    def after_task(self, task_idx: int) -> None:
        logger.info(f"SeqLoRA: finished task {task_idx}, no merging")