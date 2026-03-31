"""Base class for continual learning methods."""

from abc import ABC, abstractmethod
from typing import List

import torch.nn as nn


class ContinualMethod(ABC):
    """Interface that all CL methods must implement.

    Lifecycle per task:
        1. before_task(task_idx, task_name)   — set up LoRA init, freeze params, etc.
        2. get_trainable_params()             — return params for the optimizer
        3. [external training loop runs]
        4. after_task(task_idx)               — merge, snapshot, etc.
    """

    def __init__(self, model: nn.Module, args):
        self.model = model
        self.args = args

    @abstractmethod
    def before_task(self, task_idx: int, task_name: str) -> None:
        """Called before training on task `task_idx` (0-indexed)."""

    @abstractmethod
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return the parameters that should be optimized for the current task."""

    @abstractmethod
    def after_task(self, task_idx: int) -> None:
        """Called after training on task `task_idx` completes."""

    def get_model(self) -> nn.Module:
        """Return the model ready for inference (with merged/applied weights)."""
        return self.model
