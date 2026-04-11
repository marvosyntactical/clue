"""Riemannian Preconditioned LoRA (Extension 1).

Applies r×r preconditioners to LoRA gradients before the optimizer step:
    grad_A' = (B^T B + δI)^{-1} @ grad_A
    grad_B' = grad_B @ (A A^T + δI)^{-1}

Reference: Prakash et al., "Riemannian Preconditioned LoRA for Fine-Tuning
Foundation Models" (arXiv:2402.02347)
"""

import torch
import torch.nn as nn


class RiemannianPreconditioner:
    """Applies Riemannian preconditioning to LoRA A and B gradients."""

    def __init__(self, model: nn.Module, delta: float = 1e-6):
        self.delta = delta
        # Map each lora_A param to its partner lora_B module, and vice versa.
        self.a_params = []   # list of (a_weight, b_module)
        self.b_params = []   # list of (b_weight, a_module)
        for _name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                a_mod = module.lora_A["default"]
                b_mod = module.lora_B["default"]
                self.a_params.append((a_mod.weight, b_mod))
                self.b_params.append((b_mod.weight, a_mod))

    def precondition_grads(self):
        """Precondition all LoRA gradients in-place.

        Call after loss.backward() and grad clipping, before optimizer.step().
        """
        for a_weight, b_mod in self.a_params:
            if a_weight.grad is None:
                continue
            B = b_mod.weight.data  # (d_out, r)
            r = B.shape[1]
            # (B^T B + δI)^{-1} @ grad_A  — solve is more stable than inv
            BtB = B.T @ B  # (r, r)
            BtB.diagonal().add_(self.delta)
            # grad_A is (r, d_in), solve expects (r, r) @ X = (r, d_in)
            a_weight.grad.data = torch.linalg.solve(BtB, a_weight.grad.data)

        for b_weight, a_mod in self.b_params:
            if b_weight.grad is None:
                continue
            A = a_mod.weight.data  # (r, d_in)
            r = A.shape[0]
            # grad_B @ (A A^T + δI)^{-1}  — shape: (d_out, r) @ (r, r)
            AAt = A @ A.T  # (r, r)
            AAt.diagonal().add_(self.delta)
            AAt_inv = torch.linalg.inv(AAt)
            b_weight.grad.data = b_weight.grad.data @ AAt_inv
