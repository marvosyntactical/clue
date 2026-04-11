"""Gradient Projection Memory for LoRA Continual Learning (Extension 4).

After each task, identifies the important input-activation subspace at each
LoRA layer via SVD.  During subsequent tasks, projects lora_A gradients to be
orthogonal to the accumulated subspace, preventing interference with
previously learned representations.

Only lora_A gradients are projected (lora_B is protected by SLAO's EMA
merging and optionally by Fisher regularization).

Reference: Saha, Garg & Roy, "Gradient Projection Memory for Continual
Learning" (ICLR 2021, arXiv:2103.09762), adapted for LoRA.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class GradientProjectionMemory:
    """Per-layer gradient projection memory for LoRA continual learning."""

    def __init__(self, model: nn.Module, threshold: float = 0.95):
        self.threshold = threshold
        self.model = model
        # {layer_name: Tensor of shape (d_in, k_accumulated)} in float32
        self.memory: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update_memory(self, data_loader: DataLoader, n_samples: int = 256):
        """Extract and store activation subspaces after training a task.

        Collects input activations at each LoRA-injected layer, removes
        directions already in memory, performs SVD on the residual, and
        appends new basis vectors that meet the threshold criterion.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # 1. Register hooks to capture inputs at each LoRA layer
        activations: dict[str, list[torch.Tensor]] = {}
        hooks = []
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                act_list: list[torch.Tensor] = []
                activations[name] = act_list

                def _hook(mod, inp, out, store=act_list):
                    # inp[0] shape: (batch, seq_len, d_in) or (batch*seq, d_in)
                    x = inp[0].detach()
                    store.append(x.reshape(-1, x.shape[-1]))

                hooks.append(module.register_forward_hook(_hook))

        # 2. Forward pass on reference samples
        count = 0
        for batch in data_loader:
            if count >= n_samples:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            self.model(input_ids=input_ids, attention_mask=attention_mask)
            count += input_ids.size(0)

        for h in hooks:
            h.remove()

        # 3. Per-layer: SVD on residual activations, update memory
        for layer_name, act_list in activations.items():
            if not act_list:
                continue
            # R: (d_in, n_tokens) — each column is an activation vector
            R = torch.cat(act_list, dim=0).T.float()

            R_norm_sq = R.norm() ** 2
            if R_norm_sq < 1e-12:
                continue

            if layer_name in self.memory:
                M = self.memory[layer_name].to(R.device)  # (d_in, k_prev)
                R_proj = M @ (M.T @ R)
                R_hat = R - R_proj
                proj_norm_sq = R_proj.norm() ** 2
            else:
                R_hat = R
                proj_norm_sq = 0.0

            # Economy SVD on residual
            U, S, _ = torch.linalg.svd(R_hat, full_matrices=False)

            if S.numel() == 0:
                continue

            # Find smallest k s.t. proj_norm² + sum(S[:k]²) >= threshold * R_norm²
            cumsum = torch.cumsum(S ** 2, dim=0)
            target = self.threshold * R_norm_sq - proj_norm_sq

            if target <= 0:
                # Existing memory already covers enough
                continue

            mask = cumsum >= target
            if mask.any():
                k = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
            else:
                k = S.numel()
            k = max(1, min(k, U.shape[1]))

            new_bases = U[:, :k]  # (d_in, k)

            if layer_name in self.memory:
                self.memory[layer_name] = torch.cat(
                    [self.memory[layer_name].to(R.device), new_bases], dim=1
                )
            else:
                self.memory[layer_name] = new_bases

    def project_grads(self):
        """Project lora_A gradients orthogonal to stored subspaces.

        For each LoRA layer with memory:
            grad_A' = grad_A - grad_A @ M @ M^T

        Call after loss.backward() and grad clipping, before optimizer.step().
        """
        for name, module in self.model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if name not in self.memory:
                continue
            a_param = module.lora_A["default"].weight
            if a_param.grad is None:
                continue

            M = self.memory[name].to(a_param.grad.device)  # (d_in, k)
            grad = a_param.grad.data  # (r, d_in)
            a_param.grad.data = grad - (grad @ M) @ M.T
