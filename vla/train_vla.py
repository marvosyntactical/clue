#!/usr/bin/env python3
"""Main entrypoint for VLA continual learning on LIBERO.

Structural analogue of clue/train.py, replacing text tokenization and
generation-based evaluation with OpenVLA's image+action processing and
LIBERO rollout evaluation.

Reuses unchanged:
    methods/slao.py      — orthogonal init, asymmetric merge
    methods/fisher.py    — Fisher estimation, EWC penalty
    models/lora.py       — merge_B, extract_lora_state, etc.
    eval/metrics.py      — AccuracyMatrix, AA, BWT

Example usage:
    python -m vla.train_vla \
        --suite libero_object \
        --method slao \
        --fisher_merge_beta 0.5 \
        --fisher_lambda 0.1 \
        --fisher_gamma 0.9 \
        --epochs 3 \
        --lr 5e-4 \
        --batch_size 16 \
        --grad_accum 4 \
        --lora_rank 32 \
        --seed 42 \
        --output_dir outputs/vla_clue_libero_object
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent dir to path so we can import from clue/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import AccuracyMatrix
from methods.slao import SLAO
from methods.seq_lora import SeqLoRA
from methods.fisher import DiagonalFisher
from utils import get_logger, set_seed

from vla.data.libero_loader import (
    ActionNormStats,
    LiberoTaskDataset,
    get_libero_task_order,
)
from vla.eval_vla import evaluate_libero_task

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "slao": SLAO,
    "seq_lora": SeqLoRA,
}

OPTIMIZER_REGISTRY = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLA Continual Learning on LIBERO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_name", type=str, default="openvla/openvla-7b",
                    help="OpenVLA model name or local path")
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])

    # LoRA (OpenVLA defaults)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str, default="all-linear",
                    help="LoRA targets: 'all-linear' or comma-separated module names")

    # Method
    p.add_argument("--method", type=str, default="slao",
                    choices=list(METHOD_REGISTRY.keys()))
    p.add_argument("--a_init_method", type=str, default="qr",
                    choices=["qr", "zca"],
                    help="A-matrix init: 'qr' (Gram-Schmidt) or 'zca' (whitening)")

    # LIBERO benchmark
    p.add_argument("--suite", type=str, default="libero_object",
                    choices=["libero_object", "libero_spatial",
                             "libero_goal", "libero_long"],
                    help="LIBERO task suite")
    p.add_argument("--data_root", type=str, default="/data/libero_rlds",
                    help="Root directory for RLDS datasets")
    p.add_argument("--max_steps", type=int, default=None,
                    help="Max rollout steps (default: suite-dependent)")

    # Training
    p.add_argument("--optimizer", type=str, default="adamw",
                    choices=list(OPTIMIZER_REGISTRY.keys()))
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--image_aug", action="store_true", default=True,
                    help="Enable random crop augmentation (OpenVLA default)")
    p.add_argument("--no_image_aug", dest="image_aug", action="store_false")

    # Extensions (CLUE)
    p.add_argument("--fisher_lambda", type=float, default=0.0,
                    help="EWC regularization strength (0 = disabled)")
    p.add_argument("--fisher_gamma", type=float, default=0.9,
                    help="Fisher EMA decay (online EWC)")
    p.add_argument("--fisher_samples", type=int, default=256,
                    help="Samples for Fisher estimation per task")
    p.add_argument("--fisher_merge_beta", type=float, default=0.0,
                    help="Fisher-weighted B merge sensitivity (0 = standard SLAO)")
    p.add_argument("--bayesian_merge", action="store_true",
                    help="Use Bayesian posterior-style merge")
    p.add_argument("--bayesian_alpha_min", type=float, default=0.01)
    p.add_argument("--bayesian_alpha_max", type=float, default=0.95)
    p.add_argument("--bayesian_lambda_damping", action="store_true")
    p.add_argument("--lora_plus_ratio", type=float, default=1.0,
                    help="LoRA+ B_lr/A_lr ratio")

    # Evaluation
    p.add_argument("--n_rollouts", type=int, default=20,
                    help="Rollout episodes per task per evaluation")
    p.add_argument("--n_parallel_envs", type=int, default=8,
                    help="Parallel LIBERO environments for evaluation")
    p.add_argument("--eval_tasks", type=str, default="all",
                    help="'all' or comma-separated task indices to eval after "
                         "(e.g., '0,4,9' for iteration)")

    # Reproducibility & output
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs/vla_default")
    p.add_argument("--save_adapters", action="store_true")

    args = p.parse_args()

    # Default max_steps per suite
    if args.max_steps is None:
        args.max_steps = 600 if args.suite == "libero_long" else 300

    return args


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_openvla(args):
    """Load OpenVLA model and processor, apply LoRA.

    Returns:
        model: PEFT-wrapped OpenVLA model.
        processor: OpenVLA processor (image + text tokenization).
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    logger.info(f"Loading OpenVLA model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Parse target modules
    if args.lora_target_modules == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Freeze base model
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    return model, processor


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model, trainable_params, args) -> torch.optim.Optimizer:
    """Build optimizer with optional LoRA+ asymmetric LR."""
    cls = OPTIMIZER_REGISTRY[args.optimizer]
    opt_kwargs = {"weight_decay": args.weight_decay}

    if args.lora_plus_ratio != 1.0:
        a_params, b_params = [], []
        trainable_ids = {id(p) for p in trainable_params}
        for name, param in model.named_parameters():
            if id(param) not in trainable_ids:
                continue
            if "lora_A" in name:
                a_params.append(param)
            elif "lora_B" in name:
                b_params.append(param)
        param_groups = [
            {"params": a_params, "lr": args.lr},
            {"params": b_params, "lr": args.lr * args.lora_plus_ratio},
        ]
        return cls(param_groups, **opt_kwargs)
    else:
        return cls(trainable_params, lr=args.lr, **opt_kwargs)


# ---------------------------------------------------------------------------
# Training loop for one task
# ---------------------------------------------------------------------------

def train_one_task_vla(
    model: nn.Module,
    processor,
    dataset: LiberoTaskDataset,
    method,
    args,
    fisher: DiagonalFisher | None = None,
) -> float:
    """Fine-tune LoRA on one LIBERO task. Returns average training loss.

    The forward pass uses OpenVLA's action-token cross-entropy loss,
    identical to standard LoRA fine-tuning.
    """
    device = next(model.parameters()).device
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )

    params = method.get_trainable_params()
    optimizer = build_optimizer(model, params, args)
    optimizer.zero_grad()

    has_fisher = fisher is not None and fisher.fisher
    total_loss = 0.0
    step = 0
    global_step = 0

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            # Prepare inputs for OpenVLA forward pass
            # OpenVLA expects processed image + text inputs and action labels
            loss = _forward_vla(
                model, processor, batch, device
            )
            loss = loss / args.grad_accum

            # EWC penalty
            if has_fisher:
                loss = loss + (args.fisher_lambda / 2) * fisher.penalty() / args.grad_accum

            loss.backward()
            total_loss += loss.item() * args.grad_accum
            step += 1

            if step % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg = total_loss / step
                    logger.info(f"  step {global_step}, avg_loss={avg:.4f}")

        # Flush remaining gradients at end of epoch
        if step % args.grad_accum != 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        logger.info(f"  Epoch {epoch+1}/{args.epochs} done, avg_loss={total_loss/max(step,1):.4f}")

    return total_loss / max(step, 1)


def _forward_vla(
    model: nn.Module,
    processor,
    batch: dict,
    device: str,
) -> torch.Tensor:
    """Process a batch through OpenVLA and return the loss.

    OpenVLA predicts action tokens via cross-entropy. Each action dimension
    is discretized into 256 bins, and the model generates 7 tokens (one per
    DoF) autoregressively.

    This function handles the image + instruction → action token loss pipeline.
    """
    from PIL import Image

    images = batch["image"]           # (B, 3, H, W) float tensor
    instructions = batch["instruction"]  # list of str
    actions = batch["action"]         # (B, 7) float tensor

    batch_size = images.shape[0]

    # Convert images to PIL for processor (OpenVLA expects PIL images)
    pil_images = []
    for i in range(batch_size):
        img_np = (images[i].permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil_images.append(Image.fromarray(img_np))

    # Format prompts
    prompts = [
        f"In: What action should the robot take to {inst}?\nOut:"
        for inst in instructions
    ]

    # Process through OpenVLA's processor
    inputs = processor(
        text=prompts,
        images=pil_images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Discretize actions into token IDs for labels
    action_token_ids = _discretize_actions(actions, device=device)

    # Create labels: input tokens are -100 (ignored), action tokens are targets
    input_len = inputs["input_ids"].shape[1]
    labels = torch.full(
        (batch_size, input_len + 7),
        fill_value=-100,
        dtype=torch.long,
        device=device,
    )
    labels[:, input_len:] = action_token_ids

    # Pad input_ids with action tokens so the model sees the full sequence
    full_input_ids = torch.cat(
        [inputs["input_ids"], action_token_ids], dim=1
    )
    # Extend attention mask
    action_mask = torch.ones(
        batch_size, 7, dtype=inputs["attention_mask"].dtype, device=device
    )
    full_attention_mask = torch.cat(
        [inputs["attention_mask"], action_mask], dim=1
    )

    outputs = model(
        input_ids=full_input_ids,
        attention_mask=full_attention_mask,
        labels=labels,
        **{k: v for k, v in inputs.items()
           if k not in ("input_ids", "attention_mask")},
    )
    return outputs.loss


def _discretize_actions(
    actions: torch.Tensor,
    n_bins: int = 256,
    action_token_offset: int = 32000,
    device: str = "cuda",
) -> torch.Tensor:
    """Discretize continuous actions into OpenVLA token IDs.

    Maps each action dimension from [-1, 1] to one of n_bins token IDs,
    offset by action_token_offset in the vocabulary.

    Args:
        actions: (B, 7) float tensor in [-1, 1].
        n_bins: Number of discretization bins (OpenVLA uses 256).
        action_token_offset: Starting vocabulary index for action tokens.

    Returns:
        (B, 7) long tensor of token IDs.
    """
    # Clamp to [-1, 1] then map to [0, n_bins-1]
    actions_clamped = actions.clamp(-1.0, 1.0)
    bins = ((actions_clamped + 1.0) / 2.0 * (n_bins - 1)).round().long()
    bins = bins.clamp(0, n_bins - 1)
    return (bins + action_token_offset).to(device)


# ---------------------------------------------------------------------------
# Fisher estimation helper (uses same forward pass)
# ---------------------------------------------------------------------------

def _make_fisher_loader(dataset: LiberoTaskDataset, args) -> DataLoader:
    """Create a DataLoader for Fisher estimation."""
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        drop_last=False,
    )


def estimate_fisher_vla(
    fisher: DiagonalFisher,
    model: nn.Module,
    processor,
    dataset: LiberoTaskDataset,
    args,
    n_samples: int = 256,
) -> dict[str, torch.Tensor]:
    """Estimate diagonal Fisher on VLA action-prediction loss.

    Reimplements DiagonalFisher._compute_fisher using the VLA forward pass
    instead of the LLM forward pass.
    """
    model.eval()
    device = next(model.parameters()).device

    loader = _make_fisher_loader(dataset, args)
    fisher_new: dict[str, torch.Tensor] = {}
    count = 0

    for batch in loader:
        if count >= n_samples:
            break

        model.zero_grad()
        with torch.enable_grad():
            loss = _forward_vla(model, processor, batch, device)
            loss.backward()

        for name, param in model.named_parameters():
            if "lora_" in name and param.grad is not None:
                if name not in fisher_new:
                    fisher_new[name] = torch.zeros_like(param.data)
                fisher_new[name].add_(param.grad.data ** 2)

        count += batch["image"].shape[0]

    model.zero_grad()
    for name in fisher_new:
        fisher_new[name].div_(max(count, 1))

    return fisher_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # ---- Task order ----
    task_names = get_libero_task_order(args.suite)
    config["task_order_resolved"] = task_names
    num_tasks = len(task_names)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Suite: {args.suite}, {num_tasks} tasks")
    for i, t in enumerate(task_names):
        logger.info(f"  Task {i}: {t.replace('_', ' ')}")

    # ---- Load model ----
    model, processor = load_openvla(args)
    device = next(model.parameters()).device

    # ---- Instantiate CL method ----
    method_cls = METHOD_REGISTRY[args.method]
    method = method_cls(model, args)

    # ---- Extensions ----
    fisher = None
    needs_fisher = (
        args.fisher_lambda > 0
        or args.fisher_merge_beta > 0
        or args.bayesian_merge
    )
    if needs_fisher:
        fisher = DiagonalFisher(model, gamma=args.fisher_gamma)
        parts = []
        if args.fisher_lambda > 0:
            ewc_mode = "online" if args.fisher_gamma < 1.0 else "standard"
            parts.append(f"{ewc_mode} EWC (λ={args.fisher_lambda}, γ={args.fisher_gamma})")
        if args.fisher_merge_beta > 0:
            parts.append(f"Fisher-weighted merge (β={args.fisher_merge_beta})")
        if args.bayesian_merge:
            parts.append(
                f"Bayesian merge (α∈[{args.bayesian_alpha_min}, "
                f"{args.bayesian_alpha_max}])"
            )
        logger.info(f"Extension: Fisher enabled — {', '.join(parts)}")

    if fisher is not None and hasattr(method, "fisher"):
        method.fisher = fisher

    if args.lora_plus_ratio != 1.0:
        logger.info(f"Extension: LoRA+ enabled (ratio={args.lora_plus_ratio})")

    if args.a_init_method != "qr":
        logger.info(f"Extension: A init method = {args.a_init_method.upper()}")

    # ---- Accuracy matrix ----
    acc_matrix = AccuracyMatrix(num_tasks)

    # ---- Action normalization ----
    action_norm = ActionNormStats()

    # ---- Parse eval schedule ----
    if args.eval_tasks == "all":
        eval_after = set(range(num_tasks))
    else:
        eval_after = {int(x) for x in args.eval_tasks.split(",")}

    # ---- Main CL loop ----
    for task_idx in range(num_tasks):
        task_name = task_names[task_idx]
        task_instruction = task_name.replace("_", " ")

        logger.info(f"\n{'='*60}")
        logger.info(f"TASK {task_idx}/{num_tasks-1}: {task_instruction}")
        logger.info(f"{'='*60}")

        # 1. Before task (orthogonal init, etc.)
        method.before_task(task_idx, task_name)

        # 2. Load task training data
        train_ds = LiberoTaskDataset(
            suite_name=args.suite,
            task_idx=task_idx,
            data_root=args.data_root,
            image_aug=args.image_aug,
        )
        logger.info(f"Training samples: {len(train_ds)}")

        # Compute and store action normalization stats
        action_norm.compute_from_dataset(train_ds, task_idx)

        # 3. Train
        t0 = time.time()
        avg_loss = train_one_task_vla(
            model, processor, train_ds, method, args,
            fisher=fisher,
        )
        elapsed = time.time() - t0
        logger.info(f"Training loss: {avg_loss:.4f}, time: {elapsed:.1f}s")

        # 4. Pre-merge Fisher estimation
        fisher_new = None
        if fisher is not None:
            if args.bayesian_merge:
                fisher_new = estimate_fisher_vla(
                    fisher, model, processor, train_ds, args,
                    n_samples=args.fisher_samples,
                )
                if hasattr(method, "fisher_new"):
                    method.fisher_new = fisher_new
                logger.info("Fisher: estimated F_new for Bayesian merge")

        # 5. After task (merge)
        method.after_task(task_idx)

        # 5b. Accumulate Fisher
        if fisher is not None:
            if fisher_new is not None:
                fisher.accumulate(fisher_new)
            else:
                fisher_new = estimate_fisher_vla(
                    fisher, model, processor, train_ds, args,
                    n_samples=args.fisher_samples,
                )
                fisher.accumulate(fisher_new)
            fisher.snapshot_ref_params()
            logger.info("Fisher: accumulated, ref params snapshotted")

            # Fisher sanity check (first task only)
            if task_idx == 0:
                _log_fisher_stats(fisher)

        # 6. Evaluate via rollouts
        if task_idx in eval_after:
            logger.info(f"Evaluating on tasks 0..{task_idx}")
            eval_model = method.get_model()
            eval_model.eval()
            for eval_idx in range(task_idx + 1):
                sr = evaluate_libero_task(
                    model=eval_model,
                    processor=processor,
                    suite_name=args.suite,
                    task_idx=eval_idx,
                    action_norm_stats=action_norm,
                    n_rollouts=args.n_rollouts,
                    max_steps=args.max_steps,
                    n_parallel=args.n_parallel_envs,
                    device=str(device),
                )
                acc_matrix.update(eval_idx, task_idx, sr)
        else:
            logger.info(f"Skipping eval (not in eval schedule)")

        # 7. Save checkpoint
        if args.save_adapters:
            ckpt_dir = output_dir / f"task_{task_idx}"
            model.save_pretrained(str(ckpt_dir))
            readme_path = ckpt_dir / "README.md"
            if readme_path.exists():
                readme_path.unlink()
            logger.info(f"Saved adapter to {ckpt_dir}")

        # 8. Save results incrementally
        acc_matrix.save(str(output_dir / "results.json"))
        action_norm.save(str(output_dir / "action_norm_stats.json"))

        # 9. Log metrics
        if task_idx in eval_after:
            cur_aa = acc_matrix.current_average_accuracy(task_idx)
            cur_bwt = acc_matrix.current_backward_transfer(task_idx)
            logger.info(f"After task {task_idx}: AA={cur_aa:.4f}, BWT={cur_bwt:.4f}")

    # ---- Final results ----
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Average Success Rate: {acc_matrix.average_accuracy():.4f}")
    logger.info(f"Backward Transfer: {acc_matrix.backward_transfer():.4f}")
    logger.info(f"Full matrix:\n{acc_matrix.matrix}")

    acc_matrix.save(str(output_dir / "results.json"))
    logger.info(f"Results saved to {output_dir / 'results.json'}")


def _log_fisher_stats(fisher: DiagonalFisher):
    """Log Fisher magnitude statistics as a sanity check.

    After task 0, Fisher should be:
    - Non-zero (model learned something)
    - Non-uniform (different params have different importance)
    - Concentrated (most Fisher mass in a subset of params)
    """
    if not fisher.fisher:
        logger.warning("Fisher sanity check: Fisher dict is empty!")
        return

    all_fishers = []
    for name, f in fisher.fisher.items():
        vals = f.flatten()
        all_fishers.append(vals)
        nonzero_frac = (vals > 1e-10).float().mean().item()
        logger.info(
            f"  Fisher [{name[-50:]}]: "
            f"mean={vals.mean():.2e}, "
            f"max={vals.max():.2e}, "
            f"nonzero={nonzero_frac:.1%}"
        )

    combined = torch.cat(all_fishers)
    logger.info(
        f"  Fisher TOTAL: mean={combined.mean():.2e}, "
        f"std={combined.std():.2e}, "
        f"max={combined.max():.2e}, "
        f"nonzero={(combined > 1e-10).float().mean():.1%}"
    )
    if combined.std() < 1e-12:
        logger.warning(
            "Fisher sanity check FAILED: Fisher is near-uniform. "
            "Estimation may be degenerate."
        )


if __name__ == "__main__":
    main()
