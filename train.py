#!/usr/bin/env python3
"""Main entrypoint for continual learning experiments.

Example usage (paper's most informative eval — 15-task with BWT):

    python train.py \
        --model_name meta-llama/Llama-2-7b-chat-hf \
        --method slao \
        --task_order O4 \
        --lr 1e-4 \
        --batch_size 1 \
        --grad_accum 8 \
        --epochs 1 \
        --lora_rank 8 \
        --lora_target_modules q_proj v_proj \
        --samples_per_class_train 1000 \
        --samples_per_class_val 500 \
        --seed 42 \
        --output_dir outputs/slao_O4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.datasets import CLDataset, collate_fn, load_task_data
from data.prompts import get_task_config, get_task_order
from eval.evaluate import evaluate_task
from eval.metrics import AccuracyMatrix
from methods.slao import SLAO
from methods.seq_lora import SeqLoRA
from methods.inc_lora import IncLoRA
from methods.riemannian import RiemannianPreconditioner
from methods.fisher import DiagonalFisher
from methods.gpm import GradientProjectionMemory
from utils import get_logger, set_seed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Method registry — add new methods here
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "slao": SLAO,
    "seq_lora": SeqLoRA,
    "inc_lora": IncLoRA,
}

# ---------------------------------------------------------------------------
# Optimizer registry — easy to swap for ablation
# ---------------------------------------------------------------------------

OPTIMIZER_REGISTRY = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continual Learning with LoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                    help="HuggingFace model name or local path")
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default="auto",
                    help="Device: 'auto' (PEFT device_map), 'cpu', or 'cuda:0'")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])

    # Method
    p.add_argument("--method", type=str, default="slao",
                    choices=list(METHOD_REGISTRY.keys()))

    # Task ordering
    p.add_argument("--task_order", type=str, default="O4",
                    help="Named order (O1-O6) or comma-separated task names")

    # Training
    p.add_argument("--optimizer", type=str, default="sgd",
                    choices=list(OPTIMIZER_REGISTRY.keys()),
                    help="Optimizer (paper uses SGD)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.0,
                    help="Momentum for SGD (paper does not use momentum)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Max gradient norm for clipping (0 to disable)")

    # Data
    p.add_argument("--samples_per_class_train", type=int, default=1000)
    p.add_argument("--samples_per_class_val", type=int, default=500)
    p.add_argument("--chat_template", type=str, default="llama2",
                    choices=["llama2", "plain"])

    # Eval
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=16)

    # Extensions
    p.add_argument("--riemannian", action="store_true",
                    help="Enable Riemannian preconditioning of LoRA gradients")
    p.add_argument("--riemannian_delta", type=float, default=1e-6,
                    help="Regularization for Riemannian matrix inversion")
    p.add_argument("--fisher_lambda", type=float, default=0.0,
                    help="Fisher regularization strength (0 = disabled)")
    p.add_argument("--fisher_gamma", type=float, default=1.0,
                    help="Fisher EMA decay: 1.0 = standard EWC (sum), <1.0 = online EWC")
    p.add_argument("--fisher_samples", type=int, default=256,
                    help="Samples for Fisher estimation per task")
    p.add_argument("--fisher_merge_beta", type=float, default=0.0,
                    help="Fisher-weighted B merge sensitivity (0 = uniform/standard SLAO)")
    p.add_argument("--lora_plus_ratio", type=float, default=1.0,
                    help="LoRA+ B_lr/A_lr ratio (1.0 = standard, paper recommends 2-16)")
    p.add_argument("--gpm_threshold", type=float, default=0.0,
                    help="GPM activation subspace threshold (0 = disabled, 0.90-0.99)")
    p.add_argument("--gpm_samples", type=int, default=256,
                    help="Reference samples for GPM activation SVD")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/default")
    p.add_argument("--save_adapters", action="store_true",
                    help="Save LoRA adapter checkpoints after each task")

    return p.parse_args()


def build_optimizer(model, trainable_params, args) -> torch.optim.Optimizer:
    """Construct optimizer from args, with LoRA+ support."""
    cls = OPTIMIZER_REGISTRY[args.optimizer]
    opt_kwargs = {"weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        opt_kwargs["momentum"] = args.momentum

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


def train_one_task(
    model,
    train_dataset: CLDataset,
    method,
    args,
    fisher: DiagonalFisher | None = None,
    gpm: GradientProjectionMemory | None = None,
    preconditioner: RiemannianPreconditioner | None = None,
) -> float:
    """Train the model on a single task. Returns the training loss."""
    device = next(model.parameters()).device
    model.train()

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum

            # Fisher regularization (part of the loss, autograd handles it)
            if has_fisher:
                loss = loss + (args.fisher_lambda / 2) * fisher.penalty() / args.grad_accum

            loss.backward()
            total_loss += outputs.loss.item()
            step += 1

            if step % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                # GPM: project A gradients orthogonal to protected subspace
                if gpm is not None:
                    gpm.project_grads()
                # Riemannian: precondition surviving gradients
                if preconditioner is not None:
                    preconditioner.precondition_grads()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    avg = total_loss / step
                    logger.info(
                        f"  step {global_step}, avg_loss={avg:.4f}"
                    )

        # Flush remaining gradients
        if step % args.grad_accum != 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            if gpm is not None:
                gpm.project_grads()
            if preconditioner is not None:
                preconditioner.precondition_grads()
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(step, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility (include resolved task list)
    config = vars(args).copy()

    logger.info(f"Config: {json.dumps(vars(args), indent=2)}")

    # ---- Parse task order ----
    try:
        task_names = get_task_order(args.task_order)
        config["task_order_resolved"] = task_names
    except ValueError:
        task_names = [t.strip() for t in args.task_order.split(",")]
        config["task_order_resolved"] = task_names

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    num_tasks = len(task_names)
    logger.info(f"Task order ({num_tasks} tasks): {task_names}")

    # ---- Load model and tokenizer ----
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect whether CUDA is actually usable (PyTorch may report cuda
    # available but fail on unsupported GPU architectures like Blackwell
    # with an older CUDA toolkit).
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                device = "auto"  # real auto — CUDA works
            except RuntimeError:
                logger.warning(
                    "CUDA reported available but kernel launch failed "
                    "(likely unsupported GPU arch). Falling back to CPU."
                )
                device = "cpu"
        else:
            device = "cpu"

    load_kwargs = {"torch_dtype": torch_dtype}
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)

    # ---- Apply LoRA via PEFT ----
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Freeze base model, only LoRA is trainable
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    # ---- Instantiate CL method ----
    method_cls = METHOD_REGISTRY[args.method]
    method = method_cls(model, args)

    # ---- Instantiate extensions & wire into method ----
    preconditioner = None
    if args.riemannian:
        preconditioner = RiemannianPreconditioner(model, delta=args.riemannian_delta)
        logger.info("Extension: Riemannian preconditioning enabled")

    fisher = None
    needs_fisher = args.fisher_lambda > 0 or args.fisher_merge_beta > 0
    if needs_fisher:
        fisher = DiagonalFisher(model, gamma=args.fisher_gamma)
        parts = []
        if args.fisher_lambda > 0:
            ewc_mode = "online" if args.fisher_gamma < 1.0 else "standard"
            parts.append(f"{ewc_mode} EWC (λ={args.fisher_lambda}, γ={args.fisher_gamma})")
        if args.fisher_merge_beta > 0:
            parts.append(f"Fisher-weighted merge (β={args.fisher_merge_beta})")
        logger.info(f"Extension: Fisher enabled — {', '.join(parts)}")

    # Give SLAO access to Fisher for importance-weighted merging
    if fisher is not None and hasattr(method, "fisher"):
        method.fisher = fisher

    gpm = None
    if args.gpm_threshold > 0:
        gpm = GradientProjectionMemory(model, threshold=args.gpm_threshold)
        logger.info(f"Extension: GPM enabled (threshold={args.gpm_threshold})")

    if args.lora_plus_ratio != 1.0:
        logger.info(f"Extension: LoRA+ enabled (ratio={args.lora_plus_ratio})")

    # ---- Accuracy matrix for metrics ----
    acc_matrix = AccuracyMatrix(num_tasks)

    # ---- Load all validation sets upfront (for eval after each task) ----
    logger.info("Loading validation datasets...")
    val_datasets = {}
    for tn in task_names:
        _, val_ds = load_task_data(
            tn, tokenizer,
            samples_per_class_train=1,  # don't need train here
            samples_per_class_val=args.samples_per_class_val,
            max_length=args.max_length,
            seed=args.seed,
            chat_template=args.chat_template,
        )
        val_datasets[tn] = val_ds

    # ---- Main CL loop ----
    for task_idx, task_name in enumerate(task_names):
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK {task_idx}/{num_tasks-1}: {task_name}")
        logger.info(f"{'='*60}")

        # 1. Before task
        method.before_task(task_idx, task_name)

        # 2. Load training data
        train_ds, _ = load_task_data(
            task_name, tokenizer,
            samples_per_class_train=args.samples_per_class_train,
            samples_per_class_val=1,
            max_length=args.max_length,
            seed=args.seed,
            chat_template=args.chat_template,
        )
        logger.info(f"Training samples: {len(train_ds)}")

        # 3. Train
        t0 = time.time()
        avg_loss = train_one_task(
            model, train_ds, method, args,
            fisher=fisher, gpm=gpm, preconditioner=preconditioner,
        )
        elapsed = time.time() - t0
        logger.info(f"Training loss: {avg_loss:.4f}, time: {elapsed:.1f}s")

        # 4. After task (merging, etc.)
        method.after_task(task_idx)

        # 4b. Update extensions that need post-task processing
        if fisher is not None:
            fisher.snapshot_ref_params()
            fisher_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, drop_last=False,
            )
            fisher.estimate(fisher_loader, n_samples=args.fisher_samples)
            logger.info(f"Fisher: estimated on {args.fisher_samples} samples")

        if gpm is not None:
            gpm_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, drop_last=False,
            )
            gpm.update_memory(gpm_loader, n_samples=args.gpm_samples)
            n_bases = sum(m.shape[1] for m in gpm.memory.values())
            logger.info(f"GPM: memory updated, {n_bases} total basis vectors")

        # 5. Evaluate on all tasks seen so far
        logger.info(f"Evaluating on tasks 0..{task_idx}")
        eval_model = method.get_model()
        for eval_idx in range(task_idx + 1):
            eval_name = task_names[eval_idx]
            cfg = get_task_config(eval_name)
            acc = evaluate_task(
                eval_model, tokenizer,
                val_datasets[eval_name],
                eval_name,
                cfg["label_names"],
                batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            acc_matrix.update(eval_idx, task_idx, acc)

        # 6. Save checkpoint if requested
        if args.save_adapters:
            ckpt_dir = output_dir / f"task_{task_idx}_{task_name}"
            model.save_pretrained(str(ckpt_dir))
            # Remove PEFT-generated boilerplate README
            readme_path = ckpt_dir / "README.md"
            if readme_path.exists():
                readme_path.unlink()
            logger.info(f"Saved adapter to {ckpt_dir}")

        # 7. Save per-task eval results
        task_results_path = output_dir / f"task_{task_idx}_{task_name}" / "eval_results.json"
        acc_matrix.save_task_results(str(task_results_path), task_idx, task_names)

        # 8. Save running results (overwritten each task so partial runs are recoverable)
        acc_matrix.save(str(output_dir / "results.json"))

        # 9. Log intermediate metrics
        cur_aa = acc_matrix.current_average_accuracy(task_idx)
        cur_bwt = acc_matrix.current_backward_transfer(task_idx)
        logger.info(
            f"After task {task_idx}: AA={cur_aa:.4f}, BWT={cur_bwt:.4f}"
        )

    # ---- Final results ----
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Average Accuracy: {acc_matrix.average_accuracy():.4f}")
    logger.info(f"Backward Transfer: {acc_matrix.backward_transfer():.4f}")
    logger.info(f"Full matrix:\n{acc_matrix.matrix}")

    # Save results
    acc_matrix.save(str(output_dir / "results.json"))
    logger.info(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()