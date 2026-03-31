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

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/default")
    p.add_argument("--save_adapters", action="store_true",
                    help="Save LoRA adapter checkpoints after each task")

    return p.parse_args()


def build_optimizer(params, args) -> torch.optim.Optimizer:
    """Construct optimizer from args."""
    cls = OPTIMIZER_REGISTRY[args.optimizer]
    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        kwargs["momentum"] = args.momentum
    return cls(params, **kwargs)


def train_one_task(
    model,
    train_dataset: CLDataset,
    method,
    args,
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
    optimizer = build_optimizer(params, args)
    optimizer.zero_grad()

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
            loss.backward()
            total_loss += loss.item() * args.grad_accum
            step += 1

            if step % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
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
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(step, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Config: {json.dumps(vars(args), indent=2)}")

    # ---- Parse task order ----
    if args.task_order in ("O1", "O2", "O3", "O4", "O5", "O6"):
        task_names = get_task_order(args.task_order)
    else:
        task_names = [t.strip() for t in args.task_order.split(",")]

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
        avg_loss = train_one_task(model, train_ds, method, args)
        elapsed = time.time() - t0
        logger.info(f"Training loss: {avg_loss:.4f}, time: {elapsed:.1f}s")

        # 4. After task (merging, etc.)
        method.after_task(task_idx)

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
            logger.info(f"Saved adapter to {ckpt_dir}")

        # 7. Log intermediate metrics
        logger.info(f"After task {task_idx}: {acc_matrix}")

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