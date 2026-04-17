"""CLUE learning engine for CLUED-IN.

Wraps methods/slao.py and methods/fisher.py into a session-oriented
learning loop. Each call to learn_session() runs one CLUE task cycle:
    before_task → fine-tune → Fisher estimation → after_task (merge)

Reuses CLDataset, collate_fn, SLAO, and DiagonalFisher unchanged.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

# Ensure clue/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import CLDataset, collate_fn
from methods.slao import SLAO
from methods.fisher import DiagonalFisher
from models.lora import extract_lora_state, set_lora_state
from utils import get_logger, set_seed

logger = get_logger(__name__)


class CLUEEngine:
    """Drives the CLUE continual learning loop across chat sessions."""

    def __init__(self, model_server, config: dict):
        self.model_server = model_server
        self.config = config
        self.model = model_server.model
        self.tokenizer = model_server.tokenizer

        self._lock = threading.Lock()
        self._status = {"state": "idle", "session": None, "progress": ""}

        args = self._build_args(config)
        self.args = args

        # Instantiate SLAO method
        self.method = SLAO(self.model, args)

        # Instantiate Fisher
        self.fisher: DiagonalFisher | None = None
        needs_fisher = args.fisher_lambda > 0 or args.fisher_merge_beta > 0
        if needs_fisher:
            self.fisher = DiagonalFisher(self.model, gamma=args.fisher_gamma)
            self.method.fisher = self.fisher

        # Restore state from disk if adapter was loaded
        self.task_idx = 0
        state_path = Path(config["paths"]["current_adapter"]) / "clue_state.json"
        if state_path.exists():
            saved = json.loads(state_path.read_text())
            self.task_idx = saved.get("task_idx", 0)
            logger.info(f"Restored CLUE state: task_idx={self.task_idx}")
            # Re-extract merge state from the loaded adapter
            if self.task_idx > 0:
                self.method.merge_state = extract_lora_state(self.model)
                self.method.ft_state = extract_lora_state(self.model)
                logger.info("Restored merge_state from loaded adapter")

        set_seed(args.seed)

    @property
    def is_training(self) -> bool:
        return self._lock.locked()

    def get_status(self) -> dict:
        return self._status.copy()

    def learn_session(self, session_id: int, training_examples: list[dict]) -> None:
        """Fine-tune on one session's data, then merge via CLUE.

        Args:
            session_id: sequential session number (for logging)
            training_examples: [{"prompt": str, "label_text": str}]
        """
        with self._lock:
            self._status = {
                "state": "training",
                "session": session_id,
                "progress": "Preparing training data...",
            }

            try:
                self._learn_session_inner(session_id, training_examples)
            except Exception as e:
                logger.error(f"Learning failed for session {session_id}: {e}")
                self._status = {
                    "state": "error",
                    "session": session_id,
                    "progress": f"Error: {e}",
                }
                raise
            finally:
                if self._status["state"] != "error":
                    self._status = {
                        "state": "idle",
                        "session": None,
                        "progress": f"Learned from session {session_id}",
                    }

    def _learn_session_inner(self, session_id: int, training_examples: list[dict]):
        """Core training + merge logic."""
        args = self.args
        device = self.model_server.device

        if not training_examples:
            logger.warning(f"Session {session_id}: no training data, skipping")
            return

        logger.info(
            f"Session {session_id} (task {self.task_idx}): "
            f"{len(training_examples)} training examples"
        )

        # Build dataset
        dataset = CLDataset(training_examples, self.tokenizer, max_length=args.max_length)
        if len(dataset) == 0:
            logger.warning(f"Session {session_id}: all examples filtered, skipping")
            return

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
            generator=torch.Generator().manual_seed(args.seed + self.task_idx),
        )

        # 1. Before task (orthogonal init, etc.)
        self._status["progress"] = "Initializing LoRA for new session..."
        self.method.before_task(self.task_idx, f"session_{session_id}")

        # 2. Train
        self._status["progress"] = "Fine-tuning..."
        self._train(loader, device)

        # 3. Fisher estimation
        if self.fisher is not None:
            self._status["progress"] = "Estimating parameter importance..."
            fisher_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=False,
            )
            self.fisher.estimate(fisher_loader, n_samples=args.fisher_samples)
            self.fisher.snapshot_ref_params()

        # 4. After task (merge)
        self._status["progress"] = "Merging knowledge..."
        self.method.after_task(self.task_idx)

        # 5. Save adapter and state
        self._status["progress"] = "Saving..."
        adapter_path = self.config["paths"]["current_adapter"]
        self.model_server.save_adapter(adapter_path)

        # Also save a numbered backup
        backup_path = str(Path(self.config["paths"]["adapters_dir"]) / f"after_session_{session_id}")
        self.model_server.save_adapter(backup_path)

        # Save CLUE state
        state = {"task_idx": self.task_idx + 1, "session_id": session_id}
        state_path = Path(adapter_path) / "clue_state.json"
        state_path.write_text(json.dumps(state))

        self.task_idx += 1
        logger.info(f"Session {session_id}: learning complete, task_idx now {self.task_idx}")

    def _train(self, loader: DataLoader, device) -> float:
        """Run the training loop (adapted from train.py's train_one_task)."""
        args = self.args
        self.model.train()

        params = self.method.get_trainable_params()
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer.zero_grad()

        has_fisher = self.fisher is not None and self.fisher.fisher
        total_loss = 0.0
        step = 0
        global_step = 0

        for epoch in range(args.epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.grad_accum

                if has_fisher:
                    loss = loss + (args.fisher_lambda / 2) * self.fisher.penalty() / args.grad_accum

                loss.backward()
                total_loss += outputs.loss.item()
                step += 1

                if step % args.grad_accum == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Flush remaining gradients
            if step % args.grad_accum != 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / max(step, 1)
            logger.info(f"  Epoch {epoch + 1}/{args.epochs}, avg_loss={avg_loss:.4f}")
            self._status["progress"] = (
                f"Fine-tuning epoch {epoch + 1}/{args.epochs} "
                f"(loss={avg_loss:.3f})"
            )

        self.model.eval()
        return total_loss / max(step, 1)

    @staticmethod
    def _build_args(config: dict) -> SimpleNamespace:
        """Build a SimpleNamespace mimicking argparse output from config."""
        t = config["training"]
        c = config["clue"]
        l = config["lora"]
        return SimpleNamespace(
            # LoRA
            lora_rank=l["rank"],
            lora_alpha=l["alpha"],
            lora_dropout=l["dropout"],
            lora_target_modules=l["target_modules"],
            # Training
            optimizer=t["optimizer"],
            lr=t["lr"],
            weight_decay=t["weight_decay"],
            batch_size=t["batch_size"],
            grad_accum=t["grad_accum"],
            epochs=t["epochs"],
            max_length=t["max_length"],
            max_grad_norm=t["max_grad_norm"],
            seed=t["seed"],
            momentum=0.0,
            # CLUE / SLAO
            method=c["method"],
            a_init_method=c.get("a_init_method", "zca"),
            fisher_lambda=c.get("fisher_lambda", 0.0),
            fisher_gamma=c.get("fisher_gamma", 0.9),
            fisher_samples=c.get("fisher_samples", 128),
            fisher_merge_beta=c.get("fisher_merge_beta", 0.0),
            # Bayesian merge (not used by default, but SLAO reads these)
            bayesian_merge=c.get("bayesian_merge", False),
            bayesian_alpha_min=c.get("bayesian_alpha_min", 0.01),
            bayesian_alpha_max=c.get("bayesian_alpha_max", 0.95),
            bayesian_lambda_damping=c.get("bayesian_lambda_damping", False),
            # Extensions (disabled for clued_in)
            lora_plus_ratio=1.0,
            gpm_threshold=0.0,
            riemannian=False,
        )
