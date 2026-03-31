"""Dataset loading, sampling, and tokenization for continual learning tasks."""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from data.prompts import get_task_config


class CLDataset(Dataset):
    """A tokenized dataset for a single CL task.

    Each item is a dict with:
      - input_ids:  (seq_len,) — full sequence (prompt + label + eos)
      - attention_mask: (seq_len,)
      - labels: (seq_len,) — -100 for prompt tokens, token ids for label + eos
      - label_text: str — the ground-truth label string (for eval)
    """

    def __init__(self, examples: list, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for ex in examples:
            encoded = self._encode(ex["prompt"], ex["label_text"])
            if encoded is not None:
                self.data.append(encoded)

    def _encode(self, prompt: str, label_text: str) -> Optional[dict]:
        """Tokenize prompt + label into a causal-LM training example."""
        # Tokenize prompt (no eos)
        prompt_enc = self.tokenizer(
            prompt, add_special_tokens=True, truncation=True,
            max_length=self.max_length - 32,  # leave room for label
        )
        # Tokenize label (with eos)
        label_enc = self.tokenizer(
            label_text, add_special_tokens=False,
        )
        label_ids = label_enc["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = prompt_enc["input_ids"] + label_ids
        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            label_start = len(prompt_enc["input_ids"])
            if label_start >= self.max_length:
                return None  # prompt alone exceeds max_length
            label_ids = input_ids[label_start:]

        attention_mask = [1] * len(input_ids)
        # Labels: -100 for prompt, actual ids for label
        labels = [-100] * len(prompt_enc["input_ids"]) + label_ids
        labels = labels[: len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "label_text": label_text,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: list) -> dict:
    """Pad a batch of CLDataset items to equal length."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_id = 0  # standard padding

    input_ids = []
    attention_mask = []
    labels = []
    label_texts = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        label_texts.append(item["label_text"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "label_texts": label_texts,
    }


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(instruction: str, input_text: str, chat_template: str = "llama2") -> str:
    """Format an instruction + input into the model's expected prompt format.

    Args:
        chat_template: "llama2" for [INST] format, "plain" for no wrapping.
    """
    body = f"{instruction}\n\n{input_text}\n\nAnswer:"
    if chat_template == "llama2":
        return f"[INST] {body} [/INST] "
    elif chat_template == "plain":
        return f"{body} "
    else:
        return f"[INST] {body} [/INST] "


# ---------------------------------------------------------------------------
# Data loading and sampling
# ---------------------------------------------------------------------------

def _sample_per_class(
    dataset,
    label_key: str,
    n_per_class: int,
    seed: int,
) -> list:
    """Sample up to n_per_class examples for each label value."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, ex in enumerate(dataset):
        by_class[ex[label_key]].append(i)

    selected = []
    for cls_indices in by_class.values():
        if len(cls_indices) <= n_per_class:
            selected.extend(cls_indices)
        else:
            selected.extend(rng.sample(cls_indices, n_per_class))

    rng.shuffle(selected)
    return [dataset[i] for i in selected]


def load_task_data(
    task_name: str,
    tokenizer,
    samples_per_class_train: int = 1000,
    samples_per_class_val: int = 500,
    max_length: int = 512,
    seed: int = 42,
    chat_template: str = "llama2",
) -> Tuple[CLDataset, CLDataset]:
    """Load and tokenize train/val splits for a CL task.

    Returns (train_dataset, val_dataset).
    """
    cfg = get_task_config(task_name)
    label_names = cfg["label_names"]
    instruction = cfg["instruction"]
    input_builder = cfg["input_builder"]
    label_key = cfg["label_key"]

    # Load raw HF dataset
    kwargs = {"trust_remote_code": True}
    if cfg["hf_config"]:
        raw = load_dataset(cfg["hf_dataset"], cfg["hf_config"], **kwargs)
    else:
        raw = load_dataset(cfg["hf_dataset"], **kwargs)

    # Sample
    train_examples = _sample_per_class(
        raw[cfg["hf_split_train"]], label_key, samples_per_class_train, seed
    )
    val_examples = _sample_per_class(
        raw[cfg["hf_split_val"]], label_key, samples_per_class_val, seed + 1
    )

    # Format into prompt + label_text
    def to_formatted(examples):
        formatted = []
        for ex in examples:
            input_text = input_builder(ex)
            label_text = label_names[ex[label_key]]
            prompt = format_prompt(instruction, input_text, chat_template)
            formatted.append({"prompt": prompt, "label_text": label_text})
        return formatted

    train_formatted = to_formatted(train_examples)
    val_formatted = to_formatted(val_examples)

    train_ds = CLDataset(train_formatted, tokenizer, max_length)
    val_ds = CLDataset(val_formatted, tokenizer, max_length)

    return train_ds, val_ds
