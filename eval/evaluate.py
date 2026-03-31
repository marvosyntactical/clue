"""Evaluation loop for continual learning tasks."""

import torch
from torch.utils.data import DataLoader

from data.datasets import collate_fn
from utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_task(
    model,
    tokenizer,
    val_dataset,
    task_name: str,
    label_names: list,
    batch_size: int = 8,
    max_new_tokens: int = 16,
) -> float:
    """Evaluate a model on a single task's validation set.

    Uses greedy generation and string-matching against label_names.
    Returns accuracy as a float in [0, 1].
    """
    model.eval()
    device = next(model.parameters()).device

    # Build prompts (everything before the label)
    prompts = []
    gold_labels = []
    for item in val_dataset.data:
        gold_labels.append(item["label_text"])
        # Reconstruct prompt by decoding input_ids up to where labels != -100
        input_ids = item["input_ids"]
        label_mask = item["labels"]
        # Find where label starts
        prompt_len = (label_mask == -100).sum().item()
        prompt_ids = input_ids[:prompt_len]
        prompts.append(prompt_ids)

    correct = 0
    total = 0
    # Lowercase label names for fuzzy matching
    label_names_lower = [ln.lower().strip() for ln in label_names]

    # Process in batches
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_golds = gold_labels[start:end]

        # Pad to same length (left-pad for generation)
        max_len = max(p.size(0) for p in batch_prompts)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = torch.full(
            (len(batch_prompts), max_len), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            len(batch_prompts), max_len, dtype=torch.long, device=device
        )
        for i, p in enumerate(batch_prompts):
            # Left-pad
            offset = max_len - p.size(0)
            input_ids[i, offset:] = p.to(device)
            attention_mask[i, offset:] = 1

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

        # Decode only the generated portion
        for i in range(len(batch_prompts)):
            gen_ids = outputs[i, max_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_text = gen_text.strip().lower()

            gold = batch_golds[i].lower().strip()

            # Match: generated text starts with the gold label
            match = gen_text.startswith(gold) or gold.startswith(gen_text)
            # Also check if the first "word" matches any valid label
            if not match:
                for ln in label_names_lower:
                    if gen_text.startswith(ln) and ln == gold:
                        match = True
                        break
            if match:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"  Eval {task_name}: {correct}/{total} = {accuracy:.4f}")
    return accuracy
