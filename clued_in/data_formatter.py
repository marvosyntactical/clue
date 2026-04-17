"""Convert chat session transcripts to SFT training data for CLUE.

Each (user, assistant) turn pair becomes one training example where:
- prompt: conversation history up to the user message, formatted via chat template
- label_text: the assistant's response

The output format matches CLDataset's expected input:
    [{"prompt": str, "label_text": str}, ...]
"""

from __future__ import annotations


class DataFormatter:
    """Formats chat transcripts into SFT training pairs."""

    def __init__(self, tokenizer, system_prompt: str = "You are a helpful assistant."):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def format_session(self, messages: list[dict]) -> list[dict]:
        """Convert a chat session into SFT training examples.

        Args:
            messages: [{"role": "user"|"assistant", "content": str}, ...]

        Returns:
            [{"prompt": str, "label_text": str}, ...] — one per assistant turn.
        """
        examples = []
        sys_msg = {"role": "system", "content": self.system_prompt}

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # Build context: system + all messages before this assistant turn
            context = [sys_msg] + messages[:i]
            prompt = self._apply_chat_template(context, add_generation_prompt=True)
            label_text = msg["content"]
            examples.append({"prompt": prompt, "label_text": label_text})

        return examples

    def format_quick_teach(self, fact: str) -> list[dict]:
        """Create synthetic SFT pairs from a single fact/instruction.

        Generates two training examples:
        1. User states the fact, assistant acknowledges
        2. User asks about it, assistant recalls it
        """
        examples = []
        sys_msg = {"role": "system", "content": self.system_prompt}

        # Teaching pair
        teach_ctx = [
            sys_msg,
            {"role": "user", "content": f"Please remember this: {fact}"},
        ]
        examples.append({
            "prompt": self._apply_chat_template(teach_ctx, add_generation_prompt=True),
            "label_text": f"Understood, I'll remember that. {fact}",
        })

        # Recall pair
        recall_ctx = [
            sys_msg,
            {"role": "user", "content": f"Please remember this: {fact}"},
            {"role": "assistant", "content": f"Understood, I'll remember that. {fact}"},
            {"role": "user", "content": "What did I just tell you to remember?"},
        ]
        examples.append({
            "prompt": self._apply_chat_template(recall_ctx, add_generation_prompt=True),
            "label_text": fact,
        })

        return examples

    def _apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True
    ) -> str:
        """Format messages using the tokenizer's chat template.

        Falls back to manual ChatML if the tokenizer lacks apply_chat_template.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass

        # Manual ChatML fallback
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
