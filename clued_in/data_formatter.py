"""Convert chat session transcripts to SFT training data for CLUE.

Two modes:
1. Direct replay: train on the actual (context → response) pairs from the chat.
2. Self-annotation: use the model to extract lessons and generate diverse
   synthetic training pairs that exercise those lessons in varied contexts.
   Much more sample-efficient — trains on *what the model should do* rather
   than *what it happened to say*.

The output format matches CLDataset's expected input:
    [{"prompt": str, "label_text": str}, ...]
"""

from __future__ import annotations

from utils import get_logger

logger = get_logger(__name__)

# Prompt for self-annotation: extract lessons and generate training pairs
SELF_ANNOTATE_PROMPT = """You just had a conversation with a user. Based on that conversation, extract what the user taught you or wants you to do differently, then generate training examples.

Here is the conversation:
{conversation}

Now generate exactly {n_examples} diverse instruction-response pairs that demonstrate the lessons from this conversation. Each pair should be a DIFFERENT scenario that exercises the same skills/preferences/knowledge. Make the responses substantive (not just acknowledgments).

Format each pair exactly like this, with no other text:
INSTRUCTION: <user message>
RESPONSE: <ideal assistant response>
---"""


class DataFormatter:
    """Formats chat transcripts into SFT training pairs."""

    def __init__(self, tokenizer, system_prompt: str = "You are a helpful assistant."):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self._generate_fn = None  # set by engine after model is ready

    def set_generate_fn(self, fn):
        """Set the generation function for self-annotation.

        Args:
            fn: callable(messages, max_new_tokens) -> str
        """
        self._generate_fn = fn

    def format_session(self, messages: list[dict]) -> list[dict]:
        """Convert a chat session into SFT training examples.

        Uses self-annotation if a generate function is available,
        otherwise falls back to direct replay.
        """
        # Always include direct replay examples
        direct = self._format_direct(messages)

        # Try self-annotation for additional synthetic examples
        if self._generate_fn is not None:
            synthetic = self._format_self_annotated(messages)
            if synthetic:
                logger.info(
                    f"Self-annotation: {len(synthetic)} synthetic + "
                    f"{len(direct)} direct = {len(synthetic) + len(direct)} total"
                )
                return synthetic + direct

        return direct

    def _format_direct(self, messages: list[dict]) -> list[dict]:
        """Direct replay: one example per assistant turn."""
        examples = []
        sys_msg = {"role": "system", "content": self.system_prompt}

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            context = [sys_msg] + messages[:i]
            prompt = self._apply_chat_template(context, add_generation_prompt=True)
            label_text = msg["content"]
            examples.append({"prompt": prompt, "label_text": label_text})

        return examples

    def _format_self_annotated(self, messages: list[dict], n_examples: int = 5) -> list[dict]:
        """Use the model to generate diverse training pairs from the conversation."""
        try:
            # Build conversation text
            conv_lines = []
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conv_lines.append(f"{role}: {msg['content']}")
            conversation = "\n".join(conv_lines)

            # Ask the model to self-annotate
            annotation_prompt = SELF_ANNOTATE_PROMPT.format(
                conversation=conversation,
                n_examples=n_examples,
            )

            sys_msg = {"role": "system", "content": "You are a training data generator. Follow instructions exactly."}
            gen_messages = [sys_msg, {"role": "user", "content": annotation_prompt}]
            logger.info("Self-annotation: generating synthetic training pairs...")
            raw_output = self._generate_fn(gen_messages, max_new_tokens=1024)
            logger.info(f"Self-annotation raw output ({len(raw_output)} chars):\n{raw_output[:500]}")

            # Parse the output into pairs
            examples = self._parse_annotation(raw_output)
            logger.info(f"Self-annotation: parsed {len(examples)} examples")
            for i, ex in enumerate(examples):
                logger.info(f"  Synthetic {i}: {ex['label_text'][:100]}")
            return examples

        except Exception as e:
            logger.warning(f"Self-annotation failed: {e}")
            return []

    def _parse_annotation(self, text: str) -> list[dict]:
        """Parse INSTRUCTION/RESPONSE pairs from model output."""
        examples = []
        sys_msg = {"role": "system", "content": self.system_prompt}

        blocks = text.split("---")
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            inst_marker = "INSTRUCTION:"
            resp_marker = "RESPONSE:"

            inst_idx = block.find(inst_marker)
            resp_idx = block.find(resp_marker)

            if inst_idx == -1 or resp_idx == -1:
                continue

            instruction = block[inst_idx + len(inst_marker):resp_idx].strip()
            response = block[resp_idx + len(resp_marker):].strip()

            if not instruction or not response:
                continue

            context = [sys_msg, {"role": "user", "content": instruction}]
            prompt = self._apply_chat_template(context, add_generation_prompt=True)
            examples.append({"prompt": prompt, "label_text": response})

        return examples

    def format_quick_teach(self, fact: str) -> list[dict]:
        """Create synthetic SFT pairs from a single fact/instruction."""
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
        """Format messages using the tokenizer's chat template."""
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
