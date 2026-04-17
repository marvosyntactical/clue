"""Model loading, inference, and adapter management for CLUED-IN.

Handles:
- Loading a quantized base model with LoRA via PEFT
- Chat inference with or without the learned adapter (A/B comparison)
- Saving/loading LoRA adapters to disk for persistence
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


class ModelServer:
    """Manages the quantized base model + LoRA adapter for inference and training."""

    def __init__(self, config: dict):
        self.config = config
        model_cfg = config["model"]
        lora_cfg = config["lora"]
        inf_cfg = config["inference"]

        self.model_name = model_cfg["name"]
        self.max_new_tokens = inf_cfg["max_new_tokens"]
        self.temperature = inf_cfg["temperature"]
        self.top_p = inf_cfg["top_p"]
        self.repetition_penalty = inf_cfg["repetition_penalty"]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        compute_dtype = getattr(torch, model_cfg.get("compute_dtype", "bfloat16"))
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("quantization", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )

        # Prepare for k-bit training (gradient checkpointing, fp32 layer norms)
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA config
        target_modules = lora_cfg["target_modules"]
        if isinstance(target_modules, str) and target_modules != "all-linear":
            target_modules = [m.strip() for m in target_modules.split(",")]

        lora_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        # Check for existing adapter
        adapter_path = config["paths"]["current_adapter"]
        if Path(adapter_path).exists() and (Path(adapter_path) / "adapter_config.json").exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self._loaded_from_disk = True
        else:
            self.model = get_peft_model(self.model, lora_config)
            self._loaded_from_disk = False

        # Freeze base weights
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        self.model.print_trainable_parameters()
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        use_adapter: bool = True,
    ) -> str:
        """Generate a response given chat messages.

        Args:
            messages: [{"role": "user"|"assistant"|"system", "content": str}]
            max_new_tokens: override default generation length
            temperature: override default temperature
            use_adapter: if False, disable LoRA for base model comparison
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature

        # Format with chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_len = input_ids.shape[1]

        # Disable adapter for A/B comparison
        if not use_adapter:
            self.model.disable_adapter_layers()

        try:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            if not use_adapter:
                self.model.enable_adapter_layers()

        # Decode only new tokens
        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def generate_base(self, messages: list[dict], **kwargs) -> str:
        """Generate with adapter disabled (base model only)."""
        return self.generate(messages, use_adapter=False, **kwargs)

    def save_adapter(self, path: str) -> None:
        """Save current LoRA adapter to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        # Remove PEFT boilerplate README
        readme = Path(path) / "README.md"
        if readme.exists():
            readme.unlink()

    def loaded_from_disk(self) -> bool:
        """Whether the adapter was loaded from a previous session."""
        return self._loaded_from_disk
