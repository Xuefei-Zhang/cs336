"""PEFT LoRA and optional QLoRA helpers."""

from __future__ import annotations

from peft import LoraConfig, TaskType
import torch
from transformers import BitsAndBytesConfig

from aiinfra_e2e.config import TrainConfig


def build_lora_config(train_config: TrainConfig) -> LoraConfig:
    """Build the PEFT LoRA configuration used for SFT runs."""

    target_modules: str | list[str]
    if train_config.lora_target_modules is None:
        target_modules = "all-linear"
    else:
        target_modules = train_config.lora_target_modules

    return LoraConfig(
        r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def build_quantization_config(*, enabled: bool) -> BitsAndBytesConfig | None:
    """Return a 4-bit NF4 quantization config when QLoRA is enabled."""

    if not enabled:
        return None

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
