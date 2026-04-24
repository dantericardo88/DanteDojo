"""DanteDojo Unsloth training backend helpers.

Provides config generation for single-GPU Unsloth runs and hardware-safe
default enforcement for 4060-laptop (8 GB VRAM) constraints.
"""

from __future__ import annotations

import importlib
from typing import Any

from dojo.core.types import HALProfile, HardwareTier
from dojo.train.base import FinetuneConfig


# ---------------------------------------------------------------------------
# 4060-laptop hard limits
# ---------------------------------------------------------------------------

_4060_MAX_BATCH_SIZE = 2
_4060_MAX_SEQ_LENGTH = 4096
_4060_GRADIENT_CHECKPOINTING = "unsloth"
_4060_OPTIMIZER = "paged_adamw_8bit"


def select_unsloth_config(config: FinetuneConfig, profile: HALProfile) -> dict[str, Any]:
    """Return Unsloth training kwargs dict, enforcing 4060-safe defaults.

    For a 4060-laptop profile (8 GB VRAM) the following limits are enforced:
      - ``batch_size`` ≤ 2
      - ``max_seq_length`` ≤ 4096
      - ``use_gradient_checkpointing`` = ``"unsloth"``
      - ``optimizer`` = ``"paged_adamw_8bit"``

    All other profiles receive the values from *config* directly.

    The returned dict matches the kwargs accepted by Unsloth's
    ``FastLanguageModel.get_peft_model()`` and the accompanying
    ``SFTTrainer`` config.
    """
    is_4060 = profile.tier == HardwareTier.TIER_4060_LAPTOP

    batch_size = config.batch_size
    seq_length = config.max_seq_length
    gradient_checkpointing: str | bool = True
    optimizer = "adamw_8bit"

    if is_4060:
        batch_size = min(batch_size, _4060_MAX_BATCH_SIZE)
        seq_length = min(seq_length, _4060_MAX_SEQ_LENGTH)
        gradient_checkpointing = _4060_GRADIENT_CHECKPOINTING
        optimizer = _4060_OPTIMIZER

    return {
        # LoRA / PEFT
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": "none",
        "use_gradient_checkpointing": gradient_checkpointing,
        "random_state": config.seed,
        # Training loop
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": config.grad_accum_steps,
        "max_seq_length": seq_length,
        "num_train_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "optim": optimizer,
        "output_dir": str(config.output_dir),
        # BF16 is always safe on Ada/Ampere; FP16 avoided for stability
        "bf16": True,
        "fp16": False,
    }


def is_unsloth_available() -> bool:
    """Return ``True`` if ``unsloth`` is importable."""
    try:
        importlib.import_module("unsloth")
        return True
    except ImportError:
        return False


__all__ = ["is_unsloth_available", "select_unsloth_config"]
