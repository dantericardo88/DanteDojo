"""DanteDojo Axolotl training backend helpers.

Generates axolotl YAML-compatible config dicts for multi-GPU fine-tuning.
Includes FSDP2 settings for dual-3090 and ZeRO-3 settings for H100.
"""

from __future__ import annotations

import importlib
from typing import Any

from dojo.core.types import HALProfile, HardwareTier
from dojo.train.base import FinetuneConfig


def generate_axolotl_config(config: FinetuneConfig, profile: HALProfile) -> dict[str, Any]:
    """Generate an axolotl YAML config dict from *config* and *profile*.

    Multi-GPU strategies:
      - ``TIER_3090_DUAL`` → FSDP2 with ``full_shard`` strategy
      - ``TIER_H100`` → FSDP2 with ZeRO-3 emulation via ``hybrid_shard``
      - Single-GPU tiers → standard DDP / no FSDP

    The returned dict can be serialized directly with ``yaml.dump()``.
    """
    is_dual = profile.tier == HardwareTier.TIER_3090_DUAL
    is_h100 = profile.tier == HardwareTier.TIER_H100

    # Base config shared by all tiers
    cfg: dict[str, Any] = {
        # Model
        "base_model": config.base_model,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        # Data
        "datasets": [
            {
                "path": config.pack_id,
                "type": "alpaca",
            }
        ],
        "dataset_prepared_path": "last_run_prepared",
        "val_set_size": 0.05,
        # LoRA / PEFT
        "adapter": "lora",
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        # Training
        "sequence_len": config.max_seq_length,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "num_epochs": config.num_epochs,
        "micro_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "seed": config.seed,
        # Output
        "output_dir": str(config.output_dir),
        # Precision
        "bf16": True,
        "fp16": False,
        "tf32": True,
        # Logging
        "logging_steps": 1,
        "save_steps": 0.1,
        "eval_steps": 0.1,
        # Gradient checkpointing
        "gradient_checkpointing": True,
        # FSDP defaults — overridden below for multi-GPU
        "fsdp": None,
        "fsdp_config": None,
    }

    if is_dual:
        # FSDP2 — full sharding across both 3090s
        cfg["fsdp"] = ["full_shard", "auto_wrap"]
        cfg["fsdp_config"] = {
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_offload_params": False,
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_use_orig_params": True,
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            # FSDP2 flag
            "fsdp2": True,
        }
        cfg["world_size"] = 2
        cfg["tensor_parallel_size"] = 1

    elif is_h100:
        # ZeRO-3 emulation via FSDP hybrid_shard (single H100 or multi)
        cfg["fsdp"] = ["hybrid_shard", "auto_wrap"]
        cfg["fsdp_config"] = {
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_offload_params": False,
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_use_orig_params": True,
            "fsdp_cpu_ram_efficient_loading": False,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            # ZeRO-3 style
            "zero3_enabled": True,
        }
        # H100: use FP8 if available
        if profile.has_native_fp8:
            cfg["bf16"] = False
            cfg["fp8"] = True
        cfg["flash_attention"] = True

    return cfg


def is_axolotl_available() -> bool:
    """Return ``True`` if ``axolotl`` is importable."""
    try:
        importlib.import_module("axolotl")
        return True
    except ImportError:
        return False


__all__ = ["generate_axolotl_config", "is_axolotl_available"]
