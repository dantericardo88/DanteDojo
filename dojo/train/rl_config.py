"""DanteDojo RL training configs — GRPO and DPO.

Both configs are frozen dataclasses so they can be hashed into evidence bundles
deterministically.  Constitutional defaults are conservative for solo-GPU hardware
(4060 Ti 16 GB / 3090 24 GB).

GRPO (Group Relative Policy Optimization):
  - Used for reward-based RL fine-tuning (DeepSeek-R1 style reasoning).
  - max_kl is the constitutional ceiling; exceeding it raises KLExplosionError.

DPO (Direct Preference Optimization):
  - Used for preference-pair alignment without a reward model.
  - min_reward_margin is the constitutional floor; collapsing below it raises DPORewardCollapse.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# GRPO config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GRPOConfig:
    """Immutable configuration for a GRPO training run.

    Args:
        pack_id: Specialization pack identifier (for evidence stamping).
        base_model: HF Hub model ID or local path.
        reward_model: HF Hub ID or local path for the reward model.
        max_new_tokens: Max tokens generated per rollout.
        temperature: Sampling temperature for rollout generation.
        beta: KL penalty coefficient (higher = closer to reference policy).
        max_kl: Constitutional KL ceiling; exceeding this raises KLExplosionError.
        num_generations: Group size G — completions per prompt in GRPO.
        lora_r: LoRA rank for the policy adapter.
        lora_alpha: LoRA alpha scaling.
        batch_size: Per-device batch size (prompts per step).
        grad_accum_steps: Gradient accumulation steps.
        num_epochs: Training epochs over the pack.
        learning_rate: AdamW learning rate.
        seed: Reproducibility seed.
        output_dir: Adapter and checkpoint output directory.
    """

    pack_id: str
    base_model: str
    reward_model: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.9
    beta: float = 0.04
    max_kl: float = 0.3
    num_generations: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    batch_size: int = 1
    grad_accum_steps: int = 8
    num_epochs: int = 1
    learning_rate: float = 1e-5
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("artifacts/grpo_run"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "base_model": self.base_model,
            "reward_model": self.reward_model,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "beta": self.beta,
            "max_kl": self.max_kl,
            "num_generations": self.num_generations,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# DPO config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DPOConfig:
    """Immutable configuration for a DPO training run.

    Args:
        pack_id: Specialization pack identifier (preference pairs must be in pack).
        base_model: HF Hub model ID or local path.
        beta: DPO temperature (inverse regularization; higher = more conservative).
        min_reward_margin: Constitutional floor for chosen_reward - rejected_reward.
                           Falls below this for more than max_collapse_steps in a row
                           → DPORewardCollapse raised.
        max_collapse_steps: Consecutive steps below min_reward_margin before abort.
        loss_type: "sigmoid" (standard DPO), "ipo", or "kto_pair".
        lora_r: LoRA rank for the policy adapter.
        lora_alpha: LoRA alpha scaling.
        batch_size: Per-device batch size (preference pairs per step).
        grad_accum_steps: Gradient accumulation steps.
        max_length: Max token length for combined prompt+completion.
        max_prompt_length: Max token length for the prompt portion.
        num_epochs: Training epochs over the pack.
        learning_rate: AdamW learning rate.
        seed: Reproducibility seed.
        output_dir: Adapter and checkpoint output directory.
    """

    pack_id: str
    base_model: str
    beta: float = 0.1
    min_reward_margin: float = 0.1
    max_collapse_steps: int = 5
    loss_type: str = "sigmoid"
    lora_r: int = 16
    lora_alpha: int = 32
    batch_size: int = 1
    grad_accum_steps: int = 4
    max_length: int = 512
    max_prompt_length: int = 256
    num_epochs: int = 3
    learning_rate: float = 5e-5
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("artifacts/dpo_run"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "base_model": self.base_model,
            "beta": self.beta,
            "min_reward_margin": self.min_reward_margin,
            "max_collapse_steps": self.max_collapse_steps,
            "loss_type": self.loss_type,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RLTrainResult:
    """Immutable result returned after a GRPO or DPO run completes."""

    adapter_path: Path
    base_model: str
    training_receipt: str
    final_reward_mean: float
    steps_completed: int
    wall_time_seconds: float
    backend_used: str
    algorithm: str              # "grpo" or "dpo"

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_path": str(self.adapter_path),
            "base_model": self.base_model,
            "training_receipt": self.training_receipt,
            "final_reward_mean": self.final_reward_mean,
            "steps_completed": self.steps_completed,
            "wall_time_seconds": self.wall_time_seconds,
            "backend_used": self.backend_used,
            "algorithm": self.algorithm,
        }


__all__ = [
    "DPOConfig",
    "GRPOConfig",
    "RLTrainResult",
]
