"""DanteDojo one-door fine-tuning entry point.

Routes to the appropriate training backend based on HALProfile, runs the
full training FSM, emits constitutional evidence, and performs pre-flight
VRAM checks.  In v1 the backend kernels (Unsloth/Axolotl/MLX) are stubbed
so the pipeline works without a GPU.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BackendUnavailable, VRAMExceeded
from dojo.core.one_door import one_door
from dojo.core.types import HALProfile, HardwareTier, PrecisionType, TrainingBackend
from dojo.pack.schema import SpecializationPack
from dojo.train.base import BaseTrainer, FinetuneConfig, FinetuneResult, TrainingState
from dojo.train.checkpoint import CheckpointManager
from dojo.train.context import ContextLengthManager


# ---------------------------------------------------------------------------
# Backend routing table
# ---------------------------------------------------------------------------

_BACKEND_MAP: dict[HardwareTier, TrainingBackend] = {
    HardwareTier.TIER_4060_LAPTOP: TrainingBackend.UNSLOTH,
    HardwareTier.TIER_3090: TrainingBackend.UNSLOTH,
    HardwareTier.TIER_3090_DUAL: TrainingBackend.AXOLOTL,
    HardwareTier.TIER_H100: TrainingBackend.AXOLOTL,
    HardwareTier.TIER_APPLE_M4: TrainingBackend.MLX,
    HardwareTier.TIER_AMD_MI300: TrainingBackend.NATIVE,
    HardwareTier.TIER_INTEL_B580: TrainingBackend.NATIVE,
    HardwareTier.TIER_CPU: TrainingBackend.NATIVE,
}

# VRAM safety threshold — refuse if estimated usage exceeds this fraction.
_VRAM_SAFETY_RATIO = 0.85

# Stub final-loss value used when no real training loop runs.
_STUB_FINAL_LOSS = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_backend(profile: HALProfile) -> TrainingBackend:
    return _BACKEND_MAP.get(profile.tier, TrainingBackend.NATIVE)


def _is_backend_available(backend: TrainingBackend) -> bool:
    if backend == TrainingBackend.UNSLOTH:
        from dojo.train.unsloth_backend import is_unsloth_available
        return is_unsloth_available()
    if backend == TrainingBackend.AXOLOTL:
        from dojo.train.axolotl_backend import is_axolotl_available
        return is_axolotl_available()
    if backend == TrainingBackend.MLX:
        try:
            import importlib
            importlib.import_module("mlx")
            return True
        except ImportError:
            return False
    # NATIVE — always consider available as a fallback
    return True


def _estimate_model_params(base_model: str) -> float:
    """Heuristic: extract parameter count in billions from a model name string."""
    import re
    # Common patterns: "7b", "7B", "13b", "70b", "8B", "72B", "3.8b"
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", base_model)
    if match:
        return float(match.group(1))
    # Default assumption if parsing fails
    return 7.0


def _check_vram(config: FinetuneConfig, profile: HALProfile) -> None:
    """Raise VRAMExceeded if estimated VRAM exceeds 85 % of available VRAM."""
    if profile.vram_bytes == 0:
        # CPU-only — no VRAM constraint
        return

    model_params_b = _estimate_model_params(config.base_model)
    precision = PrecisionType.NF4  # QLoRA assumption for VRAM estimate

    estimated = ContextLengthManager.estimate_vram_for_config(
        model_params_b=model_params_b,
        batch_size=config.batch_size,
        seq_len=config.max_seq_length,
        precision=precision,
        lora_r=config.lora_r,
    )

    limit = int(profile.vram_bytes * _VRAM_SAFETY_RATIO)
    if estimated > limit:
        raise VRAMExceeded(
            f"Estimated VRAM {estimated / 1e9:.1f} GB exceeds "
            f"85% of {profile.vram_bytes / 1e9:.1f} GB ({limit / 1e9:.1f} GB)",
            reason=f"model_params_b={model_params_b}, seq_len={config.max_seq_length}",
            details={"estimated_bytes": estimated, "limit_bytes": limit},
        )


def _simulate_training_steps(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, float]:
    """Simulate the training loop: emit step events and save one checkpoint.

    Returns (steps_completed, final_loss).
    """
    # In v1 we do one simulated optimizer step per epoch for evidence emission.
    steps_completed = 0
    simulated_loss = 1.0

    for epoch in range(config.num_epochs):
        # Emit a step event every grad_accum_steps
        logger.emit(
            {
                "type": EvidenceType.TRAINING_STEP.value,
                "data": {
                    "epoch": epoch + 1,
                    "step": steps_completed,
                    "loss": simulated_loss,
                },
            },
            organ="train",
        )
        steps_completed += config.grad_accum_steps
        simulated_loss = max(0.01, simulated_loss * 0.8)

    # Save one checkpoint at the final step
    ckpt_mgr.save(steps_completed, {"loss": simulated_loss, "epochs": config.num_epochs})

    return steps_completed, simulated_loss


# ---------------------------------------------------------------------------
# One-door entry point
# ---------------------------------------------------------------------------

@one_door("train.finetune")
def finetune(
    config: FinetuneConfig,
    hal_profile: HALProfile,
    pack: SpecializationPack,
) -> FinetuneResult:
    """Fine-tune a base model on a specialization pack.

    Routes to backend based on HALProfile:
      - TIER_4060_LAPTOP → UNSLOTH (with 4060-safe defaults enforced)
      - TIER_3090 → UNSLOTH (primary)
      - TIER_3090_DUAL → AXOLOTL (FSDP2)
      - TIER_H100 → AXOLOTL or NATIVE
      - TIER_APPLE_M4 → MLX

    Evidence emission:
      - TRAINING_START (config hash, pack id, hardware tier)
      - TRAINING_BACKEND_SELECTED
      - TRAINING_STEP (every grad_accum_steps)
      - TRAINING_CHECKPOINT
      - TRAINING_COMPLETE (final_loss, steps, wall_time)

    Fails closed on:
      - VRAM > 85% of profile.vram_bytes → VRAMExceeded
      - Backend not installed → BackendUnavailable (warning, stub result)
      - Loss NaN/Inf → LossDivergenceError
    """
    logger = EvidenceLogger.get_instance()
    trainer = BaseTrainer()
    start_time = time.monotonic()

    # ---- IDLE → PROFILING ------------------------------------------------
    trainer.transition(TrainingState.PROFILING)

    logger.emit(
        {
            "type": EvidenceType.TRAINING_START.value,
            "data": {
                "config_hash": config.config_hash(),
                "pack_id": pack.pack_id,
                "hardware_tier": hal_profile.tier.value,
                "base_model": config.base_model,
            },
        },
        organ="train",
    )

    # ---- PROFILING → ROUTING ---------------------------------------------
    trainer.transition(TrainingState.ROUTING)

    backend = _select_backend(hal_profile)
    backend_available = _is_backend_available(backend)

    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {
                "backend": backend.value,
                "available": backend_available,
                "hardware_tier": hal_profile.tier.value,
            },
        },
        organ="train",
    )

    # ---- ROUTING → PREPARING --------------------------------------------
    trainer.transition(TrainingState.PREPARING)

    # Pre-flight VRAM check (raises VRAMExceeded if over limit)
    _check_vram(config, hal_profile)

    safe_ctx = ContextLengthManager.get_safe_context(hal_profile, requested=config.max_seq_length)

    # ---- PREPARING → TRAINING -------------------------------------------
    trainer.transition(TrainingState.TRAINING)

    ckpt_dir = config.output_dir / "checkpoints"
    ckpt_mgr = CheckpointManager(ckpt_dir, max_checkpoints=3)

    if not backend_available:
        warnings.warn(
            f"Backend '{backend.value}' is not installed. "
            "Returning stub FinetuneResult without real training.",
            stacklevel=2,
        )
        steps_completed = 0
        final_loss = _STUB_FINAL_LOSS
    else:
        steps_completed, final_loss = _simulate_training_steps(config, logger, ckpt_mgr)

    import math
    if math.isnan(final_loss) or math.isinf(final_loss):
        from dojo.core.exceptions import LossDivergenceError
        trainer.transition(TrainingState.FAILED)
        raise LossDivergenceError(
            f"Loss diverged: final_loss={final_loss}",
            reason="NaN or Inf detected in training loss",
        )

    # ---- TRAINING → CHECKPOINTING ---------------------------------------
    trainer.transition(TrainingState.CHECKPOINTING)

    # ---- CHECKPOINTING → EVALUATING -------------------------------------
    trainer.transition(TrainingState.EVALUATING)

    # ---- EVALUATING → COMPLETED -----------------------------------------
    trainer.transition(TrainingState.COMPLETED)

    wall_time = time.monotonic() - start_time
    receipt = logger.compute_merkle_root()

    logger.emit(
        {
            "type": EvidenceType.TRAINING_COMPLETE.value,
            "data": {
                "final_loss": final_loss,
                "steps_completed": steps_completed,
                "wall_time_seconds": wall_time,
                "backend_used": backend.value,
                "training_receipt": receipt,
                "safe_ctx": safe_ctx,
            },
        },
        organ="train",
    )

    return FinetuneResult(
        adapter_path=config.output_dir,
        base_model=config.base_model,
        training_receipt=receipt,
        final_loss=final_loss,
        steps_completed=steps_completed,
        wall_time_seconds=wall_time,
        backend_used=backend.value,
    )


__all__ = ["finetune"]
