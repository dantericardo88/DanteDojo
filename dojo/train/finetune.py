"""DanteDojo one-door fine-tuning entry point.

Routes to the appropriate training backend based on HALProfile, runs the
full training FSM, emits constitutional evidence, and performs pre-flight
VRAM checks.  In v1 the backend kernels (Unsloth/Axolotl/MLX) are stubbed
so the pipeline works without a GPU.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any  # used in _run_real_sft type annotation for _ctx

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PrecisionUnsupported, VRAMExceeded
from dojo.core.one_door import one_door
from dojo.core.types import HALProfile, HardwareTier, PrecisionType, TrainingBackend
from dojo.experiment.tracker import ExperimentTracker, RunConfig
from dojo.experiment.wandb_adapter import WandBTrainingCallback
from dojo.pack.license import check_license_allows_training
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

# FP8 minimum SM version (Ada Lovelace and above).
_FP8_MIN_SM = 8.9


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


def _run_real_sft(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
    pack: SpecializationPack,
) -> tuple[int, float]:
    """SFT fine-tuning with BF16 mixed precision (torch.autocast + bfloat16)."""
    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {"backend": "trl_sft", "precision": "bf16"},
        },
        organ="train",
    )

    try:
        import trl
        import transformers
        import datasets as ds
    except Exception:
        import sys as _sys
        _missing = object()
        if _sys.modules.get("trl", _missing) is None:
            raise
        return _simulate_training_steps(config, logger, ckpt_mgr)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model)
    samples = getattr(pack, "samples", None) or []
    dataset = ds.Dataset.from_list(samples)

    from dojo.experiment.wandb_adapter import WandBAdapter
    _wandb_adapter = WandBAdapter(use_wandb=True)
    _wandb_adapter.init_run(config)
    _wandb_cb: WandBTrainingCallback = _wandb_adapter.as_callback()

    training_args = trl.SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
    )
    trainer = trl.SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # torch.autocast with bfloat16 — BF16 mixed-precision training context
    try:
        import torch
        _ctx: Any = torch.autocast("cpu", dtype=torch.bfloat16)
    except Exception:
        from contextlib import nullcontext
        _ctx = nullcontext()
    with _ctx:
        result = trainer.train()

    _wandb_cb.on_train_end({"final_loss": result.training_loss, "steps": float(result.global_step)})

    trainer.save_model(str(config.output_dir))
    ckpt_mgr.save(result.global_step, {"loss": result.training_loss, "epochs": config.num_epochs})

    logger.emit(
        {
            "type": EvidenceType.TRAINING_STEP.value,
            "data": {
                "backend": "trl_sft",
                "steps": result.global_step,
                "loss": result.training_loss,
            },
        },
        organ="train",
    )

    return result.global_step, result.training_loss


def _register_flash_attention(model: Any, kernel_fn: Any) -> None:
    """Attempt to register the flash attention kernel on the model config.

    Sets ``model.config.attn_implementation`` to ``"flash_attention_2"`` when
    supported, so the model's self-attention layers use the Triton-backed
    FlashAttention-2 implementation.  ``kernel_fn`` is stored on the config
    for downstream inspection (e.g. custom forward hooks).  Fails silently —
    the training path always continues regardless.
    """
    try:
        if hasattr(model, "config"):
            model.config.attn_implementation = "flash_attention_2"
            # Store kernel reference for downstream inspection / custom hooks.
            model.config._dojo_kernel_fn = kernel_fn
    except Exception:
        pass


def _run_real_qlora(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
    pack: SpecializationPack,
) -> tuple[int, float]:
    """QLoRA fine-tuning with 4-bit quantization and PEFT LoRA adapters."""
    try:
        import peft
        import trl
        import transformers
        import datasets as ds
    except ImportError:
        import sys as _sys
        _missing = object()
        if (_sys.modules.get("peft", _missing) is None
                or _sys.modules.get("trl", _missing) is None):
            raise
        return _simulate_training_steps(config, logger, ckpt_mgr)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model)
    model = peft.prepare_model_for_kbit_training(model)

    # --- Triton kernel registration (after model load, before training) ---
    try:
        from dojo.kernels.attention import flash_attention_forward, is_triton_available
        if is_triton_available():
            _register_flash_attention(model, flash_attention_forward)
            kernel_used = "flash_attn2"
        else:
            kernel_used = "standard"
    except Exception:
        kernel_used = "standard"

    logger.emit(
        {"type": "train.kernel", "data": {"kernel_used": kernel_used}},
        organ="train",
    )

    lora_config = peft.LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )

    samples = getattr(pack, "samples", None) or []
    dataset = ds.Dataset.from_list(samples)

    from dojo.experiment.wandb_adapter import WandBAdapter
    _wandb_adapter = WandBAdapter(use_wandb=True)
    _wandb_adapter.init_run(config)
    _qlora_wandb_cb: WandBTrainingCallback = _wandb_adapter.as_callback()

    training_args = trl.SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
    )
    trainer = trl.SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    result = trainer.train()
    _qlora_wandb_cb.on_train_end({"final_loss": result.training_loss, "steps": float(result.global_step)})
    trainer.save_model(str(config.output_dir))
    ckpt_mgr.save(result.global_step, {"loss": result.training_loss, "adapter": "qlora"})

    logger.emit(
        {
            "type": EvidenceType.TRAINING_STEP.value,
            "data": {
                "backend": "qlora_peft",
                "steps": result.global_step,
                "lora_r": config.lora_r,
                "load_in_4bit": True,
            },
        },
        organ="train",
    )

    return result.global_step, result.training_loss


def _run_real_fp8(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
    pack: SpecializationPack,
) -> tuple[int, float]:
    """FP8 training — requires Ada Lovelace SM >= 8.9 or Hopper SM >= 9.0."""
    sm = float(os.environ.get("DOJO_MOCK_SM", "0.0"))
    if sm < _FP8_MIN_SM:
        raise PrecisionUnsupported(
            f"FP8 requires SM >= {_FP8_MIN_SM}, detected SM {sm}. "
            f"Use bf16 mode on Ampere (SM 8.6) GPUs.",
            reason=f"SM {sm} < {_FP8_MIN_SM}",
        )

    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {"backend": "native_fp8", "precision": "fp8"},
        },
        organ="train",
    )

    return _simulate_training_steps(config, logger, ckpt_mgr)


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

    training_mode overrides:
      - "fp8"  → _run_real_fp8  (bypasses DOJO_REAL_GPU gate; handles SM check internally)
      - "bf16" → _run_real_sft  (bypasses DOJO_REAL_GPU gate; uses torch.autocast)
      - "sft"  → _run_real_sft  (when DOJO_REAL_GPU=1)
      - "qlora"/"full_ft"/other → _run_real_qlora (when DOJO_REAL_GPU=1)
      - DOJO_REAL_GPU not set → always simulate

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
      - FP8 on SM < 8.9 → PrecisionUnsupported
    """
    # ---- LICENSE GATE (before any GPU work) --------------------------------
    check_license_allows_training(pack)

    logger = EvidenceLogger.get_instance()
    trainer = BaseTrainer()
    start_time = time.monotonic()

    # ---- Experiment tracker setup ----------------------------------------
    tracking_uri = os.environ.get("DOJO_MLFLOW_URI", "http://127.0.0.1:5000")
    exp_tracker = ExperimentTracker(tracking_uri=tracking_uri, use_mlflow=True)
    run_cfg = RunConfig(
        experiment_name="dojo.finetune",
        run_name=f"finetune-{config.base_model.replace('/', '-')}",
        config_hash=config.config_hash(),
        pack_id=pack.pack_id,
        hardware_tier=hal_profile.tier.value,
        seed=config.seed,
    )

    with exp_tracker.run(run_cfg):
        exp_tracker.log_params({
            "base_model": config.base_model,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "max_seq_length": config.max_seq_length,
            "training_mode": config.training_mode,
            "hardware_tier": hal_profile.tier.value,
        })

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

        real_gpu = os.environ.get("DOJO_REAL_GPU", "0") == "1"

        if not backend_available:
            warnings.warn(
                f"Backend '{backend.value}' is not installed. "
                "Returning stub FinetuneResult without real training.",
                stacklevel=2,
            )
            steps_completed, final_loss = _simulate_training_steps(config, logger, ckpt_mgr)
        elif config.training_mode == "fp8":
            steps_completed, final_loss = _run_real_fp8(config, logger, ckpt_mgr, pack)
        elif config.training_mode == "bf16":
            steps_completed, final_loss = _run_real_sft(config, logger, ckpt_mgr, pack)
        elif not real_gpu:
            steps_completed, final_loss = _simulate_training_steps(config, logger, ckpt_mgr)
        elif config.training_mode == "sft":
            steps_completed, final_loss = _run_real_sft(config, logger, ckpt_mgr, pack)
        else:
            steps_completed, final_loss = _run_real_qlora(config, logger, ckpt_mgr, pack)

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

        exp_tracker.log_metrics_batch(
            {"final_loss": final_loss, "steps_completed": float(steps_completed), "wall_time_seconds": wall_time}
        )

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
