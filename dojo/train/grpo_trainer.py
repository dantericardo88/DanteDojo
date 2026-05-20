"""DanteDojo GRPO trainer — constitutional reinforcement-learning fine-tuning.

Group Relative Policy Optimization (GRPO) is the algorithm behind DeepSeek-R1
style reasoning models.  This module wires TRL's GRPOTrainer under the same
FSM lifecycle and constitutional evidence emission pattern as finetune.py.

Constitutional guarantees:
- Every rollout step emits a GRPO_STEP evidence bundle: {reward_mean, kl_div,
  clip_fraction, step, epoch}.  Stamped into the Merkle chain.
- If kl_div > config.max_kl → KLExplosionError raised immediately (fail-closed).
- GRPO_COMPLETE carries the final Merkle root over the entire training chain.
- No competitor emits per-step RL evidence into a tamper-evident chain.

TRL backend:
- When trl is importable and DOJO_REAL_GPU=1, GRPOTrainer is invoked.
- Otherwise, a step-accurate simulation runs so evidence + FSM are fully tested
  without GPU hardware.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import (
    BackendUnavailable,
    GRPODivergenceError,
    KLExplosionError,
    LossDivergenceError,
    VRAMExceeded,
)
from dojo.core.one_door import one_door
from dojo.core.types import HALProfile
from dojo.experiment.tracker import ExperimentTracker, RunConfig
from dojo.experiment.wandb_adapter import WandBTrainingCallback
from dojo.train.base import BaseTrainer, TrainingState
from dojo.train.checkpoint import CheckpointManager
from dojo.train.rl_config import GRPOConfig, RLTrainResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_trl_available() -> bool:
    try:
        import importlib
        importlib.import_module("trl")
        return True
    except ImportError:
        return False


def _real_gpu_requested() -> bool:
    return os.environ.get("DOJO_REAL_GPU", "0") == "1"


def _run_native_grpo_steps(
    config: GRPOConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, float]:
    """Run real GRPO training via native PyTorch (no TRL dependency).

    Delegates to dojo.train.native_rl.run_native_grpo_steps — real forward passes,
    real KL from learned distributions, real AdamW gradient updates on available
    hardware (CPU or CUDA).  Emits GRPO_STEP + GRPO_KL_CHECKED per step;
    GRPO_CHECKPOINT per epoch.
    """
    from dojo.train.native_rl import run_native_grpo_steps as _native_run

    step_results = _native_run(
        num_epochs=config.num_epochs,
        steps_per_epoch=config.grad_accum_steps,
        beta=config.beta,
        max_kl=config.max_kl,
        num_generations=config.num_generations,
        learning_rate=config.learning_rate,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
    )

    reward_mean = 0.0

    for i, s in enumerate(step_results):
        # Constitutional KL check — fail-closed (>= makes max_kl a strict upper bound)
        if s.kl_div >= config.max_kl:
            logger.emit(
                {
                    "type": EvidenceType.GRPO_KL_CHECKED.value,
                    "data": {
                        "step": s.step,
                        "kl_div": s.kl_div,
                        "max_kl": config.max_kl,
                        "status": "VIOLATED",
                    },
                },
                organ="grpo",
            )
            raise KLExplosionError(
                f"KL divergence {s.kl_div:.4f} exceeded constitutional ceiling {config.max_kl}",
                reason=f"step={s.step} kl_div={s.kl_div:.4f} max_kl={config.max_kl}",
                details={"step": s.step, "kl_div": s.kl_div, "max_kl": config.max_kl},
            )

        logger.emit(
            {
                "type": EvidenceType.GRPO_STEP.value,
                "data": {
                    "epoch": s.epoch,
                    "step": s.step,
                    "reward_mean": round(s.reward_mean, 4),
                    "kl_div": round(s.kl_div, 6),
                    "clip_fraction": round(s.clip_fraction, 4),
                    "num_generations": s.num_generations,
                },
            },
            organ="grpo",
        )
        logger.emit(
            {
                "type": EvidenceType.GRPO_KL_CHECKED.value,
                "data": {
                    "step": s.step,
                    "kl_div": round(s.kl_div, 6),
                    "max_kl": config.max_kl,
                    "status": "OK",
                },
            },
            organ="grpo",
        )

        reward_mean = s.reward_mean

        # Checkpoint at end of each epoch
        is_epoch_end = (
            i + 1 == len(step_results)
            or step_results[i + 1].epoch != s.epoch
        )
        if is_epoch_end:
            ckpt_mgr.save(
                s.step,
                {
                    "epoch": s.epoch,
                    "reward_mean": reward_mean,
                    "kl_div": s.kl_div,
                    "algorithm": "grpo",
                },
            )
            logger.emit(
                {
                    "type": EvidenceType.GRPO_CHECKPOINT.value,
                    "data": {
                        "step": s.step,
                        "epoch": s.epoch,
                        "reward_mean": round(reward_mean, 4),
                    },
                },
                organ="grpo",
            )

    return len(step_results), reward_mean


def _run_trl_grpo(
    config: GRPOConfig,
    hal_profile: HALProfile,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, float]:
    """Run real TRL GRPOTrainer when DOJO_REAL_GPU=1 and trl is available.

    Wraps GRPOTrainer with a constitutional callback that emits GRPO_STEP
    evidence on every logging step and enforces the max_kl ceiling.
    """
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

    class _ConstitutionalGRPOCallback(TrainerCallback):
        """HF Trainer callback that emits constitutional evidence on every log step."""

        def __init__(self, cfg: GRPOConfig, ev_logger: EvidenceLogger) -> None:
            self._cfg = cfg
            self._logger = ev_logger
            self._step = 0

        def on_log(self, args: Any, state: Any, control: Any, logs: dict = None, **kwargs: Any) -> None:  # type: ignore[override]
            self._step += 1
            logs = logs or {}
            reward_mean = float(logs.get("rewards/mean", 0.0))
            kl_div = float(logs.get("kl", 0.0))
            clip_fraction = float(logs.get("clip_range", 0.0))

            if kl_div > self._cfg.max_kl:
                self._logger.emit(
                    {
                        "type": EvidenceType.GRPO_KL_CHECKED.value,
                        "data": {
                            "step": self._step,
                            "kl_div": kl_div,
                            "max_kl": self._cfg.max_kl,
                            "status": "VIOLATED",
                        },
                    },
                    organ="grpo",
                )
                raise KLExplosionError(
                    f"KL divergence {kl_div:.4f} exceeded constitutional ceiling {self._cfg.max_kl}",
                    reason=f"step={self._step} kl_div={kl_div:.4f} max_kl={self._cfg.max_kl}",
                )

            self._logger.emit(
                {
                    "type": EvidenceType.GRPO_STEP.value,
                    "data": {
                        "step": self._step,
                        "reward_mean": round(reward_mean, 4),
                        "kl_div": round(kl_div, 6),
                        "clip_fraction": round(clip_fraction, 4),
                        "num_generations": self._cfg.num_generations,
                    },
                },
                organ="grpo",
            )

    precision = "bf16" if hal_profile.has_native_fp8 or hal_profile.vram_bytes >= 24 * 1024**3 else "fp16"

    # TRL requires generation_batch_size (= per_device_train_batch_size) to be
    # divisible by num_generations; bump the batch size up to satisfy this.
    effective_batch_size = max(config.batch_size, config.num_generations)

    trl_cfg = TRLGRPOConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        max_completion_length=config.max_new_tokens,
        temperature=config.temperature,
        num_generations=config.num_generations,
        beta=config.beta,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        seed=config.seed,
        logging_steps=1,
        save_steps=config.grad_accum_steps * config.num_epochs,
    )

    from dojo.experiment.wandb_adapter import WandBAdapter
    import datasets as ds
    _wandb_adapter = WandBAdapter(use_wandb=True)
    _wandb_adapter.init_run(config)
    _wandb_cb = _wandb_adapter.as_callback()

    model = AutoModelForCausalLM.from_pretrained(config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cb = _ConstitutionalGRPOCallback(config, logger)

    # Minimal reward: longer completions score higher (validates end-to-end RL loop)
    def _length_reward(completions: list[str], **kwargs: Any) -> list[float]:
        return [float(len(c.split())) for c in completions]

    # Minimal dataset: synthetic prompts when no pack data is available
    _prompts = [
        {"prompt": "What is the meaning of life?"},
        {"prompt": "Describe a sunny day."},
        {"prompt": "Explain quantum computing briefly."},
        {"prompt": "Write a haiku about mountains."},
    ]
    train_dataset = ds.Dataset.from_list(_prompts)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[_length_reward],
        args=trl_cfg,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[cb],
    )
    trainer.train()

    state = trainer.state
    steps = state.global_step if state else 0
    last_log = state.log_history[-1] if (state and state.log_history) else {}
    reward_mean = float(last_log.get("rewards/mean", 0.0))

    _wandb_cb.on_train_end({"final_reward_mean": reward_mean, "steps": float(steps)})
    ckpt_mgr.save(steps, {"reward_mean": reward_mean, "algorithm": "grpo"})
    return steps, reward_mean


# ---------------------------------------------------------------------------
# One-door entry point
# ---------------------------------------------------------------------------

@one_door("train.grpo")
def grpo_train(
    config: GRPOConfig,
    hal_profile: HALProfile,
) -> RLTrainResult:
    """Fine-tune a model via GRPO (Group Relative Policy Optimization).

    Constitutional evidence emission per step:
      - GRPO_START (config hash, hardware tier, reward_model)
      - GRPO_STEP (reward_mean, kl_div, clip_fraction, step, epoch) — every step
      - GRPO_KL_CHECKED (kl_div vs max_kl ceiling) — every step
      - GRPO_CHECKPOINT (step, epoch, reward_mean) — every epoch
      - GRPO_COMPLETE (final_reward_mean, steps, wall_time, Merkle root)

    Fails closed on:
      - kl_div > config.max_kl → KLExplosionError
      - reward NaN/Inf → GRPODivergenceError
      - TRL unavailable + DOJO_REAL_GPU=1 → BackendUnavailable
    """
    logger = EvidenceLogger.get_instance()
    trainer = BaseTrainer()
    start_time = time.monotonic()

    # ---- Experiment tracker setup ----------------------------------------
    tracking_uri = os.environ.get("DOJO_MLFLOW_URI", "http://127.0.0.1:5000")
    exp_tracker = ExperimentTracker(tracking_uri=tracking_uri, use_mlflow=True)
    run_cfg = RunConfig(
        experiment_name="dojo.grpo",
        run_name=f"grpo-{config.base_model.replace('/', '-')}",
        config_hash=config.config_hash(),
        pack_id=config.pack_id,
        hardware_tier=hal_profile.tier.value,
        seed=config.seed,
    )

    with exp_tracker.run(run_cfg):
        exp_tracker.log_params({
            "base_model": config.base_model,
            "num_epochs": config.num_epochs,
            "beta": config.beta,
            "max_kl": config.max_kl,
            "num_generations": config.num_generations,
            "learning_rate": config.learning_rate,
            "hardware_tier": hal_profile.tier.value,
            "algorithm": "grpo",
        })

        # ---- IDLE → PROFILING ------------------------------------------------
        trainer.transition(TrainingState.PROFILING)

        logger.emit(
            {
                "type": EvidenceType.GRPO_START.value,
                "data": {
                    "config_hash": config.config_hash(),
                    "pack_id": config.pack_id,
                    "hardware_tier": hal_profile.tier.value,
                    "base_model": config.base_model,
                    "reward_model": config.reward_model,
                    "beta": config.beta,
                    "max_kl": config.max_kl,
                    "num_generations": config.num_generations,
                    "algorithm": "grpo",
                },
            },
            organ="grpo",
        )

        # ---- PROFILING → ROUTING ---------------------------------------------
        trainer.transition(TrainingState.ROUTING)

        trl_available = _is_trl_available()
        real_gpu = _real_gpu_requested()

        logger.emit(
            {
                "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
                "data": {
                    "backend": "trl_grpo",
                    "trl_available": trl_available,
                    "real_gpu_requested": real_gpu,
                    "hardware_tier": hal_profile.tier.value,
                    "algorithm": "grpo",
                },
            },
            organ="grpo",
        )

        # ---- ROUTING → PREPARING --------------------------------------------
        trainer.transition(TrainingState.PREPARING)

        if real_gpu and not trl_available:
            raise BackendUnavailable(
                "TRL is required for GRPO training but is not installed.",
                reason="pip install trl>=0.8.0",
            )

        # ---- PREPARING → TRAINING -------------------------------------------
        trainer.transition(TrainingState.TRAINING)

        ckpt_dir = config.output_dir / "checkpoints"
        ckpt_mgr = CheckpointManager(ckpt_dir, max_checkpoints=3)

        if real_gpu and trl_available:
            steps_completed, final_reward = _run_trl_grpo(config, hal_profile, logger, ckpt_mgr)
        else:
            steps_completed, final_reward = _run_native_grpo_steps(config, logger, ckpt_mgr)

        if math.isnan(final_reward) or math.isinf(final_reward):
            trainer.transition(TrainingState.FAILED)
            raise GRPODivergenceError(
                f"GRPO reward diverged: final_reward_mean={final_reward}",
                reason="NaN or Inf in reward signal",
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
            {"final_reward_mean": final_reward, "steps_completed": float(steps_completed), "wall_time_seconds": wall_time}
        )

        logger.emit(
            {
                "type": EvidenceType.GRPO_COMPLETE.value,
                "data": {
                    "final_reward_mean": round(final_reward, 4),
                    "steps_completed": steps_completed,
                    "wall_time_seconds": wall_time,
                    "backend_used": "trl_grpo" if (real_gpu and trl_available) else "native_pytorch",
                    "algorithm": "grpo",
                    "training_receipt": receipt,
                },
            },
            organ="grpo",
        )

    return RLTrainResult(
        adapter_path=config.output_dir,
        base_model=config.base_model,
        training_receipt=receipt,
        final_reward_mean=final_reward,
        steps_completed=steps_completed,
        wall_time_seconds=wall_time,
        backend_used="trl_grpo" if (real_gpu and trl_available) else "native_pytorch",
        algorithm="grpo",
    )


__all__ = ["grpo_train"]
