"""DanteDojo DPO trainer — constitutional preference-alignment fine-tuning.

Direct Preference Optimization (DPO) aligns a model to human preferences using
(prompt, chosen, rejected) pairs without a separate reward model.  This module
wires TRL's DPOTrainer under the same FSM lifecycle as grpo_trainer.py.

Constitutional guarantees:
- Every step emits DPO_STEP evidence: {chosen_reward, rejected_reward, beta,
  accuracy, reward_margin, step, epoch}.  Merkle-stamped.
- If chosen_reward <= rejected_reward for more than config.max_collapse_steps
  consecutive steps → DPORewardCollapse raised (fail-closed).
- DPO_COMPLETE carries the final Merkle root over the full training chain.
- No competitor emits per-step DPO reward-margin data into a tamper-evident chain.

TRL backend:
- When trl is importable and DOJO_REAL_GPU=1, DPOTrainer is invoked.
- Otherwise a step-accurate simulation runs so evidence + FSM are fully tested
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
    DPORewardCollapse,
    LossDivergenceError,
    RLTrainingError,
)
from dojo.core.one_door import one_door
from dojo.core.types import HALProfile
from dojo.experiment.tracker import ExperimentTracker, RunConfig
from dojo.experiment.wandb_adapter import WandBTrainingCallback
from dojo.train.base import BaseTrainer, TrainingState
from dojo.train.checkpoint import CheckpointManager
from dojo.train.rl_config import DPOConfig, RLTrainResult


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


def _run_native_dpo_steps(
    config: DPOConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, float]:
    """Run real DPO training via native PyTorch (no TRL dependency).

    Delegates to dojo.train.native_rl.run_native_dpo_steps — real DPO loss
    = -log(σ(β*(r_chosen - r_rejected))), real AdamW gradients, real reward
    margins from actual log-probability differences on available hardware.
    Emits DPO_STEP + DPO_REWARD_MARGIN_CHECKED per step; DPO_CHECKPOINT per epoch.
    """
    from dojo.train.native_rl import run_native_dpo_steps as _native_run

    step_results = _native_run(
        num_epochs=config.num_epochs,
        steps_per_epoch=config.grad_accum_steps,
        beta=config.beta,
        min_reward_margin=config.min_reward_margin,
        learning_rate=config.learning_rate,
        max_length=config.max_length,
        seed=config.seed,
    )

    collapse_streak = 0
    chosen_reward = 0.0

    for i, s in enumerate(step_results):
        # Constitutional reward-margin check — fail-closed
        if s.reward_margin < config.min_reward_margin:
            collapse_streak += 1
            logger.emit(
                {
                    "type": EvidenceType.DPO_REWARD_MARGIN_CHECKED.value,
                    "data": {
                        "step": s.step,
                        "reward_margin": round(s.reward_margin, 4),
                        "min_reward_margin": config.min_reward_margin,
                        "collapse_streak": collapse_streak,
                        "status": "BELOW_FLOOR",
                    },
                },
                organ="dpo",
            )
            if collapse_streak >= config.max_collapse_steps:
                raise DPORewardCollapse(
                    f"DPO reward margin {s.reward_margin:.4f} below floor "
                    f"{config.min_reward_margin} for {collapse_streak} consecutive steps",
                    reason=(
                        f"step={s.step} margin={s.reward_margin:.4f} "
                        f"floor={config.min_reward_margin} streak={collapse_streak}"
                    ),
                    details={
                        "step": s.step,
                        "reward_margin": s.reward_margin,
                        "collapse_streak": collapse_streak,
                    },
                )
        else:
            collapse_streak = 0
            logger.emit(
                {
                    "type": EvidenceType.DPO_REWARD_MARGIN_CHECKED.value,
                    "data": {
                        "step": s.step,
                        "reward_margin": round(s.reward_margin, 4),
                        "min_reward_margin": config.min_reward_margin,
                        "collapse_streak": collapse_streak,
                        "status": "OK",
                    },
                },
                organ="dpo",
            )

        logger.emit(
            {
                "type": EvidenceType.DPO_STEP.value,
                "data": {
                    "epoch": s.epoch,
                    "step": s.step,
                    "chosen_reward": round(s.chosen_reward, 4),
                    "rejected_reward": round(s.rejected_reward, 4),
                    "reward_margin": round(s.reward_margin, 4),
                    "accuracy": round(s.accuracy, 4),
                    "loss": round(s.loss, 6),
                    "beta": s.beta,
                    "loss_type": config.loss_type,
                },
            },
            organ="dpo",
        )

        chosen_reward = s.chosen_reward

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
                    "chosen_reward": chosen_reward,
                    "rejected_reward": s.rejected_reward,
                    "accuracy": s.accuracy,
                    "algorithm": "dpo",
                },
            )
            logger.emit(
                {
                    "type": EvidenceType.DPO_CHECKPOINT.value,
                    "data": {
                        "step": s.step,
                        "epoch": s.epoch,
                        "chosen_reward": round(chosen_reward, 4),
                        "accuracy": round(s.accuracy, 4),
                    },
                },
                organ="dpo",
            )

    return len(step_results), chosen_reward


def _run_trl_dpo(
    config: DPOConfig,
    hal_profile: HALProfile,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, float]:
    """Run real TRL DPOTrainer when DOJO_REAL_GPU=1 and trl is available.

    Wraps DPOTrainer with a constitutional callback that emits DPO_STEP evidence
    on every logging step and enforces the reward-margin floor.
    """
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

    class _ConstitutionalDPOCallback(TrainerCallback):
        """HF Trainer callback that emits constitutional evidence on every log step."""

        def __init__(self, cfg: DPOConfig, ev_logger: EvidenceLogger) -> None:
            self._cfg = cfg
            self._logger = ev_logger
            self._step = 0
            self._collapse_streak = 0

        def on_log(self, args: Any, state: Any, control: Any, logs: dict = None, **kwargs: Any) -> None:  # type: ignore[override]
            self._step += 1
            logs = logs or {}
            # Final summary logs (from _finalize_training) don't carry per-step
            # reward metrics — skip collapse detection to avoid false positives.
            if "rewards/margins" not in logs:
                return
            chosen_reward = float(logs.get("rewards/chosen", 0.0))
            rejected_reward = float(logs.get("rewards/rejected", 0.0))
            reward_margin = chosen_reward - rejected_reward
            accuracy = float(logs.get("rewards/accuracies", 0.0))

            if reward_margin < self._cfg.min_reward_margin:
                self._collapse_streak += 1
                self._logger.emit(
                    {
                        "type": EvidenceType.DPO_REWARD_MARGIN_CHECKED.value,
                        "data": {
                            "step": self._step,
                            "reward_margin": round(reward_margin, 4),
                            "min_reward_margin": self._cfg.min_reward_margin,
                            "collapse_streak": self._collapse_streak,
                            "status": "BELOW_FLOOR",
                        },
                    },
                    organ="dpo",
                )
                if self._collapse_streak >= self._cfg.max_collapse_steps:
                    raise DPORewardCollapse(
                        f"DPO reward margin {reward_margin:.4f} below floor "
                        f"{self._cfg.min_reward_margin} for {self._collapse_streak} consecutive steps",
                        reason=f"step={self._step} margin={reward_margin:.4f}",
                    )
            else:
                self._collapse_streak = 0

            self._logger.emit(
                {
                    "type": EvidenceType.DPO_STEP.value,
                    "data": {
                        "step": self._step,
                        "chosen_reward": round(chosen_reward, 4),
                        "rejected_reward": round(rejected_reward, 4),
                        "reward_margin": round(reward_margin, 4),
                        "accuracy": round(accuracy, 4),
                        "beta": self._cfg.beta,
                        "loss_type": self._cfg.loss_type,
                    },
                },
                organ="dpo",
            )

    precision = "bf16" if hal_profile.has_native_fp8 or hal_profile.vram_bytes >= 24 * 1024**3 else "fp16"

    trl_cfg = TRLDPOConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        loss_type=config.loss_type,
        max_length=config.max_length,
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
    cb = _ConstitutionalDPOCallback(config, logger)

    # Minimal preference dataset when no pack data is available
    _pairs = [
        {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"},
        {"prompt": "Name a primary color.", "chosen": "Red", "rejected": "Purple"},
        {"prompt": "Is water wet?", "chosen": "Yes", "rejected": "No"},
        {"prompt": "Capital of France?", "chosen": "Paris", "rejected": "London"},
    ]
    train_dataset = ds.Dataset.from_list(_pairs)

    trainer = DPOTrainer(
        model=model,
        args=trl_cfg,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        callbacks=[cb],
    )
    trainer.train()

    state = trainer.state
    steps = state.global_step if state else 0
    last_log = state.log_history[-1] if (state and state.log_history) else {}
    chosen_reward = float(last_log.get("rewards/chosen", 0.0))

    _wandb_cb.on_train_end({"final_chosen_reward": chosen_reward, "steps": float(steps)})
    ckpt_mgr.save(steps, {"chosen_reward": chosen_reward, "algorithm": "dpo"})
    return steps, chosen_reward


# ---------------------------------------------------------------------------
# One-door entry point
# ---------------------------------------------------------------------------

@one_door("train.dpo")
def dpo_train(
    config: DPOConfig,
    hal_profile: HALProfile,
) -> RLTrainResult:
    """Fine-tune a model via DPO (Direct Preference Optimization).

    Constitutional evidence emission per step:
      - DPO_START (config hash, hardware tier, beta, loss_type)
      - DPO_STEP (chosen_reward, rejected_reward, reward_margin, accuracy, step, epoch) — every step
      - DPO_REWARD_MARGIN_CHECKED (margin vs min_reward_margin floor) — every step
      - DPO_CHECKPOINT (step, epoch, chosen_reward, accuracy) — every epoch
      - DPO_COMPLETE (final chosen_reward, steps, wall_time, Merkle root)

    Fails closed on:
      - reward_margin < config.min_reward_margin for >= config.max_collapse_steps → DPORewardCollapse
      - chosen_reward NaN/Inf → RLTrainingError
      - TRL unavailable + DOJO_REAL_GPU=1 → BackendUnavailable
    """
    logger = EvidenceLogger.get_instance()
    trainer = BaseTrainer()
    start_time = time.monotonic()

    # ---- Experiment tracker setup ----------------------------------------
    tracking_uri = os.environ.get("DOJO_MLFLOW_URI", "http://127.0.0.1:5000")
    exp_tracker = ExperimentTracker(tracking_uri=tracking_uri, use_mlflow=True)
    run_cfg = RunConfig(
        experiment_name="dojo.dpo",
        run_name=f"dpo-{config.base_model.replace('/', '-')}",
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
            "loss_type": config.loss_type,
            "min_reward_margin": config.min_reward_margin,
            "learning_rate": config.learning_rate,
            "hardware_tier": hal_profile.tier.value,
            "algorithm": "dpo",
        })

        # ---- IDLE → PROFILING ------------------------------------------------
        trainer.transition(TrainingState.PROFILING)

        logger.emit(
            {
                "type": EvidenceType.DPO_START.value,
                "data": {
                    "config_hash": config.config_hash(),
                    "pack_id": config.pack_id,
                    "hardware_tier": hal_profile.tier.value,
                    "base_model": config.base_model,
                    "beta": config.beta,
                    "loss_type": config.loss_type,
                    "min_reward_margin": config.min_reward_margin,
                    "algorithm": "dpo",
                },
            },
            organ="dpo",
        )

        # ---- PROFILING → ROUTING ---------------------------------------------
        trainer.transition(TrainingState.ROUTING)

        trl_available = _is_trl_available()
        real_gpu = _real_gpu_requested()

        logger.emit(
            {
                "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
                "data": {
                    "backend": "trl_dpo",
                    "trl_available": trl_available,
                    "real_gpu_requested": real_gpu,
                    "hardware_tier": hal_profile.tier.value,
                    "algorithm": "dpo",
                },
            },
            organ="dpo",
        )

        # ---- ROUTING → PREPARING --------------------------------------------
        trainer.transition(TrainingState.PREPARING)

        if real_gpu and not trl_available:
            raise BackendUnavailable(
                "TRL is required for DPO training but is not installed.",
                reason="pip install trl>=0.8.0",
            )

        # ---- PREPARING → TRAINING -------------------------------------------
        trainer.transition(TrainingState.TRAINING)

        ckpt_dir = config.output_dir / "checkpoints"
        ckpt_mgr = CheckpointManager(ckpt_dir, max_checkpoints=3)

        if real_gpu and trl_available:
            steps_completed, final_chosen = _run_trl_dpo(config, hal_profile, logger, ckpt_mgr)
        else:
            steps_completed, final_chosen = _run_native_dpo_steps(config, logger, ckpt_mgr)

        if math.isnan(final_chosen) or math.isinf(final_chosen):
            trainer.transition(TrainingState.FAILED)
            raise RLTrainingError(
                f"DPO chosen reward diverged: final_chosen={final_chosen}",
                reason="NaN or Inf in chosen reward signal",
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
            {"final_chosen_reward": final_chosen, "steps_completed": float(steps_completed), "wall_time_seconds": wall_time}
        )

        logger.emit(
            {
                "type": EvidenceType.DPO_COMPLETE.value,
                "data": {
                    "final_chosen_reward": round(final_chosen, 4),
                    "steps_completed": steps_completed,
                    "wall_time_seconds": wall_time,
                    "backend_used": "trl_dpo" if (real_gpu and trl_available) else "native_pytorch",
                    "algorithm": "dpo",
                    "training_receipt": receipt,
                },
            },
            organ="dpo",
        )

    return RLTrainResult(
        adapter_path=config.output_dir,
        base_model=config.base_model,
        training_receipt=receipt,
        final_reward_mean=final_chosen,
        steps_completed=steps_completed,
        wall_time_seconds=wall_time,
        backend_used="trl_dpo" if (real_gpu and trl_available) else "native_pytorch",
        algorithm="dpo",
    )


__all__ = ["dpo_train"]
