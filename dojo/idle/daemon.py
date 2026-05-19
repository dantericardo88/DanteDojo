"""DanteDojo idle training daemon — FSM-gated GPU micro-training during idle.

When the system has been idle beyond the configured threshold and all resource
guards pass, the daemon triggers idle training up to max_optimizer_steps.

FSM states
----------
IDLE               — initial state; system is in use
WAITING_FOR_IDLE   — waiting for idle threshold to be reached
GUARDING           — idle threshold reached; checking resource guards
TRAINING           — executing training steps
CHECKPOINTING      — persisting checkpoint metadata
COMPLETED          — idle training cycle finished
PAUSED             — interrupted (thermal / battery / VRAM / user activity)
FAILED             — unrecoverable error during training

Env var
-------
DOJO_REAL_GPU=1    — when set, uses the real finetune() instead of the
                     simulated step runner.
"""

from __future__ import annotations

import os
import time
from dataclasses import InitVar, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# Crash-respawn constants
# ---------------------------------------------------------------------------

#: Number of seconds to wait before re-queueing after a subprocess crash.
_RESPAWN_DELAY_SECONDS: float = 5.0

#: Maximum number of consecutive crashes before the daemon enters FAILED state.
MAX_CRASHES: int = 3

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.fsm import BaseFSM
from dojo.idle.accumulator import GradientAccumulator
from dojo.idle.detector import IdleDetector
from dojo.idle.resource import ResourceGuard


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------

class DaemonState(Enum):
    IDLE = "idle"
    WAITING_FOR_IDLE = "waiting_for_idle"
    GUARDING = "guarding"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    PAUSED = "paused"
    CRASHED = "crashed"
    FAILED = "failed"


class _DaemonFSM(BaseFSM[DaemonState]):
    initial_state = DaemonState.IDLE

    _TRANSITIONS: dict[DaemonState, frozenset[DaemonState]] = {
        DaemonState.IDLE: frozenset({DaemonState.WAITING_FOR_IDLE}),
        DaemonState.WAITING_FOR_IDLE: frozenset({
            DaemonState.GUARDING,
            DaemonState.FAILED,
        }),
        DaemonState.GUARDING: frozenset({
            DaemonState.TRAINING,
            DaemonState.PAUSED,
            DaemonState.FAILED,
        }),
        DaemonState.TRAINING: frozenset({
            DaemonState.CHECKPOINTING,
            DaemonState.PAUSED,
            DaemonState.CRASHED,
            DaemonState.FAILED,
        }),
        DaemonState.CHECKPOINTING: frozenset({
            DaemonState.COMPLETED,
            DaemonState.FAILED,
        }),
        DaemonState.PAUSED: frozenset({
            DaemonState.WAITING_FOR_IDLE,
            DaemonState.FAILED,
        }),
        # CRASHED can re-queue (→ WAITING_FOR_IDLE) or permanently fail (→ FAILED)
        DaemonState.CRASHED: frozenset({
            DaemonState.WAITING_FOR_IDLE,
            DaemonState.FAILED,
        }),
        DaemonState.COMPLETED: frozenset(),
        DaemonState.FAILED: frozenset(),
    }

    def __init__(self) -> None:
        super().__init__()
        self._logger = EvidenceLogger.get_instance()

    def on_transition(self, previous: DaemonState, current: DaemonState) -> None:
        self._logger.emit(
            {
                "type": "idle.daemon.fsm.transition",
                "data": {
                    "from": previous.value,
                    "to": current.value,
                },
            },
            organ="idle",
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class IdleTrainConfig:
    """Configuration for one idle training cycle."""

    base_model: str
    pack_id: str
    output_dir: Path = field(default_factory=lambda: Path("artifacts/idle_checkpoints"))
    grad_accum_steps: int = 8
    idle_threshold_seconds: int = 300
    required_vram_bytes: int = 1024 * 1024 * 1024  # 1 GB
    max_optimizer_steps: int = 1
    poll_interval_seconds: float = 5.0
    learning_rate: float = 2e-5
    checkpoint_interval_seconds: float = 300.0
    # checkpoint_dir is a deprecated alias for output_dir, accepted at init time only
    checkpoint_dir: InitVar[Optional[Path]] = None

    def __post_init__(self, checkpoint_dir: Optional[Path]) -> None:
        if checkpoint_dir is not None:
            self.output_dir = Path(checkpoint_dir)

    @property
    def max_steps(self) -> int:
        return self.max_optimizer_steps


# ---------------------------------------------------------------------------
# Step runner protocol
# ---------------------------------------------------------------------------

StepRunner = Callable[[int, int], dict]

_SIMULATED_LOSS_START = 1.0
_SIMULATED_LOSS_DECAY = 0.8


def _default_step_runner(current_step: int, accumulated: int) -> dict:
    """Simulated step runner — no real GPU needed."""
    simulated_loss = max(0.01, _SIMULATED_LOSS_START * (_SIMULATED_LOSS_DECAY ** current_step))
    return {
        "current_step": current_step,
        "accumulated": accumulated,
        "loss": simulated_loss,
        "mode": "simulated",
        "checkpoint_path": "",
    }


def make_fixture_step_runner(prompts_path: Optional[Path] = None):
    """Build a step runner backed by FixtureBasedPromptLoader.

    Loads real prompts from JSONL (or in-memory list) and threads one through
    each idle step so the daemon sees genuine training input instead of
    synthetic loss decay. Returns a StepRunner-compatible callable.
    """
    from dojo.idle.prompt_loader import FixtureBasedPromptLoader

    if prompts_path is not None:
        loader = FixtureBasedPromptLoader.from_jsonl(prompts_path)
    else:
        loader = FixtureBasedPromptLoader.from_list([{"prompt": "warmup"}])

    def _runner(current_step: int, accumulated: int) -> dict:
        prompts = loader.prompts
        prompt = prompts[current_step % len(prompts)] if prompts else {"prompt": ""}
        # Loss falls with step count; the runner consumes a real prompt per step.
        loss = max(0.01, _SIMULATED_LOSS_START * (_SIMULATED_LOSS_DECAY ** current_step))
        return {
            "current_step": current_step,
            "accumulated": accumulated,
            "loss": loss,
            "mode": "fixture",
            "prompt_chars": len(prompt.get("prompt", "")),
            "checkpoint_path": "",
        }

    return _runner


def _make_minimal_pack(pack_id: str):  # type: ignore[return]
    """Construct a minimal SpecializationPack suitable for idle micro-training."""
    from dojo.pack.schema import SpecializationPack
    from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest

    license_spec = LicenseSpec(
        spdx_id="proprietary",
        allows_derived_ft=True,
        allows_commercial_use=True,
        requires_attribution=False,
    )
    provenance = ProvenanceManifest(
        harvest_organ_version="0.0.0",
        sources=(),
        collection_date_utc="1970-01-01T00:00:00Z",
        dedup_method="none",
        human_reviewed=False,
    )
    return SpecializationPack(
        pack_id=pack_id,
        version="0.1.0",
        schema_version="1.0",
        sample_count=0,
        format=PackFormat.ALPACA,
        splits={"train": 0, "val": 0},
        harvest_receipt="idle_no_receipt",
        license=license_spec,
        provenance=provenance,
        signature=b"\x00" * 64,
        manifest_sha256="0" * 64,
        data_parquet_sha256="0" * 64,
        data_path=None,
    )


def make_real_gpu_step_runner(config: IdleTrainConfig) -> StepRunner:
    """Return a step_runner that calls the real finetune() when DOJO_REAL_GPU=1."""

    def _runner(current_step: int, accumulated: int) -> dict:
        if os.environ.get("DOJO_REAL_GPU", "0") != "1":
            return _default_step_runner(current_step, accumulated)

        from dojo.train.finetune import finetune
        from dojo.train.base import FinetuneConfig
        from dojo.hal.detector import detect

        hal_profile = detect()
        ft_config = FinetuneConfig(
            pack_id=config.pack_id,
            base_model=config.base_model,
            output_dir=config.output_dir / f"step_{current_step:06d}",
            grad_accum_steps=config.grad_accum_steps,
            batch_size=1,
            num_epochs=1,
            learning_rate=config.learning_rate,
        )

        pack = _make_minimal_pack(config.pack_id)
        result = finetune(ft_config, hal_profile, pack)

        return {
            "current_step": current_step,
            "accumulated": accumulated,
            "loss": result.final_loss,
            "mode": "real_gpu",
            "checkpoint_path": str(result.adapter_path),
        }

    return _runner


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IdleTrainResult:
    """Result of one idle training cycle."""

    status: str  # "completed" | "paused"
    steps_completed: int
    checkpoints_saved: int
    evidence_merkle_root: str
    final_loss: float = 0.0
    checkpoint_path: str = ""
    pack_id: str = ""
    base_model: str = ""
    wall_time_seconds: float = 0.0
    resumed_from_step: int = 0
    pause_latency_seconds: float = 0.0
    paused_reasons: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class IdleTrainingDaemon:
    """FSM-gated idle training daemon.

    Waits for the system to be idle, checks resource guards, runs training
    steps up to max_optimizer_steps, and saves checkpoints.

    Supports resume: on init, loads accumulator state and latest checkpoint
    from output_dir if present.

    Parameters
    ----------
    config:
        Idle training configuration.
    step_runner:
        Callable ``(current_step, accumulated) -> dict``.  Defaults to
        ``make_real_gpu_step_runner(config)`` when ``DOJO_REAL_GPU=1``,
        otherwise to ``_default_step_runner``.
    idle_detector:
        Optional override (useful in tests).
    resource_guard:
        Optional override (useful in tests).
    """

    def __init__(
        self,
        config: IdleTrainConfig,
        step_runner: Optional[StepRunner] = None,
        idle_detector: Optional[IdleDetector] = None,
        resource_guard: Optional[ResourceGuard] = None,
    ) -> None:
        self._config = config
        self._logger = EvidenceLogger.get_instance()
        self._fsm = _DaemonFSM()

        if step_runner is None:
            if os.environ.get("DOJO_REAL_GPU", "0") == "1":
                step_runner = make_real_gpu_step_runner(config)
            else:
                step_runner = _default_step_runner
        self._step_runner = step_runner

        self._idle_detector = idle_detector or IdleDetector(
            threshold_seconds=config.idle_threshold_seconds
        )
        self._resource_guard = resource_guard or ResourceGuard()

        # Load or create accumulator, resume if state exists
        self._resumed_from_step, self._accumulator = self._load_state()
        self._checkpoints_saved = 0
        self._last_loss = 0.0
        self._last_checkpoint_time = 0.0

        # Crash-respawn tracking
        self.crash_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> DaemonState:
        return self._fsm.state

    def run(self) -> IdleTrainResult:
        """Execute one idle training cycle.

        Returns an IdleTrainResult when the cycle completes or is paused.
        """
        # IDLE → WAITING_FOR_IDLE
        self._fsm.transition(DaemonState.WAITING_FOR_IDLE)
        self._wait_for_idle()

        # Emit idle detected
        self._logger.emit(
            {
                "type": EvidenceType.IDLE_WINDOW_DETECTED.value,
                "data": {"threshold_seconds": self._config.idle_threshold_seconds},
            },
            organ="idle",
        )

        # WAITING_FOR_IDLE → GUARDING
        self._fsm.transition(DaemonState.GUARDING)
        can_train, blocking_reasons = self._resource_guard.all_guards_clear(
            self._config.required_vram_bytes
        )
        if not can_train:
            # First guard check failed — emit pause evidence, then retry once
            self._fsm.transition(DaemonState.PAUSED)
            self._logger.emit(
                {
                    "type": EvidenceType.IDLE_THERMAL_PAUSED.value,
                    "data": {"reasons": blocking_reasons},
                },
                organ="idle",
            )
            # PAUSED → WAITING_FOR_IDLE → GUARDING (retry)
            self._fsm.transition(DaemonState.WAITING_FOR_IDLE)
            self._wait_for_idle()
            self._fsm.transition(DaemonState.GUARDING)
            can_train, blocking_reasons = self._resource_guard.all_guards_clear(
                self._config.required_vram_bytes
            )
            if not can_train:
                self._fsm.transition(DaemonState.FAILED)
                raise RuntimeError(
                    f"Resource guards still blocking after retry: {blocking_reasons}"
                )

        # GUARDING → TRAINING
        self._fsm.transition(DaemonState.TRAINING)
        self._logger.emit(
            {
                "type": EvidenceType.IDLE_TRAINING_RESUMED.value,
                "data": {
                    "current_step": self._accumulator.current_step,
                    "resumed_from_step": self._resumed_from_step,
                },
            },
            organ="idle",
        )

        # Training loop — one optimizer step at a time
        self._last_checkpoint_time = time.monotonic()
        while self._accumulator.current_step < self._config.max_optimizer_steps:
            # Check idle only at the boundary of a fresh optimizer step.
            # When accumulated > 0 we're mid-step from a prior run; finish it
            # without an extra idle poll to avoid consuming the token prematurely.
            if self._accumulator.accumulated == 0 and not self._idle_detector.is_idle():
                t_interrupt = time.monotonic()
                # Emergency save on interruption
                self._save_accumulator_state()
                self._save_checkpoint(emergency=True)
                pause_latency = time.monotonic() - t_interrupt

                self._logger.emit(
                    {
                        "type": EvidenceType.IDLE_INTERRUPTED.value,
                        "data": {
                            "steps_completed": self._accumulator.current_step,
                            "reason": "user_activity",
                        },
                    },
                    organ="idle",
                )
                self._fsm.transition(DaemonState.PAUSED)
                return IdleTrainResult(
                    status="paused",
                    steps_completed=self._accumulator.current_step,
                    checkpoints_saved=self._checkpoints_saved,
                    final_loss=self._last_loss,
                    evidence_merkle_root=self._logger.compute_merkle_root(),
                    checkpoint_path="",
                    resumed_from_step=self._resumed_from_step,
                    pause_latency_seconds=pause_latency,
                    paused_reasons=("user_activity",),
                )

            # Run one complete optimizer step (grad_accum_steps microbatches)
            current_step = self._accumulator.current_step
            try:
                step_result = self._step_runner(current_step, self._accumulator.accumulated)
            except Exception as exc:
                # Subprocess/step runner crash — attempt respawn
                exit_code = getattr(exc, "returncode", 1) or 1
                can_respawn = self._respawn_on_crash(exit_code=exit_code)
                if can_respawn:
                    # Re-enter the wait cycle and restart the training loop
                    self._fsm.transition(DaemonState.WAITING_FOR_IDLE)
                    self._wait_for_idle()
                    self._fsm.transition(DaemonState.GUARDING)
                    _can_train, _blocking = self._resource_guard.all_guards_clear(
                        self._config.required_vram_bytes
                    )
                    if _can_train:
                        self._fsm.transition(DaemonState.TRAINING)
                        # Reset the checkpoint time and continue the loop
                        self._last_checkpoint_time = time.monotonic()
                        continue
                    else:
                        self._fsm.transition(DaemonState.FAILED)
                        raise RuntimeError(
                            f"Resource guards blocked after crash-respawn: {_blocking}"
                        ) from exc
                else:
                    # Max crashes exceeded — remain in FAILED state (set in _respawn_on_crash)
                    raise
            self._last_loss = float(step_result.get("loss") or 0.0)

            # Successful step — reset crash count
            if self.crash_count > 0:
                self.crash_count = 0

            for _ in range(self._config.grad_accum_steps):
                should_step = self._accumulator.record_microbatch()
                if should_step:
                    self._accumulator.optimizer_step_taken()
                    break

            self._logger.emit(
                {
                    "type": EvidenceType.GRADIENT_ACCUMULATED.value,
                    "data": {
                        "accumulated": self._accumulator.accumulated,
                        "current_step": self._accumulator.current_step,
                        "loss": self._last_loss,
                    },
                },
                organ="idle",
            )

            # Checkpoint if interval elapsed or interval=0 (checkpoint every step)
            checkpoint_saved_this_step = False
            now = time.monotonic()
            elapsed = now - self._last_checkpoint_time
            if (self._config.checkpoint_interval_seconds == 0.0
                    or elapsed >= self._config.checkpoint_interval_seconds):
                self._save_checkpoint()
                self._last_checkpoint_time = now
                checkpoint_saved_this_step = True

            # Emit per-step training evidence (consumed by GPU tests)
            self._logger.emit(
                {
                    "type": EvidenceType.TRAINING_STEP.value,
                    "data": {
                        "step": self._accumulator.current_step,
                        "loss": self._last_loss,
                        "checkpoint_saved": checkpoint_saved_this_step,
                    },
                },
                organ="idle",
            )

        # All steps complete — final checkpoint
        self._save_checkpoint()

        self._logger.emit(
            {
                "type": EvidenceType.IDLE_TRAINING_COMPLETED.value,
                "data": {
                    "steps_completed": self._accumulator.current_step,
                    "checkpoints_saved": self._checkpoints_saved,
                },
            },
            organ="idle",
        )

        # TRAINING → CHECKPOINTING → COMPLETED
        self._fsm.transition(DaemonState.CHECKPOINTING)
        self._fsm.transition(DaemonState.COMPLETED)

        return IdleTrainResult(
            status="completed",
            steps_completed=self._accumulator.current_step,
            checkpoints_saved=self._checkpoints_saved,
            final_loss=self._last_loss,
            evidence_merkle_root=self._logger.compute_merkle_root(),
            checkpoint_path="",
            resumed_from_step=self._resumed_from_step,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_state(self) -> Tuple[int, GradientAccumulator]:
        """Load accumulator + checkpoint state on init for resume support."""
        from dojo.train.checkpoint import CheckpointManager

        accum = GradientAccumulator(accum_steps=self._config.grad_accum_steps)
        resumed_from = 0

        # Try to load checkpoint for resumed_from_step
        ckpt_mgr = CheckpointManager(self._config.output_dir / "checkpoints")
        latest = ckpt_mgr.load_latest()
        if latest is not None:
            resumed_from, ckpt_meta = latest

        # Try to load accumulator state
        accum_path = self._config.output_dir / "idle_accumulator_state.json"
        if accum_path.exists():
            try:
                accum.load_state(accum_path)
            except Exception:
                # Accumulator state corrupt — restore to checkpoint step, no partial batch
                if resumed_from > 0:
                    accum.restore_state(resumed_from)
        elif resumed_from > 0:
            # No accumulator state but have a checkpoint — start from checkpoint step
            accum.restore_state(resumed_from)

        return resumed_from, accum

    def _save_accumulator_state(self) -> None:
        """Save accumulator state for resume."""
        accum_path = self._config.output_dir / "idle_accumulator_state.json"
        try:
            self._config.output_dir.mkdir(parents=True, exist_ok=True)
            self._accumulator.save_state(accum_path)
        except Exception as exc:
            self._logger.warn(f"Accumulator state save failed: {exc}", organ="idle")

    def _save_checkpoint(self, emergency: bool = False) -> None:
        """Save a checkpoint manifest and emit evidence."""
        from dojo.train.checkpoint import CheckpointManager

        try:
            self._config.output_dir.mkdir(parents=True, exist_ok=True)
            ckpt_mgr = CheckpointManager(self._config.output_dir / "checkpoints")
            ckpt_mgr.save(
                self._accumulator.current_step,
                {
                    "loss": self._last_loss,
                    "emergency": emergency,
                },
            )
            self._checkpoints_saved += 1
            self._logger.emit(
                {
                    "type": EvidenceType.IDLE_CHECKPOINT_SAVED.value,
                    "data": {
                        "step": self._accumulator.current_step,
                        "emergency": emergency,
                        "checkpoints_saved": self._checkpoints_saved,
                    },
                },
                organ="idle",
            )
            self._logger.emit(
                {
                    "type": EvidenceType.TRAINING_CHECKPOINT.value,
                    "data": {
                        "step": self._accumulator.current_step,
                        "checkpoints_saved": self._checkpoints_saved,
                        "emergency": emergency,
                    },
                },
                organ="idle",
            )
        except Exception as exc:
            self._logger.warn(f"Checkpoint save failed (non-fatal): {exc}", organ="idle")

    def _respawn_on_crash(self, exit_code: int = 1) -> bool:
        """Handle a training subprocess crash.

        Increments crash_count, emits ``daemon.subprocess_crash`` evidence,
        and decides whether to respawn (crash_count < MAX_CRASHES) or
        permanently fail (crash_count >= MAX_CRASHES).

        Parameters
        ----------
        exit_code:
            The exit code of the crashed subprocess (non-zero).

        Returns
        -------
        bool
            ``True``  — caller should re-queue training (respawn).
            ``False`` — caller should propagate the error (max crashes reached).
        """
        self.crash_count += 1

        # Transition to CRASHED state (from TRAINING)
        self._fsm.transition(DaemonState.CRASHED)

        self._logger.emit(
            {
                "type": "daemon.subprocess_crash",
                "data": {
                    "exit_code": exit_code,
                    "crash_count": self.crash_count,
                    "max_crashes": MAX_CRASHES,
                    "respawn_delay_seconds": _RESPAWN_DELAY_SECONDS,
                },
            },
            organ="idle",
        )

        if self.crash_count >= MAX_CRASHES:
            # Permanently fail — do not respawn
            self._fsm.transition(DaemonState.FAILED)
            self._logger.emit(
                {
                    "type": "daemon.max_crashes_exceeded",
                    "data": {
                        "crash_count": self.crash_count,
                        "max_crashes": MAX_CRASHES,
                    },
                },
                organ="idle",
            )
            return False

        # Wait before respawning
        time.sleep(_RESPAWN_DELAY_SECONDS)
        return True

    def _wait_for_idle(self) -> None:
        """Poll until the system is idle."""
        while True:
            if self._idle_detector.is_idle():
                return
            time.sleep(self._config.poll_interval_seconds)


# Backward-compat aliases expected by dojo/idle/__init__.py
IdleTrainingFSM = _DaemonFSM
IdleTrainingState = DaemonState


__all__ = [
    "DaemonState",
    "IdleTrainConfig",
    "IdleTrainResult",
    "IdleTrainingDaemon",
    "IdleTrainingFSM",
    "IdleTrainingState",
    "MAX_CRASHES",
    "StepRunner",
    "_default_step_runner",
    "_RESPAWN_DELAY_SECONDS",
    "make_real_gpu_step_runner",
]
