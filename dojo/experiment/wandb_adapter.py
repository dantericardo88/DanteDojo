"""W&B (Weights & Biases) adapter for DanteDojo experiment tracking — d45.

Matches the ExperimentTracker interface and falls back to in-memory tracking
when wandb is not installed. All operations emit constitutional evidence
regardless of W&B availability.

Design:
- wandb is NEVER imported at module level — always guarded with try/except.
- Works correctly in CI environments without wandb installed.
- WandBTrainingCallback hooks into SFT/DPO/GRPO training loops.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType


# ---------------------------------------------------------------------------
# Forward-reference for type hints only
# ---------------------------------------------------------------------------

# We use string annotations throughout to avoid importing wandb at top level.


# ---------------------------------------------------------------------------
# WandBAdapter
# ---------------------------------------------------------------------------

class WandBAdapter:
    """W&B adapter matching ExperimentTracker interface.

    Falls back to in-memory tracking when wandb is unavailable or when
    ``use_wandb=False`` is passed.

    Args:
        project: W&B project name.
        entity: W&B entity (username or team). Optional.
        use_wandb: If False, always use in-memory fallback.
    """

    def __init__(
        self,
        project: str = "dojo",
        entity: Optional[str] = None,
        use_wandb: bool = True,
    ) -> None:
        self._project = project
        self._entity = entity
        self._use_wandb = use_wandb
        self._wandb_available = False
        self._run: Any = None                   # wandb.Run or None
        self._in_memory: Dict[str, Any] = {}    # fallback storage
        self._active_run_id: Optional[str] = None

        if use_wandb:
            try:
                import wandb  # noqa: F401
                self._wandb_available = True
            except ImportError:
                self._wandb_available = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _use_live_wandb(self) -> bool:
        return self._use_wandb and self._wandb_available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_run(
        self,
        config: Any,  # RunConfig or plain dict
        run_id: Optional[str] = None,
    ) -> str:
        """Initialize a W&B run.

        Falls back to in-memory when wandb is unavailable.
        Emits EXPERIMENT_RUN_STARTED evidence unconditionally.

        Args:
            config: RunConfig (or any object with a ``__dict__``) describing
                    the run, or a plain dict of hyperparameters.
            run_id: Optional explicit run ID. A UUID is generated if omitted.

        Returns:
            Non-empty run_id string.
        """
        resolved_run_id = run_id or str(uuid.uuid4())

        # Resolve config to a plain dict for W&B
        if hasattr(config, "__dict__"):
            config_dict: Dict[str, Any] = {
                k: v for k, v in config.__dict__.items()
                if not k.startswith("_")
            }
        elif isinstance(config, dict):
            config_dict = dict(config)
        else:
            config_dict = {}

        if self._use_live_wandb():
            try:
                import wandb  # noqa: F811
                self._run = wandb.init(
                    project=self._project,
                    entity=self._entity,
                    config=config_dict,
                    id=resolved_run_id,
                    reinit=True,
                )
            except Exception:
                # W&B not reachable — fall back to in-memory
                self._wandb_available = False
                self._run = None

        # Always maintain in-memory state as fallback / local record
        self._in_memory = {
            "run_id": resolved_run_id,
            "config": config_dict,
            "metrics": [],
            "status": "running",
        }
        self._active_run_id = resolved_run_id

        # Resolve experiment_name / run_name from config if available
        experiment_name = config_dict.get("experiment_name", "")
        run_name = config_dict.get("run_name", "")

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_RUN_STARTED.value,
                "data": {
                    "run_id": resolved_run_id,
                    "project": self._project,
                    "entity": self._entity,
                    "experiment_name": experiment_name,
                    "run_name": run_name,
                    "wandb_available": self._wandb_available,
                },
            },
            organ="experiment",
        )

        return resolved_run_id

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dict of metrics.

        Emits one EXPERIMENT_METRIC evidence bundle covering all keys.

        Args:
            metrics: Metric name → float value.
            step: Optional training step / epoch number.
        """
        if self._use_live_wandb() and self._run is not None:
            try:
                log_kwargs: Dict[str, Any] = dict(metrics)
                if step is not None:
                    self._run.log(log_kwargs, step=step)
                else:
                    self._run.log(log_kwargs)
            except Exception:
                pass

        if self._in_memory:
            self._in_memory.setdefault("metrics", []).append(
                {"metrics": dict(metrics), "step": step}
            )

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_METRIC.value,
                "data": {
                    "run_id": self._active_run_id or "",
                    "metrics": dict(metrics),
                    "step": step,
                    "keys": list(metrics.keys()),
                },
            },
            organ="experiment",
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log / update hyperparameters for the active run.

        Args:
            config: Dict of hyperparameter name → value.
        """
        if self._use_live_wandb() and self._run is not None:
            try:
                import wandb  # noqa: F811
                wandb.config.update(config)
            except Exception:
                pass

        if self._in_memory:
            self._in_memory.setdefault("config", {}).update(config)

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_METRIC.value,
                "data": {
                    "run_id": self._active_run_id or "",
                    "action": "log_config",
                    "config_keys": list(config.keys()),
                },
            },
            organ="experiment",
        )

    def finish(self, status: str = "success") -> None:
        """Finish the active run.

        Emits EXPERIMENT_RUN_COMPLETE evidence unconditionally.

        Args:
            status: Final status string (e.g. "success", "failed").
        """
        if self._use_live_wandb() and self._run is not None:
            try:
                import wandb  # noqa: F811
                wandb.finish()
            except Exception:
                pass
            self._run = None

        if self._in_memory:
            self._in_memory["status"] = status

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_RUN_COMPLETE.value,
                "data": {
                    "run_id": self._active_run_id or "",
                    "status": status,
                    "wandb_available": self._wandb_available,
                },
            },
            organ="experiment",
        )

        self._active_run_id = None

    def as_callback(self) -> "WandBTrainingCallback":
        """Return a WandBTrainingCallback bound to this adapter.

        Returns:
            WandBTrainingCallback suitable for use in SFT/DPO/GRPO training loops.
        """
        return WandBTrainingCallback(self)


# ---------------------------------------------------------------------------
# WandBTrainingCallback
# ---------------------------------------------------------------------------

class WandBTrainingCallback:
    """Callback for training loops that logs to a WandBAdapter.

    Attach to any SFT, DPO, or GRPO training loop that supports step/epoch
    callbacks.

    Args:
        adapter: The WandBAdapter instance to log to.
    """

    def __init__(self, adapter: WandBAdapter) -> None:
        self._adapter = adapter

    def on_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Called each training step.

        Args:
            step: Current global training step.
            metrics: Metrics to log at this step.
        """
        self._adapter.log(metrics, step=step)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch.

        Prefixes all metric keys with ``epoch_`` for easy filtering in W&B.

        Args:
            epoch: Current epoch number (0-indexed or 1-indexed, as the caller
                   prefers).
            metrics: Epoch-level metrics.
        """
        self._adapter.log(
            {f"epoch_{k}": v for k, v in metrics.items()},
            step=epoch,
        )

    def on_train_end(self, final_metrics: Dict[str, float]) -> None:
        """Called when training finishes.

        Logs final metrics and calls adapter.finish().

        Args:
            final_metrics: Final aggregate metrics for the whole training run.
        """
        self._adapter.log(final_metrics)
        self._adapter.finish()


__all__ = [
    "WandBAdapter",
    "WandBTrainingCallback",
]
