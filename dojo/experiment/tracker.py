"""MLflow 3.x GenAI tracing wrapper for DanteDojo experiment tracking.

Falls back to an in-memory tracker when MLflow is unavailable (e.g. in CI).
All metric logging emits constitutional evidence regardless of MLflow status.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Configuration for a single experiment run."""

    experiment_name: str
    run_name: str
    tags: Dict[str, str] = field(default_factory=dict)
    config_hash: str = ""
    pack_id: str = ""
    hardware_tier: str = ""
    git_sha: str = ""
    seed: int = 42


# ---------------------------------------------------------------------------
# In-memory fallback backend
# ---------------------------------------------------------------------------

class _InMemoryRun:
    """Minimal in-memory experiment tracking fallback."""

    def __init__(self, run_id: str, config: RunConfig) -> None:
        self.run_id = run_id
        self.config = config
        self.params: Dict[str, Any] = {}
        self.metrics: List[Dict[str, Any]] = []
        self.artifacts: List[str] = []
        self.status: str = "RUNNING"

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        self.metrics.append({"key": key, "value": value, "step": step})

    def end(self, status: str = "FINISHED") -> None:
        self.status = status


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """MLflow 3.x GenAI tracing wrapper with in-memory fallback.

    Args:
        tracking_uri: MLflow tracking server URI.
        use_mlflow: If False, always use the in-memory fallback.
    """

    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5000",
        use_mlflow: bool = True,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._use_mlflow = use_mlflow
        self._mlflow_available = False
        self._active_run_id: Optional[str] = None
        self._in_memory_run: Optional[_InMemoryRun] = None
        self._mlflow_run: Optional[Any] = None  # mlflow.ActiveRun or None

        if use_mlflow:
            try:
                import mlflow  # noqa: F401
                self._mlflow_available = True
            except ImportError:
                self._mlflow_available = False

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _get_mlflow(self) -> Any:
        import mlflow
        return mlflow

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, config: RunConfig) -> str:
        """Start an MLflow run. Returns run_id. Falls back to in-memory if MLflow unavailable.

        Args:
            config: RunConfig describing the experiment and run metadata.

        Returns:
            A non-empty run_id string.
        """
        run_id = str(uuid.uuid4())

        if self._mlflow_available and self._use_mlflow:
            try:
                mlflow = self._get_mlflow()
                mlflow.set_tracking_uri(self._tracking_uri)
                mlflow.set_experiment(config.experiment_name)
                active = mlflow.start_run(run_name=config.run_name, tags=config.tags)
                run_id = active.info.run_id
                self._mlflow_run = active

                # Log standard tags/params
                if config.config_hash:
                    mlflow.log_param("config_hash", config.config_hash)
                if config.pack_id:
                    mlflow.log_param("pack_id", config.pack_id)
                if config.hardware_tier:
                    mlflow.log_param("hardware_tier", config.hardware_tier)
                if config.git_sha:
                    mlflow.log_param("git_sha", config.git_sha)
                mlflow.log_param("seed", config.seed)
            except Exception:
                # MLflow not reachable — fall through to in-memory
                self._mlflow_available = False
                self._mlflow_run = None

        if not self._mlflow_available or not self._use_mlflow:
            self._in_memory_run = _InMemoryRun(run_id, config)

        self._active_run_id = run_id

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_RUN_STARTED.value,
                "data": {
                    "run_id": run_id,
                    "experiment_name": config.experiment_name,
                    "run_name": config.run_name,
                    "seed": config.seed,
                    "config_hash": config.config_hash,
                },
            },
            organ="experiment",
        )

        return run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a batch of parameters to the active run. Emits evidence for each call.

        Args:
            params: Dict of parameter name → value.
        """
        if self._mlflow_available and self._mlflow_run is not None:
            try:
                mlflow = self._get_mlflow()
                mlflow.log_params({k: str(v) for k, v in params.items()})
            except Exception:
                pass

        if self._in_memory_run is not None:
            for k, v in params.items():
                self._in_memory_run.log_param(k, v)

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_METRIC.value,
                "data": {
                    "run_id": self._active_run_id or "",
                    "type": "param",
                    "params": {k: str(v) for k, v in params.items()},
                    "param_count": len(params),
                },
            },
            organ="experiment",
        )

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value. Emits EXPERIMENT_METRIC evidence and logs to MLflow.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional training step / epoch number.
        """
        if self._mlflow_available and self._mlflow_run is not None:
            try:
                mlflow = self._get_mlflow()
                if step is not None:
                    mlflow.log_metric(key, value, step=step)
                else:
                    mlflow.log_metric(key, value)
            except Exception:
                pass

        if self._in_memory_run is not None:
            self._in_memory_run.log_metric(key, value, step)

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_METRIC.value,
                "data": {
                    "run_id": self._active_run_id or "",
                    "key": key,
                    "value": value,
                    "step": step,
                },
            },
            organ="experiment",
        )

    def log_metrics_batch(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once. Each metric emits EXPERIMENT_METRIC evidence.

        Args:
            metrics: Dict of metric name → value.
            step: Optional training step / epoch number applied to all metrics.
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_metric_series(self, key: str, values: List[float], start_step: int = 0) -> None:
        """Log a list of values as sequential steps. Emits one evidence event per value.

        Args:
            key: Metric name.
            values: Ordered list of metric values.
            start_step: Step index for the first value; increments by 1 per value.
        """
        for i, value in enumerate(values):
            self.log_metric(key, value, step=start_step + i)

    def get_run_summary(self) -> Dict[str, Any]:
        """Return a summary dict for the active run.

        Returns a dict with: run_id, status, param_count, metric_count, artifact_count.
        Falls back to zeros when no active run exists.
        """
        if self._in_memory_run is not None:
            r = self._in_memory_run
            return {
                "run_id": r.run_id,
                "status": r.status,
                "param_count": len(r.params),
                "metric_count": len(r.metrics),
                "artifact_count": len(r.artifacts),
            }

        if self._mlflow_available and self._mlflow_run is not None:
            try:
                mlflow = self._get_mlflow()
                info = self._mlflow_run.info
                data = mlflow.get_run(info.run_id).data
                return {
                    "run_id": info.run_id,
                    "status": info.status,
                    "param_count": len(data.params),
                    "metric_count": len(data.metrics),
                    "artifact_count": 0,
                }
            except Exception:
                pass

        return {
            "run_id": self._active_run_id or "",
            "status": "UNKNOWN",
            "param_count": 0,
            "metric_count": 0,
            "artifact_count": 0,
        }

    def log_artifact(self, local_path: Path, artifact_path: str = "") -> None:
        """Log a local file as an artifact.

        Args:
            local_path: Path to the local file to upload.
            artifact_path: Optional subdirectory path within the artifact store.
        """
        if self._mlflow_available and self._mlflow_run is not None:
            try:
                mlflow = self._get_mlflow()
                mlflow.log_artifact(str(local_path), artifact_path or None)
            except Exception:
                pass

        if self._in_memory_run is not None:
            self._in_memory_run.artifacts.append(str(local_path))

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active run and emit EXPERIMENT_RUN_COMPLETE evidence.

        Args:
            status: Final run status string (e.g. "FINISHED", "FAILED").
        """
        if self._mlflow_available and self._mlflow_run is not None:
            try:
                mlflow = self._get_mlflow()
                mlflow.end_run(status=status)
            except Exception:
                pass
            self._mlflow_run = None

        if self._in_memory_run is not None:
            self._in_memory_run.end(status)

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_RUN_COMPLETE.value,
                "data": {
                    "run_id": self._active_run_id or "",
                    "status": status,
                },
            },
            organ="experiment",
        )

        self._active_run_id = None
        self._in_memory_run = None

    def list_runs(self, experiment_name: str = "") -> List[Dict[str, Any]]:
        """List completed runs. Returns MLflow runs when available, empty list otherwise."""
        if self._mlflow_available:
            try:
                mlflow = self._get_mlflow()
                filter_str = f"attributes.experiment_name = '{experiment_name}'" if experiment_name else ""
                runs = mlflow.search_runs(filter_string=filter_str, output_format="list")
                return [
                    {
                        "run_id": r.info.run_id,
                        "name": r.info.run_name or "",
                        "status": r.info.status,
                        "start_time": str(r.info.start_time),
                    }
                    for r in runs
                ]
            except Exception:
                return []
        return []

    @contextmanager
    def run(self, config: RunConfig) -> Iterator[str]:
        """Context manager: start_run → yield run_id → end_run.

        Ensures end_run is always called, even on exceptions.

        Args:
            config: RunConfig describing the experiment.

        Yields:
            The run_id string.
        """
        run_id = self.start_run(config)
        try:
            yield run_id
        except Exception:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")


__all__ = [
    "RunConfig",
    "ExperimentTracker",
    "_InMemoryRun",
]
