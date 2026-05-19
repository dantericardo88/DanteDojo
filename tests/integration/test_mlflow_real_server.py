"""Integration test: ExperimentTracker against a real MLflow tracking server.

Validates d42 (MLflow experiment integration) and d43 (run logging).
Skipped automatically when the `mlflow` package is not installed or when
`mlflow server` cannot start within the timeout.

The test:
1. Starts `mlflow server` on a random high port via subprocess.
2. Creates an ExperimentTracker pointed at that server.
3. Logs params, metrics (step-level), and a text artifact.
4. Queries the MLflow REST API directly to assert the data landed.
5. Verifies constitutional evidence was emitted unconditionally.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Generator

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.experiment.tracker import ExperimentTracker, RunConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Poll GET *url* until it returns 200 or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixture: real MLflow server process
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mlflow_server() -> Generator[str, None, None]:
    """Start an mlflow server subprocess; yield its tracking URI; teardown."""
    if not _mlflow_available():
        pytest.skip("mlflow not installed")

    port = _free_port()
    tracking_uri = f"http://127.0.0.1:{port}"

    with tempfile.TemporaryDirectory(prefix="dojo_mlflow_", ignore_cleanup_errors=True) as tmpdir:
        backend_store = Path(tmpdir) / "mlruns"
        artifact_root = Path(tmpdir) / "artifacts"
        backend_store.mkdir()
        artifact_root.mkdir()

        # Use file:// URI so MLflow 3.x recognises the artifact root on Windows
        artifact_root_uri = artifact_root.as_uri()

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "mlflow", "server",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--backend-store-uri", f"sqlite:///{backend_store}/mlflow.db",
                "--default-artifact-root", artifact_root_uri,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        healthy = _wait_healthy(f"{tracking_uri}/health", timeout=30.0)
        if not healthy:
            proc.terminate()
            proc.wait(timeout=5)
            pytest.skip("mlflow server did not start within 30 s")

        try:
            yield tracking_uri
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# Test: d42 — MLflow experiment integration (real server)
# ---------------------------------------------------------------------------

class TestMLflowRealServer:
    """d42: ExperimentTracker writes to a live MLflow tracking server."""

    def test_start_run_creates_mlflow_run(self, mlflow_server: str) -> None:
        """start_run() must return a non-empty run_id that exists in MLflow."""
        tracker = ExperimentTracker(tracking_uri=mlflow_server)
        config = RunConfig(
            experiment_name="dojo_d42_test",
            run_name="run_start_test",
            seed=42,
        )
        run_id = tracker.start_run(config)
        assert run_id, "run_id must be non-empty"

        # Verify the run actually exists in the MLflow server
        import mlflow
        mlflow.set_tracking_uri(mlflow_server)
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        assert run.info.run_id == run_id

        tracker.end_run()

    def test_log_params_persisted_to_mlflow(self, mlflow_server: str) -> None:
        """log_params() must persist parameter key/value pairs to MLflow."""
        tracker = ExperimentTracker(tracking_uri=mlflow_server)
        config = RunConfig(
            experiment_name="dojo_d42_test",
            run_name="run_params_test",
            seed=7,
        )
        tracker.start_run(config)
        tracker.log_params({"learning_rate": 2e-4, "batch_size": 8, "model": "Qwen-7B"})
        tracker.end_run()

        import mlflow
        mlflow.set_tracking_uri(mlflow_server)
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=client.get_experiment_by_name("dojo_d42_test").experiment_id,
            filter_string="tags.mlflow.runName = 'run_params_test'",
        )
        assert runs, "Run must exist in MLflow"
        params = runs[0].data.params
        assert params.get("learning_rate") == "0.0002"
        assert params.get("batch_size") == "8"

    def test_evidence_emitted_regardless_of_mlflow_availability(
        self, mlflow_server: str
    ) -> None:
        """Evidence must be emitted even when using real MLflow (d02 integration)."""
        tracker = ExperimentTracker(tracking_uri=mlflow_server)
        config = RunConfig(
            experiment_name="dojo_d42_evidence",
            run_name="run_evidence_test",
        )
        tracker.start_run(config)
        tracker.end_run()

        chain = EvidenceLogger.get_instance().list_events()
        types_seen = {e.get("evidence", {}).get("type") for e in chain}
        assert EvidenceType.EXPERIMENT_RUN_STARTED.value in types_seen


# ---------------------------------------------------------------------------
# Test: d43 — Run logging (params, metrics, artifacts) — real server
# ---------------------------------------------------------------------------

class TestRunLoggingRealServer:
    """d43: step-level metrics and artifacts are stored in a live MLflow server."""

    def test_log_metrics_step_level(self, mlflow_server: str) -> None:
        """log_metric() with step values must produce a multi-step metric history."""
        tracker = ExperimentTracker(tracking_uri=mlflow_server)
        config = RunConfig(
            experiment_name="dojo_d43_metrics",
            run_name="run_metrics_step",
        )
        run_id = tracker.start_run(config)

        for step in range(5):
            tracker.log_metric("train_loss", 2.0 - step * 0.3, step=step)

        tracker.end_run()

        import mlflow
        mlflow.set_tracking_uri(mlflow_server)
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(run_id, "train_loss")
        assert len(history) == 5, "All 5 steps must be persisted"
        # Loss should decrease across steps
        values = [h.value for h in sorted(history, key=lambda h: h.step)]
        assert values[0] > values[-1], "Loss must decrease"

    def test_log_artifact_text(self, mlflow_server: str) -> None:
        """log_artifact() must store a text artifact in MLflow."""
        tracker = ExperimentTracker(tracking_uri=mlflow_server)
        config = RunConfig(
            experiment_name="dojo_d43_artifacts",
            run_name="run_artifact_test",
        )
        run_id = tracker.start_run(config)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("eval_receipt: test_merkle_root_abc123\n")
            artifact_path = f.name

        tracker.log_artifact(artifact_path)
        tracker.end_run()

        import mlflow
        mlflow.set_tracking_uri(mlflow_server)
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        assert artifacts, "At least one artifact must be listed"
        names = [a.path for a in artifacts]
        assert any(Path(artifact_path).name in n for n in names)

        os.unlink(artifact_path)

    def test_params_metrics_artifacts_all_logged(self, mlflow_server: str) -> None:
        """Full run: params + step metrics + artifact all land in one MLflow run."""
        tracker = ExperimentTracker(tracking_uri=mlflow_server)
        config = RunConfig(
            experiment_name="dojo_d43_full",
            run_name="run_full",
            config_hash="sha256:deadbeef",
            hardware_tier="TIER_4060",
            seed=99,
        )
        run_id = tracker.start_run(config)

        tracker.log_params({"epochs": 3, "lr": 1e-3})
        for s in range(3):
            tracker.log_metric("eval_f1", 0.70 + s * 0.05, step=s)
        tracker.log_metrics_batch({"final_f1": 0.80, "final_loss": 0.42})

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"passport_id": "test-passport", "merkle": "abc"}, f)
            art_path = f.name

        tracker.log_artifact(art_path)
        tracker.end_run()
        os.unlink(art_path)

        import mlflow
        mlflow.set_tracking_uri(mlflow_server)
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        assert run.data.params.get("epochs") == "3"
        f1_history = client.get_metric_history(run_id, "eval_f1")
        assert len(f1_history) == 3
        artifacts = client.list_artifacts(run_id)
        assert artifacts

        # Constitutional evidence must still be emitted
        chain = EvidenceLogger.get_instance().list_events()
        assert chain, "Evidence chain must be non-empty"
