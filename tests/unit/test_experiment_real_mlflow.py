"""Real MLflow integration tests for d42 (run lifecycle) and d43 (params/metrics tracking).

Uses a local file-based tracking store via a temp directory — no server required.
All tests are skipped automatically if mlflow is not installed.
"""

from __future__ import annotations

import pytest

mlflow = pytest.importorskip("mlflow")

from dojo.experiment.tracker import ExperimentTracker, RunConfig  # noqa: E402
from dojo.experiment.leaderboard import RunLeaderboard  # noqa: E402
from dojo.core.evidence import EvidenceLogger  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def mlflow_tracking_dir(tmp_path):
    """Point MLflow at a fresh local file store for each test."""
    uri = f"file:///{tmp_path.as_posix()}"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri("memory://")


def _make_tracker(tracking_uri: str) -> ExperimentTracker:
    """Return a real-MLflow tracker pointed at the given URI."""
    return ExperimentTracker(use_mlflow=True, tracking_uri=tracking_uri)


def _make_config(
    experiment_name: str = "dojo-real-test",
    run_name: str = "run-real",
    seed: int = 42,
) -> RunConfig:
    return RunConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        tags={"env": "ci"},
        config_hash="cafebabe",
        pack_id="pack-real",
        hardware_tier="tier_cpu",
        git_sha="def456",
        seed=seed,
    )


# ---------------------------------------------------------------------------
# d42 — Real MLflow run lifecycle
# ---------------------------------------------------------------------------

class TestRealMLflowRunLifecycle:

    def test_start_run_creates_mlflow_run(self, mlflow_tracking_dir):
        """start_run() must create a run that mlflow.search_runs can find."""
        tracker = _make_tracker(mlflow_tracking_dir)
        config = _make_config()
        run_id = tracker.start_run(config)
        tracker.end_run()

        runs = mlflow.search_runs(output_format="list")
        found_ids = [r.info.run_id for r in runs]
        assert run_id in found_ids

    def test_log_params_persisted(self, mlflow_tracking_dir):
        """Params logged via log_params must be visible in mlflow.get_run()."""
        tracker = _make_tracker(mlflow_tracking_dir)
        config = _make_config()
        run_id = tracker.start_run(config)
        tracker.log_params({"lr": 1e-4, "batch_size": 32})
        tracker.end_run()

        data = mlflow.get_run(run_id).data
        assert data.params.get("lr") == str(1e-4)
        assert data.params.get("batch_size") == str(32)

    def test_log_metrics_persisted(self, mlflow_tracking_dir):
        """Metrics logged via log_metric must be visible in mlflow.get_run()."""
        tracker = _make_tracker(mlflow_tracking_dir)
        config = _make_config()
        run_id = tracker.start_run(config)
        tracker.log_metric("accuracy", 0.95)
        tracker.log_metric("loss", 0.12)
        tracker.end_run()

        data = mlflow.get_run(run_id).data
        assert abs(data.metrics.get("accuracy", -1) - 0.95) < 1e-6
        assert abs(data.metrics.get("loss", -1) - 0.12) < 1e-6

    def test_finalize_sets_run_status(self, mlflow_tracking_dir):
        """After end_run(), the MLflow run status must be FINISHED."""
        tracker = _make_tracker(mlflow_tracking_dir)
        config = _make_config()
        run_id = tracker.start_run(config)
        tracker.end_run()

        run = mlflow.get_run(run_id)
        assert run.info.status == "FINISHED"

    def test_list_runs_returns_real_run(self, mlflow_tracking_dir):
        """list_runs() must return a dict containing the run_id after start+end."""
        tracker = _make_tracker(mlflow_tracking_dir)
        config = _make_config()
        run_id = tracker.start_run(config)
        tracker.end_run()

        runs = tracker.list_runs()
        found_ids = [r["run_id"] for r in runs]
        assert run_id in found_ids


# ---------------------------------------------------------------------------
# d43 — Real MLflow leaderboard
# ---------------------------------------------------------------------------

class TestRealMLflowLeaderboard:

    def _log_run(
        self,
        tracker: ExperimentTracker,
        experiment_name: str,
        run_name: str,
        loss: float,
    ) -> str:
        config = _make_config(experiment_name=experiment_name, run_name=run_name)
        run_id = tracker.start_run(config)
        tracker.log_metric("loss", loss)
        tracker.end_run()
        return run_id

    def test_from_tracker_builds_leaderboard(self, mlflow_tracking_dir):
        """from_tracker() must return a leaderboard with 3 entries after logging 3 runs."""
        tracker = _make_tracker(mlflow_tracking_dir)
        exp = "lb-exp"
        self._log_run(tracker, exp, "run-a", loss=0.9)
        self._log_run(tracker, exp, "run-b", loss=0.5)
        self._log_run(tracker, exp, "run-c", loss=0.3)

        lb = RunLeaderboard.from_tracker(tracker, primary_metric="loss", higher_is_better=False)
        assert len(lb._entries) == 3

    def test_leaderboard_rank_order(self, mlflow_tracking_dir):
        """Lowest-loss run must receive rank 1 when higher_is_better=False."""
        tracker = _make_tracker(mlflow_tracking_dir)
        exp = "lb-rank-exp"
        self._log_run(tracker, exp, "run-high", loss=0.9)
        self._log_run(tracker, exp, "run-mid", loss=0.5)
        self._log_run(tracker, exp, "run-low", loss=0.1)

        lb = RunLeaderboard.from_tracker(tracker, primary_metric="loss", higher_is_better=False)
        ranked = lb.rank()
        assert ranked[0].metrics["loss"] == pytest.approx(0.1)
        assert ranked[0].rank == 1

    def test_leaderboard_best_returns_lowest_loss(self, mlflow_tracking_dir):
        """lb.best() must return the run with the minimum loss value."""
        tracker = _make_tracker(mlflow_tracking_dir)
        exp = "lb-best-exp"
        self._log_run(tracker, exp, "run-worst", loss=1.5)
        self._log_run(tracker, exp, "run-ok", loss=0.7)
        run_id_best = self._log_run(tracker, exp, "run-best", loss=0.05)

        lb = RunLeaderboard.from_tracker(tracker, primary_metric="loss", higher_is_better=False)
        best = lb.best()
        assert best is not None
        assert best.run_id == run_id_best
        assert best.metrics["loss"] == pytest.approx(0.05)
