"""Comprehensive tests for d42 (MLflow run lifecycle) and d43 (params/metrics tracking).

All tests use use_mlflow=False so they run without an MLflow server.
They validate evidence chain content, not just that functions run.
"""

from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.experiment.tracker import ExperimentTracker, RunConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset the evidence logger before and after each test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def tracker() -> ExperimentTracker:
    """ExperimentTracker with MLflow disabled (in-memory only)."""
    return ExperimentTracker(use_mlflow=False)


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        experiment_name="dojo-unit-test",
        run_name="run-alpha",
        tags={"env": "ci"},
        config_hash="deadbeef42",
        pack_id="pack-001",
        hardware_tier="tier_cpu",
        git_sha="abc123",
        seed=7,
    )


def _events_of_type(ev_type: EvidenceType) -> list[dict]:
    """Return all evidence events matching the given type."""
    return [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == ev_type.value
    ]


# ---------------------------------------------------------------------------
# d42 — MLflow run lifecycle (12 tests)
# ---------------------------------------------------------------------------

class TestRunLifecycle:

    def test_start_run_emits_experiment_run_started_evidence(self, tracker, config):
        tracker.start_run(config)
        tracker.end_run()
        started = _events_of_type(EvidenceType.EXPERIMENT_RUN_STARTED)
        assert len(started) >= 1

    def test_start_run_evidence_has_experiment_name(self, tracker, config):
        tracker.start_run(config)
        tracker.end_run()
        started = _events_of_type(EvidenceType.EXPERIMENT_RUN_STARTED)
        assert started[0]["data"]["experiment_name"] == "dojo-unit-test"

    def test_start_run_evidence_has_run_name(self, tracker, config):
        tracker.start_run(config)
        tracker.end_run()
        started = _events_of_type(EvidenceType.EXPERIMENT_RUN_STARTED)
        assert started[0]["data"]["run_name"] == "run-alpha"

    def test_start_run_evidence_has_seed(self, tracker, config):
        tracker.start_run(config)
        tracker.end_run()
        started = _events_of_type(EvidenceType.EXPERIMENT_RUN_STARTED)
        assert started[0]["data"]["seed"] == 7

    def test_start_run_evidence_has_config_hash(self, tracker, config):
        tracker.start_run(config)
        tracker.end_run()
        started = _events_of_type(EvidenceType.EXPERIMENT_RUN_STARTED)
        assert started[0]["data"]["config_hash"] == "deadbeef42"

    def test_end_run_status_finished(self, tracker, config):
        tracker.start_run(config)
        tracker.end_run("FINISHED")
        complete = _events_of_type(EvidenceType.EXPERIMENT_RUN_COMPLETE)
        assert len(complete) >= 1
        assert complete[0]["data"]["status"] == "FINISHED"

    def test_end_run_status_failed(self, tracker, config):
        tracker.start_run(config)
        tracker.end_run("FAILED")
        complete = _events_of_type(EvidenceType.EXPERIMENT_RUN_COMPLETE)
        assert complete[0]["data"]["status"] == "FAILED"

    def test_context_manager_run_id_consistent(self, tracker, config):
        with tracker.run(config) as run_id:
            assert run_id == tracker._active_run_id

    def test_context_manager_exception_sets_failed(self, tracker, config):
        with pytest.raises(RuntimeError):
            with tracker.run(config):
                raise RuntimeError("boom")
        complete = _events_of_type(EvidenceType.EXPERIMENT_RUN_COMPLETE)
        assert complete[0]["data"]["status"] == "FAILED"

    def test_multiple_sequential_runs(self, tracker, config):
        run_id_1 = tracker.start_run(config)
        tracker.end_run()
        run_id_2 = tracker.start_run(config)
        tracker.end_run()
        assert run_id_1 != run_id_2

    def test_run_id_unique_per_run(self, tracker, config):
        run_ids = set()
        for _ in range(5):
            rid = tracker.start_run(config)
            tracker.end_run()
            run_ids.add(rid)
        assert len(run_ids) == 5

    def test_log_artifact_path_stored(self, tracker, config):
        from pathlib import Path
        tracker.start_run(config)
        tracker.log_artifact(Path("/tmp/model.pt"))
        tracker.end_run()
        # _in_memory_run is cleared after end_run, so check via the run before end
        # We check through a fresh run retained before end_run
        tracker.start_run(config)
        tracker.log_artifact(Path("/tmp/checkpoint.bin"))
        in_mem = tracker._in_memory_run
        assert any("checkpoint.bin" in a for a in in_mem.artifacts)
        tracker.end_run()


# ---------------------------------------------------------------------------
# d43 — Params and metrics (13 tests)
# ---------------------------------------------------------------------------

class TestParamsAndMetrics:

    def test_log_params_all_params_stored(self, tracker, config):
        tracker.start_run(config)
        tracker.log_params({"lr": 1e-4, "batch_size": 32, "epochs": 10})
        in_mem = tracker._in_memory_run
        assert "lr" in in_mem.params
        assert "batch_size" in in_mem.params
        assert "epochs" in in_mem.params
        tracker.end_run()

    def test_log_params_emits_evidence(self, tracker, config):
        tracker.start_run(config)
        tracker.log_params({"lr": 1e-4, "dropout": 0.1})
        tracker.end_run()
        # log_params emits EXPERIMENT_METRIC with type="param"
        metric_events = _events_of_type(EvidenceType.EXPERIMENT_METRIC)
        param_events = [e for e in metric_events if e["data"].get("type") == "param"]
        assert len(param_events) >= 1

    def test_log_metric_key_stored(self, tracker, config):
        tracker.start_run(config)
        tracker.log_metric("accuracy", 0.95)
        in_mem = tracker._in_memory_run
        keys = [m["key"] for m in in_mem.metrics]
        assert "accuracy" in keys
        tracker.end_run()

    def test_log_metric_value_stored(self, tracker, config):
        tracker.start_run(config)
        tracker.log_metric("f1", 0.87)
        in_mem = tracker._in_memory_run
        stored = next(m for m in in_mem.metrics if m["key"] == "f1")
        assert abs(stored["value"] - 0.87) < 1e-9
        tracker.end_run()

    def test_log_metric_step_stored(self, tracker, config):
        tracker.start_run(config)
        tracker.log_metric("loss", 0.42, step=50)
        in_mem = tracker._in_memory_run
        stored = next(m for m in in_mem.metrics if m["key"] == "loss")
        assert stored["step"] == 50
        tracker.end_run()

    def test_log_metric_step_none_when_not_provided(self, tracker, config):
        tracker.start_run(config)
        tracker.log_metric("bleu", 0.31)
        in_mem = tracker._in_memory_run
        stored = next(m for m in in_mem.metrics if m["key"] == "bleu")
        assert stored["step"] is None
        tracker.end_run()

    def test_log_metrics_batch_emits_one_evidence_per_metric(self, tracker, config):
        tracker.start_run(config)
        EvidenceLogger.clear()  # zero out prior events
        tracker.log_metrics_batch({"p": 0.9, "r": 0.8, "f1": 0.85})
        metric_events = _events_of_type(EvidenceType.EXPERIMENT_METRIC)
        # Each of the 3 metrics emits its own evidence event
        assert len(metric_events) == 3
        tracker.end_run()

    def test_log_metrics_batch_all_keys_in_evidence(self, tracker, config):
        tracker.start_run(config)
        EvidenceLogger.clear()
        tracker.log_metrics_batch({"rouge": 0.55, "meteor": 0.44})
        metric_events = _events_of_type(EvidenceType.EXPERIMENT_METRIC)
        keys_in_evidence = {e["data"]["key"] for e in metric_events}
        assert {"rouge", "meteor"} == keys_in_evidence
        tracker.end_run()

    def test_log_metric_series_all_steps_emitted(self, tracker, config):
        tracker.start_run(config)
        EvidenceLogger.clear()
        tracker.log_metric_series("train_loss", [0.9, 0.8, 0.7, 0.6, 0.5])
        metric_events = _events_of_type(EvidenceType.EXPERIMENT_METRIC)
        assert len(metric_events) == 5
        tracker.end_run()

    def test_log_metric_series_steps_sequential(self, tracker, config):
        tracker.start_run(config)
        EvidenceLogger.clear()
        tracker.log_metric_series("val_loss", [0.5, 0.4, 0.3], start_step=10)
        metric_events = _events_of_type(EvidenceType.EXPERIMENT_METRIC)
        steps = [e["data"]["step"] for e in metric_events]
        assert steps == [10, 11, 12]
        tracker.end_run()

    def test_get_run_summary_has_run_id(self, tracker, config):
        run_id = tracker.start_run(config)
        summary = tracker.get_run_summary()
        assert summary["run_id"] == run_id
        tracker.end_run()

    def test_get_run_summary_metric_count(self, tracker, config):
        tracker.start_run(config)
        tracker.log_metric("a", 1.0)
        tracker.log_metric("b", 2.0)
        tracker.log_metric("c", 3.0)
        summary = tracker.get_run_summary()
        assert summary["metric_count"] == 3
        tracker.end_run()

    def test_get_run_summary_param_count(self, tracker, config):
        tracker.start_run(config)
        tracker.log_params({"alpha": 1, "beta": 2})
        summary = tracker.get_run_summary()
        assert summary["param_count"] == 2
        tracker.end_run()
