"""Unit tests for dojo/experiment/ — Phase 4 experiment tracking."""

from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.experiment.metrics import aggregate_metrics, compare_runs
from dojo.experiment.tracker import ExperimentTracker, RunConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset the evidence logger before each test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def tracker() -> ExperimentTracker:
    """ExperimentTracker with MLflow disabled (in-memory only)."""
    return ExperimentTracker(use_mlflow=False)


@pytest.fixture
def basic_config() -> RunConfig:
    return RunConfig(
        experiment_name="test-experiment",
        run_name="test-run-001",
        tags={"env": "ci"},
        config_hash="abc123",
        pack_id="pack-v1",
        hardware_tier="tier_cpu",
        git_sha="deadbeef",
        seed=42,
    )


# ---------------------------------------------------------------------------
# ExperimentTracker tests
# ---------------------------------------------------------------------------

class TestExperimentTracker:
    def test_start_run_returns_non_empty_run_id(self, tracker, basic_config):
        run_id = tracker.start_run(basic_config)
        assert isinstance(run_id, str)
        assert len(run_id) > 0
        tracker.end_run()

    def test_start_run_run_id_is_uuid_like(self, tracker, basic_config):
        run_id = tracker.start_run(basic_config)
        # UUIDs are 36 chars (with dashes)
        assert len(run_id) >= 32
        tracker.end_run()

    def test_log_metric_emits_experiment_metric_evidence(self, tracker, basic_config):
        tracker.start_run(basic_config)
        tracker.log_metric("f1_score", 0.85)
        tracker.end_run()

        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_METRIC.value in types

    def test_log_metric_with_step(self, tracker, basic_config):
        tracker.start_run(basic_config)
        tracker.log_metric("loss", 0.42, step=100)
        tracker.end_run()

        events = EvidenceLogger.get_events()
        metric_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
        ]
        assert len(metric_events) >= 1
        # Check step is captured
        data = metric_events[0].get("data", {})
        assert data.get("step") == 100

    def test_end_run_emits_experiment_run_complete_evidence(self, tracker, basic_config):
        tracker.start_run(basic_config)
        tracker.end_run()

        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_RUN_COMPLETE.value in types

    def test_context_manager_run_opens_and_closes_cleanly(self, tracker, basic_config):
        with tracker.run(basic_config) as run_id:
            assert isinstance(run_id, str)
            assert len(run_id) > 0
            tracker.log_metric("accuracy", 0.92)

        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_RUN_STARTED.value in types
        assert EvidenceType.EXPERIMENT_RUN_COMPLETE.value in types

    def test_context_manager_ends_run_on_exception(self, tracker, basic_config):
        with pytest.raises(ValueError, match="test error"):
            with tracker.run(basic_config) as run_id:
                assert run_id
                raise ValueError("test error")

        events = EvidenceLogger.get_events()
        complete_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_RUN_COMPLETE.value
        ]
        assert len(complete_events) >= 1
        # Should have FAILED status
        data = complete_events[0].get("data", {})
        assert data.get("status") == "FAILED"

    def test_log_params_stores_without_error(self, tracker, basic_config):
        tracker.start_run(basic_config)
        tracker.log_params({"lr": 1e-4, "batch_size": 8, "epochs": 3})
        tracker.end_run()

    def test_multiple_metrics_all_emitted(self, tracker, basic_config):
        tracker.start_run(basic_config)
        tracker.log_metric("f1", 0.85)
        tracker.log_metric("precision", 0.88)
        tracker.log_metric("recall", 0.82)
        tracker.end_run()

        events = EvidenceLogger.get_events()
        metric_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
        ]
        metric_keys = {e.get("data", {}).get("key") for e in metric_events}
        assert {"f1", "precision", "recall"}.issubset(metric_keys)


# ---------------------------------------------------------------------------
# aggregate_metrics tests
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def test_empty_runs_returns_empty_dict(self):
        assert aggregate_metrics([]) == {}

    def test_single_run(self):
        result = aggregate_metrics([{"f1": 0.8, "precision": 0.9}])
        assert abs(result["f1"]["mean"] - 0.8) < 1e-9
        assert result["f1"]["std"] == 0.0
        assert result["f1"]["min"] == 0.8
        assert result["f1"]["max"] == 0.8

    def test_correct_mean(self):
        runs = [{"score": 0.7}, {"score": 0.8}, {"score": 0.9}]
        result = aggregate_metrics(runs)
        assert abs(result["score"]["mean"] - 0.8) < 1e-9

    def test_correct_std(self):
        # population std of [0.7, 0.8, 0.9]: mean=0.8, variance=((−0.1)²+(0)²+(0.1)²)/3 = 0.02/3
        import math
        runs = [{"score": 0.7}, {"score": 0.8}, {"score": 0.9}]
        result = aggregate_metrics(runs)
        expected_std = math.sqrt(0.02 / 3)
        assert abs(result["score"]["std"] - expected_std) < 1e-9

    def test_correct_min_max(self):
        runs = [{"score": 0.5}, {"score": 0.7}, {"score": 0.6}]
        result = aggregate_metrics(runs)
        assert result["score"]["min"] == 0.5
        assert result["score"]["max"] == 0.7

    def test_multiple_metrics(self):
        runs = [
            {"f1": 0.8, "bleu": 0.3},
            {"f1": 0.9, "bleu": 0.4},
        ]
        result = aggregate_metrics(runs)
        assert "f1" in result
        assert "bleu" in result
        assert abs(result["f1"]["mean"] - 0.85) < 1e-9
        assert abs(result["bleu"]["mean"] - 0.35) < 1e-9

    def test_missing_metric_in_some_runs(self):
        runs = [{"f1": 0.8}, {"f1": 0.9, "extra": 0.5}]
        result = aggregate_metrics(runs)
        assert "f1" in result
        assert "extra" in result
        # extra only has one value
        assert result["extra"]["mean"] == 0.5


# ---------------------------------------------------------------------------
# compare_runs tests
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def test_correct_delta(self):
        result = compare_runs({"f1": 0.8}, {"f1": 0.85})
        assert abs(result["f1"]["delta"] - 0.05) < 1e-9

    def test_correct_delta_pct(self):
        result = compare_runs({"f1": 0.8}, {"f1": 0.88})
        expected_pct = (0.08 / 0.8) * 100.0
        assert abs(result["f1"]["delta_pct"] - expected_pct) < 1e-9

    def test_baseline_and_candidate_values(self):
        result = compare_runs({"score": 0.6}, {"score": 0.75})
        assert result["score"]["baseline"] == 0.6
        assert result["score"]["candidate"] == 0.75

    def test_zero_baseline_delta_pct_is_zero(self):
        result = compare_runs({"score": 0.0}, {"score": 0.5})
        assert result["score"]["delta_pct"] == 0.0

    def test_negative_delta(self):
        result = compare_runs({"score": 0.9}, {"score": 0.8})
        assert result["score"]["delta"] < 0

    def test_metric_only_in_baseline(self):
        result = compare_runs({"f1": 0.8, "extra": 0.5}, {"f1": 0.9})
        assert "extra" in result
        assert result["extra"]["baseline"] == 0.5
        assert result["extra"]["candidate"] == 0.0

    def test_metric_only_in_candidate(self):
        result = compare_runs({"f1": 0.8}, {"f1": 0.9, "new_metric": 0.7})
        assert "new_metric" in result
        assert result["new_metric"]["baseline"] == 0.0
        assert result["new_metric"]["candidate"] == 0.7

    def test_multiple_metrics(self):
        baseline = {"f1": 0.80, "bleu": 0.30, "rouge": 0.50}
        candidate = {"f1": 0.85, "bleu": 0.28, "rouge": 0.55}
        result = compare_runs(baseline, candidate)
        assert abs(result["f1"]["delta"] - 0.05) < 1e-9
        assert abs(result["bleu"]["delta"] - (-0.02)) < 1e-9
        assert abs(result["rouge"]["delta"] - 0.05) < 1e-9
