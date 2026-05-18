"""Tests for ExperimentTracker filter_runs(), list_runs() history, and _InMemoryRun.to_dict().

Covers d43: run history query/filter API with in-memory fallback.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from dojo.experiment.tracker import ExperimentTracker, RunConfig, _InMemoryRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker() -> ExperimentTracker:
    return ExperimentTracker(use_mlflow=False)


def _run(tracker: ExperimentTracker, name: str, experiment: str = "exp", **metrics: float) -> str:
    cfg = RunConfig(experiment_name=experiment, run_name=name)
    run_id = tracker.start_run(cfg)
    for k, v in metrics.items():
        tracker.log_metric(k, v)
    tracker.end_run()
    return run_id


# ---------------------------------------------------------------------------
# _InMemoryRun.to_dict
# ---------------------------------------------------------------------------

class TestInMemoryRunToDict:
    def test_basic_fields_present(self):
        r = _InMemoryRun("r1", RunConfig(experiment_name="e", run_name="n"))
        d = r.to_dict()
        assert d["run_id"] == "r1"
        assert d["name"] == "n"
        assert d["experiment_name"] == "e"
        assert "status" in d
        assert "start_time" in d

    def test_end_time_none_before_end(self):
        r = _InMemoryRun("r1", RunConfig(experiment_name="e", run_name="n"))
        assert r.to_dict()["end_time"] is None

    def test_end_time_set_after_end(self):
        r = _InMemoryRun("r1", RunConfig(experiment_name="e", run_name="n"))
        r.end("FINISHED")
        assert r.to_dict()["end_time"] is not None

    def test_metrics_flattened_as_top_level_keys(self):
        r = _InMemoryRun("r1", RunConfig(experiment_name="e", run_name="n"))
        r.log_metric("accuracy", 0.95)
        r.log_metric("loss", 0.3)
        d = r.to_dict()
        assert d["accuracy"] == 0.95
        assert d["loss"] == 0.3

    def test_latest_value_wins_for_repeated_metric(self):
        r = _InMemoryRun("r1", RunConfig(experiment_name="e", run_name="n"))
        r.log_metric("loss", 1.0, step=0)
        r.log_metric("loss", 0.5, step=1)
        r.log_metric("loss", 0.2, step=2)
        assert r.to_dict()["loss"] == 0.2


# ---------------------------------------------------------------------------
# list_runs() — in-memory history
# ---------------------------------------------------------------------------

class TestListRunsInMemory:
    def test_empty_before_any_runs(self):
        tracker = _make_tracker()
        assert tracker.list_runs() == []

    def test_active_run_not_in_history(self):
        tracker = _make_tracker()
        tracker.start_run(RunConfig(experiment_name="e", run_name="r"))
        assert tracker.list_runs() == []

    def test_finished_run_appears_in_history(self):
        tracker = _make_tracker()
        _run(tracker, "r1", accuracy=0.9)
        runs = tracker.list_runs()
        assert len(runs) == 1
        assert runs[0]["name"] == "r1"

    def test_multiple_runs_all_listed(self):
        tracker = _make_tracker()
        for i in range(5):
            _run(tracker, f"run{i}")
        assert len(tracker.list_runs()) == 5

    def test_filter_by_experiment_name(self):
        tracker = _make_tracker()
        _run(tracker, "a1", experiment="alpha")
        _run(tracker, "b1", experiment="beta")
        _run(tracker, "a2", experiment="alpha")
        results = tracker.list_runs(experiment_name="alpha")
        assert len(results) == 2
        assert all(r["experiment_name"] == "alpha" for r in results)


# ---------------------------------------------------------------------------
# filter_runs() — filtering logic
# ---------------------------------------------------------------------------

class TestFilterRuns:
    def test_filter_by_status_finished(self):
        tracker = _make_tracker()
        _run(tracker, "ok")
        cfg = RunConfig(experiment_name="exp", run_name="fail")
        tracker.start_run(cfg)
        tracker.end_run(status="FAILED")
        finished = tracker.filter_runs(status="FINISHED")
        failed = tracker.filter_runs(status="FAILED")
        assert len(finished) == 1 and finished[0]["name"] == "ok"
        assert len(failed) == 1 and failed[0]["name"] == "fail"

    def test_filter_by_metric_min_val(self):
        tracker = _make_tracker()
        _run(tracker, "low", accuracy=0.5)
        _run(tracker, "high", accuracy=0.9)
        results = tracker.filter_runs(metric_name="accuracy", min_val=0.8)
        assert len(results) == 1
        assert results[0]["name"] == "high"

    def test_filter_by_metric_max_val(self):
        tracker = _make_tracker()
        _run(tracker, "low", loss=0.2)
        _run(tracker, "high", loss=2.0)
        results = tracker.filter_runs(metric_name="loss", max_val=0.5)
        assert len(results) == 1
        assert results[0]["name"] == "low"

    def test_filter_by_metric_range(self):
        tracker = _make_tracker()
        _run(tracker, "r1", score=1.0)
        _run(tracker, "r2", score=5.0)
        _run(tracker, "r3", score=9.0)
        results = tracker.filter_runs(metric_name="score", min_val=2.0, max_val=8.0)
        assert len(results) == 1
        assert results[0]["name"] == "r2"

    def test_missing_metric_excludes_run(self):
        tracker = _make_tracker()
        _run(tracker, "with_metric", accuracy=0.9)
        _run(tracker, "without_metric")
        results = tracker.filter_runs(metric_name="accuracy")
        assert len(results) == 1
        assert results[0]["name"] == "with_metric"

    def test_sort_by_metric_descending(self):
        tracker = _make_tracker()
        _run(tracker, "r1", score=3.0)
        _run(tracker, "r2", score=9.0)
        _run(tracker, "r3", score=6.0)
        results = tracker.filter_runs(sort_by="score", higher_is_better=True)
        assert [r["name"] for r in results] == ["r2", "r3", "r1"]

    def test_sort_by_metric_ascending(self):
        tracker = _make_tracker()
        _run(tracker, "r1", loss=0.9)
        _run(tracker, "r2", loss=0.1)
        _run(tracker, "r3", loss=0.5)
        results = tracker.filter_runs(sort_by="loss", higher_is_better=False)
        assert [r["name"] for r in results] == ["r2", "r3", "r1"]

    def test_limit_caps_results(self):
        tracker = _make_tracker()
        for i in range(10):
            _run(tracker, f"r{i}", score=float(i))
        results = tracker.filter_runs(sort_by="score", limit=3)
        assert len(results) == 3

    def test_filter_after_timestamp(self):
        tracker = _make_tracker()
        past_ts = time.time() - 100.0
        _run(tracker, "old")
        future_ts = time.time() + 1.0
        results = tracker.filter_runs(after_ts=future_ts)
        assert results == []
        results_all = tracker.filter_runs(after_ts=past_ts)
        assert len(results_all) == 1

    def test_combined_filters(self):
        tracker = _make_tracker()
        _run(tracker, "good", accuracy=0.95)
        _run(tracker, "bad_acc", accuracy=0.4)
        cfg = RunConfig(experiment_name="exp", run_name="failed")
        tracker.start_run(cfg)
        tracker.log_metric("accuracy", 0.99)
        tracker.end_run(status="FAILED")
        results = tracker.filter_runs(
            metric_name="accuracy", min_val=0.9, status="FINISHED"
        )
        assert len(results) == 1
        assert results[0]["name"] == "good"

    def test_empty_result_on_no_match(self):
        tracker = _make_tracker()
        _run(tracker, "r1", score=5.0)
        results = tracker.filter_runs(metric_name="score", min_val=9.0)
        assert results == []

    def test_filter_emits_evidence(self):
        tracker = _make_tracker()
        _run(tracker, "r1", score=5.0)
        emitted = []
        with patch("dojo.experiment.tracker.EvidenceLogger") as mock_ev:
            inst = MagicMock()
            mock_ev.get_instance.return_value = inst
            inst.emit.side_effect = lambda ev, **kw: emitted.append(ev)
            tracker.filter_runs(metric_name="score")
        assert len(emitted) == 1
        assert emitted[0]["data"]["type"] == "filter_runs"

    def test_filter_by_experiment_name_in_filter_runs(self):
        tracker = _make_tracker()
        _run(tracker, "a", experiment="proj_a", accuracy=0.8)
        _run(tracker, "b", experiment="proj_b", accuracy=0.9)
        results = tracker.filter_runs(experiment_name="proj_a")
        assert len(results) == 1 and results[0]["name"] == "a"

    def test_no_sort_preserves_insertion_order(self):
        tracker = _make_tracker()
        names = ["first", "second", "third"]
        for n in names:
            _run(tracker, n, score=1.0)
        results = tracker.filter_runs()
        assert [r["name"] for r in results] == names

    def test_run_id_in_every_result(self):
        tracker = _make_tracker()
        rid = _run(tracker, "r1", score=1.0)
        results = tracker.filter_runs()
        assert results[0]["run_id"] == rid
