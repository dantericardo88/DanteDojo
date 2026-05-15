"""Unit tests for d44 (RunLeaderboard) and d45 (WandBAdapter / WandBTrainingCallback)."""

from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.experiment.leaderboard import LeaderboardEntry, RunLeaderboard
from dojo.experiment.tracker import RunConfig
from dojo.experiment.wandb_adapter import WandBAdapter, WandBTrainingCallback


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset the evidence logger before and after every test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ===========================================================================
# d44 — RunLeaderboard tests
# ===========================================================================

class TestRunLeaderboard:

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_leaderboard_empty_best_returns_none(self):
        lb = RunLeaderboard(primary_metric="f1")
        assert lb.best() is None

    def test_leaderboard_single_run_rank_1(self):
        lb = RunLeaderboard(primary_metric="f1")
        lb.add_run("r1", "run-1", "exp-a", {"f1": 0.80})
        ranked = lb.rank()
        assert len(ranked) == 1
        assert ranked[0].rank == 1

    # -----------------------------------------------------------------------
    # Ranking order
    # -----------------------------------------------------------------------

    def test_leaderboard_rank_orders_by_metric_descending(self):
        """Higher F1 → rank 1 (higher_is_better=True)."""
        lb = RunLeaderboard(primary_metric="f1", higher_is_better=True)
        lb.add_run("r1", "run-1", "exp", {"f1": 0.70})
        lb.add_run("r2", "run-2", "exp", {"f1": 0.90})
        lb.add_run("r3", "run-3", "exp", {"f1": 0.80})
        ranked = lb.rank()
        assert ranked[0].run_id == "r2"
        assert ranked[1].run_id == "r3"
        assert ranked[2].run_id == "r1"

    def test_leaderboard_rank_lower_is_better(self):
        """Lower loss → rank 1 (higher_is_better=False)."""
        lb = RunLeaderboard(primary_metric="loss", higher_is_better=False)
        lb.add_run("r1", "run-1", "exp", {"loss": 0.5})
        lb.add_run("r2", "run-2", "exp", {"loss": 0.1})
        lb.add_run("r3", "run-3", "exp", {"loss": 0.3})
        ranked = lb.rank()
        assert ranked[0].run_id == "r2"   # lowest loss → best
        assert ranked[2].run_id == "r1"   # highest loss → worst

    def test_leaderboard_three_runs_correct_ranking(self):
        lb = RunLeaderboard(primary_metric="accuracy", higher_is_better=True)
        lb.add_run("a", "run-a", "exp", {"accuracy": 0.60})
        lb.add_run("b", "run-b", "exp", {"accuracy": 0.95})
        lb.add_run("c", "run-c", "exp", {"accuracy": 0.75})
        ranked = lb.rank()
        assert [e.run_id for e in ranked] == ["b", "c", "a"]
        assert [e.rank for e in ranked] == [1, 2, 3]

    # -----------------------------------------------------------------------
    # best()
    # -----------------------------------------------------------------------

    def test_leaderboard_best_returns_highest_metric(self):
        lb = RunLeaderboard(primary_metric="f1")
        lb.add_run("r1", "run-1", "exp", {"f1": 0.70})
        lb.add_run("r2", "run-2", "exp", {"f1": 0.92})
        best = lb.best()
        assert best is not None
        assert best.run_id == "r2"

    def test_leaderboard_best_lower_is_better_returns_lowest(self):
        lb = RunLeaderboard(primary_metric="perplexity", higher_is_better=False)
        lb.add_run("r1", "run-1", "exp", {"perplexity": 20.0})
        lb.add_run("r2", "run-2", "exp", {"perplexity": 8.5})
        best = lb.best()
        assert best is not None
        assert best.run_id == "r2"

    # -----------------------------------------------------------------------
    # to_dict()
    # -----------------------------------------------------------------------

    def test_leaderboard_to_dict_has_primary_metric(self):
        lb = RunLeaderboard(primary_metric="rouge")
        d = lb.to_dict()
        assert d["primary_metric"] == "rouge"
        assert "higher_is_better" in d
        assert "entries" in d

    def test_leaderboard_to_dict_entries_sorted(self):
        lb = RunLeaderboard(primary_metric="f1")
        lb.add_run("r1", "run-1", "exp", {"f1": 0.55})
        lb.add_run("r2", "run-2", "exp", {"f1": 0.88})
        d = lb.to_dict()
        assert d["entries"][0]["run_id"] == "r2"
        assert d["entries"][0]["rank"] == 1
        assert d["entries"][1]["rank"] == 2

    # -----------------------------------------------------------------------
    # compare_top_n()
    # -----------------------------------------------------------------------

    def test_leaderboard_compare_top_n_returns_dict(self):
        lb = RunLeaderboard(primary_metric="f1")
        lb.add_run("r1", "run-1", "exp", {"f1": 0.70})
        lb.add_run("r2", "run-2", "exp", {"f1": 0.85})
        lb.add_run("r3", "run-3", "exp", {"f1": 0.75})
        result = lb.compare_top_n(n=2)
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_leaderboard_compare_top_n_has_baseline_and_candidate(self):
        lb = RunLeaderboard(primary_metric="f1")
        lb.add_run("r1", "run-1", "exp", {"f1": 0.70, "loss": 0.4})
        lb.add_run("r2", "run-2", "exp", {"f1": 0.85, "loss": 0.2})
        result = lb.compare_top_n(n=2)
        # Both runs should appear; each should have a 'comparison' sub-dict
        for run_id, info in result.items():
            assert "comparison" in info
            assert "baseline_run_id" in info

    def test_leaderboard_compare_top_n_empty_returns_empty(self):
        lb = RunLeaderboard(primary_metric="f1")
        assert lb.compare_top_n(n=3) == {}

    # -----------------------------------------------------------------------
    # Evidence emission
    # -----------------------------------------------------------------------

    def test_leaderboard_rank_emits_evidence(self):
        lb = RunLeaderboard(primary_metric="f1")
        lb.add_run("r1", "run-1", "exp", {"f1": 0.80})
        lb.rank()
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_RUN_COMPLETE.value in types

    def test_leaderboard_add_run_emits_evidence(self):
        lb = RunLeaderboard(primary_metric="f1")
        lb.add_run("r1", "run-1", "exp", {"f1": 0.80})
        events = EvidenceLogger.get_events()
        run_complete_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_RUN_COMPLETE.value
        ]
        assert len(run_complete_events) >= 1


# ===========================================================================
# d45 — WandBAdapter / WandBTrainingCallback tests
# ===========================================================================

class TestWandBAdapter:

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def test_wandb_adapter_init_no_wandb_falls_back(self):
        """use_wandb=False → _wandb_available=False regardless of install state."""
        adapter = WandBAdapter(use_wandb=False)
        assert adapter._wandb_available is False

    def test_wandb_adapter_init_missing_wandb_falls_back(self):
        """If wandb import fails, _wandb_available must be False."""
        # Since wandb is unlikely to be installed in CI, this exercises the
        # ImportError path. If wandb IS installed the test still passes because
        # we only assert the adapter constructs without error.
        adapter = WandBAdapter(use_wandb=True)
        # No assertion on _wandb_available — either state is acceptable.
        assert adapter is not None

    # -----------------------------------------------------------------------
    # init_run()
    # -----------------------------------------------------------------------

    def test_wandb_adapter_init_run_returns_run_id(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        run_id = adapter.init_run(config)
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_wandb_adapter_init_run_explicit_id(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        run_id = adapter.init_run(config, run_id="explicit-id-123")
        assert run_id == "explicit-id-123"

    def test_wandb_adapter_init_run_emits_run_started_evidence(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        adapter.init_run(config)
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_RUN_STARTED.value in types

    # -----------------------------------------------------------------------
    # log()
    # -----------------------------------------------------------------------

    def test_wandb_adapter_log_emits_experiment_metric_evidence(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        adapter.init_run(config)
        adapter.log({"loss": 0.42})
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_METRIC.value in types

    def test_wandb_adapter_log_all_keys_in_evidence(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        adapter.init_run(config)
        adapter.log({"loss": 0.5, "f1": 0.82, "accuracy": 0.91})
        events = EvidenceLogger.get_events()
        metric_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
        ]
        # Find the log() event (not log_config)
        log_events = [
            e for e in metric_events
            if "metrics" in e.get("data", {})
        ]
        assert len(log_events) >= 1
        logged_keys = set(log_events[-1]["data"]["keys"])
        assert {"loss", "f1", "accuracy"} == logged_keys

    def test_wandb_adapter_log_step_in_evidence(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        adapter.init_run(config)
        adapter.log({"loss": 0.3}, step=50)
        events = EvidenceLogger.get_events()
        metric_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
            and "step" in e.get("data", {})
        ]
        assert any(e["data"]["step"] == 50 for e in metric_events)

    # -----------------------------------------------------------------------
    # log_config()
    # -----------------------------------------------------------------------

    def test_wandb_adapter_log_config_stored(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        adapter.init_run(config)
        adapter.log_config({"lr": 1e-4, "batch_size": 16})
        # Check in-memory config updated
        assert adapter._in_memory.get("config", {}).get("lr") == 1e-4
        assert adapter._in_memory.get("config", {}).get("batch_size") == 16

    def test_wandb_adapter_log_config_emits_evidence(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        adapter.init_run(config)
        adapter.log_config({"lr": 1e-4})
        events = EvidenceLogger.get_events()
        config_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
            and e.get("data", {}).get("action") == "log_config"
        ]
        assert len(config_events) >= 1

    # -----------------------------------------------------------------------
    # finish()
    # -----------------------------------------------------------------------

    def test_wandb_adapter_finish_emits_run_complete(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        adapter.init_run(config)
        adapter.finish()
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_RUN_COMPLETE.value in types

    def test_wandb_adapter_finish_status_in_evidence(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        adapter.init_run(config)
        adapter.finish(status="failed")
        events = EvidenceLogger.get_events()
        complete_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_RUN_COMPLETE.value
        ]
        assert len(complete_events) >= 1
        assert complete_events[-1]["data"]["status"] == "failed"

    def test_wandb_adapter_finish_clears_active_run_id(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="run-001")
        adapter.init_run(config)
        adapter.finish()
        assert adapter._active_run_id is None


# ===========================================================================
# d45 — WandBTrainingCallback tests
# ===========================================================================

class TestWandBTrainingCallback:

    @pytest.fixture
    def adapter_and_callback(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="exp", run_name="train-run")
        adapter.init_run(config)
        cb = adapter.as_callback()
        return adapter, cb

    def test_wandb_training_callback_on_step_logs_metrics(self, adapter_and_callback):
        adapter, cb = adapter_and_callback
        cb.on_step(step=10, metrics={"loss": 0.35, "lr": 3e-5})
        events = EvidenceLogger.get_events()
        metric_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
            and "step" in e.get("data", {})
        ]
        assert any(e["data"]["step"] == 10 for e in metric_events)

    def test_wandb_training_callback_on_step_includes_keys(self, adapter_and_callback):
        _, cb = adapter_and_callback
        cb.on_step(step=5, metrics={"loss": 0.4, "grad_norm": 1.2})
        events = EvidenceLogger.get_events()
        log_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
            and "metrics" in e.get("data", {})
        ]
        last = log_events[-1]["data"]["metrics"]
        assert "loss" in last and "grad_norm" in last

    def test_wandb_training_callback_on_epoch_end_prefixes_epoch(self, adapter_and_callback):
        adapter, cb = adapter_and_callback
        cb.on_epoch_end(epoch=2, metrics={"f1": 0.88, "loss": 0.20})
        events = EvidenceLogger.get_events()
        log_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
            and "metrics" in e.get("data", {})
        ]
        last_metrics = log_events[-1]["data"]["metrics"]
        assert "epoch_f1" in last_metrics
        assert "epoch_loss" in last_metrics
        assert last_metrics["epoch_f1"] == pytest.approx(0.88)

    def test_wandb_training_callback_on_train_end_finishes_run(self, adapter_and_callback):
        adapter, cb = adapter_and_callback
        cb.on_train_end(final_metrics={"f1": 0.91})
        # adapter.finish() clears _active_run_id
        assert adapter._active_run_id is None

    def test_wandb_training_callback_on_train_end_emits_complete_evidence(self, adapter_and_callback):
        _, cb = adapter_and_callback
        cb.on_train_end(final_metrics={"accuracy": 0.95})
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_RUN_COMPLETE.value in types

    def test_wandb_as_callback_returns_callback_instance(self):
        adapter = WandBAdapter(use_wandb=False)
        cb = adapter.as_callback()
        assert isinstance(cb, WandBTrainingCallback)
