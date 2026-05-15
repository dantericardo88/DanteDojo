"""Comprehensive tests for d42/d43/d44/d45 — experiment tracking final.

Covers:
- d42/d43: ExperimentTracker wired into finetune/grpo/dpo; evidence emission
- d44: RunLeaderboard.from_tracker(); compare_top_n(); CLI compare command
- d45: WandBTrainingCallback.on_step/on_epoch_end emit experiment.wandb_step;
       DOJO_WANDB=1 env var gates callback registration
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.experiment.leaderboard import RunLeaderboard
from dojo.experiment.tracker import ExperimentTracker, RunConfig
from dojo.experiment.wandb_adapter import WandBAdapter, WandBTrainingCallback


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def tracker() -> ExperimentTracker:
    return ExperimentTracker(use_mlflow=False)


@pytest.fixture
def run_config() -> RunConfig:
    return RunConfig(
        experiment_name="dojo.test",
        run_name="test-run",
        config_hash="abc123",
        pack_id="pack-test",
        hardware_tier="tier_cpu",
        seed=99,
    )


def _events_of_type(type_str: str) -> list[dict]:
    return [e for e in EvidenceLogger.get_events() if e.get("type") == type_str]


def _make_finetune_args(tmp_path: Path):
    from datetime import datetime, timezone as _tz
    from dojo.core.types import HALProfile, HardwareTier, LicenseSpec, PackFormat, ProvenanceManifest
    from dojo.pack.schema import SpecializationPack
    from dojo.train.base import FinetuneConfig

    profile = HALProfile(
        tier=HardwareTier.TIER_CPU,
        gpu_name="none",
        vram_bytes=0,
        compute_capability=0.0,
        driver_version="0",
        has_native_fp8=False,
        has_native_fp4=False,
    )
    cfg = FinetuneConfig(
        pack_id="test-pack",
        base_model="gpt2",
        output_dir=tmp_path / "out",
        num_epochs=2,
        seed=42,
    )
    pack = SpecializationPack(
        pack_id="test-pack",
        version="0.0.0",
        schema_version="1.0",
        sample_count=0,
        format=PackFormat.INSTRUCTION,
        splits={"train": 0, "val": 0, "held_out_gold": 0},
        harvest_receipt="test",
        license=LicenseSpec(
            spdx_id="Apache-2.0",
            allows_derived_ft=True,
            allows_commercial_use=True,
            requires_attribution=False,
        ),
        provenance=ProvenanceManifest(
            harvest_organ_version="0.0.0",
            sources=(),
            collection_date_utc=datetime.now(_tz.utc).date().isoformat(),
            dedup_method="none",
            human_reviewed=False,
        ),
        signature=b"sig",
        manifest_sha256="0" * 64,
        data_parquet_sha256="0" * 64,
    )
    return cfg, profile, pack


# ---------------------------------------------------------------------------
# d42/d43 — ExperimentTracker wired into finetune()
# ---------------------------------------------------------------------------

class TestFinetuneTrackerWiringFinal:
    """d42/d43: tracker.run() called, params and metrics logged inside finetune."""

    def test_finetune_emits_experiment_run_started(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg, profile, pack = _make_finetune_args(tmp_path)
        # Use real in-memory tracker (no MLflow) to verify evidence emission
        real_tracker = ExperimentTracker(use_mlflow=False)
        with patch("dojo.train.finetune.ExperimentTracker", return_value=real_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        events = _events_of_type(EvidenceType.EXPERIMENT_RUN_STARTED.value)
        assert len(events) >= 1, "EXPERIMENT_RUN_STARTED must be emitted"

    def test_finetune_emits_experiment_run_complete(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg, profile, pack = _make_finetune_args(tmp_path)
        real_tracker = ExperimentTracker(use_mlflow=False)
        with patch("dojo.train.finetune.ExperimentTracker", return_value=real_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        events = _events_of_type(EvidenceType.EXPERIMENT_RUN_COMPLETE.value)
        assert len(events) >= 1, "EXPERIMENT_RUN_COMPLETE must be emitted"

    def test_finetune_tracker_run_called_once(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg, profile, pack = _make_finetune_args(tmp_path)
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.run.return_value.__enter__ = MagicMock(return_value="run-id")
        mock_tracker.run.return_value.__exit__ = MagicMock(return_value=False)
        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        mock_tracker.run.assert_called_once()

    def test_finetune_log_params_includes_base_model(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg, profile, pack = _make_finetune_args(tmp_path)
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.run.return_value.__enter__ = MagicMock(return_value="run-id")
        mock_tracker.run.return_value.__exit__ = MagicMock(return_value=False)
        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        logged = mock_tracker.log_params.call_args[0][0]
        assert "base_model" in logged

    def test_finetune_log_params_includes_num_epochs(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg, profile, pack = _make_finetune_args(tmp_path)
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.run.return_value.__enter__ = MagicMock(return_value="run-id")
        mock_tracker.run.return_value.__exit__ = MagicMock(return_value=False)
        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        logged = mock_tracker.log_params.call_args[0][0]
        assert "num_epochs" in logged

    def test_finetune_log_metrics_batch_includes_final_loss(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg, profile, pack = _make_finetune_args(tmp_path)
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.run.return_value.__enter__ = MagicMock(return_value="run-id")
        mock_tracker.run.return_value.__exit__ = MagicMock(return_value=False)
        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        logged = mock_tracker.log_metrics_batch.call_args[0][0]
        assert "final_loss" in logged

    def test_finetune_log_metrics_batch_includes_steps_completed(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg, profile, pack = _make_finetune_args(tmp_path)
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.run.return_value.__enter__ = MagicMock(return_value="run-id")
        mock_tracker.run.return_value.__exit__ = MagicMock(return_value=False)
        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        logged = mock_tracker.log_metrics_batch.call_args[0][0]
        assert "steps_completed" in logged

    def test_tracker_inmemory_fallback_works_when_mlflow_unavailable(self, tracker, run_config):
        """MLflow unavailable → in-memory fallback used, run_id still returned."""
        run_id = tracker.start_run(run_config)
        assert run_id != ""
        assert tracker._in_memory_run is not None
        tracker.end_run()


# ---------------------------------------------------------------------------
# d44 — RunLeaderboard wired + compare_top_n
# ---------------------------------------------------------------------------

class TestLeaderboardCompareTopN:
    """d44: compare_top_n() returns N best runs ranked by primary metric."""

    def _make_lb(self, metric="f1", higher=True) -> RunLeaderboard:
        lb = RunLeaderboard(primary_metric=metric, higher_is_better=higher)
        lb.add_run("r1", "run-1", "exp", {metric: 0.90})
        lb.add_run("r2", "run-2", "exp", {metric: 0.75})
        lb.add_run("r3", "run-3", "exp", {metric: 0.80})
        lb.add_run("r4", "run-4", "exp", {metric: 0.60})
        lb.add_run("r5", "run-5", "exp", {metric: 0.95})
        return lb

    def test_compare_top_n_returns_3_runs(self):
        lb = self._make_lb()
        result = lb.compare_top_n(n=3)
        assert len(result) == 3

    def test_compare_top_n_best_run_is_rank_1(self):
        lb = self._make_lb()
        result = lb.compare_top_n(n=3)
        ranks = {info["rank"] for info in result.values()}
        assert 1 in ranks

    def test_compare_top_n_best_run_is_baseline(self):
        lb = self._make_lb()
        result = lb.compare_top_n(n=3)
        # Rank-1 entry uses itself as baseline
        rank1_entry = next(
            (rid, info) for rid, info in result.items() if info["rank"] == 1
        )
        assert rank1_entry[0] == rank1_entry[1]["baseline_run_id"]

    def test_compare_top_n_higher_is_better_selects_max(self):
        lb = self._make_lb(metric="f1", higher=True)
        result = lb.compare_top_n(n=1)
        # Only the highest f1=0.95 should be in top 1
        assert "r5" in result

    def test_compare_top_n_lower_is_better_selects_min(self):
        lb = RunLeaderboard(primary_metric="loss", higher_is_better=False)
        lb.add_run("a", "run-a", "exp", {"loss": 0.5})
        lb.add_run("b", "run-b", "exp", {"loss": 0.2})
        lb.add_run("c", "run-c", "exp", {"loss": 0.8})
        result = lb.compare_top_n(n=1)
        assert "b" in result  # lowest loss wins

    def test_compare_top_n_emits_leaderboard_evidence(self):
        lb = self._make_lb()
        EvidenceLogger.clear()
        lb.compare_top_n(n=3)
        events = _events_of_type(EvidenceType.EXPERIMENT_METRIC.value)
        compare_events = [
            e for e in events
            if e.get("data", {}).get("action") == "leaderboard.compare_top_n"
        ]
        assert len(compare_events) >= 1

    def test_compare_top_n_evidence_has_n_value(self):
        lb = self._make_lb()
        EvidenceLogger.clear()
        lb.compare_top_n(n=3)
        events = _events_of_type(EvidenceType.EXPERIMENT_METRIC.value)
        compare_events = [
            e for e in events
            if e.get("data", {}).get("action") == "leaderboard.compare_top_n"
        ]
        assert compare_events[0]["data"]["n"] == 3

    def test_from_tracker_populates_correct_run_count(self):
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = [
            {"run_id": f"r{i}", "name": f"run-{i}", "status": "FINISHED",
             "start_time": "", "loss": float(i) * 0.1}
            for i in range(5)
        ]
        lb = RunLeaderboard.from_tracker(mock_tracker, primary_metric="loss", higher_is_better=False)
        ranked = lb.rank()
        assert len(ranked) == 5

    def test_from_tracker_best_entry_has_correct_metric(self):
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = [
            {"run_id": "best", "name": "best-run", "status": "FINISHED",
             "start_time": "", "loss": 0.05},
            {"run_id": "worse", "name": "worse-run", "status": "FINISHED",
             "start_time": "", "loss": 0.50},
        ]
        lb = RunLeaderboard.from_tracker(mock_tracker, primary_metric="loss", higher_is_better=False)
        best = lb.best()
        assert best is not None
        assert best.run_id == "best"


# ---------------------------------------------------------------------------
# d44 — CLI experiments compare command
# ---------------------------------------------------------------------------

class TestExperimentsCompareCommand:

    def test_compare_command_exits_ok_no_runs(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app
        runner = CliRunner()
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = []
        with patch("dojo.cli.main.ExperimentTracker", return_value=mock_tracker):
            result = runner.invoke(app, ["experiments", "compare"])
        assert result.exit_code == 0

    def test_compare_command_json_output_empty_when_no_runs(self):
        import json as _json
        from typer.testing import CliRunner
        from dojo.cli.main import app
        runner = CliRunner()
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = []
        with patch("dojo.cli.main.ExperimentTracker", return_value=mock_tracker):
            result = runner.invoke(app, ["experiments", "compare", "--json"])
        assert result.exit_code == 0
        parsed = _json.loads(result.output)
        assert parsed == {}

    def test_compare_command_emits_leaderboard_compare_evidence(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app
        runner = CliRunner()
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = [
            {"run_id": "r1", "name": "run-1", "status": "FINISHED",
             "start_time": "", "loss": 0.3},
            {"run_id": "r2", "name": "run-2", "status": "FINISHED",
             "start_time": "", "loss": 0.5},
        ]
        with patch("dojo.cli.main.ExperimentTracker", return_value=mock_tracker):
            runner.invoke(app, ["experiments", "compare", "--metric", "loss"])
        events = _events_of_type("experiment.leaderboard_compare")
        assert len(events) >= 1

    def test_compare_command_top_n_option(self):
        import json as _json
        from typer.testing import CliRunner
        from dojo.cli.main import app
        runner = CliRunner()
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = [
            {"run_id": f"r{i}", "name": f"run-{i}", "status": "FINISHED",
             "start_time": "", "loss": float(i) * 0.1}
            for i in range(10)
        ]
        with patch("dojo.cli.main.ExperimentTracker", return_value=mock_tracker):
            result = runner.invoke(app, ["experiments", "compare", "--json", "--top-n", "3"])
        assert result.exit_code == 0
        parsed = _json.loads(result.output)
        assert len(parsed) == 3


# ---------------------------------------------------------------------------
# d45 — WandBTrainingCallback evidence emission
# ---------------------------------------------------------------------------

class TestWandBCallbackEvidence:
    """d45: on_step and on_epoch_end emit experiment.wandb_step evidence."""

    def _make_callback(self) -> WandBTrainingCallback:
        adapter = WandBAdapter(use_wandb=False)
        cfg = RunConfig(experiment_name="wandb-test", run_name="cb-run")
        adapter.init_run(cfg)
        return adapter.as_callback()

    def test_on_step_emits_wandb_step_evidence(self):
        cb = self._make_callback()
        EvidenceLogger.clear()
        cb.on_step(step=1, metrics={"loss": 0.5})
        events = _events_of_type("experiment.wandb_step")
        assert len(events) >= 1

    def test_on_step_evidence_has_correct_step(self):
        cb = self._make_callback()
        EvidenceLogger.clear()
        cb.on_step(step=77, metrics={"loss": 0.3})
        events = _events_of_type("experiment.wandb_step")
        assert events[0]["data"]["step"] == 77

    def test_on_step_evidence_has_on_step_event_label(self):
        cb = self._make_callback()
        EvidenceLogger.clear()
        cb.on_step(step=1, metrics={"lr": 1e-4})
        events = _events_of_type("experiment.wandb_step")
        assert events[0]["data"]["event"] == "on_step"

    def test_on_epoch_end_emits_wandb_step_evidence(self):
        cb = self._make_callback()
        EvidenceLogger.clear()
        cb.on_epoch_end(epoch=2, metrics={"loss": 0.4, "f1": 0.7})
        events = _events_of_type("experiment.wandb_step")
        assert len(events) >= 1

    def test_on_epoch_end_evidence_has_on_epoch_end_label(self):
        cb = self._make_callback()
        EvidenceLogger.clear()
        cb.on_epoch_end(epoch=3, metrics={"loss": 0.2})
        events = _events_of_type("experiment.wandb_step")
        assert events[0]["data"]["event"] == "on_epoch_end"

    def test_on_epoch_end_evidence_has_epoch_prefixed_metrics(self):
        cb = self._make_callback()
        EvidenceLogger.clear()
        cb.on_epoch_end(epoch=1, metrics={"loss": 0.4, "accuracy": 0.9})
        events = _events_of_type("experiment.wandb_step")
        metrics_keys = set(events[0]["data"]["metrics"].keys())
        assert "epoch_loss" in metrics_keys
        assert "epoch_accuracy" in metrics_keys

    def test_on_step_also_emits_experiment_metric(self):
        cb = self._make_callback()
        EvidenceLogger.clear()
        cb.on_step(step=5, metrics={"loss": 0.25})
        metric_events = _events_of_type(EvidenceType.EXPERIMENT_METRIC.value)
        assert len(metric_events) >= 1


# ---------------------------------------------------------------------------
# d45 — DOJO_WANDB=1 env var gates callback in simulate path
# ---------------------------------------------------------------------------

class TestDOJOWandBEnvVar:
    """d45: DOJO_WANDB=1 causes wandb_step evidence to be emitted in simulate path."""

    def _mock_tracker(self):
        """Return a mock ExperimentTracker that avoids MLflow network calls."""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.run.return_value.__enter__ = MagicMock(return_value="mock-run-id")
        mock_tracker.run.return_value.__exit__ = MagicMock(return_value=False)
        return mock_tracker

    def test_dojo_wandb_1_triggers_wandb_step_evidence(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_WANDB", "1")
        cfg, profile, pack = _make_finetune_args(tmp_path)
        mock_tracker = self._mock_tracker()
        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        events = _events_of_type("experiment.wandb_step")
        assert len(events) >= 1, "DOJO_WANDB=1 must produce wandb_step evidence"

    def test_dojo_wandb_0_does_not_trigger_wandb_step_evidence(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_WANDB", "0")
        cfg, profile, pack = _make_finetune_args(tmp_path)
        mock_tracker = self._mock_tracker()
        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        events = _events_of_type("experiment.wandb_step")
        assert len(events) == 0, "DOJO_WANDB=0 must not produce wandb_step evidence"

    def test_dojo_wandb_absent_does_not_trigger_wandb_step_evidence(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.delenv("DOJO_WANDB", raising=False)
        cfg, profile, pack = _make_finetune_args(tmp_path)
        mock_tracker = self._mock_tracker()
        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)
        events = _events_of_type("experiment.wandb_step")
        assert len(events) == 0
