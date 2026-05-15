"""Tests verifying experiment tracker + WandB callback wiring into training loops.

Covers:
- ExperimentTracker.start_run called during finetune() (d42)
- WandBTrainingCallback.on_step called during training simulation (d45)
- RunLeaderboard.from_tracker() populates correctly from mock tracker (d44)
- CLI `experiments list` command works end-to-end (d42)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call
import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.experiment.leaderboard import RunLeaderboard
from dojo.experiment.tracker import ExperimentTracker, RunConfig
from dojo.experiment.wandb_adapter import WandBAdapter, WandBTrainingCallback


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# ExperimentTracker wired into finetune()
# ---------------------------------------------------------------------------

class TestFinetuneTrackerWiring:

    def _make_finetune_args(self, tmp_path: Path):
        from datetime import datetime, timezone as _tz
        from dojo.core.types import (
            HALProfile, HardwareTier, LicenseSpec, PackFormat, ProvenanceManifest,
        )
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
            num_epochs=1,
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

    def test_finetune_calls_tracker_start_run(self, tmp_path):
        from dojo.train.finetune import finetune

        cfg, profile, pack = self._make_finetune_args(tmp_path)

        mock_tracker = MagicMock(spec=ExperimentTracker)
        # Make run() context manager work
        mock_tracker.run.return_value.__enter__ = MagicMock(return_value="mock-run-id")
        mock_tracker.run.return_value.__exit__ = MagicMock(return_value=False)

        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)

        mock_tracker.run.assert_called_once()

    def test_finetune_logs_params(self, tmp_path):
        from dojo.train.finetune import finetune

        cfg, profile, pack = self._make_finetune_args(tmp_path)

        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.run.return_value.__enter__ = MagicMock(return_value="mock-run-id")
        mock_tracker.run.return_value.__exit__ = MagicMock(return_value=False)

        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)

        mock_tracker.log_params.assert_called_once()
        logged = mock_tracker.log_params.call_args[0][0]
        assert "base_model" in logged
        assert "num_epochs" in logged

    def test_finetune_logs_final_metrics(self, tmp_path):
        from dojo.train.finetune import finetune

        cfg, profile, pack = self._make_finetune_args(tmp_path)

        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.run.return_value.__enter__ = MagicMock(return_value="mock-run-id")
        mock_tracker.run.return_value.__exit__ = MagicMock(return_value=False)

        with patch("dojo.train.finetune.ExperimentTracker", return_value=mock_tracker):
            finetune(config=cfg, hal_profile=profile, pack=pack)

        mock_tracker.log_metrics_batch.assert_called_once()
        logged = mock_tracker.log_metrics_batch.call_args[0][0]
        assert "final_loss" in logged
        assert "steps_completed" in logged

    def test_finetune_emits_run_started_evidence(self, tmp_path):
        """ExperimentTracker (real, in-memory) emits EXPERIMENT_RUN_STARTED during finetune."""
        from dojo.train.finetune import finetune

        cfg, profile, pack = self._make_finetune_args(tmp_path)
        finetune(config=cfg, hal_profile=profile, pack=pack)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.EXPERIMENT_RUN_STARTED.value in types

    def test_finetune_emits_run_complete_evidence(self, tmp_path):
        from dojo.train.finetune import finetune

        cfg, profile, pack = self._make_finetune_args(tmp_path)
        finetune(config=cfg, hal_profile=profile, pack=pack)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.EXPERIMENT_RUN_COMPLETE.value in types


# ---------------------------------------------------------------------------
# WandBTrainingCallback.on_step called during training simulation
# ---------------------------------------------------------------------------

class TestWandBCallbackOnStep:

    def test_on_step_emits_metric_evidence(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="test-exp", run_name="cb-test")
        adapter.init_run(config)
        cb = adapter.as_callback()

        EvidenceLogger.clear()
        cb.on_step(step=1, metrics={"loss": 0.5, "lr": 1e-4})

        events = EvidenceLogger.get_events()
        metric_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
        ]
        assert len(metric_events) >= 1

    def test_on_step_step_value_recorded(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="test-exp", run_name="cb-test")
        adapter.init_run(config)
        cb = adapter.as_callback()

        EvidenceLogger.clear()
        cb.on_step(step=42, metrics={"loss": 0.3})

        events = EvidenceLogger.get_events()
        step_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
            and e.get("data", {}).get("step") == 42
        ]
        assert len(step_events) >= 1

    def test_multiple_on_step_calls_accumulate(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="test-exp", run_name="multi-step")
        adapter.init_run(config)
        cb = adapter.as_callback()

        EvidenceLogger.clear()
        for step in range(5):
            cb.on_step(step=step, metrics={"loss": 1.0 - step * 0.1})

        assert len(adapter._in_memory.get("metrics", [])) == 5

    def test_on_epoch_end_prefixes_metrics(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="test-exp", run_name="epoch-test")
        adapter.init_run(config)
        cb = adapter.as_callback()

        EvidenceLogger.clear()
        cb.on_epoch_end(epoch=1, metrics={"loss": 0.4, "f1": 0.7})

        events = EvidenceLogger.get_events()
        log_events = [
            e for e in events
            if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
            and "metrics" in e.get("data", {})
        ]
        assert len(log_events) >= 1
        keys = set(log_events[-1]["data"]["metrics"].keys())
        assert "epoch_loss" in keys
        assert "epoch_f1" in keys

    def test_on_train_end_finishes_adapter(self):
        adapter = WandBAdapter(use_wandb=False)
        config = RunConfig(experiment_name="test-exp", run_name="end-test")
        adapter.init_run(config)
        cb = adapter.as_callback()
        cb.on_train_end({"final_loss": 0.1})
        assert adapter._active_run_id is None


# ---------------------------------------------------------------------------
# RunLeaderboard.from_tracker()
# ---------------------------------------------------------------------------

class TestLeaderboardFromTracker:

    def test_from_tracker_returns_leaderboard_instance(self):
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = []

        lb = RunLeaderboard.from_tracker(mock_tracker, experiment_name="exp-a")
        assert isinstance(lb, RunLeaderboard)

    def test_from_tracker_calls_list_runs_with_name(self):
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = []

        RunLeaderboard.from_tracker(mock_tracker, experiment_name="my-exp")
        mock_tracker.list_runs.assert_called_once_with("my-exp")

    def test_from_tracker_empty_runs_gives_empty_leaderboard(self):
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = []

        lb = RunLeaderboard.from_tracker(mock_tracker)
        assert lb.best() is None

    def test_from_tracker_populates_entries_from_run_dicts(self):
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = [
            {"run_id": "r1", "name": "run-one", "status": "FINISHED", "start_time": "2025-01-01"},
            {"run_id": "r2", "name": "run-two", "status": "FINISHED", "start_time": "2025-01-02"},
        ]

        lb = RunLeaderboard.from_tracker(mock_tracker, experiment_name="exp-x")
        ranked = lb.rank()
        assert len(ranked) == 2
        run_ids = {e.run_id for e in ranked}
        assert {"r1", "r2"} == run_ids

    def test_from_tracker_uses_primary_metric_and_direction(self):
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = [
            {"run_id": "r1", "name": "run-1", "status": "FINISHED", "start_time": "", "f1": 0.90},
            {"run_id": "r2", "name": "run-2", "status": "FINISHED", "start_time": "", "f1": 0.75},
        ]

        lb = RunLeaderboard.from_tracker(
            mock_tracker,
            primary_metric="f1",
            higher_is_better=True,
        )
        best = lb.best()
        assert best is not None
        assert best.run_id == "r1"

    def test_from_tracker_with_real_in_memory_tracker(self):
        tracker = ExperimentTracker(use_mlflow=False)
        # list_runs returns empty list with in-memory backend
        lb = RunLeaderboard.from_tracker(tracker, experiment_name="any")
        assert isinstance(lb, RunLeaderboard)
        assert lb.best() is None


# ---------------------------------------------------------------------------
# CLI experiments commands
# ---------------------------------------------------------------------------

class TestExperimentsCliList:

    def test_experiments_list_no_runs(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app

        runner = CliRunner()
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = []

        with patch("dojo.cli.main.ExperimentTracker", return_value=mock_tracker):
            result = runner.invoke(app, ["experiments", "list"])

        assert result.exit_code == 0
        assert "No experiment runs" in result.output

    def test_experiments_list_json_empty(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app
        import json as _json

        runner = CliRunner()
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = []

        with patch("dojo.cli.main.ExperimentTracker", return_value=mock_tracker):
            result = runner.invoke(app, ["experiments", "list", "--json"])

        assert result.exit_code == 0
        parsed = _json.loads(result.output)
        assert parsed == []

    def test_experiments_list_json_with_runs(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app
        import json as _json

        runner = CliRunner()
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = [
            {"run_id": "abc123", "name": "run-1", "status": "FINISHED", "start_time": "2025-01-01"},
        ]

        with patch("dojo.cli.main.ExperimentTracker", return_value=mock_tracker):
            result = runner.invoke(app, ["experiments", "list", "--json"])

        assert result.exit_code == 0
        parsed = _json.loads(result.output)
        assert len(parsed) == 1
        assert parsed[0]["run_id"] == "abc123"

    def test_experiments_list_table_with_runs(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app

        runner = CliRunner()
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = [
            {"run_id": "abc123", "name": "run-1", "status": "FINISHED", "start_time": "2025-01-01"},
        ]

        with patch("dojo.cli.main.ExperimentTracker", return_value=mock_tracker):
            result = runner.invoke(app, ["experiments", "list"])

        assert result.exit_code == 0
        assert "abc123" in result.output

    def test_experiments_leaderboard_json_empty(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app
        import json as _json

        runner = CliRunner()
        mock_tracker = MagicMock(spec=ExperimentTracker)
        mock_tracker.list_runs.return_value = []

        with patch("dojo.cli.main.ExperimentTracker", return_value=mock_tracker):
            result = runner.invoke(app, ["experiments", "leaderboard", "--json"])

        assert result.exit_code == 0
        parsed = _json.loads(result.output)
        assert "primary_metric" in parsed
        assert parsed["entries"] == []
