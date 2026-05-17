"""Comprehensive mock-mode tests for SFT (d16) and QLoRA (d17) fine-tuning.

All tests run with DOJO_REAL_GPU=0 (default) so no GPU or model weights are
needed.  They exercise the simulation code paths, evidence emission, checkpoint
saving, ExperimentTracker wiring, and WandB callback firing.

Coverage targets (score 7 requirements):
  - Per-step evidence (step, loss, epoch fields)
  - Run start / end evidence (EXPERIMENT_RUN_STARTED, EXPERIMENT_RUN_COMPLETE)
  - Loss curve decreases across steps
  - ExperimentTracker.log_metrics_batch called with final_loss / steps_completed
  - WandB callback: experiment.wandb_step evidence when DOJO_WANDB=1
  - QLoRA-specific: quant.nf4_config evidence before training steps
  - SFT-specific: no quant.nf4_config evidence
  - Error handling: TRAINING_FAILED evidence on exception
  - Checkpoint: TRAINING_CHECKPOINT evidence with path field
  - Evidence chain ordering
"""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.types import HALProfile, HardwareTier
from dojo.train.base import FinetuneConfig, FinetuneResult


# ---------------------------------------------------------------------------
# Sentinel (same pattern as existing tests)
# ---------------------------------------------------------------------------

_SENTINEL = object()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_hal(
    tier: HardwareTier = HardwareTier.TIER_CPU,
    vram_bytes: int = 0,
) -> HALProfile:
    """Return a CPU HALProfile so VRAM check always passes (vram_bytes=0)."""
    return HALProfile(
        tier=tier,
        gpu_name="Mock CPU",
        vram_bytes=vram_bytes,
        compute_capability="0.0",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=0,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="0.0",
        profile_sha256="a" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_pack():
    from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest
    from dojo.pack.schema import SpecializationPack

    return SpecializationPack(
        pack_id="test-pack-final",
        version="1.0.0",
        schema_version="1.0",
        sample_count=10,
        format=PackFormat.ALPACA,
        splits={"train": 8, "val": 2},
        harvest_receipt="abc123",
        license=LicenseSpec(
            spdx_id="MIT",
            allows_derived_ft=True,
            allows_commercial_use=True,
            requires_attribution=False,
        ),
        provenance=ProvenanceManifest(
            harvest_organ_version="1.0",
            sources=("https://example.com",),
            collection_date_utc="2026-01-01",
            dedup_method="minhash",
            human_reviewed=False,
        ),
        signature=b"\x00" * 64,
        manifest_sha256="d" * 64,
        data_parquet_sha256="e" * 64,
    )


def _make_mock_mlflow() -> types.ModuleType:
    m = types.ModuleType("mlflow")
    mock_run = MagicMock()
    mock_run.info.run_id = "mock-run-id-final"
    mock_run.info.run_name = "mock-run-final"
    mock_run.info.status = "RUNNING"
    mock_run.info.start_time = "0"
    mock_run.data.metrics = {}
    mock_run.data.params = {}
    m.set_tracking_uri = MagicMock()
    m.set_experiment = MagicMock()
    m.start_run = MagicMock(return_value=mock_run)
    m.end_run = MagicMock()
    m.log_param = MagicMock()
    m.log_params = MagicMock()
    m.log_metric = MagicMock()
    m.log_artifact = MagicMock()
    m.get_run = MagicMock(return_value=mock_run)
    m.search_runs = MagicMock(return_value=[])
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    return m


# ---------------------------------------------------------------------------
# Mock-mode context manager (DOJO_REAL_GPU=0 — pure simulation path)
# ---------------------------------------------------------------------------

@contextmanager
def _mock_sim_env(
    training_mode: str = "qlora",
    num_epochs: int = 3,
    tmp_path: Path | None = None,
    wandb_enabled: bool = False,
    num_steps: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager that sets up the DOJO_REAL_GPU=0 simulation environment.

    Mocks mlflow only (to avoid network timeouts).  Does NOT set DOJO_REAL_GPU
    (defaults to 0) so the `not real_gpu` branch in finetune() is taken.
    """
    mock_mlflow = _make_mock_mlflow()

    output_dir = tmp_path or Path(os.environ.get("TEMP", "/tmp")) / "dojo_sim_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_kwargs: dict[str, Any] = dict(
        pack_id="mock-pack",
        base_model="mock/model-7b",
        output_dir=output_dir,
        num_epochs=num_epochs,
        training_mode=training_mode,
    )
    if num_steps is not None:
        cfg_kwargs["num_steps"] = num_steps

    cfg = FinetuneConfig(**cfg_kwargs)
    pack = _make_pack()
    hal = _make_hal()

    old_mlflow = sys.modules.get("mlflow", _SENTINEL)
    sys.modules["mlflow"] = mock_mlflow

    old_real_gpu = os.environ.get("DOJO_REAL_GPU")
    old_wandb = os.environ.get("DOJO_WANDB")
    old_mlflow_uri = os.environ.get("DOJO_MLFLOW_URI")

    # Ensure DOJO_REAL_GPU=0 (simulation path)
    os.environ["DOJO_REAL_GPU"] = "0"
    os.environ["DOJO_MLFLOW_URI"] = "http://127.0.0.1:19999"
    os.environ["DOJO_WANDB"] = "1" if wandb_enabled else "0"

    try:
        yield {
            "config": cfg,
            "pack": pack,
            "hal": hal,
            "mlflow": mock_mlflow,
        }
    finally:
        if old_mlflow is _SENTINEL:
            sys.modules.pop("mlflow", None)
        else:
            sys.modules["mlflow"] = old_mlflow

        if old_real_gpu is None:
            os.environ.pop("DOJO_REAL_GPU", None)
        else:
            os.environ["DOJO_REAL_GPU"] = old_real_gpu

        if old_wandb is None:
            os.environ.pop("DOJO_WANDB", None)
        else:
            os.environ["DOJO_WANDB"] = old_wandb

        if old_mlflow_uri is None:
            os.environ.pop("DOJO_MLFLOW_URI", None)
        else:
            os.environ["DOJO_MLFLOW_URI"] = old_mlflow_uri


# ---------------------------------------------------------------------------
# Helper: collect typed events
# ---------------------------------------------------------------------------

def _events_of_type(event_type: str) -> list[dict[str, Any]]:
    return [
        e for e in EvidenceLogger.get_events()
        if e.get("evidence", {}).get("type") == event_type
        or e.get("type") == event_type  # top-level type (raw event)
    ]


def _all_event_types() -> set[str]:
    evts = EvidenceLogger.get_events()
    types_found: set[str] = set()
    for e in evts:
        # Events are stored with evidence nested
        ev = e.get("evidence", e)
        t = ev.get("type", "")
        if t:
            types_found.add(t)
    return types_found


def _events_data(event_type: str) -> list[dict[str, Any]]:
    """Return the data payloads for all events matching event_type."""
    results = []
    for e in EvidenceLogger.get_events():
        ev = e.get("evidence", e)
        if ev.get("type") == event_type:
            results.append(ev.get("data", {}))
    return results


def _event_seq(event_type: str) -> list[int]:
    """Return seq numbers for events of the given type."""
    seqs = []
    for e in EvidenceLogger.get_events():
        ev = e.get("evidence", e)
        if ev.get("type") == event_type:
            seqs.append(e.get("seq", -1))
    return seqs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# SFT Tests (d16) — training_mode="sft"
# ---------------------------------------------------------------------------

class TestSFTSimulationPath:
    """8 tests for the SFT (training_mode='sft') simulation path."""

    def test_sft_completes_without_error(self, tmp_path):
        """finetune() with training_mode='sft', DOJO_REAL_GPU=0 completes."""
        with _mock_sim_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert isinstance(result, FinetuneResult)

    def test_sft_emits_experiment_run_complete(self, tmp_path):
        """EXPERIMENT_RUN_COMPLETE evidence emitted after SFT run."""
        with _mock_sim_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert EvidenceType.EXPERIMENT_RUN_COMPLETE.value in _all_event_types()

    def test_sft_per_step_evidence_has_step_field(self, tmp_path):
        """Per-step evidence contains a 'step' field."""
        with _mock_sim_env(training_mode="sft", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        step_data = _events_data(EvidenceType.TRAINING_STEP.value)
        assert len(step_data) > 0, "Expected at least one TRAINING_STEP event"
        for data in step_data:
            assert "step" in data, f"step field missing from {data}"

    def test_sft_per_step_evidence_has_loss_field(self, tmp_path):
        """Per-step evidence contains a 'loss' field that is a float."""
        with _mock_sim_env(training_mode="sft", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        step_data = _events_data(EvidenceType.TRAINING_STEP.value)
        assert len(step_data) > 0
        for data in step_data:
            assert "loss" in data, f"loss field missing from {data}"
            assert isinstance(data["loss"], (int, float)), f"loss is not numeric: {data['loss']}"

    def test_sft_steps_completed_matches_epochs(self, tmp_path):
        """steps_completed in result reflects num_epochs * grad_accum_steps."""
        num_epochs = 3
        with _mock_sim_env(training_mode="sft", num_epochs=num_epochs, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])
        # Simulation: steps_completed = num_epochs * grad_accum_steps
        expected = num_epochs * ctx["config"].grad_accum_steps
        assert result.steps_completed == expected

    def test_sft_experiment_tracker_records_run(self, tmp_path):
        """ExperimentTracker.log_metrics_batch is called with final_loss."""
        with _mock_sim_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            with patch("dojo.train.finetune.ExperimentTracker") as MockTracker:
                inst = MagicMock()
                MockTracker.return_value = inst
                inst.run.return_value.__enter__ = MagicMock(return_value="run-id")
                inst.run.return_value.__exit__ = MagicMock(return_value=False)
                from dojo.train.finetune import finetune
                finetune(ctx["config"], ctx["hal"], ctx["pack"])
        # log_metrics_batch should have been called with final_loss
        inst.log_metrics_batch.assert_called_once()
        call_args = inst.log_metrics_batch.call_args
        metrics_dict = call_args[0][0] if call_args[0] else call_args[1].get("metrics", {})
        assert "final_loss" in metrics_dict

    def test_sft_does_not_emit_nf4_config(self, tmp_path):
        """SFT run does NOT emit quant.nf4_config evidence (SFT uses full precision)."""
        with _mock_sim_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert "quant.nf4_config" not in _all_event_types()

    def test_sft_emits_experiment_run_started(self, tmp_path):
        """EXPERIMENT_RUN_STARTED evidence emitted before training steps."""
        with _mock_sim_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert EvidenceType.EXPERIMENT_RUN_STARTED.value in _all_event_types()


# ---------------------------------------------------------------------------
# QLoRA Tests (d17) — training_mode="qlora"
# ---------------------------------------------------------------------------

class TestQLoRASimulationPath:
    """6 tests for the QLoRA (training_mode='qlora') simulation path."""

    def test_qlora_completes_without_error(self, tmp_path):
        """finetune() with training_mode='qlora', DOJO_REAL_GPU=0 completes."""
        with _mock_sim_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert isinstance(result, FinetuneResult)

    def test_qlora_emits_nf4_config_before_training_steps(self, tmp_path):
        """quant.nf4_config evidence emitted before training steps in QLoRA run."""
        with _mock_sim_env(training_mode="qlora", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert "quant.nf4_config" in _all_event_types(), "quant.nf4_config not emitted in QLoRA run"

    def test_qlora_nf4_config_has_correct_quant_type(self, tmp_path):
        """quant.nf4_config evidence has bnb_4bit_quant_type='nf4'."""
        with _mock_sim_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        nf4_events = _events_data("quant.nf4_config")
        assert len(nf4_events) > 0, "No quant.nf4_config events found"
        for data in nf4_events:
            assert data.get("bnb_4bit_quant_type") == "nf4", f"Wrong quant type: {data}"

    def test_qlora_per_step_evidence_has_loss(self, tmp_path):
        """Per-step evidence has 'loss' field in QLoRA run."""
        with _mock_sim_env(training_mode="qlora", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        step_data = _events_data(EvidenceType.TRAINING_STEP.value)
        assert len(step_data) > 0
        for data in step_data:
            assert "loss" in data

    def test_qlora_experiment_tracker_records_metrics(self, tmp_path):
        """ExperimentTracker.log_metrics_batch called with steps_completed for QLoRA."""
        with _mock_sim_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            with patch("dojo.train.finetune.ExperimentTracker") as MockTracker:
                inst = MagicMock()
                MockTracker.return_value = inst
                inst.run.return_value.__enter__ = MagicMock(return_value="run-id")
                inst.run.return_value.__exit__ = MagicMock(return_value=False)
                from dojo.train.finetune import finetune
                finetune(ctx["config"], ctx["hal"], ctx["pack"])
        inst.log_metrics_batch.assert_called_once()
        call_args = inst.log_metrics_batch.call_args
        metrics_dict = call_args[0][0] if call_args[0] else call_args[1].get("metrics", {})
        assert "steps_completed" in metrics_dict

    def test_qlora_emits_experiment_run_complete(self, tmp_path):
        """EXPERIMENT_RUN_COMPLETE emitted after QLoRA run."""
        with _mock_sim_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert EvidenceType.EXPERIMENT_RUN_COMPLETE.value in _all_event_types()


# ---------------------------------------------------------------------------
# WandB Integration Tests
# ---------------------------------------------------------------------------

class TestWandBIntegration:
    """2 tests for WandB callback integration."""

    def test_wandb_enabled_emits_wandb_step_evidence(self, tmp_path):
        """With DOJO_WANDB=1, experiment.wandb_step evidence is emitted during training."""
        with _mock_sim_env(
            training_mode="qlora", num_epochs=2, tmp_path=tmp_path, wandb_enabled=True
        ) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert "experiment.wandb_step" in _all_event_types(), \
            "experiment.wandb_step not emitted when DOJO_WANDB=1"

    def test_wandb_disabled_no_wandb_step_evidence(self, tmp_path):
        """With DOJO_WANDB=0, no experiment.wandb_step evidence emitted."""
        with _mock_sim_env(
            training_mode="qlora", num_epochs=2, tmp_path=tmp_path, wandb_enabled=False
        ) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert "experiment.wandb_step" not in _all_event_types(), \
            "experiment.wandb_step should NOT be emitted when DOJO_WANDB=0"


# ---------------------------------------------------------------------------
# Evidence Chain Ordering Tests
# ---------------------------------------------------------------------------

class TestEvidenceChainOrdering:
    """3 tests verifying the ordering of evidence events."""

    def _seq_of_type(self, event_type: str) -> list[int]:
        """Return seq numbers for all events matching event_type."""
        seqs = []
        for i, e in enumerate(EvidenceLogger.get_events()):
            ev = e.get("evidence", e)
            if ev.get("type") == event_type:
                seqs.append(i)  # use insertion order as proxy for seq
        return seqs

    def test_run_started_before_training_steps(self, tmp_path):
        """EXPERIMENT_RUN_STARTED comes before first train.finetune.step event."""
        with _mock_sim_env(training_mode="qlora", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        started_idxs = self._seq_of_type(EvidenceType.EXPERIMENT_RUN_STARTED.value)
        step_idxs = self._seq_of_type(EvidenceType.TRAINING_STEP.value)
        assert started_idxs, "No EXPERIMENT_RUN_STARTED events"
        assert step_idxs, "No TRAINING_STEP events"
        assert min(started_idxs) < min(step_idxs), \
            "EXPERIMENT_RUN_STARTED must come before first TRAINING_STEP"

    def test_run_complete_after_all_training_steps(self, tmp_path):
        """EXPERIMENT_RUN_COMPLETE comes after all train.finetune.step events."""
        with _mock_sim_env(training_mode="qlora", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        complete_idxs = self._seq_of_type(EvidenceType.EXPERIMENT_RUN_COMPLETE.value)
        step_idxs = self._seq_of_type(EvidenceType.TRAINING_STEP.value)
        assert complete_idxs, "No EXPERIMENT_RUN_COMPLETE events"
        assert step_idxs, "No TRAINING_STEP events"
        assert max(complete_idxs) > max(step_idxs), \
            "EXPERIMENT_RUN_COMPLETE must come after last TRAINING_STEP"

    def test_qlora_nf4_config_before_first_training_step(self, tmp_path):
        """In QLoRA mode, quant.nf4_config comes before the first train.finetune.step."""
        with _mock_sim_env(training_mode="qlora", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        nf4_idxs = self._seq_of_type("quant.nf4_config")
        step_idxs = self._seq_of_type(EvidenceType.TRAINING_STEP.value)
        assert nf4_idxs, "No quant.nf4_config events in QLoRA run"
        assert step_idxs, "No TRAINING_STEP events"
        assert min(nf4_idxs) < min(step_idxs), \
            "quant.nf4_config must be emitted before first TRAINING_STEP"


# ---------------------------------------------------------------------------
# Error Path Test
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """1 test verifying error state evidence on training failure."""

    def test_training_failed_evidence_on_loss_divergence(self, tmp_path):
        """If training produces NaN loss, error evidence or exception is raised.

        The FSM transitions to FAILED and the error should be traceable
        through either a TRAINING_FAILED evidence event or by the exception
        propagating (LossDivergenceError).
        """
        from dojo.core.exceptions import LossDivergenceError

        with _mock_sim_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            # Patch _simulate_training_steps to return NaN loss
            with patch(
                "dojo.train.finetune._simulate_training_steps",
                return_value=(1, float("nan")),
            ):
                from dojo.train.finetune import finetune
                with pytest.raises((LossDivergenceError, Exception)) as exc_info:
                    finetune(ctx["config"], ctx["hal"], ctx["pack"])

        # Either exception propagates (LossDivergenceError) or TRAINING_FAILED evidence
        raised = exc_info.value
        # Acceptable: LossDivergenceError raised, or error evidence emitted
        event_types = _all_event_types()
        loss_divergence_raised = isinstance(raised, LossDivergenceError)
        failed_evidence_emitted = "train.training_failed" in event_types
        fsm_failed_emitted = any(
            "failed" in str(e.get("evidence", e).get("data", {})).lower()
            for e in EvidenceLogger.get_events()
        )
        assert loss_divergence_raised or failed_evidence_emitted or fsm_failed_emitted, \
            f"Expected LossDivergenceError or failure evidence. Got: {type(raised).__name__}"


# ---------------------------------------------------------------------------
# Additional Scenario Tests
# ---------------------------------------------------------------------------

class TestAdditionalScenarios:
    """Additional tests for checkpoint, loss curve, and result integrity."""

    def test_checkpoint_evidence_emitted_with_path_field(self, tmp_path):
        """TRAINING_CHECKPOINT evidence is emitted with a 'path' field."""
        with _mock_sim_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        checkpoint_data = _events_data(EvidenceType.TRAINING_CHECKPOINT.value)
        assert len(checkpoint_data) > 0, "No TRAINING_CHECKPOINT events emitted"
        for data in checkpoint_data:
            assert "path" in data, f"checkpoint event missing 'path': {data}"

    def test_loss_decreases_across_simulation_steps(self, tmp_path):
        """Simulated loss curve decreases across epochs (loss[n+1] < loss[n])."""
        with _mock_sim_env(training_mode="sft", num_epochs=4, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        step_data = _events_data(EvidenceType.TRAINING_STEP.value)
        losses = [d["loss"] for d in step_data if "loss" in d]
        assert len(losses) >= 2, f"Need at least 2 steps to verify loss curve, got: {losses}"
        # Loss should be strictly decreasing (each step is 80% of previous)
        for i in range(1, len(losses)):
            assert losses[i] < losses[i - 1], \
                f"Loss did not decrease at step {i}: {losses[i - 1]} -> {losses[i]}"

    def test_final_loss_in_training_complete_evidence(self, tmp_path):
        """TRAINING_COMPLETE evidence contains final_loss field."""
        with _mock_sim_env(training_mode="qlora", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        complete_data = _events_data(EvidenceType.TRAINING_COMPLETE.value)
        assert len(complete_data) > 0, "No TRAINING_COMPLETE events emitted"
        for data in complete_data:
            assert "final_loss" in data, f"final_loss missing from TRAINING_COMPLETE: {data}"

    def test_result_has_training_receipt(self, tmp_path):
        """FinetuneResult.training_receipt is a 64-char hex string."""
        with _mock_sim_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert isinstance(result.training_receipt, str)
        assert len(result.training_receipt) == 64
        assert all(c in "0123456789abcdef" for c in result.training_receipt)

    def test_sft_result_steps_completed_positive(self, tmp_path):
        """SFT result.steps_completed > 0."""
        with _mock_sim_env(training_mode="sft", num_epochs=1, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert result.steps_completed > 0

    def test_qlora_result_final_loss_is_positive_float(self, tmp_path):
        """QLoRA result.final_loss is a finite positive float."""
        import math
        with _mock_sim_env(training_mode="qlora", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])
        assert isinstance(result.final_loss, float)
        assert math.isfinite(result.final_loss)
        assert result.final_loss > 0

    def test_training_start_evidence_has_pack_id(self, tmp_path):
        """TRAINING_START evidence contains pack_id field matching the pack."""
        with _mock_sim_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        start_data = _events_data(EvidenceType.TRAINING_START.value)
        assert len(start_data) > 0, "No TRAINING_START events"
        assert start_data[0].get("pack_id") == "test-pack-final"

    def test_qlora_per_step_evidence_has_epoch_field(self, tmp_path):
        """Per-step evidence in QLoRA run contains 'epoch' field."""
        with _mock_sim_env(training_mode="qlora", num_epochs=2, tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])
        step_data = _events_data(EvidenceType.TRAINING_STEP.value)
        assert len(step_data) > 0
        for data in step_data:
            assert "epoch" in data, f"epoch field missing from step event: {data}"
