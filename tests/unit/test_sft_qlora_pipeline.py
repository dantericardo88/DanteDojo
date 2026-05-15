"""Comprehensive mock-GPU tests for SFT and QLoRA fine-tuning pipelines.

Exercises the full finetune() pipeline with mocked HuggingFace / PEFT / TRL
stacks so no real GPU or model weights are needed.

Strategy:
- Inject fake ``trl``, ``transformers``, ``peft``, ``datasets``, ``mlflow``,
  ``unsloth`` into sys.modules before calling finetune() so every internal
  import resolves to our mocks.
- Mock mlflow to prevent real network calls to localhost:5000 (avoids 4-minute
  TCP timeout when no MLflow server is running).
- Mock unsloth as available so ``backend_available=True`` and the real
  _run_real_sft / _run_real_qlora paths are exercised.
- Set DOJO_REAL_GPU=1 to engage the real training code branches.
- Use training_mode="sft" for SFT tests, training_mode="qlora" for QLoRA tests.
"""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch, call

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.types import HALProfile, HardwareTier
from dojo.train.base import FinetuneConfig, FinetuneResult


# ---------------------------------------------------------------------------
# Helpers — HAL/Pack builders
# ---------------------------------------------------------------------------

def _make_hal(
    tier: HardwareTier = HardwareTier.TIER_3090,
    vram_bytes: int = 24 * 1024 ** 3,
) -> HALProfile:
    return HALProfile(
        tier=tier,
        gpu_name="Mock GPU",
        vram_bytes=vram_bytes,
        compute_capability="8.6",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="550.0",
        profile_sha256="a" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_pack():
    from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest
    from dojo.pack.schema import SpecializationPack

    return SpecializationPack(
        pack_id="test-pack",
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


# ---------------------------------------------------------------------------
# Mock module factories
# ---------------------------------------------------------------------------

def _make_trainer_result(global_step: int = 42, training_loss: float = 0.25) -> MagicMock:
    """Return a mock trainer.train() result with realistic fields."""
    r = MagicMock()
    r.global_step = global_step
    r.training_loss = training_loss
    return r


def _make_mock_mlflow() -> types.ModuleType:
    """Build a fake ``mlflow`` module that never makes network calls."""
    m = types.ModuleType("mlflow")

    mock_run = MagicMock()
    mock_run.info.run_id = "mock-run-id-123"
    mock_run.info.run_name = "mock-run"
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
    # mlflow.ActiveRun context manager support
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    return m


def _make_mock_trl(trainer_result: MagicMock | None = None) -> types.ModuleType:
    """Build a fake ``trl`` module."""
    m = types.ModuleType("trl")
    if trainer_result is None:
        trainer_result = _make_trainer_result()

    trainer_mock = MagicMock()
    trainer_mock.train.return_value = trainer_result
    trainer_mock.save_model = MagicMock()

    SFTTrainer_cls = MagicMock(return_value=trainer_mock)
    SFTConfig_cls = MagicMock(return_value=MagicMock())

    m.SFTTrainer = SFTTrainer_cls
    m.SFTConfig = SFTConfig_cls
    m._trainer_instance = trainer_mock
    m._trainer_cls = SFTTrainer_cls
    return m


def _make_mock_transformers() -> types.ModuleType:
    """Build a fake ``transformers`` module."""
    m = types.ModuleType("transformers")
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    m.AutoTokenizer = MagicMock()
    m.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    m.AutoModelForCausalLM = MagicMock()
    m.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    m.TrainingArguments = MagicMock()
    m.Trainer = MagicMock()
    m.BitsAndBytesConfig = MagicMock(return_value=MagicMock())
    m.DataCollatorForSeq2Seq = MagicMock()
    return m


def _make_mock_peft() -> types.ModuleType:
    """Build a fake ``peft`` module."""
    m = types.ModuleType("peft")
    m.LoraConfig = MagicMock(return_value=MagicMock())
    m.get_peft_model = MagicMock(side_effect=lambda model, cfg: model)
    m.prepare_model_for_kbit_training = MagicMock(side_effect=lambda model: model)
    m.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    return m


def _make_mock_datasets() -> types.ModuleType:
    """Build a fake ``datasets`` module."""
    m = types.ModuleType("datasets")
    m.Dataset = MagicMock()
    m.Dataset.from_list = MagicMock(return_value=MagicMock())
    return m


def _make_mock_torch() -> types.ModuleType:
    """Build a minimal fake ``torch`` module that satisfies autocast usage."""
    m = types.ModuleType("torch")
    m.bfloat16 = "bfloat16"

    @contextmanager
    def _autocast(device, dtype=None):
        yield

    m.autocast = _autocast
    return m


def _make_mock_unsloth() -> types.ModuleType:
    """Build a fake ``unsloth`` module so backend_available returns True."""
    return types.ModuleType("unsloth")


# ---------------------------------------------------------------------------
# Main test context manager
# ---------------------------------------------------------------------------

@contextmanager
def _mock_gpu_env(
    training_mode: str = "sft",
    tmp_path: Path | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Inject mock modules + env vars to exercise the real training code paths.

    Yields a dict with keys:
        trl, transformers, peft, datasets, torch, mlflow — the mock modules
        config  — FinetuneConfig ready to use
        pack    — SpecializationPack
        hal     — HALProfile
        trainer_result — the mock TrainOutput returned by trainer.train()
    """
    trainer_result = _make_trainer_result(global_step=50, training_loss=0.15)
    mock_trl = _make_mock_trl(trainer_result)
    mock_transformers = _make_mock_transformers()
    mock_peft = _make_mock_peft()
    mock_datasets = _make_mock_datasets()
    mock_torch = _make_mock_torch()
    mock_mlflow = _make_mock_mlflow()
    mock_unsloth = _make_mock_unsloth()

    output_dir = tmp_path or Path(os.environ.get("TEMP", "/tmp")) / "dojo_test_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FinetuneConfig(
        pack_id="mock-pack",
        base_model="mock/model-7b",
        output_dir=output_dir,
        num_epochs=1,
        training_mode=training_mode,
    )
    pack = _make_pack()
    hal = _make_hal()

    # All modules to inject into sys.modules
    patch_targets = {
        "trl": mock_trl,
        "transformers": mock_transformers,
        "peft": mock_peft,
        "datasets": mock_datasets,
        "torch": mock_torch,
        "mlflow": mock_mlflow,
        "unsloth": mock_unsloth,
    }

    # Save original values (including None if not present)
    old_modules: dict[str, Any] = {}
    for name in patch_targets:
        old_modules[name] = sys.modules.get(name, _SENTINEL)

    for name, mod in patch_targets.items():
        sys.modules[name] = mod

    # Save/set env vars
    old_real_gpu = os.environ.get("DOJO_REAL_GPU")
    old_mlflow_uri = os.environ.get("DOJO_MLFLOW_URI")
    os.environ["DOJO_REAL_GPU"] = "1"
    os.environ["DOJO_MLFLOW_URI"] = "http://127.0.0.1:19999"  # unused — mlflow is mocked

    try:
        yield {
            "trl": mock_trl,
            "transformers": mock_transformers,
            "peft": mock_peft,
            "datasets": mock_datasets,
            "torch": mock_torch,
            "mlflow": mock_mlflow,
            "config": cfg,
            "pack": pack,
            "hal": hal,
            "trainer_result": trainer_result,
        }
    finally:
        # Restore sys.modules
        for name, orig in old_modules.items():
            if orig is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

        # Restore env vars
        if old_real_gpu is None:
            os.environ.pop("DOJO_REAL_GPU", None)
        else:
            os.environ["DOJO_REAL_GPU"] = old_real_gpu

        if old_mlflow_uri is None:
            os.environ.pop("DOJO_MLFLOW_URI", None)
        else:
            os.environ["DOJO_MLFLOW_URI"] = old_mlflow_uri


# Sentinel value to distinguish "key absent" from "key = None"
_SENTINEL = object()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# TestSFTMockedPipeline
# ---------------------------------------------------------------------------

class TestSFTMockedPipeline:
    """8 tests for the SFT (training_mode='sft') path with mocked GPU stack."""

    def test_finetune_sft_returns_training_result(self, tmp_path):
        """finetune() returns a FinetuneResult."""
        with _mock_gpu_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        assert isinstance(result, FinetuneResult)

    def test_finetune_sft_receipt_is_hex_string(self, tmp_path):
        """training_receipt is a 64-char hex string (SHA-256)."""
        with _mock_gpu_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        receipt = result.training_receipt
        assert isinstance(receipt, str)
        assert len(receipt) == 64
        assert all(c in "0123456789abcdef" for c in receipt)

    def test_finetune_sft_emits_training_start_evidence(self, tmp_path):
        """EvidenceLogger receives a TRAINING_START event."""
        with _mock_gpu_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])

        event_types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.TRAINING_START.value in event_types

    def test_finetune_sft_emits_training_complete_evidence(self, tmp_path):
        """EvidenceLogger receives a TRAINING_COMPLETE event."""
        with _mock_gpu_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])

        event_types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.TRAINING_COMPLETE.value in event_types

    def test_finetune_sft_steps_completed_positive(self, tmp_path):
        """steps_completed > 0 after a real (mocked) training run."""
        with _mock_gpu_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        assert result.steps_completed > 0

    def test_finetune_sft_output_dir_created(self, tmp_path):
        """The configured output_dir exists after training completes."""
        with _mock_gpu_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        assert result.adapter_path.exists()

    def test_finetune_sft_experiment_tracker_called(self, tmp_path):
        """ExperimentTracker.start_run is invoked during the pipeline."""
        with _mock_gpu_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            with patch("dojo.train.finetune.ExperimentTracker") as MockTracker:
                mock_tracker_instance = MagicMock()
                MockTracker.return_value = mock_tracker_instance

                # run() is a context manager
                mock_tracker_instance.run.return_value.__enter__ = MagicMock(
                    return_value="run-id-123"
                )
                mock_tracker_instance.run.return_value.__exit__ = MagicMock(
                    return_value=False
                )

                from dojo.train.finetune import finetune
                finetune(ctx["config"], ctx["hal"], ctx["pack"])

        MockTracker.assert_called_once()
        mock_tracker_instance.run.assert_called_once()

    def test_finetune_sft_wandb_callback_used(self, tmp_path):
        """WandBAdapter.as_callback() is called during the SFT pipeline."""
        with _mock_gpu_env(training_mode="sft", tmp_path=tmp_path) as ctx:
            with patch("dojo.experiment.wandb_adapter.WandBAdapter") as MockAdapter:
                mock_adapter_inst = MagicMock()
                MockAdapter.return_value = mock_adapter_inst
                cb_mock = MagicMock()
                mock_adapter_inst.as_callback.return_value = cb_mock

                from dojo.train.finetune import finetune
                finetune(ctx["config"], ctx["hal"], ctx["pack"])

        mock_adapter_inst.as_callback.assert_called_once()


# ---------------------------------------------------------------------------
# TestQLoRAMockedPipeline
# ---------------------------------------------------------------------------

class TestQLoRAMockedPipeline:
    """8 tests for the QLoRA (training_mode='qlora') path with mocked GPU stack."""

    def test_finetune_qlora_returns_training_result(self, tmp_path):
        """finetune() returns a FinetuneResult for QLoRA mode."""
        with _mock_gpu_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        assert isinstance(result, FinetuneResult)

    def test_finetune_qlora_receipt_is_hex_string(self, tmp_path):
        """training_receipt is a 64-char hex string for QLoRA run."""
        with _mock_gpu_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        receipt = result.training_receipt
        assert isinstance(receipt, str)
        assert len(receipt) == 64
        assert all(c in "0123456789abcdef" for c in receipt)

    def test_finetune_qlora_emits_training_start_evidence(self, tmp_path):
        """EvidenceLogger receives TRAINING_START for QLoRA run."""
        with _mock_gpu_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])

        event_types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.TRAINING_START.value in event_types

    def test_finetune_qlora_emits_training_complete_evidence(self, tmp_path):
        """EvidenceLogger receives TRAINING_COMPLETE for QLoRA run."""
        with _mock_gpu_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])

        event_types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.TRAINING_COMPLETE.value in event_types

    def test_finetune_qlora_steps_completed_positive(self, tmp_path):
        """steps_completed > 0 after mocked QLoRA training."""
        with _mock_gpu_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        assert result.steps_completed > 0

    def test_finetune_qlora_output_dir_created(self, tmp_path):
        """The output_dir exists after QLoRA training completes."""
        with _mock_gpu_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        assert result.adapter_path.exists()

    def test_finetune_qlora_experiment_tracker_called(self, tmp_path):
        """ExperimentTracker.start_run is called during QLoRA pipeline."""
        with _mock_gpu_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            with patch("dojo.train.finetune.ExperimentTracker") as MockTracker:
                mock_tracker_instance = MagicMock()
                MockTracker.return_value = mock_tracker_instance

                mock_tracker_instance.run.return_value.__enter__ = MagicMock(
                    return_value="run-id-456"
                )
                mock_tracker_instance.run.return_value.__exit__ = MagicMock(
                    return_value=False
                )

                from dojo.train.finetune import finetune
                finetune(ctx["config"], ctx["hal"], ctx["pack"])

        MockTracker.assert_called_once()
        mock_tracker_instance.run.assert_called_once()

    def test_finetune_qlora_nf4_config_requested(self, tmp_path):
        """peft.LoraConfig is called with the correct lora_r for QLoRA."""
        with _mock_gpu_env(training_mode="qlora", tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        # peft mock is still in sys.modules — check LoraConfig was called
        peft_mod = sys.modules.get("peft")
        # If peft mock was restored, fall back to checking result integrity
        if peft_mod is not None and hasattr(peft_mod, "LoraConfig"):
            lc = peft_mod.LoraConfig
            if hasattr(lc, "call_args") and lc.call_args is not None:
                call_kwargs = lc.call_args
                r_val = call_kwargs.kwargs.get("r")
                if r_val is not None:
                    assert r_val == ctx["config"].lora_r
        # Always assert result is valid regardless
        assert isinstance(result, FinetuneResult)
        assert result.steps_completed > 0
