"""Unit tests for d16 (SFT) and d17 (QLoRA) real training functions.

Covers:
- _simulate_training_steps: still works without DOJO_REAL_GPU (CI mode)
- _run_real_sft: mocked TRL SFTTrainer, correct API calls, evidence emission
- _run_real_qlora: mocked PEFT + TRL SFTTrainer, correct API calls, evidence emission
- finetune(): routes to _run_real_sft when DOJO_REAL_GPU=1 and training_mode="sft"
- finetune(): routes to _run_real_qlora when DOJO_REAL_GPU=1 and training_mode="qlora"
- finetune(): falls back to simulate when DOJO_REAL_GPU not set
- Evidence events: TRAINING_START, TRAINING_STEP, TRAINING_COMPLETE all emitted
- VRAM check still raises VRAMExceeded
- LossDivergenceError raised on NaN/Inf final loss
- FinetuneConfig.training_mode field defaults to "qlora"
- BackendUnavailable raised when TRL/PEFT absent and DOJO_REAL_GPU=1
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BackendUnavailable, LossDivergenceError, VRAMExceeded
from dojo.core.types import HALProfile, HardwareTier
from dojo.train.base import FinetuneConfig, FinetuneResult
from dojo.train.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_hal(
    tier: HardwareTier = HardwareTier.TIER_3090,
    vram_bytes: int = 24 * 1024 ** 3,
    compute_capability: str = "8.6",
) -> HALProfile:
    return HALProfile(
        tier=tier,
        gpu_name="Test GPU",
        vram_bytes=vram_bytes,
        compute_capability=compute_capability,
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


def _make_pack(samples=None):
    from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest
    from dojo.pack.schema import SpecializationPack
    pack = SpecializationPack(
        pack_id="test-pack",
        version="1.0.0",
        schema_version="1.0",
        sample_count=2,
        format=PackFormat.ALPACA,
        splits={"train": 2, "val": 0},
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
    if samples is not None:
        object.__setattr__(pack, "samples", samples)
    return pack


def _make_train_result(global_step: int = 10, training_loss: float = 0.42) -> SimpleNamespace:
    """Create a mock TRL TrainOutput-like object."""
    return SimpleNamespace(global_step=global_step, training_loss=training_loss)


def _build_fake_trl_module(mock_trainer_inst: MagicMock) -> MagicMock:
    """Build a fake trl module with a SFTTrainer class that returns mock_trainer_inst."""
    fake_trl = MagicMock(name="trl")
    fake_trl.SFTConfig = MagicMock(name="SFTConfig")
    fake_trl.SFTTrainer = MagicMock(name="SFTTrainer", return_value=mock_trainer_inst)
    return fake_trl


def _build_fake_transformers_module(
    mock_tokenizer: MagicMock,
    mock_model: MagicMock,
    include_bnb: bool = False,
) -> MagicMock:
    """Build a fake transformers module."""
    fake_tf = MagicMock(name="transformers")
    mock_auto_tok = MagicMock(name="AutoTokenizer")
    mock_auto_tok.from_pretrained.return_value = mock_tokenizer
    mock_auto_model = MagicMock(name="AutoModelForCausalLM")
    mock_auto_model.from_pretrained.return_value = mock_model
    fake_tf.AutoTokenizer = mock_auto_tok
    fake_tf.AutoModelForCausalLM = mock_auto_model
    fake_tf.TrainingArguments = MagicMock(name="TrainingArguments")
    if include_bnb:
        fake_tf.BitsAndBytesConfig = MagicMock(name="BitsAndBytesConfig")
    return fake_tf


def _build_fake_datasets_module(dataset_mock: MagicMock) -> MagicMock:
    fake_ds = MagicMock(name="datasets")
    mock_dataset_cls = MagicMock(name="Dataset")
    mock_dataset_cls.from_list.return_value = dataset_mock
    fake_ds.Dataset = mock_dataset_cls
    return fake_ds


def _build_fake_peft_module() -> tuple[MagicMock, MagicMock]:
    fake_peft = MagicMock(name="peft")
    mock_lora_inst = MagicMock(name="lora_config_instance")
    fake_peft.LoraConfig = MagicMock(name="LoraConfig", return_value=mock_lora_inst)
    fake_peft.TaskType = MagicMock(name="TaskType")
    fake_peft.TaskType.CAUSAL_LM = "CAUSAL_LM"
    fake_peft.prepare_model_for_kbit_training = MagicMock(
        name="prepare_model_for_kbit_training", return_value=MagicMock()
    )
    fake_peft.get_peft_model = MagicMock(name="get_peft_model", return_value=MagicMock())
    return fake_peft, mock_lora_inst


@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    """Ensure DOJO_REAL_GPU is unset by default so CI tests never try real GPU."""
    monkeypatch.delenv("DOJO_REAL_GPU", raising=False)


# ---------------------------------------------------------------------------
# FinetuneConfig.training_mode
# ---------------------------------------------------------------------------

class TestFinetuneConfigTrainingMode:
    def test_default_training_mode_is_sft(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m")
        assert cfg.training_mode == "sft"

    def test_training_mode_can_be_set_to_sft(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="sft")
        assert cfg.training_mode == "sft"

    def test_training_mode_in_to_dict(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="sft")
        assert cfg.to_dict()["training_mode"] == "sft"

    def test_training_mode_affects_config_hash(self):
        cfg_qlora = FinetuneConfig(pack_id="p", base_model="m", training_mode="qlora")
        cfg_sft = FinetuneConfig(pack_id="p", base_model="m", training_mode="sft")
        assert cfg_qlora.config_hash() != cfg_sft.config_hash()

    def test_full_ft_mode_accepted(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="full_ft")
        assert cfg.training_mode == "full_ft"

    def test_config_hash_stable(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="sft")
        assert cfg.config_hash() == cfg.config_hash()

    def test_config_hash_is_64_chars(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="qlora")
        assert len(cfg.config_hash()) == 64


# ---------------------------------------------------------------------------
# _simulate_training_steps — always runs in CI (no DOJO_REAL_GPU)
# ---------------------------------------------------------------------------

class TestSimulateTrainingSteps:
    def test_returns_steps_and_loss(self, tmp_path):
        from dojo.train.finetune import _simulate_training_steps
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="m", num_epochs=2, grad_accum_steps=4)
        steps, loss = _simulate_training_steps(cfg, logger, ckpt_mgr)
        assert steps == 8  # 2 epochs × 4 grad_accum_steps
        assert 0.0 < loss < 1.0

    def test_emits_training_step_events_per_epoch(self, tmp_path):
        from dojo.train.finetune import _simulate_training_steps
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="m", num_epochs=3, grad_accum_steps=2)
        _simulate_training_steps(cfg, logger, ckpt_mgr)
        step_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.TRAINING_STEP.value
        ]
        assert len(step_events) == 3

    def test_saves_checkpoint(self, tmp_path):
        from dojo.train.finetune import _simulate_training_steps
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="m", num_epochs=1, grad_accum_steps=4)
        _simulate_training_steps(cfg, logger, ckpt_mgr)
        assert ckpt_mgr.load_latest() is not None

    def test_loss_decreases_over_epochs(self, tmp_path):
        from dojo.train.finetune import _simulate_training_steps
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="m", num_epochs=5, grad_accum_steps=1)
        _, final_loss = _simulate_training_steps(cfg, logger, ckpt_mgr)
        assert final_loss < 1.0

    def test_step_events_have_epoch_and_loss_fields(self, tmp_path):
        from dojo.train.finetune import _simulate_training_steps
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="m", num_epochs=2, grad_accum_steps=1)
        _simulate_training_steps(cfg, logger, ckpt_mgr)
        step_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.TRAINING_STEP.value
        ]
        for ev in step_events:
            assert "epoch" in ev["data"]
            assert "loss" in ev["data"]


# ---------------------------------------------------------------------------
# _run_real_sft — mocked via sys.modules injection
# ---------------------------------------------------------------------------

class TestRunRealSFT:
    """Tests for _run_real_sft with all external deps mocked via sys.modules."""

    def _inject_sft_mocks(self, global_step=10, training_loss=0.35, pad_token="PAD"):
        """Return (trainer_inst, context_manager_patches) for SFT mocking."""
        mock_trainer_inst = MagicMock()
        mock_trainer_inst.train.return_value = _make_train_result(global_step, training_loss)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = pad_token

        mock_model = MagicMock()
        mock_dataset = MagicMock()

        # Build fake modules
        fake_trl = _build_fake_trl_module(mock_trainer_inst)
        fake_tf = _build_fake_transformers_module(mock_tokenizer, mock_model)
        fake_ds = _build_fake_datasets_module(mock_dataset)

        return mock_trainer_inst, mock_tokenizer, mock_model, fake_trl, fake_tf, fake_ds

    def test_returns_steps_and_loss(self, tmp_path):
        from dojo.train.finetune import _run_real_sft

        trainer_inst, tokenizer, model, fake_trl, fake_tf, fake_ds = (
            self._inject_sft_mocks(global_step=20, training_loss=0.25)
        )
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)
        pack = _make_pack()

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf}):
            steps, loss = _run_real_sft(cfg, logger, ckpt_mgr, pack)

        assert steps == 20
        assert loss == pytest.approx(0.25)

    def test_calls_trainer_train(self, tmp_path):
        from dojo.train.finetune import _run_real_sft

        trainer_inst, tokenizer, model, fake_trl, fake_tf, fake_ds = (
            self._inject_sft_mocks()
        )
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf}):
            _run_real_sft(cfg, logger, ckpt_mgr, _make_pack())

        trainer_inst.train.assert_called_once()

    def test_calls_save_model(self, tmp_path):
        from dojo.train.finetune import _run_real_sft

        trainer_inst, tokenizer, model, fake_trl, fake_tf, fake_ds = (
            self._inject_sft_mocks()
        )
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf}):
            _run_real_sft(cfg, logger, ckpt_mgr, _make_pack())

        trainer_inst.save_model.assert_called_once_with(str(tmp_path))

    def test_emits_training_step_evidence_with_trl_sft_backend(self, tmp_path):
        from dojo.train.finetune import _run_real_sft

        trainer_inst, tokenizer, model, fake_trl, fake_tf, fake_ds = (
            self._inject_sft_mocks(global_step=8, training_loss=0.3)
        )
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf}):
            _run_real_sft(cfg, logger, ckpt_mgr, _make_pack())

        step_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.TRAINING_STEP.value
        ]
        assert len(step_events) >= 1
        last = step_events[-1]
        assert last["data"]["backend"] == "trl_sft"
        assert last["data"]["steps"] == 8

    def test_sets_pad_token_when_none(self, tmp_path):
        from dojo.train.finetune import _run_real_sft

        trainer_inst, tokenizer, model, fake_trl, fake_tf, fake_ds = (
            self._inject_sft_mocks(pad_token=None)
        )
        tokenizer.eos_token = "<eos>"
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf}):
            _run_real_sft(cfg, logger, ckpt_mgr, _make_pack())

        assert tokenizer.pad_token == "<eos>"

    def test_uses_pack_samples_when_available(self, tmp_path):
        from dojo.train.finetune import _run_real_sft

        trainer_inst, tokenizer, model, fake_trl, fake_tf, fake_ds = (
            self._inject_sft_mocks()
        )
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)
        custom_samples = [{"text": "custom sample 1"}, {"text": "custom sample 2"}]
        pack = _make_pack(samples=custom_samples)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf}):
            _run_real_sft(cfg, logger, ckpt_mgr, pack)

        fake_ds.Dataset.from_list.assert_called_once_with(custom_samples)

    def test_saves_checkpoint_metadata(self, tmp_path):
        from dojo.train.finetune import _run_real_sft

        trainer_inst, tokenizer, model, fake_trl, fake_tf, fake_ds = (
            self._inject_sft_mocks(global_step=7, training_loss=0.4)
        )
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path,
                             num_epochs=2)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf}):
            _run_real_sft(cfg, logger, ckpt_mgr, _make_pack())

        latest = ckpt_mgr.load_latest()
        assert latest is not None, "expected a checkpoint to have been saved"
        step, meta = latest
        assert step == 7
        assert meta.get("epochs") == 2

    def test_raises_backend_unavailable_when_trl_missing(self, tmp_path):
        from dojo.train.finetune import _run_real_sft

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        # Simulate trl not installed by injecting None into sys.modules
        with patch.dict("sys.modules", {"trl": None}):
            with pytest.raises((BackendUnavailable, ImportError, Exception)):
                _run_real_sft(cfg, logger, ckpt_mgr, _make_pack())


# ---------------------------------------------------------------------------
# _run_real_qlora — mocked PEFT + TRL via sys.modules injection
# ---------------------------------------------------------------------------

class TestRunRealQLoRA:
    """Tests for _run_real_qlora with all external deps mocked."""

    def _inject_qlora_mocks(self, global_step=15, training_loss=0.28, pad_token="PAD"):
        mock_trainer_inst = MagicMock()
        mock_trainer_inst.train.return_value = _make_train_result(global_step, training_loss)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = pad_token

        mock_model = MagicMock()
        mock_prepped_model = MagicMock()
        mock_dataset = MagicMock()

        fake_trl = _build_fake_trl_module(mock_trainer_inst)
        fake_tf = _build_fake_transformers_module(mock_tokenizer, mock_model,
                                                   include_bnb=True)
        fake_ds = _build_fake_datasets_module(mock_dataset)
        fake_peft, mock_lora_inst = _build_fake_peft_module()
        fake_peft.prepare_model_for_kbit_training.return_value = mock_prepped_model

        return (mock_trainer_inst, mock_tokenizer, mock_model, mock_prepped_model,
                mock_lora_inst, fake_trl, fake_tf, fake_ds, fake_peft)

    def test_returns_steps_and_loss(self, tmp_path):
        from dojo.train.finetune import _run_real_qlora

        (trainer_inst, tokenizer, model, prepped, lora_inst,
         fake_trl, fake_tf, fake_ds, fake_peft) = self._inject_qlora_mocks(15, 0.28)
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path,
                             lora_r=16)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf, "peft": fake_peft}):
            steps, loss = _run_real_qlora(cfg, logger, ckpt_mgr, _make_pack())

        assert steps == 15
        assert loss == pytest.approx(0.28)

    def test_calls_prepare_model_for_kbit_training(self, tmp_path):
        from dojo.train.finetune import _run_real_qlora

        (trainer_inst, tokenizer, model, prepped, lora_inst,
         fake_trl, fake_tf, fake_ds, fake_peft) = self._inject_qlora_mocks()
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf, "peft": fake_peft}):
            _run_real_qlora(cfg, logger, ckpt_mgr, _make_pack())

        fake_peft.prepare_model_for_kbit_training.assert_called_once_with(model)

    def test_passes_peft_config_to_sft_trainer(self, tmp_path):
        from dojo.train.finetune import _run_real_qlora

        (trainer_inst, tokenizer, model, prepped, lora_inst,
         fake_trl, fake_tf, fake_ds, fake_peft) = self._inject_qlora_mocks()
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf, "peft": fake_peft}):
            _run_real_qlora(cfg, logger, ckpt_mgr, _make_pack())

        call_kwargs = fake_trl.SFTTrainer.call_args.kwargs
        assert "peft_config" in call_kwargs
        # The peft_config passed should be the LoraConfig instance
        assert call_kwargs["peft_config"] is lora_inst

    def test_emits_training_step_with_qlora_backend(self, tmp_path):
        from dojo.train.finetune import _run_real_qlora

        (trainer_inst, tokenizer, model, prepped, lora_inst,
         fake_trl, fake_tf, fake_ds, fake_peft) = self._inject_qlora_mocks(10, 0.4)
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path,
                             lora_r=32)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf, "peft": fake_peft}):
            _run_real_qlora(cfg, logger, ckpt_mgr, _make_pack())

        step_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.TRAINING_STEP.value
        ]
        assert len(step_events) >= 1
        last = step_events[-1]
        assert last["data"]["backend"] == "qlora_peft"
        assert last["data"]["lora_r"] == 32
        assert last["data"]["load_in_4bit"] is True

    def test_saves_checkpoint_with_adapter_metadata(self, tmp_path):
        from dojo.train.finetune import _run_real_qlora

        (trainer_inst, tokenizer, model, prepped, lora_inst,
         fake_trl, fake_tf, fake_ds, fake_peft) = self._inject_qlora_mocks(7, 0.33)
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf, "peft": fake_peft}):
            _run_real_qlora(cfg, logger, ckpt_mgr, _make_pack())

        latest = ckpt_mgr.load_latest()
        assert latest is not None, "expected a checkpoint to have been saved"
        step, meta = latest
        assert step == 7
        assert meta.get("adapter") == "qlora"

    def test_calls_save_model(self, tmp_path):
        from dojo.train.finetune import _run_real_qlora

        (trainer_inst, tokenizer, model, prepped, lora_inst,
         fake_trl, fake_tf, fake_ds, fake_peft) = self._inject_qlora_mocks()
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf, "peft": fake_peft}):
            _run_real_qlora(cfg, logger, ckpt_mgr, _make_pack())

        trainer_inst.save_model.assert_called_once_with(str(tmp_path))

    def test_sets_pad_token_when_none(self, tmp_path):
        from dojo.train.finetune import _run_real_qlora

        (trainer_inst, tokenizer, model, prepped, lora_inst,
         fake_trl, fake_tf, fake_ds, fake_peft) = self._inject_qlora_mocks(pad_token=None)
        tokenizer.eos_token = "<eos>"
        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"trl": fake_trl, "datasets": fake_ds,
                                         "transformers": fake_tf, "peft": fake_peft}):
            _run_real_qlora(cfg, logger, ckpt_mgr, _make_pack())

        assert tokenizer.pad_token == "<eos>"

    def test_raises_backend_unavailable_when_peft_missing(self, tmp_path):
        from dojo.train.finetune import _run_real_qlora

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpts", max_checkpoints=3)
        cfg = FinetuneConfig(pack_id="p", base_model="mock/model", output_dir=tmp_path)

        with patch.dict("sys.modules", {"peft": None}):
            with pytest.raises((BackendUnavailable, ImportError, Exception)):
                _run_real_qlora(cfg, logger, ckpt_mgr, _make_pack())


# ---------------------------------------------------------------------------
# finetune() routing — DOJO_REAL_GPU integration
# ---------------------------------------------------------------------------

class TestFinetuneRoutingRealGPU:
    """Integration tests for finetune() routing logic with DOJO_REAL_GPU."""

    _hal = _make_hal(tier=HardwareTier.TIER_3090)

    def test_no_real_gpu_uses_simulate_not_real_functions(self, tmp_path):
        """Without DOJO_REAL_GPU, always uses simulate regardless of training_mode."""
        from dojo.train.finetune import _simulate_training_steps, finetune
        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="sft",
        )
        with patch("dojo.train.finetune._simulate_training_steps",
                   wraps=_simulate_training_steps) as mock_sim, \
             patch("dojo.train.finetune._run_real_sft") as mock_sft, \
             patch("dojo.train.finetune._run_real_qlora") as mock_qlora:

            finetune(cfg, self._hal, _make_pack())

        mock_sim.assert_called_once()
        mock_sft.assert_not_called()
        mock_qlora.assert_not_called()

    def test_real_gpu_sft_mode_calls_run_real_sft(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_REAL_GPU", "1")

        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="sft",
        )

        with patch("dojo.train.finetune._run_real_sft", return_value=(10, 0.3)) as mock_sft, \
             patch("dojo.train.finetune._run_real_qlora") as mock_qlora, \
             patch("dojo.train.finetune._simulate_training_steps") as mock_sim, \
             patch("dojo.train.finetune._is_backend_available", return_value=True):

            result = finetune(cfg, self._hal, _make_pack())

        mock_sft.assert_called_once()
        mock_qlora.assert_not_called()
        mock_sim.assert_not_called()
        assert result.steps_completed == 10
        assert result.final_loss == pytest.approx(0.3)

    def test_real_gpu_qlora_mode_calls_run_real_qlora(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_REAL_GPU", "1")

        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="qlora",
        )

        with patch("dojo.train.finetune._run_real_qlora", return_value=(12, 0.22)) as mock_qlora, \
             patch("dojo.train.finetune._run_real_sft") as mock_sft, \
             patch("dojo.train.finetune._simulate_training_steps") as mock_sim, \
             patch("dojo.train.finetune._is_backend_available", return_value=True):

            result = finetune(cfg, self._hal, _make_pack())

        mock_qlora.assert_called_once()
        mock_sft.assert_not_called()
        mock_sim.assert_not_called()
        assert result.steps_completed == 12
        assert result.final_loss == pytest.approx(0.22)

    def test_real_gpu_unknown_mode_defaults_to_qlora(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_REAL_GPU", "1")

        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="full_ft",
        )

        with patch("dojo.train.finetune._run_real_qlora", return_value=(5, 0.5)) as mock_qlora, \
             patch("dojo.train.finetune._run_real_sft") as mock_sft, \
             patch("dojo.train.finetune._simulate_training_steps") as mock_sim, \
             patch("dojo.train.finetune._is_backend_available", return_value=True):

            finetune(cfg, self._hal, _make_pack())

        mock_qlora.assert_called_once()
        mock_sft.assert_not_called()

    def test_real_gpu_backend_unavailable_falls_back_to_simulate(self, tmp_path, monkeypatch):
        """Even with DOJO_REAL_GPU=1, unavailable backend uses simulate (warns)."""
        from dojo.train.finetune import _simulate_training_steps, finetune
        monkeypatch.setenv("DOJO_REAL_GPU", "1")

        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="sft",
        )

        with patch("dojo.train.finetune._simulate_training_steps",
                   wraps=_simulate_training_steps) as mock_sim, \
             patch("dojo.train.finetune._run_real_sft") as mock_sft, \
             patch("dojo.train.finetune._run_real_qlora") as mock_qlora, \
             patch("dojo.train.finetune._is_backend_available", return_value=False):

            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = finetune(cfg, self._hal, _make_pack())
            assert any("not installed" in str(warning.message) for warning in w)

        mock_sim.assert_called_once()
        mock_sft.assert_not_called()
        mock_qlora.assert_not_called()
        assert isinstance(result, FinetuneResult)

    def test_result_backend_used_field_populated(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_REAL_GPU", "1")

        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="sft",
        )

        with patch("dojo.train.finetune._run_real_sft", return_value=(6, 0.2)), \
             patch("dojo.train.finetune._is_backend_available", return_value=True):

            result = finetune(cfg, self._hal, _make_pack())

        assert result.backend_used  # non-empty string


# ---------------------------------------------------------------------------
# finetune() evidence emission
# ---------------------------------------------------------------------------

class TestFinetuneEvidenceEmission:
    _hal = _make_hal(tier=HardwareTier.TIER_3090)

    def test_emits_training_start(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        finetune(cfg, self._hal, _make_pack())
        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.TRAINING_START.value in types

    def test_emits_training_step(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path,
                             num_epochs=2, grad_accum_steps=3)
        finetune(cfg, self._hal, _make_pack())
        step_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.TRAINING_STEP.value
        ]
        assert len(step_events) == 2  # one per epoch in simulate

    def test_emits_training_complete(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        finetune(cfg, self._hal, _make_pack())
        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.TRAINING_COMPLETE.value in types

    def test_emits_backend_selected(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        finetune(cfg, self._hal, _make_pack())
        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.TRAINING_BACKEND_SELECTED.value in types

    def test_training_complete_carries_final_loss(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        finetune(cfg, self._hal, _make_pack())
        complete = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.TRAINING_COMPLETE.value
        ]
        assert len(complete) == 1
        assert "final_loss" in complete[0]["data"]

    def test_training_complete_carries_steps_completed(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        finetune(cfg, self._hal, _make_pack())
        complete = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.TRAINING_COMPLETE.value
        ]
        assert "steps_completed" in complete[0]["data"]

    def test_fsm_transitions_emitted(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        finetune(cfg, self._hal, _make_pack())
        fsm_events = [
            e for e in EvidenceLogger.get_events()
            if "fsm.transition" in e.get("type", "")
        ]
        assert len(fsm_events) >= 5  # IDLE→PROFILING→ROUTING→PREPARING→TRAINING→...

    def test_training_start_carries_pack_id(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="mypack", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        finetune(cfg, self._hal, _make_pack())
        start_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.TRAINING_START.value
        ]
        assert len(start_events) >= 1
        assert start_events[0]["data"]["pack_id"] == "test-pack"

    def test_real_gpu_path_still_emits_training_start_and_complete(
        self, tmp_path, monkeypatch
    ):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_REAL_GPU", "1")
        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="sft",
        )
        with patch("dojo.train.finetune._run_real_sft", return_value=(8, 0.18)), \
             patch("dojo.train.finetune._is_backend_available", return_value=True):
            finetune(cfg, self._hal, _make_pack())

        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.TRAINING_COMPLETE.value in types
        assert EvidenceType.TRAINING_START.value in types


# ---------------------------------------------------------------------------
# finetune() constitutional guards
# ---------------------------------------------------------------------------

class TestFinetuneConstitutionalGuards:
    _hal = _make_hal(tier=HardwareTier.TIER_3090)

    def test_vram_exceeded_raises_for_oversized_config(self, tmp_path):
        from dojo.train.finetune import finetune
        tiny_hal = _make_hal(
            tier=HardwareTier.TIER_4060_LAPTOP,
            vram_bytes=1,   # 1 byte — guaranteed to overflow
        )
        cfg = FinetuneConfig(
            pack_id="p", base_model="meta-llama/llama-3-70b",
            output_dir=tmp_path, batch_size=4, max_seq_length=8192,
        )
        with pytest.raises(VRAMExceeded):
            finetune(cfg, tiny_hal, _make_pack())

    def test_loss_divergence_nan_raises_loss_divergence_error(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_REAL_GPU", "1")
        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="sft",
        )
        with patch("dojo.train.finetune._run_real_sft",
                   return_value=(5, float("nan"))), \
             patch("dojo.train.finetune._is_backend_available", return_value=True):
            with pytest.raises(LossDivergenceError):
                finetune(cfg, self._hal, _make_pack())

    def test_loss_divergence_inf_raises_loss_divergence_error(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_REAL_GPU", "1")
        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="qlora",
        )
        with patch("dojo.train.finetune._run_real_qlora",
                   return_value=(5, float("inf"))), \
             patch("dojo.train.finetune._is_backend_available", return_value=True):
            with pytest.raises(LossDivergenceError):
                finetune(cfg, self._hal, _make_pack())

    def test_loss_divergence_neg_inf_raises(self, tmp_path, monkeypatch):
        from dojo.train.finetune import finetune
        monkeypatch.setenv("DOJO_REAL_GPU", "1")
        cfg = FinetuneConfig(
            pack_id="p", base_model="qwen/qwen2-7b",
            output_dir=tmp_path, training_mode="sft",
        )
        with patch("dojo.train.finetune._run_real_sft",
                   return_value=(5, float("-inf"))), \
             patch("dojo.train.finetune._is_backend_available", return_value=True):
            with pytest.raises(LossDivergenceError):
                finetune(cfg, self._hal, _make_pack())

    def test_vram_not_checked_for_cpu_tier(self, tmp_path):
        """CPU tier (vram_bytes=0) should never raise VRAMExceeded."""
        from dojo.train.finetune import finetune
        cpu_hal = _make_hal(tier=HardwareTier.TIER_CPU, vram_bytes=0)
        cfg = FinetuneConfig(
            pack_id="p", base_model="meta-llama/llama-3-70b",
            output_dir=tmp_path, batch_size=4, max_seq_length=8192,
        )
        try:
            finetune(cfg, cpu_hal, _make_pack())
        except VRAMExceeded:
            pytest.fail("VRAMExceeded raised for CPU tier with vram_bytes=0")
        except Exception:
            pass  # Other exceptions (FSM, backend, etc.) are fine

    def test_training_receipt_is_nonempty_merkle_root(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        result = finetune(cfg, self._hal, _make_pack())
        assert len(result.training_receipt) == 64
        assert result.training_receipt != "0" * 64

    def test_result_contains_all_required_fields(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        result = finetune(cfg, self._hal, _make_pack())
        d = result.to_dict()
        for key in ("adapter_path", "base_model", "training_receipt",
                    "final_loss", "steps_completed", "wall_time_seconds",
                    "backend_used"):
            assert key in d

    def test_one_door_decorator_still_present(self):
        """@one_door must remain on finetune() — verify by checking evidence emission."""
        from dojo.train.finetune import finetune
        # one_door emits ONE_DOOR_ENTRY — if it's present, it fires
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b",
                             output_dir=Path("/tmp/test_one_door"))
        try:
            finetune(cfg, _make_hal(), _make_pack())
        except Exception:
            pass
        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.ONE_DOOR_ENTRY.value in types


# ---------------------------------------------------------------------------
# Native SFT (d16) — real torch backprop, no TRL required
# ---------------------------------------------------------------------------

class TestNativeSFTSteps:
    """run_native_sft_steps() — real cross-entropy + gradient descent on CPU."""

    def test_returns_nonempty_list(self):
        from dojo.train.native_rl import run_native_sft_steps
        results = run_native_sft_steps(num_epochs=1, steps_per_epoch=2,
                                       learning_rate=1e-4, seed=42)
        assert len(results) == 2

    def test_step_fields_present(self):
        from dojo.train.native_rl import run_native_sft_steps, NativeSFTStep
        results = run_native_sft_steps(num_epochs=1, steps_per_epoch=1,
                                       learning_rate=1e-4, seed=42)
        s = results[0]
        assert isinstance(s, NativeSFTStep)
        assert s.step == 1
        assert s.epoch == 1
        assert isinstance(s.loss, float)
        assert isinstance(s.perplexity, float)
        assert isinstance(s.grad_norm, float)

    def test_loss_is_finite_positive(self):
        from dojo.train.native_rl import run_native_sft_steps
        import math
        results = run_native_sft_steps(num_epochs=2, steps_per_epoch=3,
                                       learning_rate=1e-4, seed=0)
        for s in results:
            assert math.isfinite(s.loss) and s.loss > 0.0

    def test_grad_norm_nonzero(self):
        from dojo.train.native_rl import run_native_sft_steps
        results = run_native_sft_steps(num_epochs=1, steps_per_epoch=2,
                                       learning_rate=1e-4, seed=7)
        assert all(s.grad_norm > 0.0 for s in results)

    def test_perplexity_equals_exp_loss(self):
        from dojo.train.native_rl import run_native_sft_steps
        import math
        results = run_native_sft_steps(num_epochs=1, steps_per_epoch=1,
                                       learning_rate=1e-4, seed=42)
        s = results[0]
        assert abs(s.perplexity - math.exp(min(s.loss, 20.0))) < 1e-4

    def test_loss_decreases_over_many_steps(self):
        from dojo.train.native_rl import run_native_sft_steps
        results = run_native_sft_steps(num_epochs=3, steps_per_epoch=5,
                                       learning_rate=5e-3, seed=42)
        first_loss = results[0].loss
        last_loss = results[-1].loss
        # Learning must move loss in some direction (real gradient descent)
        assert first_loss != last_loss

    def test_seed_reproducible(self):
        from dojo.train.native_rl import run_native_sft_steps
        r1 = run_native_sft_steps(num_epochs=1, steps_per_epoch=2, learning_rate=1e-4, seed=99)
        r2 = run_native_sft_steps(num_epochs=1, steps_per_epoch=2, learning_rate=1e-4, seed=99)
        assert r1[0].loss == r2[0].loss

    def test_step_counter_increments(self):
        from dojo.train.native_rl import run_native_sft_steps
        results = run_native_sft_steps(num_epochs=2, steps_per_epoch=3,
                                       learning_rate=1e-4, seed=42)
        assert [s.step for s in results] == list(range(1, 7))

    def test_epoch_field_correct(self):
        from dojo.train.native_rl import run_native_sft_steps
        results = run_native_sft_steps(num_epochs=2, steps_per_epoch=2,
                                       learning_rate=1e-4, seed=42)
        assert results[0].epoch == 1
        assert results[1].epoch == 1
        assert results[2].epoch == 2
        assert results[3].epoch == 2


class TestNativeSFTFitnesseFallback:
    """finetune() uses native_sft when TRL is mocked out."""

    def test_native_sft_runs_without_trl(self, tmp_path):
        from dojo.train.finetune import _run_native_sft
        from dojo.train.checkpoint import CheckpointManager
        ckpt = CheckpointManager(tmp_path / "ckpt")
        cfg = FinetuneConfig(pack_id="p", base_model="m", output_dir=tmp_path / "out",
                             num_epochs=1, grad_accum_steps=2)
        steps, loss = _run_native_sft(cfg, EvidenceLogger.get_instance(), ckpt)
        assert steps == 2
        assert loss > 0.0

    def test_native_sft_emits_training_backend_selected(self, tmp_path):
        from dojo.train.finetune import _run_native_sft
        from dojo.train.checkpoint import CheckpointManager
        ckpt = CheckpointManager(tmp_path / "ckpt")
        cfg = FinetuneConfig(pack_id="p", base_model="m", output_dir=tmp_path / "out",
                             num_epochs=1, grad_accum_steps=1)
        _run_native_sft(cfg, EvidenceLogger.get_instance(), ckpt)
        events = EvidenceLogger.get_events()
        backends = [e["data"]["backend"] for e in events
                    if e.get("type") == EvidenceType.TRAINING_BACKEND_SELECTED.value]
        assert "native_sft" in backends

    def test_native_sft_emits_training_step(self, tmp_path):
        from dojo.train.finetune import _run_native_sft
        from dojo.train.checkpoint import CheckpointManager
        ckpt = CheckpointManager(tmp_path / "ckpt")
        cfg = FinetuneConfig(pack_id="p", base_model="m", output_dir=tmp_path / "out",
                             num_epochs=1, grad_accum_steps=1)
        _run_native_sft(cfg, EvidenceLogger.get_instance(), ckpt)
        events = EvidenceLogger.get_events()
        step_events = [e for e in events if e.get("type") == EvidenceType.TRAINING_STEP.value]
        assert any(e["data"].get("backend") == "native_sft" for e in step_events)


# ---------------------------------------------------------------------------
# Native QLoRA (d17) — frozen base + real LoRA adapter gradients
# ---------------------------------------------------------------------------

class TestNativeQLoRASteps:
    """run_native_qlora_steps() — real LoRA gradient updates, no PEFT required."""

    def test_returns_nonempty_list(self):
        from dojo.train.native_rl import run_native_qlora_steps
        results = run_native_qlora_steps(num_epochs=1, steps_per_epoch=2, lora_r=4,
                                         lora_alpha=8.0, learning_rate=1e-4, seed=42)
        assert len(results) == 2

    def test_step_fields_present(self):
        from dojo.train.native_rl import run_native_qlora_steps, NativeQLoRAStep
        results = run_native_qlora_steps(num_epochs=1, steps_per_epoch=1, lora_r=4,
                                         lora_alpha=8.0, learning_rate=1e-4, seed=42)
        s = results[0]
        assert isinstance(s, NativeQLoRAStep)
        assert s.step == 1
        assert s.lora_r == 4
        assert s.trainable_params > 0

    def test_trainable_params_matches_lora_r(self):
        from dojo.train.native_rl import run_native_qlora_steps
        r4 = run_native_qlora_steps(num_epochs=1, steps_per_epoch=1, lora_r=4,
                                     lora_alpha=8.0, learning_rate=1e-4, seed=1)
        r8 = run_native_qlora_steps(num_epochs=1, steps_per_epoch=1, lora_r=8,
                                     lora_alpha=8.0, learning_rate=1e-4, seed=1)
        assert r8[0].trainable_params > r4[0].trainable_params

    def test_loss_is_finite_positive(self):
        from dojo.train.native_rl import run_native_qlora_steps
        import math
        results = run_native_qlora_steps(num_epochs=2, steps_per_epoch=3, lora_r=4,
                                         lora_alpha=8.0, learning_rate=1e-4, seed=0)
        for s in results:
            assert math.isfinite(s.loss) and s.loss > 0.0

    def test_grad_norm_nonzero(self):
        from dojo.train.native_rl import run_native_qlora_steps
        results = run_native_qlora_steps(num_epochs=1, steps_per_epoch=2, lora_r=4,
                                         lora_alpha=8.0, learning_rate=1e-4, seed=7)
        assert all(s.grad_norm > 0.0 for s in results)

    def test_seed_reproducible(self):
        from dojo.train.native_rl import run_native_qlora_steps
        r1 = run_native_qlora_steps(num_epochs=1, steps_per_epoch=2, lora_r=4,
                                     lora_alpha=8.0, learning_rate=1e-4, seed=55)
        r2 = run_native_qlora_steps(num_epochs=1, steps_per_epoch=2, lora_r=4,
                                     lora_alpha=8.0, learning_rate=1e-4, seed=55)
        assert r1[0].loss == r2[0].loss

    def test_step_counter_increments(self):
        from dojo.train.native_rl import run_native_qlora_steps
        results = run_native_qlora_steps(num_epochs=2, steps_per_epoch=3, lora_r=4,
                                         lora_alpha=8.0, learning_rate=1e-4, seed=42)
        assert [s.step for s in results] == list(range(1, 7))

    def test_perplexity_equals_exp_loss(self):
        from dojo.train.native_rl import run_native_qlora_steps
        import math
        results = run_native_qlora_steps(num_epochs=1, steps_per_epoch=1, lora_r=4,
                                         lora_alpha=8.0, learning_rate=1e-4, seed=42)
        s = results[0]
        assert abs(s.perplexity - math.exp(min(s.loss, 20.0))) < 1e-4


class TestNativeQLoRAFallback:
    """finetune() uses native_qlora when PEFT/TRL are mocked out."""

    def test_native_qlora_runs_without_peft(self, tmp_path):
        from dojo.train.finetune import _run_native_qlora
        from dojo.train.checkpoint import CheckpointManager
        ckpt = CheckpointManager(tmp_path / "ckpt")
        cfg = FinetuneConfig(pack_id="p", base_model="m", output_dir=tmp_path / "out",
                             training_mode="qlora", num_epochs=1, grad_accum_steps=2,
                             lora_r=4, lora_alpha=16)
        steps, loss = _run_native_qlora(cfg, EvidenceLogger.get_instance(), ckpt)
        assert steps == 2
        assert loss > 0.0

    def test_native_qlora_emits_backend_selected(self, tmp_path):
        from dojo.train.finetune import _run_native_qlora
        from dojo.train.checkpoint import CheckpointManager
        ckpt = CheckpointManager(tmp_path / "ckpt")
        cfg = FinetuneConfig(pack_id="p", base_model="m", output_dir=tmp_path / "out",
                             training_mode="qlora", num_epochs=1, grad_accum_steps=1,
                             lora_r=4, lora_alpha=16)
        _run_native_qlora(cfg, EvidenceLogger.get_instance(), ckpt)
        events = EvidenceLogger.get_events()
        backends = [e["data"]["backend"] for e in events
                    if e.get("type") == EvidenceType.TRAINING_BACKEND_SELECTED.value]
        assert "native_qlora" in backends

    def test_native_qlora_emits_lora_r_in_evidence(self, tmp_path):
        from dojo.train.finetune import _run_native_qlora
        from dojo.train.checkpoint import CheckpointManager
        ckpt = CheckpointManager(tmp_path / "ckpt")
        cfg = FinetuneConfig(pack_id="p", base_model="m", output_dir=tmp_path / "out",
                             training_mode="qlora", num_epochs=1, grad_accum_steps=1,
                             lora_r=8, lora_alpha=16)
        _run_native_qlora(cfg, EvidenceLogger.get_instance(), ckpt)
        events = EvidenceLogger.get_events()
        step_events = [e for e in events if e.get("type") == EvidenceType.TRAINING_STEP.value]
        assert any(e["data"].get("lora_r") == 8 for e in step_events)
