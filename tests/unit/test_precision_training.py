"""Unit tests for d10 (FP8) and d11 (BF16) precision training — Wave 2.

Covers:
BF16 (d11):
- test_bf16_autocast_context_used         — torch.autocast called with dtype=bfloat16
- test_bf16_evidence_emitted              — TRAINING_BACKEND_SELECTED has precision="bf16"
- test_bf16_training_completes_ci_mode    — DOJO_REAL_GPU=0; returns (steps, loss)

FP8 (d10):
- test_fp8_sm89_allowed                   — SM=8.9 proceeds without PrecisionUnsupported
- test_fp8_sm86_blocked                   — SM=8.6 raises PrecisionUnsupported
- test_fp8_evidence_emitted               — TRAINING_BACKEND_SELECTED has precision="fp8"
- test_fp8_training_completes_ci_mode     — DOJO_REAL_GPU=0, SM=8.9 mocked; completes
- test_fp8_below_sm89_raises_precision_unsupported — SM=7.5; informative error message
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PrecisionUnsupported
from dojo.core.types import HALProfile, HardwareTier
from dojo.train.base import FinetuneConfig
from dojo.train.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_hal(
    tier: HardwareTier = HardwareTier.TIER_4060_LAPTOP,
    vram_bytes: int = 16 * 1024 ** 3,
    compute_capability: str = "8.9",
    has_native_fp8: bool = True,
) -> HALProfile:
    return HALProfile(
        tier=tier,
        gpu_name="RTX 4060 Ti",
        vram_bytes=vram_bytes,
        compute_capability=compute_capability,
        has_native_fp8=has_native_fp8,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="555.0",
        profile_sha256="b" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_pack():
    from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest
    from dojo.pack.schema import SpecializationPack
    return SpecializationPack(
        pack_id="test-pack-precision",
        version="1.0.0",
        schema_version="1.0",
        sample_count=100,
        format=PackFormat.ALPACA,
        splits={"train": 80, "val": 20},
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


def _make_config(training_mode: str = "sft", tmp_path: Path | None = None) -> FinetuneConfig:
    return FinetuneConfig(
        pack_id="test-pack-precision",
        base_model="qwen/qwen2-1.5b",
        output_dir=tmp_path or Path("artifacts/test_precision"),
        training_mode=training_mode,
        num_epochs=1,
    )


@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    """Ensure tests never accidentally trigger real GPU training."""
    monkeypatch.delenv("DOJO_REAL_GPU", raising=False)


# ---------------------------------------------------------------------------
# d11 — BF16 tests
# ---------------------------------------------------------------------------

class TestBF16Training:
    """d11: BF16 mixed-precision training via torch.autocast."""

    def test_bf16_autocast_context_used(self, tmp_path):
        """_run_real_sft source must contain torch.autocast with bfloat16."""
        import inspect
        from dojo.train.finetune import _run_real_sft

        src = inspect.getsource(_run_real_sft)
        assert "torch.autocast" in src, "autocast must be present in _run_real_sft"
        assert "bfloat16" in src, "bfloat16 dtype must be specified in _run_real_sft"

    def test_bf16_evidence_emitted(self, tmp_path):
        """TRAINING_BACKEND_SELECTED event must include precision='bf16'."""
        from dojo.train.finetune import _run_real_sft

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("sft", tmp_path)

        _run_real_sft(cfg, logger, ckpt_mgr, _make_pack())

        events = EvidenceLogger.get_events()
        backend_events = [
            e for e in events
            if e.get("type") == EvidenceType.TRAINING_BACKEND_SELECTED.value
        ]
        assert backend_events, "TRAINING_BACKEND_SELECTED must be emitted"
        assert any(
            e.get("data", {}).get("precision") == "bf16"
            for e in backend_events
        ), f"precision='bf16' not found in backend events: {backend_events}"

    def test_bf16_training_completes_ci_mode(self, tmp_path):
        """DOJO_REAL_GPU=0: BF16 mode must return (steps, loss) without error."""
        from dojo.train.finetune import _run_real_sft

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("sft", tmp_path)

        steps, loss = _run_real_sft(cfg, logger, ckpt_mgr, _make_pack())

        assert isinstance(steps, int) and steps >= 0
        assert isinstance(loss, float) and loss >= 0.0

    def test_bf16_via_finetune_entrypoint(self, tmp_path):
        """finetune() with training_mode='bf16' routes to BF16 path end-to-end."""
        from unittest.mock import patch
        from dojo.train.finetune import finetune

        hal = _make_hal()
        cfg = _make_config("bf16", tmp_path)

        with patch("dojo.train.finetune._is_backend_available", return_value=True):
            result = finetune(cfg, hal, _make_pack())

        assert result.steps_completed >= 0
        assert result.final_loss >= 0.0

        events = EvidenceLogger.get_events()
        bf16_events = [
            e for e in events
            if e.get("data", {}).get("precision") == "bf16"
        ]
        assert bf16_events, "BF16 evidence must appear when training_mode='bf16'"


# ---------------------------------------------------------------------------
# d10 — FP8 tests
# ---------------------------------------------------------------------------

class TestFP8Training:
    """d10: FP8 training on Ada Lovelace SM >= 8.9."""

    def test_fp8_sm89_allowed(self, tmp_path, monkeypatch):
        """SM=8.9 must not raise PrecisionUnsupported."""
        monkeypatch.setenv("DOJO_MOCK_SM", "8.9")

        from dojo.train.finetune import _run_real_fp8

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp8", tmp_path)

        # Should not raise
        steps, loss = _run_real_fp8(cfg, logger, ckpt_mgr, _make_pack())
        assert steps >= 0
        assert loss >= 0.0

    def test_fp8_sm86_blocked(self, tmp_path, monkeypatch):
        """SM=8.6 (Ampere, e.g. RTX 3090) must raise PrecisionUnsupported."""
        monkeypatch.setenv("DOJO_MOCK_SM", "8.6")

        from dojo.train.finetune import _run_real_fp8

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp8", tmp_path)

        with pytest.raises(PrecisionUnsupported):
            _run_real_fp8(cfg, logger, ckpt_mgr, _make_pack())

    def test_fp8_evidence_emitted(self, tmp_path, monkeypatch):
        """TRAINING_BACKEND_SELECTED must include precision='fp8'."""
        monkeypatch.setenv("DOJO_MOCK_SM", "8.9")

        from dojo.train.finetune import _run_real_fp8

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp8", tmp_path)

        _run_real_fp8(cfg, logger, ckpt_mgr, _make_pack())

        events = EvidenceLogger.get_events()
        fp8_events = [
            e for e in events
            if e.get("data", {}).get("precision") == "fp8"
        ]
        assert fp8_events, "TRAINING_BACKEND_SELECTED with precision='fp8' must be emitted"

    def test_fp8_training_completes_ci_mode(self, tmp_path, monkeypatch):
        """DOJO_REAL_GPU=0, SM=8.9: FP8 stub must return (steps, loss)."""
        monkeypatch.setenv("DOJO_MOCK_SM", "8.9")

        from dojo.train.finetune import _run_real_fp8

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp8", tmp_path)

        steps, loss = _run_real_fp8(cfg, logger, ckpt_mgr, _make_pack())

        assert isinstance(steps, int) and steps > 0
        assert isinstance(loss, float) and loss > 0.0

    def test_fp8_below_sm89_raises_precision_unsupported(self, tmp_path, monkeypatch):
        """SM=7.5 (Turing) must raise PrecisionUnsupported with informative message."""
        monkeypatch.setenv("DOJO_MOCK_SM", "7.5")

        from dojo.train.finetune import _run_real_fp8

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp8", tmp_path)

        with pytest.raises(PrecisionUnsupported) as exc_info:
            _run_real_fp8(cfg, logger, ckpt_mgr, _make_pack())

        err = str(exc_info.value)
        assert "8.9" in err, "Error message must mention the required SM version 8.9"
        assert "7.5" in err or "SM" in err, "Error message must reference the detected SM"

    def test_fp8_sm90_hopper_allowed(self, tmp_path, monkeypatch):
        """SM=9.0 (Hopper H100) must also be allowed for FP8."""
        monkeypatch.setenv("DOJO_MOCK_SM", "9.0")

        from dojo.train.finetune import _run_real_fp8

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp8", tmp_path)

        steps, loss = _run_real_fp8(cfg, logger, ckpt_mgr, _make_pack())
        assert steps >= 0

    def test_fp8_via_finetune_entrypoint(self, tmp_path, monkeypatch):
        """finetune() with training_mode='fp8' routes to FP8 path end-to-end."""
        monkeypatch.setenv("DOJO_MOCK_SM", "8.9")

        from unittest.mock import patch
        from dojo.train.finetune import finetune

        hal = _make_hal()
        cfg = _make_config("fp8", tmp_path)

        with patch("dojo.train.finetune._is_backend_available", return_value=True):
            result = finetune(cfg, hal, _make_pack())

        assert result.steps_completed > 0
        assert result.final_loss > 0.0

        events = EvidenceLogger.get_events()
        fp8_events = [
            e for e in events
            if e.get("data", {}).get("precision") == "fp8"
        ]
        assert fp8_events, "FP8 evidence must appear when training_mode='fp8'"

    def test_fp8_sm86_via_finetune_raises(self, tmp_path, monkeypatch):
        """finetune() with FP8 mode and SM=8.6 must propagate PrecisionUnsupported."""
        monkeypatch.setenv("DOJO_MOCK_SM", "8.6")

        from unittest.mock import patch
        from dojo.train.finetune import finetune

        hal = _make_hal()
        cfg = _make_config("fp8", tmp_path)

        with patch("dojo.train.finetune._is_backend_available", return_value=True):
            with pytest.raises(PrecisionUnsupported):
                finetune(cfg, hal, _make_pack())


# ---------------------------------------------------------------------------
# FinetuneConfig training_mode field
# ---------------------------------------------------------------------------

class TestFinetuneConfigTrainingMode:
    """Verify training_mode is serialised and has correct defaults."""

    def test_default_training_mode_is_sft(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m")
        assert cfg.training_mode == "sft"

    def test_fp8_training_mode_accepted(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="fp8")
        assert cfg.training_mode == "fp8"

    def test_bf16_training_mode_accepted(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="bf16")
        assert cfg.training_mode == "bf16"

    def test_training_mode_in_to_dict(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="fp8")
        assert cfg.to_dict()["training_mode"] == "fp8"

    def test_training_mode_affects_config_hash(self):
        cfg_sft = FinetuneConfig(pack_id="p", base_model="m", training_mode="sft")
        cfg_fp8 = FinetuneConfig(pack_id="p", base_model="m", training_mode="fp8")
        assert cfg_sft.config_hash() != cfg_fp8.config_hash()

    def test_fp4_training_mode_accepted(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="fp4")
        assert cfg.training_mode == "fp4"

    def test_fp4_training_mode_in_to_dict(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", training_mode="fp4")
        assert cfg.to_dict()["training_mode"] == "fp4"


# ---------------------------------------------------------------------------
# d12 — FP4 tests
# ---------------------------------------------------------------------------

class TestFP4Training:
    """d12: FP4 (E2M1) training requiring SM >= 10.0 (Blackwell)."""

    def test_fp4_sm100_allowed(self, tmp_path, monkeypatch):
        """SM=10.0 (Blackwell) must not raise PrecisionUnsupported."""
        monkeypatch.setenv("DOJO_MOCK_SM", "10.0")

        from dojo.train.finetune import _run_real_fp4

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp4", tmp_path)

        steps, loss = _run_real_fp4(cfg, logger, ckpt_mgr, _make_pack())
        assert steps >= 0
        assert loss >= 0.0

    def test_fp4_sm89_blocked(self, tmp_path, monkeypatch):
        """SM=8.9 (Ada Lovelace) must raise PrecisionUnsupported for FP4."""
        monkeypatch.setenv("DOJO_MOCK_SM", "8.9")

        from dojo.train.finetune import _run_real_fp4

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp4", tmp_path)

        with pytest.raises(PrecisionUnsupported):
            _run_real_fp4(cfg, logger, ckpt_mgr, _make_pack())

    def test_fp4_sm86_blocked(self, tmp_path, monkeypatch):
        """SM=8.6 (Ampere) must raise PrecisionUnsupported for FP4."""
        monkeypatch.setenv("DOJO_MOCK_SM", "8.6")

        from dojo.train.finetune import _run_real_fp4

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp4", tmp_path)

        with pytest.raises(PrecisionUnsupported):
            _run_real_fp4(cfg, logger, ckpt_mgr, _make_pack())

    def test_fp4_evidence_emitted(self, tmp_path, monkeypatch):
        """TRAINING_BACKEND_SELECTED must include precision='fp4'."""
        monkeypatch.setenv("DOJO_MOCK_SM", "10.0")

        from dojo.train.finetune import _run_real_fp4

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp4", tmp_path)

        _run_real_fp4(cfg, logger, ckpt_mgr, _make_pack())

        events = EvidenceLogger.get_events()
        fp4_events = [
            e for e in events
            if e.get("data", {}).get("precision") == "fp4"
        ]
        assert fp4_events, "TRAINING_BACKEND_SELECTED with precision='fp4' must be emitted"

    def test_fp4_backend_is_native_fp4(self, tmp_path, monkeypatch):
        """Evidence backend field must be 'native_fp4'."""
        monkeypatch.setenv("DOJO_MOCK_SM", "10.0")

        from dojo.train.finetune import _run_real_fp4

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp4", tmp_path)

        _run_real_fp4(cfg, logger, ckpt_mgr, _make_pack())

        events = EvidenceLogger.get_events()
        native_events = [
            e for e in events
            if e.get("data", {}).get("backend") == "native_fp4"
        ]
        assert native_events, "backend='native_fp4' must appear in evidence"

    def test_fp4_training_completes_ci_mode(self, tmp_path, monkeypatch):
        """DOJO_REAL_GPU=0, SM=10.0: FP4 stub must return (steps, loss)."""
        monkeypatch.setenv("DOJO_MOCK_SM", "10.0")

        from dojo.train.finetune import _run_real_fp4

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp4", tmp_path)

        steps, loss = _run_real_fp4(cfg, logger, ckpt_mgr, _make_pack())

        assert isinstance(steps, int) and steps > 0
        assert isinstance(loss, float) and loss > 0.0

    def test_fp4_sm_below_100_raises_with_informative_message(self, tmp_path, monkeypatch):
        """SM=9.0 (Hopper) must raise PrecisionUnsupported with SM 10.0 in message."""
        monkeypatch.setenv("DOJO_MOCK_SM", "9.0")

        from dojo.train.finetune import _run_real_fp4

        logger = EvidenceLogger.get_instance()
        ckpt_mgr = CheckpointManager(tmp_path / "ckpt", max_checkpoints=3)
        cfg = _make_config("fp4", tmp_path)

        with pytest.raises(PrecisionUnsupported) as exc_info:
            _run_real_fp4(cfg, logger, ckpt_mgr, _make_pack())

        err = str(exc_info.value)
        assert "10.0" in err, "Error message must mention required SM 10.0"
        assert "9.0" in err or "SM" in err, "Error message must reference detected SM"

    def test_fp4_via_finetune_entrypoint(self, tmp_path, monkeypatch):
        """finetune() with training_mode='fp4' routes to FP4 path end-to-end."""
        monkeypatch.setenv("DOJO_MOCK_SM", "10.0")

        from unittest.mock import patch
        from dojo.train.finetune import finetune

        hal = _make_hal()
        cfg = _make_config("fp4", tmp_path)

        with patch("dojo.train.finetune._is_backend_available", return_value=True):
            result = finetune(cfg, hal, _make_pack())

        assert result.steps_completed >= 0
        assert result.final_loss >= 0.0

        events = EvidenceLogger.get_events()
        fp4_events = [
            e for e in events
            if e.get("data", {}).get("precision") == "fp4"
        ]
        assert fp4_events, "FP4 evidence must appear when training_mode='fp4'"

    def test_fp4_sm89_via_finetune_raises(self, tmp_path, monkeypatch):
        """finetune() with FP4 mode and SM=8.9 must propagate PrecisionUnsupported."""
        monkeypatch.setenv("DOJO_MOCK_SM", "8.9")

        from unittest.mock import patch
        from dojo.train.finetune import finetune

        hal = _make_hal()
        cfg = _make_config("fp4", tmp_path)

        with patch("dojo.train.finetune._is_backend_available", return_value=True):
            with pytest.raises(PrecisionUnsupported):
                finetune(cfg, hal, _make_pack())
