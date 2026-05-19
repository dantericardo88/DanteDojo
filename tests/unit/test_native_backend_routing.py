"""d16/d17 — Native backend routing integration tests.

Proves that finetune() routes TIER_CPU → TrainingBackend.NATIVE →
run_native_sft / run_native_qlora WITHOUT requiring DOJO_REAL_GPU=1.

The _TinyDecoderLM runs on CPU only (no GPU hardware needed).
Every test here is fully CI-safe.

Coverage:
1.  CPU tier → native_sft called (not simulate, not real_sft)
2.  CPU tier → native_qlora called for non-sft mode
3.  native_sft result: loss > 0, steps == grad_accum_steps
4.  native_qlora result: loss > 0, lora_r propagated
5.  Evidence: TRAINING_START + TRAINING_COMPLETE always emitted on native path
6.  Evidence: TRAINING_STEP events carry backend="native_sft"
7.  Evidence: TRAINING_STEP events carry backend="native_qlora"
8.  FinetuneResult fields all populated on native path
9.  training_receipt is 64-char Merkle root
10. Checkpoint saved on native_sft path
11. Checkpoint saved on native_qlora path
12. native_sft loss is finite and positive
13. native_qlora grad_norm > 0 in step events
14. Multi-epoch native_sft: correct number of TRAINING_STEP events
15. Multi-epoch native_qlora: correct number of TRAINING_STEP events
16. BACKEND_SELECTED evidence shows native_sft backend
17. BACKEND_SELECTED evidence shows native_qlora backend
18. Loss NaN from native path still raises LossDivergenceError
19. VRAMExceeded still raised before native path executes
20. finetune() routing: NATIVE + training_mode=None falls back to qlora path
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import LossDivergenceError, VRAMExceeded
from dojo.core.types import HALProfile, HardwareTier, LicenseSpec, PackFormat, ProvenanceManifest
from dojo.train.base import FinetuneConfig, FinetuneResult
from dojo.pack.schema import SpecializationPack


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _cpu_hal(vram_bytes: int = 0) -> HALProfile:
    return HALProfile(
        tier=HardwareTier.TIER_CPU,
        gpu_name="No GPU",
        vram_bytes=vram_bytes,
        compute_capability="0.0",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="0.0",
        profile_sha256="a" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_pack() -> SpecializationPack:
    return SpecializationPack(
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


@pytest.fixture(autouse=True)
def _reset_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    monkeypatch.delenv("DOJO_REAL_GPU", raising=False)


# ---------------------------------------------------------------------------
# 1. CPU tier → native_sft called (not simulate)
# ---------------------------------------------------------------------------

def test_cpu_tier_sft_calls_run_native_sft(tmp_path):
    from dojo.train.finetune import _run_native_sft, finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=1, grad_accum_steps=2,
    )
    with patch("dojo.train.finetune._run_native_sft", wraps=_run_native_sft) as mock_native, \
         patch("dojo.train.finetune._simulate_training_steps") as mock_sim, \
         patch("dojo.train.finetune._run_real_sft") as mock_real:
        finetune(cfg, _cpu_hal(), _make_pack())
    mock_native.assert_called_once()
    mock_sim.assert_not_called()
    mock_real.assert_not_called()


# ---------------------------------------------------------------------------
# 2. CPU tier → native_qlora called for non-sft mode
# ---------------------------------------------------------------------------

def test_cpu_tier_qlora_calls_run_native_qlora(tmp_path):
    from dojo.train.finetune import _run_native_qlora, finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="qlora", num_epochs=1, grad_accum_steps=2,
        lora_r=4, lora_alpha=8,
    )
    with patch("dojo.train.finetune._run_native_qlora", wraps=_run_native_qlora) as mock_q, \
         patch("dojo.train.finetune._simulate_training_steps") as mock_sim, \
         patch("dojo.train.finetune._run_real_qlora") as mock_real:
        finetune(cfg, _cpu_hal(), _make_pack())
    mock_q.assert_called_once()
    mock_sim.assert_not_called()
    mock_real.assert_not_called()


# ---------------------------------------------------------------------------
# 3. native_sft: loss > 0, steps == grad_accum_steps
# ---------------------------------------------------------------------------

def test_cpu_tier_sft_returns_real_loss(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=1, grad_accum_steps=3,
    )
    result = finetune(cfg, _cpu_hal(), _make_pack())
    assert math.isfinite(result.final_loss)
    assert result.final_loss > 0.0
    assert result.steps_completed == 3


# ---------------------------------------------------------------------------
# 4. native_qlora: loss > 0, steps propagated
# ---------------------------------------------------------------------------

def test_cpu_tier_qlora_returns_real_loss(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="qlora", num_epochs=1, grad_accum_steps=2,
        lora_r=8, lora_alpha=16,
    )
    result = finetune(cfg, _cpu_hal(), _make_pack())
    assert math.isfinite(result.final_loss)
    assert result.final_loss > 0.0
    assert result.steps_completed == 2


# ---------------------------------------------------------------------------
# 5. Evidence: TRAINING_START + TRAINING_COMPLETE on native path
# ---------------------------------------------------------------------------

def test_cpu_sft_emits_start_and_complete(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=1, grad_accum_steps=1,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    types = {e.get("type") for e in EvidenceLogger.get_events()}
    assert EvidenceType.TRAINING_START.value in types
    assert EvidenceType.TRAINING_COMPLETE.value in types


# ---------------------------------------------------------------------------
# 6. TRAINING_STEP events carry backend="native_sft"
# ---------------------------------------------------------------------------

def test_cpu_sft_step_events_show_native_sft_backend(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=2, grad_accum_steps=1,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    step_events = [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == EvidenceType.TRAINING_STEP.value
    ]
    assert len(step_events) >= 1
    assert any(e["data"].get("backend") == "native_sft" for e in step_events)


# ---------------------------------------------------------------------------
# 7. TRAINING_STEP events carry backend="native_qlora"
# ---------------------------------------------------------------------------

def test_cpu_qlora_step_events_show_native_qlora_backend(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="qlora", num_epochs=2, grad_accum_steps=1,
        lora_r=4, lora_alpha=8,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    step_events = [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == EvidenceType.TRAINING_STEP.value
    ]
    assert len(step_events) >= 1
    assert any(e["data"].get("backend") == "native_qlora" for e in step_events)


# ---------------------------------------------------------------------------
# 8. FinetuneResult all required fields populated
# ---------------------------------------------------------------------------

def test_cpu_sft_result_has_all_fields(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=1, grad_accum_steps=1,
    )
    result = finetune(cfg, _cpu_hal(), _make_pack())
    d = result.to_dict()
    for key in ("adapter_path", "base_model", "training_receipt",
                "final_loss", "steps_completed", "wall_time_seconds", "backend_used"):
        assert key in d, f"Missing field: {key}"


# ---------------------------------------------------------------------------
# 9. training_receipt is 64-char Merkle root
# ---------------------------------------------------------------------------

def test_cpu_sft_training_receipt_is_merkle_root(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=1, grad_accum_steps=1,
    )
    result = finetune(cfg, _cpu_hal(), _make_pack())
    assert len(result.training_receipt) == 64
    assert result.training_receipt != "0" * 64


# ---------------------------------------------------------------------------
# 10. Checkpoint saved on native_sft path
# ---------------------------------------------------------------------------

def test_cpu_sft_checkpoint_saved(tmp_path):
    from dojo.train.finetune import finetune
    from dojo.train.checkpoint import CheckpointManager
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=1, grad_accum_steps=2,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    ckpt_mgr = CheckpointManager(tmp_path / "checkpoints")
    latest = ckpt_mgr.load_latest()
    assert latest is not None


# ---------------------------------------------------------------------------
# 11. Checkpoint saved on native_qlora path
# ---------------------------------------------------------------------------

def test_cpu_qlora_checkpoint_saved(tmp_path):
    from dojo.train.finetune import finetune
    from dojo.train.checkpoint import CheckpointManager
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="qlora", num_epochs=1, grad_accum_steps=2,
        lora_r=4, lora_alpha=8,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    ckpt_mgr = CheckpointManager(tmp_path / "checkpoints")
    latest = ckpt_mgr.load_latest()
    assert latest is not None


# ---------------------------------------------------------------------------
# 12. native_sft loss is finite and positive
# ---------------------------------------------------------------------------

def test_cpu_sft_loss_finite_positive(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=3, grad_accum_steps=1,
    )
    result = finetune(cfg, _cpu_hal(), _make_pack())
    assert math.isfinite(result.final_loss)
    assert result.final_loss > 0.0


# ---------------------------------------------------------------------------
# 13. native_qlora: lora_r in step event data
# ---------------------------------------------------------------------------

def test_cpu_qlora_step_events_carry_lora_r(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="qlora", num_epochs=1, grad_accum_steps=1,
        lora_r=16, lora_alpha=32,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    step_events = [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == EvidenceType.TRAINING_STEP.value
        and e["data"].get("backend") == "native_qlora"
    ]
    assert len(step_events) >= 1
    assert step_events[0]["data"].get("lora_r") == 16


# ---------------------------------------------------------------------------
# 14. Multi-epoch native_sft: N TRAINING_STEP events (one per epoch)
# ---------------------------------------------------------------------------

def test_cpu_sft_multi_epoch_step_count(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=3, grad_accum_steps=1,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    step_events = [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == EvidenceType.TRAINING_STEP.value
        and e["data"].get("backend") == "native_sft"
    ]
    assert len(step_events) == 3


# ---------------------------------------------------------------------------
# 15. Multi-epoch native_qlora: N TRAINING_STEP events (one per epoch)
# ---------------------------------------------------------------------------

def test_cpu_qlora_multi_epoch_step_count(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="qlora", num_epochs=4, grad_accum_steps=1,
        lora_r=4, lora_alpha=8,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    step_events = [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == EvidenceType.TRAINING_STEP.value
        and e["data"].get("backend") == "native_qlora"
    ]
    assert len(step_events) == 4


# ---------------------------------------------------------------------------
# 16. BACKEND_SELECTED evidence shows native_sft backend
# ---------------------------------------------------------------------------

def test_cpu_sft_backend_selected_event_shows_native_sft(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=1, grad_accum_steps=1,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    sel_events = [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == EvidenceType.TRAINING_BACKEND_SELECTED.value
    ]
    assert any(e["data"].get("backend") == "native_sft" for e in sel_events)


# ---------------------------------------------------------------------------
# 17. BACKEND_SELECTED evidence shows native_qlora backend
# ---------------------------------------------------------------------------

def test_cpu_qlora_backend_selected_event_shows_native_qlora(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="qlora", num_epochs=1, grad_accum_steps=1,
        lora_r=4, lora_alpha=8,
    )
    finetune(cfg, _cpu_hal(), _make_pack())
    sel_events = [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == EvidenceType.TRAINING_BACKEND_SELECTED.value
    ]
    assert any(e["data"].get("backend") == "native_qlora" for e in sel_events)


# ---------------------------------------------------------------------------
# 18. Loss NaN from native path still raises LossDivergenceError
# ---------------------------------------------------------------------------

def test_cpu_sft_nan_loss_raises_loss_divergence(tmp_path):
    from dojo.train.finetune import finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="sft", num_epochs=1, grad_accum_steps=1,
    )
    with patch("dojo.train.finetune._run_native_sft", return_value=(1, float("nan"))):
        with pytest.raises(LossDivergenceError):
            finetune(cfg, _cpu_hal(), _make_pack())


# ---------------------------------------------------------------------------
# 19. VRAMExceeded raised before native path executes (when vram_bytes > 0)
# ---------------------------------------------------------------------------

def test_cpu_tier_vram_exceeded_when_vram_set(tmp_path):
    from dojo.train.finetune import finetune
    # CPU tier with tiny vram_bytes set to non-zero → overflow on 70B
    constrained_hal = _cpu_hal(vram_bytes=1)
    cfg = FinetuneConfig(
        pack_id="p", base_model="meta-llama/llama-3-70b",
        output_dir=tmp_path, batch_size=4, max_seq_length=8192,
    )
    with pytest.raises(VRAMExceeded):
        finetune(cfg, constrained_hal, _make_pack())


# ---------------------------------------------------------------------------
# 20. NATIVE + training_mode="full_ft" falls back to qlora path
# ---------------------------------------------------------------------------

def test_cpu_full_ft_mode_calls_native_qlora(tmp_path):
    from dojo.train.finetune import _run_native_qlora, finetune
    cfg = FinetuneConfig(
        pack_id="p", base_model="tiny", output_dir=tmp_path,
        training_mode="full_ft", num_epochs=1, grad_accum_steps=1,
    )
    with patch("dojo.train.finetune._run_native_qlora", wraps=_run_native_qlora) as mock_q, \
         patch("dojo.train.finetune._simulate_training_steps") as mock_sim:
        finetune(cfg, _cpu_hal(), _make_pack())
    mock_q.assert_called_once()
    mock_sim.assert_not_called()
