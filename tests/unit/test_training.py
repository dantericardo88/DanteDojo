"""Unit tests for dojo/train/ — Phase 2 training engine.

Covers:
- BaseTrainer FSM: valid transition sequence completes to COMPLETED
- Illegal transition from IDLE → TRAINING raises FSMTransitionError
- FinetuneConfig defaults are 4060-safe
- select_unsloth_config enforces 4060 batch_size limit
- generate_axolotl_config includes FSDP2 for TIER_3090_DUAL
- CheckpointManager.save creates a manifest file
- CheckpointManager.load_latest returns correct step
- CheckpointManager.prune removes oldest beyond max
- ContextLengthManager.get_safe_context respects 4060 limit
- finetune() runs FSM, emits TRAINING_START and TRAINING_COMPLETE evidence
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import FSMTransitionError
from dojo.core.types import HALProfile, HardwareTier, PrecisionType
from dojo.train.base import BaseTrainer, FinetuneConfig, FinetuneResult, TrainingState
from dojo.train.checkpoint import CheckpointManager
from dojo.train.context import ContextLengthManager
from dojo.train.unsloth_backend import select_unsloth_config
from dojo.train.axolotl_backend import generate_axolotl_config


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_hal(
    tier: HardwareTier = HardwareTier.TIER_3090,
    vram_bytes: int = 24 * 1024 ** 3,
    compute_capability: str = "8.6",
    has_native_fp8: bool = False,
) -> HALProfile:
    return HALProfile(
        tier=tier, gpu_name="Test GPU", vram_bytes=vram_bytes,
        compute_capability=compute_capability, has_native_fp8=has_native_fp8,
        has_native_fp4=False, has_nvlink=False, pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0, memory_bandwidth_gb_s=0.0,
        driver_version="550.0", profile_sha256="a" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_pack():
    from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest
    from dojo.pack.schema import SpecializationPack
    return SpecializationPack(
        pack_id="test-pack", version="1.0.0", schema_version="1.0",
        sample_count=100, format=PackFormat.ALPACA,
        splits={"train": 80, "val": 20}, harvest_receipt="abc123",
        license=LicenseSpec(spdx_id="MIT", allows_derived_ft=True,
                            allows_commercial_use=True, requires_attribution=False),
        provenance=ProvenanceManifest(
            harvest_organ_version="1.0", sources=("https://example.com",),
            collection_date_utc="2026-01-01", dedup_method="minhash",
            human_reviewed=False,
        ),
        signature=b"\x00" * 64, manifest_sha256="d" * 64,
        data_parquet_sha256="e" * 64,
    )


@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# BaseTrainer FSM
# ---------------------------------------------------------------------------

class TestBaseTrainerFSM:
    def test_valid_full_transition_sequence_reaches_completed(self):
        trainer = BaseTrainer()
        for state in [
            TrainingState.PROFILING, TrainingState.ROUTING,
            TrainingState.PREPARING, TrainingState.TRAINING,
            TrainingState.CHECKPOINTING, TrainingState.EVALUATING,
            TrainingState.COMPLETED,
        ]:
            trainer.transition(state)
        assert trainer.state == TrainingState.COMPLETED
        assert trainer.is_terminal()

    def test_illegal_idle_to_training_raises(self):
        with pytest.raises(FSMTransitionError):
            BaseTrainer().transition(TrainingState.TRAINING)

    def test_illegal_transition_from_terminal_raises(self):
        trainer = BaseTrainer()
        trainer.transition(TrainingState.PROFILING)
        trainer.transition(TrainingState.FAILED)
        with pytest.raises(FSMTransitionError):
            trainer.transition(TrainingState.IDLE)

    def test_on_transition_emits_evidence(self):
        trainer = BaseTrainer()
        trainer.transition(TrainingState.PROFILING)
        events = EvidenceLogger.get_events()
        assert any("fsm.transition" in e.get("type", "") for e in events)

    def test_history_records_transitions(self):
        trainer = BaseTrainer()
        trainer.transition(TrainingState.PROFILING)
        trainer.transition(TrainingState.ROUTING)
        assert (TrainingState.IDLE, TrainingState.PROFILING) in trainer.history
        assert (TrainingState.PROFILING, TrainingState.ROUTING) in trainer.history

    def test_can_fail_from_preparing(self):
        trainer = BaseTrainer()
        for s in [TrainingState.PROFILING, TrainingState.ROUTING,
                  TrainingState.PREPARING, TrainingState.FAILED]:
            trainer.transition(s)
        assert trainer.state == TrainingState.FAILED

    def test_promoting_path_to_completed(self):
        trainer = BaseTrainer()
        for s in [TrainingState.PROFILING, TrainingState.ROUTING,
                  TrainingState.PREPARING, TrainingState.TRAINING,
                  TrainingState.CHECKPOINTING, TrainingState.EVALUATING,
                  TrainingState.PROMOTING, TrainingState.COMPLETED]:
            trainer.transition(s)
        assert trainer.state == TrainingState.COMPLETED


# ---------------------------------------------------------------------------
# FinetuneConfig defaults
# ---------------------------------------------------------------------------

class TestFinetuneConfigDefaults:
    def test_defaults_are_4060_safe(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m")
        assert cfg.batch_size <= 2
        assert cfg.max_seq_length <= 2048
        assert cfg.grad_accum_steps == 8

    def test_config_hash_stable_and_unique(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m")
        assert cfg.config_hash() == cfg.config_hash()
        assert len(cfg.config_hash()) == 64
        assert cfg.config_hash() != FinetuneConfig(pack_id="q", base_model="m").config_hash()

    def test_output_dir_is_path_and_to_dict_has_keys(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m")
        assert isinstance(cfg.output_dir, Path)
        for key in ("pack_id", "base_model", "lora_r", "batch_size", "seed"):
            assert key in cfg.to_dict()


# ---------------------------------------------------------------------------
# select_unsloth_config — 4060 enforcement
# ---------------------------------------------------------------------------

class TestSelectUnslothConfig:
    _4060 = _make_hal(tier=HardwareTier.TIER_4060_LAPTOP,
                      vram_bytes=8*1024**3, compute_capability="8.9",
                      has_native_fp8=True)
    _3090 = _make_hal(tier=HardwareTier.TIER_3090)

    def test_4060_enforces_batch_seq_opts(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", batch_size=4, max_seq_length=8192)
        r = select_unsloth_config(cfg, self._4060)
        assert r["per_device_train_batch_size"] <= 2
        assert r["max_seq_length"] <= 4096
        assert r["use_gradient_checkpointing"] == "unsloth"
        assert r["optim"] == "paged_adamw_8bit"

    def test_3090_does_not_cap_batch_size(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", batch_size=4)
        assert select_unsloth_config(cfg, self._3090)["per_device_train_batch_size"] == 4

    def test_lora_r_passed_through(self):
        cfg = FinetuneConfig(pack_id="p", base_model="m", lora_r=32)
        assert select_unsloth_config(cfg, self._3090)["r"] == 32


# ---------------------------------------------------------------------------
# generate_axolotl_config — FSDP2 for dual 3090, ZeRO-3 for H100
# ---------------------------------------------------------------------------

class TestGenerateAxolotlConfig:
    _dual = _make_hal(tier=HardwareTier.TIER_3090_DUAL, vram_bytes=48*1024**3)
    _h100 = _make_hal(tier=HardwareTier.TIER_H100, vram_bytes=80*1024**3,
                      compute_capability="9.0", has_native_fp8=True)
    _3090 = _make_hal(tier=HardwareTier.TIER_3090)
    _cfg = FinetuneConfig(pack_id="p", base_model="meta-llama/Llama-2-7b-hf")

    def test_dual_3090_fsdp2_full_shard(self):
        r = generate_axolotl_config(self._cfg, self._dual)
        assert "full_shard" in r["fsdp"]
        assert r["fsdp_config"]["fsdp2"] is True
        assert r["world_size"] == 2

    def test_h100_zero3_and_fp8(self):
        r = generate_axolotl_config(self._cfg, self._h100)
        assert r["fsdp_config"]["zero3_enabled"] is True
        assert r.get("fp8") is True

    def test_single_3090_no_fsdp(self):
        assert generate_axolotl_config(self._cfg, self._3090)["fsdp"] is None

    def test_base_model_included(self):
        cfg = FinetuneConfig(pack_id="p", base_model="my-model")
        assert generate_axolotl_config(cfg, self._3090)["base_model"] == "my-model"


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    def test_save_creates_json_with_step(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "c", max_checkpoints=3)
        path = mgr.save(100, {"loss": 0.5})
        assert path.exists() and path.suffix == ".json"
        assert json.loads(path.read_text())["step"] == 100

    def test_load_latest_none_when_empty(self, tmp_path):
        assert CheckpointManager(tmp_path / "c").load_latest() is None

    def test_load_latest_returns_highest_step(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "c", max_checkpoints=5)
        for step in [100, 200, 300]:
            mgr.save(step, {"loss": 0.1})
        step, _ = mgr.load_latest()
        assert step == 300

    def test_list_checkpoints_sorted(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "c", max_checkpoints=10)
        for step in [300, 100, 200]:
            mgr.save(step, {})
        steps = [s for s, _ in mgr.list_checkpoints()]
        assert steps == sorted(steps)

    def test_prune_removes_oldest_beyond_max(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "c", max_checkpoints=2)
        for step in [100, 200, 300]:
            mgr.save(step, {})
        remaining = [s for s, _ in mgr.list_checkpoints()]
        assert 100 not in remaining
        assert len(remaining) == 2
        assert 200 in remaining and 300 in remaining

    def test_save_emits_checkpoint_evidence(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "c", max_checkpoints=3)
        mgr.save(50, {"loss": 0.7})
        events = EvidenceLogger.get_events()
        assert any(EvidenceType.TRAINING_CHECKPOINT.value in e.get("type", "")
                   for e in events)


# ---------------------------------------------------------------------------
# ContextLengthManager
# ---------------------------------------------------------------------------

class TestContextLengthManager:
    def test_4060_context_capped_at_2048(self):
        p = _make_hal(tier=HardwareTier.TIER_4060_LAPTOP, vram_bytes=8*1024**3)
        assert ContextLengthManager.get_safe_context(p, requested=8192) <= 2048

    def test_3090_allows_8192_and_request_caps(self):
        p = _make_hal(tier=HardwareTier.TIER_3090)
        assert ContextLengthManager.get_safe_context(p, requested=8192) == 8192
        assert ContextLengthManager.get_safe_context(p, requested=512) == 512

    def test_h100_allows_32768(self):
        p = _make_hal(tier=HardwareTier.TIER_H100, vram_bytes=80*1024**3)
        assert ContextLengthManager.get_safe_context(p, requested=32768) == 32768

    def test_estimate_vram_scaling(self):
        def est(params_b, seq):
            return ContextLengthManager.estimate_vram_for_config(
                params_b, 1, seq, PrecisionType.NF4, 16)
        assert isinstance(est(7.0, 2048), int) and est(7.0, 2048) > 0
        assert est(13.0, 2048) > est(3.0, 2048)
        assert est(7.0, 4096) > est(7.0, 512)


# ---------------------------------------------------------------------------
# finetune() — FSM and evidence integration
# ---------------------------------------------------------------------------

class TestFinetune:
    _3090 = _make_hal(tier=HardwareTier.TIER_3090)
    _4060 = _make_hal(tier=HardwareTier.TIER_4060_LAPTOP,
                      vram_bytes=8*1024**3, compute_capability="8.9")

    def test_finetune_returns_result_with_fields(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        result = finetune(cfg, self._3090, _make_pack())
        assert isinstance(result, FinetuneResult)
        assert result.base_model == "qwen/qwen2-7b"
        assert result.adapter_path == tmp_path
        assert result.wall_time_seconds >= 0.0
        assert len(result.training_receipt) > 0

    def test_finetune_emits_required_evidence(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        finetune(cfg, self._3090, _make_pack())
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert any(EvidenceType.TRAINING_START.value in t for t in types)
        assert any(EvidenceType.TRAINING_COMPLETE.value in t for t in types)
        assert any(EvidenceType.TRAINING_BACKEND_SELECTED.value in t for t in types)

    def test_finetune_backend_routing(self, tmp_path):
        from dojo.train.finetune import finetune
        cfg7b = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path / "a")
        cfg1b = FinetuneConfig(pack_id="p", base_model="qwen/qwen2-1.5b", output_dir=tmp_path / "b")
        assert finetune(cfg7b, self._3090, _make_pack()).backend_used == "unsloth"
        EvidenceLogger.clear()
        assert finetune(cfg1b, self._4060, _make_pack()).backend_used == "unsloth"
