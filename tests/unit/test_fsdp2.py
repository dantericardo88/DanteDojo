"""Unit tests for FSDP2 multi-GPU training — 33 tests, 0 GPU required.

Covers:
- MultiGPUProfile: mock detection, strategy selection, profile hash
- RankEvidenceCollector: validation, record, merkle, seal
- aggregate_rank_merkle_roots: edge cases and ordering stability
- FSDP2Config: defaults, use_orig_params, hash stability
- fsdp2_train FSM: happy path, invalid config, NaN loss, world_size auto-detect
- fsdp2_train Evidence: FSDP2_PROFILE_DETECTED, RANK_INITIALIZED, TRAINING_COMPLETE, RANK_MERKLE_ROOT
- fsdp2_train Result: receipt length, backend, steps
- All 4 sharding strategies complete without error
- cpu_offload_params=True accepted in mock mode
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import FSDP2ConfigError, LossDivergenceError
from dojo.core.types import HALProfile, HardwareTier
from dojo.train.base import FSDP2Config, FinetuneConfig, FinetuneResult
from dojo.train.fsdp2 import fsdp2_train
from dojo.train.multi_gpu import (
    STRATEGY_FULL_SHARD,
    STRATEGY_HYBRID_SHARD,
    STRATEGY_NO_SHARD,
    STRATEGY_SHARD_GRAD_OP,
    MultiGPUProfile,
    _select_strategy,
    detect_multi_gpu,
)
from dojo.train.rank_evidence import (
    RankEvidenceBundle,
    RankEvidenceCollector,
    aggregate_rank_merkle_roots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GIB = 1024 ** 3


def _make_hal(
    tier: HardwareTier = HardwareTier.TIER_3090,
    vram_bytes: int = 24 * _GIB,
) -> HALProfile:
    return HALProfile(
        tier=tier,
        gpu_name="Test GPU",
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


def _make_cfg(tmp_path: Path, num_epochs: int = 1, grad_accum_steps: int = 2) -> FinetuneConfig:
    return FinetuneConfig(
        pack_id="test-pack",
        base_model="test-model-7b",
        num_epochs=num_epochs,
        grad_accum_steps=grad_accum_steps,
        output_dir=tmp_path,
    )


def _make_bundle(rank: int = 0, world_size: int = 1, strategy: str = "FULL_SHARD") -> RankEvidenceBundle:
    return RankEvidenceBundle(
        rank=rank,
        world_size=world_size,
        steps_completed=10,
        final_loss=0.5,
        merkle_root=hashlib.sha256(f"rank{rank}".encode()).hexdigest(),
        event_count=10,
        sharding_strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    monkeypatch.delenv("DOJO_REAL_GPU", raising=False)


# ---------------------------------------------------------------------------
# TestMultiGPUDetection (4)
# ---------------------------------------------------------------------------

class TestMultiGPUDetection:
    def test_mock_profile_returns_two_gpus(self):
        profile = detect_multi_gpu(mock_mode=True)
        assert profile.device_count == 2
        assert len(profile.device_names) == 2

    def test_mock_profile_recommends_full_shard(self):
        profile = detect_multi_gpu(mock_mode=True)
        assert profile.recommended_strategy == STRATEGY_FULL_SHARD

    def test_profile_hash_is_64_hex_chars(self):
        profile = detect_multi_gpu(mock_mode=True)
        h = profile.profile_hash()
        assert len(h) == 64
        int(h, 16)  # raises ValueError if not valid hex

    def test_single_gpu_selects_no_shard(self):
        strategy = _select_strategy(device_count=1, per_device_vram_bytes=24 * _GIB)
        assert strategy == STRATEGY_NO_SHARD


# ---------------------------------------------------------------------------
# TestRankEvidenceCollector (6)
# ---------------------------------------------------------------------------

class TestRankEvidenceCollector:
    def test_invalid_rank_raises_dojo_error(self):
        from dojo.core.exceptions import DojoError
        with pytest.raises(DojoError):
            RankEvidenceCollector(rank=5, world_size=2)

    def test_record_appends_to_events(self):
        collector = RankEvidenceCollector(rank=0, world_size=2, mock_mode=True)
        collector.record({"step": 1, "loss": 0.9})
        assert len(collector._events) == 1

    def test_record_emits_fsdp2_training_step_evidence(self):
        collector = RankEvidenceCollector(rank=0, world_size=2, mock_mode=True)
        collector.record({"step": 1, "loss": 0.9})
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.FSDP2_TRAINING_STEP.value in types

    def test_merkle_root_changes_as_events_added(self):
        collector = RankEvidenceCollector(rank=0, world_size=1, mock_mode=True)
        root_empty = collector.compute_rank_merkle_root()
        collector.record({"step": 1})
        root_one = collector.compute_rank_merkle_root()
        assert root_empty != root_one

    def test_seal_returns_rank_evidence_bundle(self):
        collector = RankEvidenceCollector(rank=0, world_size=2, mock_mode=True)
        collector.record({"step": 1, "loss": 0.5})
        bundle = collector.seal(steps_completed=1, final_loss=0.5, sharding_strategy="FULL_SHARD")
        assert isinstance(bundle, RankEvidenceBundle)
        assert bundle.rank == 0
        assert bundle.world_size == 2
        assert bundle.steps_completed == 1
        assert bundle.final_loss == 0.5
        assert bundle.sharding_strategy == "FULL_SHARD"

    def test_seal_emits_fsdp2_rank_merkle_root(self):
        collector = RankEvidenceCollector(rank=0, world_size=1, mock_mode=True)
        collector.record({"step": 1})
        collector.seal(steps_completed=1, final_loss=0.5, sharding_strategy="NO_SHARD")
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.FSDP2_RANK_MERKLE_ROOT.value in types


# ---------------------------------------------------------------------------
# TestRankEvidenceAggregation (4)
# ---------------------------------------------------------------------------

class TestRankEvidenceAggregation:
    def test_empty_list_returns_valid_hash(self):
        root = aggregate_rank_merkle_roots([])
        assert len(root) == 64
        int(root, 16)

    def test_single_bundle_produces_stable_hash(self):
        b = _make_bundle()
        assert aggregate_rank_merkle_roots([b]) == aggregate_rank_merkle_roots([b])

    def test_two_bundles_differ_from_one(self):
        b0 = _make_bundle(rank=0, world_size=2)
        b1 = _make_bundle(rank=1, world_size=2)
        root_one = aggregate_rank_merkle_roots([b0])
        root_two = aggregate_rank_merkle_roots([b0, b1])
        assert root_one != root_two

    def test_order_independent(self):
        b0 = _make_bundle(rank=0, world_size=2)
        b1 = _make_bundle(rank=1, world_size=2)
        assert aggregate_rank_merkle_roots([b0, b1]) == aggregate_rank_merkle_roots([b1, b0])


# ---------------------------------------------------------------------------
# TestFSDP2Config (3)
# ---------------------------------------------------------------------------

class TestFSDP2Config:
    def test_default_sharding_strategy_is_full_shard(self):
        cfg = FSDP2Config()
        assert cfg.sharding_strategy == "FULL_SHARD"

    def test_use_orig_params_true_by_default(self):
        # Required for QLoRA+FSDP2 compat — gradients must flow through LoRA adapters.
        cfg = FSDP2Config()
        assert cfg.use_orig_params is True

    def test_config_hash_is_stable_64_chars(self):
        cfg = FSDP2Config()
        h1 = cfg.config_hash()
        h2 = cfg.config_hash()
        assert h1 == h2
        assert len(h1) == 64


# ---------------------------------------------------------------------------
# TestFSDP2Train_FSM (4)
# ---------------------------------------------------------------------------

class TestFSDP2Train_FSM:
    _hal = _make_hal()

    def test_happy_path_returns_finetune_result(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        result = fsdp2_train(cfg, self._hal, mock_mode=True)
        assert isinstance(result, FinetuneResult)

    def test_invalid_strategy_raises_fsdp2_config_error(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        bad_cfg = FSDP2Config(sharding_strategy="MADE_UP_STRATEGY")
        with pytest.raises(FSDP2ConfigError):
            fsdp2_train(cfg, self._hal, fsdp2_cfg=bad_cfg, mock_mode=True)

    def test_nan_loss_raises_loss_divergence_error(self, tmp_path, monkeypatch):
        def _fake_simulate(*args, **kwargs):
            return (1, float("nan"))
        monkeypatch.setattr("dojo.train.fsdp2._simulate_fsdp2_steps", _fake_simulate)
        cfg = _make_cfg(tmp_path)
        with pytest.raises(LossDivergenceError):
            fsdp2_train(cfg, self._hal, mock_mode=True)

    def test_world_size_zero_auto_detects_from_mock_profile(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        fsdp2_cfg = FSDP2Config(world_size=0)
        result = fsdp2_train(cfg, self._hal, fsdp2_cfg=fsdp2_cfg, mock_mode=True)
        # mock_mode detect_multi_gpu returns 2 GPUs → world_size=2 → 2 rank bundles
        assert isinstance(result, FinetuneResult)
        assert result.steps_completed > 0


# ---------------------------------------------------------------------------
# TestFSDP2Train_Evidence (4)
# ---------------------------------------------------------------------------

class TestFSDP2Train_Evidence:
    _hal = _make_hal()

    def test_emits_fsdp2_profile_detected(self, tmp_path):
        fsdp2_train(_make_cfg(tmp_path), self._hal, mock_mode=True)
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.FSDP2_PROFILE_DETECTED.value in types

    def test_emits_fsdp2_rank_initialized(self, tmp_path):
        fsdp2_train(_make_cfg(tmp_path), self._hal, mock_mode=True)
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.FSDP2_RANK_INITIALIZED.value in types

    def test_emits_fsdp2_training_complete(self, tmp_path):
        fsdp2_train(_make_cfg(tmp_path), self._hal, mock_mode=True)
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.FSDP2_TRAINING_COMPLETE.value in types

    def test_emits_fsdp2_rank_merkle_root(self, tmp_path):
        fsdp2_train(_make_cfg(tmp_path), self._hal, mock_mode=True)
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.FSDP2_RANK_MERKLE_ROOT.value in types


# ---------------------------------------------------------------------------
# TestFSDP2Train_Result (3)
# ---------------------------------------------------------------------------

class TestFSDP2Train_Result:
    _hal = _make_hal()

    def test_training_receipt_is_64_hex_chars(self, tmp_path):
        result = fsdp2_train(_make_cfg(tmp_path), self._hal, mock_mode=True)
        assert len(result.training_receipt) == 64
        int(result.training_receipt, 16)

    def test_backend_used_is_fsdp2(self, tmp_path):
        result = fsdp2_train(_make_cfg(tmp_path), self._hal, mock_mode=True)
        assert result.backend_used == "fsdp2"

    def test_steps_completed_greater_than_zero(self, tmp_path):
        cfg = _make_cfg(tmp_path, num_epochs=1, grad_accum_steps=3)
        result = fsdp2_train(cfg, self._hal, mock_mode=True)
        assert result.steps_completed > 0


# ---------------------------------------------------------------------------
# TestFSDP2AllStrategies (4)
# ---------------------------------------------------------------------------

class TestFSDP2AllStrategies:
    _hal = _make_hal()

    def test_full_shard_completes(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        result = fsdp2_train(cfg, self._hal,
                             fsdp2_cfg=FSDP2Config(sharding_strategy="FULL_SHARD"),
                             mock_mode=True)
        assert isinstance(result, FinetuneResult)

    def test_shard_grad_op_completes(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        result = fsdp2_train(cfg, self._hal,
                             fsdp2_cfg=FSDP2Config(sharding_strategy="SHARD_GRAD_OP"),
                             mock_mode=True)
        assert isinstance(result, FinetuneResult)

    def test_no_shard_completes(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        result = fsdp2_train(cfg, self._hal,
                             fsdp2_cfg=FSDP2Config(sharding_strategy="NO_SHARD"),
                             mock_mode=True)
        assert isinstance(result, FinetuneResult)

    def test_hybrid_shard_completes(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        result = fsdp2_train(cfg, self._hal,
                             fsdp2_cfg=FSDP2Config(sharding_strategy="HYBRID_SHARD"),
                             mock_mode=True)
        assert isinstance(result, FinetuneResult)


# ---------------------------------------------------------------------------
# TestFSDP2CPUOffload (1)
# ---------------------------------------------------------------------------

class TestFSDP2CPUOffload:
    def test_cpu_offload_params_accepted_in_mock_mode(self, tmp_path):
        hal = _make_hal()
        cfg = _make_cfg(tmp_path)
        fsdp2_cfg = FSDP2Config(cpu_offload_params=True, cpu_offload_activations=True)
        result = fsdp2_train(cfg, hal, fsdp2_cfg=fsdp2_cfg, mock_mode=True)
        assert isinstance(result, FinetuneResult)
