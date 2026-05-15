"""Unit tests for dojo/train/grpo_trainer.py and dojo/train/dpo_trainer.py.

Covers:
- GRPOConfig: defaults, config_hash stability, to_dict completeness
- DPOConfig: defaults, config_hash stability, to_dict completeness
- grpo_train(): returns RLTrainResult with correct fields
- grpo_train(): runs FSM from IDLE to COMPLETED
- grpo_train(): emits GRPO_START, GRPO_STEP, GRPO_KL_CHECKED, GRPO_CHECKPOINT, GRPO_COMPLETE
- grpo_train(): every step has reward_mean and kl_div in evidence
- grpo_train(): raises KLExplosionError when max_kl is breached
- grpo_train(): BackendUnavailable when DOJO_REAL_GPU=1 and TRL absent
- dpo_train(): returns RLTrainResult with correct fields
- dpo_train(): runs FSM from IDLE to COMPLETED
- dpo_train(): emits DPO_START, DPO_STEP, DPO_REWARD_MARGIN_CHECKED, DPO_CHECKPOINT, DPO_COMPLETE
- dpo_train(): every step has chosen_reward, rejected_reward, reward_margin
- dpo_train(): raises DPORewardCollapse when margin below floor for max_collapse_steps
- dpo_train(): BackendUnavailable when DOJO_REAL_GPU=1 and TRL absent
- RLTrainResult: to_dict has all required keys
- Constitutional: training_receipt is non-empty Merkle root in both trainers
- Constitutional: GRPO_STEP count matches expected steps
- Constitutional: DPO_STEP count matches expected steps
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import (
    BackendUnavailable,
    DPORewardCollapse,
    KLExplosionError,
)
from dojo.core.types import HALProfile, HardwareTier
from dojo.train.dpo_trainer import dpo_train
from dojo.train.grpo_trainer import grpo_train
from dojo.train.rl_config import DPOConfig, GRPOConfig, RLTrainResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
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


@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    monkeypatch.delenv("DOJO_REAL_GPU", raising=False)


# ---------------------------------------------------------------------------
# GRPOConfig
# ---------------------------------------------------------------------------

class TestGRPOConfig:
    def test_defaults_are_conservative(self):
        cfg = GRPOConfig(pack_id="p", base_model="m")
        assert cfg.batch_size <= 2
        assert cfg.num_generations >= 4
        assert cfg.max_kl > 0.0
        assert cfg.beta > 0.0

    def test_config_hash_stable_and_64_chars(self):
        cfg = GRPOConfig(pack_id="p", base_model="m")
        assert cfg.config_hash() == cfg.config_hash()
        assert len(cfg.config_hash()) == 64

    def test_config_hash_differs_by_pack_id(self):
        assert (
            GRPOConfig(pack_id="a", base_model="m").config_hash()
            != GRPOConfig(pack_id="b", base_model="m").config_hash()
        )

    def test_to_dict_has_all_required_keys(self):
        cfg = GRPOConfig(pack_id="p", base_model="m")
        d = cfg.to_dict()
        for key in ("pack_id", "base_model", "beta", "max_kl", "num_generations",
                    "lora_r", "batch_size", "learning_rate", "output_dir"):
            assert key in d

    def test_output_dir_is_path(self):
        cfg = GRPOConfig(pack_id="p", base_model="m")
        assert isinstance(cfg.output_dir, Path)


# ---------------------------------------------------------------------------
# DPOConfig
# ---------------------------------------------------------------------------

class TestDPOConfig:
    def test_defaults_are_conservative(self):
        cfg = DPOConfig(pack_id="p", base_model="m")
        assert cfg.batch_size <= 2
        assert cfg.beta > 0.0
        assert cfg.min_reward_margin >= 0.0
        assert cfg.max_collapse_steps >= 1

    def test_config_hash_stable_and_64_chars(self):
        cfg = DPOConfig(pack_id="p", base_model="m")
        assert cfg.config_hash() == cfg.config_hash()
        assert len(cfg.config_hash()) == 64

    def test_config_hash_differs_by_loss_type(self):
        a = DPOConfig(pack_id="p", base_model="m", loss_type="sigmoid")
        b = DPOConfig(pack_id="p", base_model="m", loss_type="ipo")
        assert a.config_hash() != b.config_hash()

    def test_to_dict_has_all_required_keys(self):
        cfg = DPOConfig(pack_id="p", base_model="m")
        d = cfg.to_dict()
        for key in ("pack_id", "base_model", "beta", "min_reward_margin",
                    "loss_type", "lora_r", "batch_size", "learning_rate", "output_dir"):
            assert key in d

    def test_loss_type_options(self):
        for lt in ("sigmoid", "ipo", "kto_pair"):
            DPOConfig(pack_id="p", base_model="m", loss_type=lt)


# ---------------------------------------------------------------------------
# grpo_train() — happy path
# ---------------------------------------------------------------------------

class TestGRPOTrainHappyPath:
    _hal = _make_hal()

    def test_returns_rl_train_result(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        result = grpo_train(cfg, self._hal)
        assert isinstance(result, RLTrainResult)

    def test_result_fields_populated(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        r = grpo_train(cfg, self._hal)
        assert r.base_model == "qwen/qwen2-7b"
        assert r.adapter_path == tmp_path
        assert r.algorithm == "grpo"
        assert r.wall_time_seconds >= 0.0
        assert r.steps_completed > 0
        assert r.backend_used == "native_pytorch"

    def test_training_receipt_is_nonempty_merkle_root(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        r = grpo_train(cfg, self._hal)
        assert len(r.training_receipt) == 64
        assert r.training_receipt != "0" * 64

    def test_result_to_dict_has_all_keys(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        r = grpo_train(cfg, self._hal)
        d = r.to_dict()
        for key in ("adapter_path", "base_model", "training_receipt",
                    "final_reward_mean", "steps_completed", "algorithm"):
            assert key in d


# ---------------------------------------------------------------------------
# grpo_train() — constitutional evidence emission
# ---------------------------------------------------------------------------

class TestGRPOTrainEvidence:
    _hal = _make_hal()

    def test_emits_grpo_start(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        grpo_train(cfg, self._hal)
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.GRPO_START.value in types

    def test_emits_grpo_step_events(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path,
                         num_epochs=1, grad_accum_steps=3)
        grpo_train(cfg, self._hal)
        step_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.GRPO_STEP.value
        ]
        assert len(step_events) == 3

    def test_grpo_step_events_have_reward_and_kl(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path,
                         num_epochs=1, grad_accum_steps=2)
        grpo_train(cfg, self._hal)
        step_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.GRPO_STEP.value
        ]
        for ev in step_events:
            data = ev.get("data", {})
            assert "reward_mean" in data
            assert "kl_div" in data
            assert "clip_fraction" in data
            assert "num_generations" in data

    def test_emits_grpo_kl_checked_every_step(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path,
                         num_epochs=1, grad_accum_steps=4)
        grpo_train(cfg, self._hal)
        kl_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.GRPO_KL_CHECKED.value
        ]
        assert len(kl_events) == 4

    def test_emits_grpo_checkpoint_per_epoch(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path,
                         num_epochs=2, grad_accum_steps=2)
        grpo_train(cfg, self._hal)
        ckpt_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.GRPO_CHECKPOINT.value
        ]
        assert len(ckpt_events) == 2

    def test_emits_grpo_complete(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        grpo_train(cfg, self._hal)
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.GRPO_COMPLETE.value in types

    def test_grpo_complete_carries_receipt(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        grpo_train(cfg, self._hal)
        complete_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.GRPO_COMPLETE.value
        ]
        assert len(complete_events) == 1
        assert len(complete_events[0].get("data", {}).get("training_receipt", "")) == 64

    def test_fsm_transitions_emitted_as_evidence(self, tmp_path):
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        grpo_train(cfg, self._hal)
        fsm_events = [
            e for e in EvidenceLogger.get_events()
            if "fsm.transition" in e.get("type", "")
        ]
        assert len(fsm_events) >= 6


# ---------------------------------------------------------------------------
# grpo_train() — constitutional guards
# ---------------------------------------------------------------------------

class TestGRPOTrainGuards:
    _hal = _make_hal()

    def test_kl_explosion_error_when_max_kl_too_low(self, tmp_path):
        # max_kl=0.0: step 1 kl≈0 (policy≡reference), step 2 kl>0 after first
        # optimizer update → KLExplosionError fires on the second step.
        cfg = GRPOConfig(
            pack_id="p", base_model="m", output_dir=tmp_path,
            max_kl=0.0, num_epochs=1, grad_accum_steps=2,
        )
        with pytest.raises(KLExplosionError) as exc_info:
            grpo_train(cfg, self._hal)
        assert "kl" in str(exc_info.value).lower()

    def test_kl_explosion_emits_violated_evidence(self, tmp_path):
        cfg = GRPOConfig(
            pack_id="p", base_model="m", output_dir=tmp_path,
            max_kl=0.0, num_epochs=1, grad_accum_steps=2,
        )
        with pytest.raises(KLExplosionError):
            grpo_train(cfg, self._hal)
        violated = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.GRPO_KL_CHECKED.value
            and e.get("data", {}).get("status") == "VIOLATED"
        ]
        assert len(violated) >= 1

    def test_backend_unavailable_when_real_gpu_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DOJO_REAL_GPU", "1")
        monkeypatch.setattr(
            "dojo.train.grpo_trainer._is_trl_available", lambda: False
        )
        cfg = GRPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        with pytest.raises(BackendUnavailable):
            grpo_train(cfg, self._hal)


# ---------------------------------------------------------------------------
# dpo_train() — happy path
# ---------------------------------------------------------------------------

class TestDPOTrainHappyPath:
    _hal = _make_hal()

    def test_returns_rl_train_result(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        result = dpo_train(cfg, self._hal)
        assert isinstance(result, RLTrainResult)

    def test_result_fields_populated(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="qwen/qwen2-7b", output_dir=tmp_path)
        r = dpo_train(cfg, self._hal)
        assert r.base_model == "qwen/qwen2-7b"
        assert r.adapter_path == tmp_path
        assert r.algorithm == "dpo"
        assert r.wall_time_seconds >= 0.0
        assert r.steps_completed > 0
        assert r.backend_used == "native_pytorch"

    def test_training_receipt_is_nonempty_merkle_root(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        r = dpo_train(cfg, self._hal)
        assert len(r.training_receipt) == 64
        assert r.training_receipt != "0" * 64

    def test_result_to_dict_has_all_keys(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        r = dpo_train(cfg, self._hal)
        d = r.to_dict()
        for key in ("adapter_path", "base_model", "training_receipt",
                    "final_reward_mean", "steps_completed", "algorithm"):
            assert key in d

    def test_final_reward_mean_is_positive(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path,
                        num_epochs=2, grad_accum_steps=4)
        r = dpo_train(cfg, self._hal)
        assert r.final_reward_mean > 0.0


# ---------------------------------------------------------------------------
# dpo_train() — constitutional evidence emission
# ---------------------------------------------------------------------------

class TestDPOTrainEvidence:
    _hal = _make_hal()

    def test_emits_dpo_start(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        dpo_train(cfg, self._hal)
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.DPO_START.value in types

    def test_emits_dpo_step_events(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path,
                        num_epochs=1, grad_accum_steps=3)
        dpo_train(cfg, self._hal)
        step_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.DPO_STEP.value
        ]
        assert len(step_events) == 3

    def test_dpo_step_events_have_chosen_and_rejected_rewards(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path,
                        num_epochs=1, grad_accum_steps=2)
        dpo_train(cfg, self._hal)
        step_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.DPO_STEP.value
        ]
        for ev in step_events:
            data = ev.get("data", {})
            assert "chosen_reward" in data
            assert "rejected_reward" in data
            assert "reward_margin" in data
            assert "accuracy" in data
            assert "beta" in data

    def test_emits_dpo_reward_margin_checked_every_step(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path,
                        num_epochs=1, grad_accum_steps=4)
        dpo_train(cfg, self._hal)
        margin_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.DPO_REWARD_MARGIN_CHECKED.value
        ]
        assert len(margin_events) == 4

    def test_emits_dpo_checkpoint_per_epoch(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path,
                        num_epochs=2, grad_accum_steps=2)
        dpo_train(cfg, self._hal)
        ckpt_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.DPO_CHECKPOINT.value
        ]
        assert len(ckpt_events) == 2

    def test_emits_dpo_complete(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        dpo_train(cfg, self._hal)
        types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.DPO_COMPLETE.value in types

    def test_dpo_complete_carries_receipt(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        dpo_train(cfg, self._hal)
        complete_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.DPO_COMPLETE.value
        ]
        assert len(complete_events) == 1
        assert len(complete_events[0].get("data", {}).get("training_receipt", "")) == 64

    def test_fsm_transitions_emitted_as_evidence(self, tmp_path):
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        dpo_train(cfg, self._hal)
        fsm_events = [
            e for e in EvidenceLogger.get_events()
            if "fsm.transition" in e.get("type", "")
        ]
        assert len(fsm_events) >= 6


# ---------------------------------------------------------------------------
# dpo_train() — constitutional guards
# ---------------------------------------------------------------------------

class TestDPOTrainGuards:
    _hal = _make_hal()

    def test_reward_collapse_when_margin_floor_too_high(self, tmp_path):
        # Set min_reward_margin so high and max_collapse_steps so low that
        # the first steps trigger DPORewardCollapse immediately.
        cfg = DPOConfig(
            pack_id="p", base_model="m", output_dir=tmp_path,
            min_reward_margin=100.0,    # impossible to satisfy
            max_collapse_steps=1,
            num_epochs=1, grad_accum_steps=3,
        )
        with pytest.raises(DPORewardCollapse) as exc_info:
            dpo_train(cfg, self._hal)
        assert "collapse" in str(exc_info.value).lower() or "margin" in str(exc_info.value).lower()

    def test_reward_collapse_emits_below_floor_evidence(self, tmp_path):
        cfg = DPOConfig(
            pack_id="p", base_model="m", output_dir=tmp_path,
            min_reward_margin=100.0, max_collapse_steps=1,
            num_epochs=1, grad_accum_steps=2,
        )
        with pytest.raises(DPORewardCollapse):
            dpo_train(cfg, self._hal)
        below_floor = [
            e for e in EvidenceLogger.get_events()
            if e.get("type", "") == EvidenceType.DPO_REWARD_MARGIN_CHECKED.value
            and e.get("data", {}).get("status") == "BELOW_FLOOR"
        ]
        assert len(below_floor) >= 1

    def test_backend_unavailable_when_real_gpu_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DOJO_REAL_GPU", "1")
        monkeypatch.setattr(
            "dojo.train.dpo_trainer._is_trl_available", lambda: False
        )
        cfg = DPOConfig(pack_id="p", base_model="m", output_dir=tmp_path)
        with pytest.raises(BackendUnavailable):
            dpo_train(cfg, self._hal)


# ---------------------------------------------------------------------------
# Cross-trainer: separate Merkle roots (isolated evidence chains)
# ---------------------------------------------------------------------------

class TestCrossTrainerIsolation:
    _hal = _make_hal()

    def test_grpo_and_dpo_receipts_differ(self, tmp_path):
        """Each trainer run produces a unique Merkle root."""
        EvidenceLogger.clear()
        grpo_cfg = GRPOConfig(
            pack_id="p", base_model="m", output_dir=tmp_path / "g",
            num_epochs=1, grad_accum_steps=2,
        )
        grpo_result = grpo_train(grpo_cfg, self._hal)

        EvidenceLogger.clear()
        dpo_cfg = DPOConfig(
            pack_id="p", base_model="m", output_dir=tmp_path / "d",
            num_epochs=1, grad_accum_steps=2,
        )
        dpo_result = dpo_train(dpo_cfg, self._hal)

        assert grpo_result.training_receipt != dpo_result.training_receipt

    def test_both_results_are_rl_train_result(self, tmp_path):
        grpo_cfg = GRPOConfig(pack_id="p", base_model="m",
                              output_dir=tmp_path / "g")
        EvidenceLogger.clear()
        gr = grpo_train(grpo_cfg, self._hal)
        assert isinstance(gr, RLTrainResult)

        dpo_cfg = DPOConfig(pack_id="p", base_model="m",
                            output_dir=tmp_path / "d")
        EvidenceLogger.clear()
        dr = dpo_train(dpo_cfg, self._hal)
        assert isinstance(dr, RLTrainResult)
