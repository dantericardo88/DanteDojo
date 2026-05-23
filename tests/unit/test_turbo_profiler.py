"""Unit tests for dojo/turbo/ — mock HALProfile → assert correct config selected."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _clear_evidence():
    from dojo.core.evidence import EvidenceLogger
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


class TestPolicySelection:
    """Verify that each hardware tier maps to the expected policies."""

    def test_ada_gets_fp8(self):
        from dojo.core.types import HALProfile, HardwareTier
        from dojo.turbo.policy import PolicyType, get_policies_for_profile

        ada = HALProfile(
            tier=HardwareTier.TIER_4060_LAPTOP,
            gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
            vram_bytes=8 * 1024 ** 3,
            compute_capability="8.9",
            has_native_fp8=True,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=4,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="test",
            profile_sha256="a" * 64,
        )
        policies = get_policies_for_profile(ada)
        assert PolicyType.FP8_QUANT in policies
        assert PolicyType.BF16_DEFAULT not in policies

    def test_ampere_gets_bf16_not_fp8(self):
        from dojo.core.types import HALProfile, HardwareTier
        from dojo.turbo.policy import PolicyType, get_policies_for_profile

        ampere = HALProfile(
            tier=HardwareTier.TIER_3090,
            gpu_name="NVIDIA GeForce RTX 3090",
            vram_bytes=24 * 1024 ** 3,
            compute_capability="8.6",
            has_native_fp8=False,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=4,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="test",
            profile_sha256="b" * 64,
        )
        policies = get_policies_for_profile(ampere)
        assert PolicyType.BF16_DEFAULT in policies
        assert PolicyType.FP8_QUANT not in policies

    def test_h100_gets_fp8(self):
        from dojo.core.types import HALProfile, HardwareTier
        from dojo.turbo.policy import PolicyType, get_policies_for_profile

        h100 = HALProfile(
            tier=HardwareTier.TIER_H100,
            gpu_name="NVIDIA H100 SXM5",
            vram_bytes=80 * 1024 ** 3,
            compute_capability="9.0",
            has_native_fp8=True,
            has_native_fp4=False,
            has_nvlink=True,
            pcie_gen=5,
            interconnect_bandwidth_gb_s=900.0,
            memory_bandwidth_gb_s=3350.0,
            driver_version="test",
            profile_sha256="c" * 64,
        )
        policies = get_policies_for_profile(h100)
        assert PolicyType.FP8_QUANT in policies
        assert PolicyType.HET_COMM in policies


class TestParetoOptimizer:
    """Verify Pareto selection logic."""

    def _make_result(self, precision, tps, ppl, quant="none", success=True):
        from dojo.turbo.config import ProfileResult
        from dojo.turbo.policy import PolicyType
        return ProfileResult(
            policy=PolicyType.BF16_DEFAULT,
            quant_scheme=quant,
            precision=precision,
            tokens_per_sec=tps,
            ppl=ppl,
            vram_gb=2.0,
            success=success,
        )

    def test_selects_fastest_within_budget(self):
        from dojo.turbo.optimizer import select_optimal

        bf16 = self._make_result("fp16", tps=1000.0, ppl=10.0)
        fp8 = self._make_result("fp8", tps=1200.0, ppl=10.5)   # +5% PPL, within 15%

        winner = select_optimal([bf16, fp8], ppl_budget=0.15)
        assert winner.precision == "fp8"
        assert winner.tokens_per_sec == 1200.0

    def test_rejects_fp8_if_ppl_exceeds_budget(self):
        from dojo.turbo.optimizer import select_optimal

        bf16 = self._make_result("fp16", tps=1000.0, ppl=10.0)
        fp8 = self._make_result("fp8", tps=1200.0, ppl=12.0)   # +20% PPL, over 15%

        winner = select_optimal([bf16, fp8], ppl_budget=0.15)
        assert winner.precision == "fp16"  # falls back to baseline

    def test_fallback_to_baseline_if_all_quantized_fail(self):
        from dojo.turbo.optimizer import select_optimal

        bf16 = self._make_result("fp16", tps=1000.0, ppl=10.0)
        fp8_fail = self._make_result("fp8", tps=0.0, ppl=9999.0, success=False)

        winner = select_optimal([bf16, fp8_fail], ppl_budget=0.15)
        assert winner.precision == "fp16"

    def test_empty_results_raises(self):
        from dojo.turbo.optimizer import select_optimal
        with pytest.raises(ValueError):
            select_optimal([], ppl_budget=0.15)

    def test_all_failed_results_raises(self):
        from dojo.turbo.optimizer import select_optimal
        fp8_fail = self._make_result("fp8", tps=0.0, ppl=9999.0, success=False)
        with pytest.raises(ValueError):
            select_optimal([fp8_fail], ppl_budget=0.15)


class TestTurbochargeOrchestrator:
    """Verify turbocharge() orchestrator wires profiler + optimizer + evidence."""

    def test_turbocharge_emits_evidence_and_returns_result(self, tmp_path, monkeypatch):
        """Turbocharge with mocked profilers should emit 3+ evidence events."""
        from dojo.turbo.config import ProfileResult, TurboConfig
        from dojo.turbo.policy import PolicyType

        bf16_result = ProfileResult(
            policy=PolicyType.BF16_DEFAULT,
            quant_scheme="none",
            precision="fp16",
            tokens_per_sec=800.0,
            ppl=10.0,
            vram_gb=3.0,
        )
        fp8_result = ProfileResult(
            policy=PolicyType.FP8_QUANT,
            quant_scheme="none",
            precision="fp8",
            tokens_per_sec=1100.0,
            ppl=10.3,
            vram_gb=2.8,
        )

        import dojo.turbo.turbocharge as tc_mod

        monkeypatch.setattr(tc_mod, "profile_bf16", lambda m, c, p: bf16_result)
        monkeypatch.setattr(tc_mod, "profile_fp8", lambda m, c, p: fp8_result)

        # Mock HAL detect to return Ada GPU
        from dojo.core.types import HALProfile, HardwareTier
        mock_hal = HALProfile(
            tier=HardwareTier.TIER_4060_LAPTOP,
            gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
            vram_bytes=8 * 1024 ** 3,
            compute_capability="8.9",
            has_native_fp8=True,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=4,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="test",
            profile_sha256="d" * 64,
        )

        import dojo.hal.detector as det_mod
        monkeypatch.setattr(det_mod, "detect", lambda: mock_hal)

        # Mock torch.cuda
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 4060 Laptop GPU")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))

        # Mock config path write
        monkeypatch.setattr(
            "pathlib.Path.home",
            lambda: tmp_path,
        )

        cfg = TurboConfig(
            model="qwen/Qwen2-1.5B",
            ppl_budget=0.15,
            optimize="inference",
            device="cuda:0",
            bench_steps=5,
        )

        result = tc_mod.turbocharge(cfg)

        assert result.winner.precision == "fp8"
        assert result.winner.tokens_per_sec == 1100.0
        assert len(result.all_results) == 2

        from dojo.core.evidence import EvidenceLogger
        events = EvidenceLogger.get_events()
        event_types = [e.get("type", "") for e in events]
        assert "TURBOCHARGE_STARTED" in event_types
        assert "TURBOCHARGE_PROFILE_RESULT" in event_types
        assert "TURBOCHARGE_OPTIMAL_CONFIG" in event_types
        assert "TURBOCHARGE_APPLIED" in event_types
