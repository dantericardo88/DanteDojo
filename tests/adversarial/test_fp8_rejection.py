"""Adversarial tests: FP8 rejection on Ampere and below.

These tests verify that the FP8 precision gate is hard — no environment
variable, monkey-patch path, or CC override can bypass the rejection.
All negative cases must raise PrecisionUnsupported; no other exception type
is acceptable.
"""

from __future__ import annotations

import os

import pytest

from dojo.core.exceptions import PrecisionUnsupported
from dojo.core.types import HALProfile, HardwareTier
from dojo.precision.fp8 import FP8Handler, is_fp8_supported


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _profile(cc: str, tier: HardwareTier = HardwareTier.TIER_3090, gpu_name: str = "Test GPU") -> HALProfile:
    """Build a minimal HALProfile with the given compute capability."""
    return HALProfile(
        tier=tier,
        gpu_name=gpu_name,
        vram_bytes=24 * 1024 ** 3,
        compute_capability=cc,
        has_native_fp8=float(cc) >= 8.9,
        has_native_fp4=float(cc) >= 10.0,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="test",
        profile_sha256="a" * 64,
    )


# ---------------------------------------------------------------------------
# Ampere rejection
# ---------------------------------------------------------------------------

class TestAmpereRejection:
    """FP8 must be categorically rejected on Ampere (SM < 8.9)."""

    def test_rtx_3090_raises_precision_unsupported(self):
        ampere = _profile("8.6", tier=HardwareTier.TIER_3090, gpu_name="NVIDIA GeForce RTX 3090")
        with pytest.raises(PrecisionUnsupported):
            FP8Handler.ensure_supported(ampere)

    def test_cc_8_6_raises(self):
        with pytest.raises(PrecisionUnsupported):
            FP8Handler.ensure_supported(_profile("8.6"))

    def test_cc_8_0_raises(self):
        with pytest.raises(PrecisionUnsupported):
            FP8Handler.ensure_supported(_profile("8.0"))

    def test_cc_7_5_raises(self):
        """Turing (SM 7.5) also lacks FP8."""
        with pytest.raises(PrecisionUnsupported):
            FP8Handler.ensure_supported(_profile("7.5"))

    def test_cc_8_6_error_message_mentions_sm(self):
        """The error message must reference the actual SM version for debuggability."""
        ampere = _profile("8.6")
        with pytest.raises(PrecisionUnsupported) as exc_info:
            FP8Handler.ensure_supported(ampere)
        assert "8.6" in str(exc_info.value) or "8.6" in str(exc_info.value.args[0])

    def test_cc_8_6_error_has_remediation(self):
        """The error reason must mention BF16 or Marlin as the routing alternative."""
        ampere = _profile("8.6")
        with pytest.raises(PrecisionUnsupported) as exc_info:
            FP8Handler.ensure_supported(ampere)
        reason = exc_info.value.reason.lower()
        assert "bf16" in reason or "marlin" in reason or "w8a16" in reason

    def test_is_fp8_supported_false_for_ampere(self):
        assert is_fp8_supported(_profile("8.6")) is False

    def test_is_fp8_supported_false_for_cc_8_0(self):
        assert is_fp8_supported(_profile("8.0")) is False


# ---------------------------------------------------------------------------
# Ada / Hopper boundary (SM 8.9+)
# ---------------------------------------------------------------------------

class TestAdaBoundary:
    """SM 8.9 is the exact boundary — must not raise."""

    def test_cc_8_9_does_not_raise(self):
        ada = _profile("8.9", tier=HardwareTier.TIER_4060_LAPTOP, gpu_name="NVIDIA GeForce RTX 4060")
        FP8Handler.ensure_supported(ada)  # must not raise

    def test_cc_8_9_is_supported(self):
        assert is_fp8_supported(_profile("8.9")) is True

    def test_cc_9_0_hopper_does_not_raise(self):
        hopper = _profile("9.0", tier=HardwareTier.TIER_H100, gpu_name="NVIDIA H100")
        FP8Handler.ensure_supported(hopper)  # must not raise

    def test_cc_9_0_is_supported(self):
        assert is_fp8_supported(_profile("9.0")) is True

    def test_cc_10_0_blackwell_does_not_raise(self):
        blackwell = _profile("10.0", tier=HardwareTier.TIER_H100, gpu_name="NVIDIA B200")
        FP8Handler.ensure_supported(blackwell)  # must not raise

    def test_cc_10_0_is_supported(self):
        assert is_fp8_supported(_profile("10.0")) is True


# ---------------------------------------------------------------------------
# No environment-variable bypass exists
# ---------------------------------------------------------------------------

class TestNoEnvBypass:
    """Verify there is no undocumented environment override for FP8 gating."""

    def test_env_override_has_no_effect(self, monkeypatch):
        """Setting any plausible bypass env var must not suppress the error."""
        monkeypatch.setenv("DOJO_FORCE_FP8", "1")
        monkeypatch.setenv("DOJO_BYPASS_FP8", "1")
        monkeypatch.setenv("FORCE_FP8", "1")
        monkeypatch.setenv("DOJO_BYPASS", "")  # empty → not set, safe

        ampere = _profile("8.6")
        with pytest.raises(PrecisionUnsupported):
            FP8Handler.ensure_supported(ampere)

    def test_is_fp8_supported_unaffected_by_env(self, monkeypatch):
        monkeypatch.setenv("DOJO_FORCE_FP8", "1")
        assert is_fp8_supported(_profile("8.6")) is False


# ---------------------------------------------------------------------------
# Double-dispatch: ensure raise type is always PrecisionUnsupported
# ---------------------------------------------------------------------------

class TestExceptionType:
    """PrecisionUnsupported (not DojoError base or RuntimeError) must be raised."""

    @pytest.mark.parametrize("cc", ["7.0", "7.5", "8.0", "8.6", "8.7", "8.8"])
    def test_correct_exception_type_below_89(self, cc: str):
        profile = _profile(cc)
        exc = None
        try:
            FP8Handler.ensure_supported(profile)
        except PrecisionUnsupported as e:
            exc = e
        except Exception as e:
            pytest.fail(f"Expected PrecisionUnsupported, got {type(e).__name__}: {e}")
        assert exc is not None, f"Expected PrecisionUnsupported for CC {cc}, but no exception raised"

    @pytest.mark.parametrize("cc", ["8.9", "9.0", "10.0"])
    def test_no_exception_at_or_above_89(self, cc: str):
        profile = _profile(cc)
        # Must not raise any exception
        FP8Handler.ensure_supported(profile)
