"""Unit tests for dojo/hal — detector, profile signing, and policy routing.

All pynvml, rocm-smi, and platform calls are mocked so tests run on any machine.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import PrecisionUnsupported
from dojo.core.types import HALProfile, HardwareTier
from dojo.hal.detector import detect
from dojo.hal.profile import sign_profile, verify_profile_signature
from dojo.precision.fp8 import FP8Handler, is_fp8_supported
from dojo.turbo.policy import PolicyType, get_policies_for_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    tier: HardwareTier = HardwareTier.TIER_3090,
    gpu_name: str = "NVIDIA GeForce RTX 3090",
    compute_capability: str = "8.6",
    has_native_fp8: bool = False,
    has_native_fp4: bool = False,
    vram_bytes: int = 24 * 1024 ** 3,
    driver_version: str = "550.0",
) -> HALProfile:
    """Build a HALProfile with a valid (self-consistent) sha256.

    sign_profile() excludes the profile_sha256 field from the signed payload,
    so the sha is a pure function of all other fields.  We build a placeholder
    with profile_sha256="" (value irrelevant), compute the real sha, then
    rebuild with the correct sha embedded.  signed_at is pinned to a fixed
    value so the sha is stable across multiple calls.
    """
    from datetime import datetime, timezone

    fixed_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Any placeholder sha — sign_profile ignores it.
    placeholder = HALProfile(
        tier=tier,
        gpu_name=gpu_name,
        vram_bytes=vram_bytes,
        compute_capability=compute_capability,
        has_native_fp8=has_native_fp8,
        has_native_fp4=has_native_fp4,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version=driver_version,
        profile_sha256="",
        signed_at=fixed_at,
    )
    sha = sign_profile(placeholder)

    return HALProfile(
        tier=tier,
        gpu_name=gpu_name,
        vram_bytes=vram_bytes,
        compute_capability=compute_capability,
        has_native_fp8=has_native_fp8,
        has_native_fp4=has_native_fp4,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version=driver_version,
        profile_sha256=sha,
        signed_at=fixed_at,
    )


_AMPERE_PROFILE = _make_profile(
    tier=HardwareTier.TIER_3090,
    gpu_name="NVIDIA GeForce RTX 3090",
    compute_capability="8.6",
    has_native_fp8=False,
)

_ADA_PROFILE = _make_profile(
    tier=HardwareTier.TIER_4060_LAPTOP,
    gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
    compute_capability="8.9",
    has_native_fp8=True,
)

_CPU_PROFILE = HALProfile(
    tier=HardwareTier.TIER_CPU,
    gpu_name="CPU-only",
    vram_bytes=0,
    compute_capability="0.0",
    has_native_fp8=False,
    has_native_fp4=False,
    has_nvlink=False,
    pcie_gen=0,
    interconnect_bandwidth_gb_s=0.0,
    memory_bandwidth_gb_s=0.0,
    driver_version="none",
    profile_sha256=hashlib.sha256(b"cpu_fallback").hexdigest(),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    """Reset the evidence logger singleton between tests."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# detect() — CPU fallback path
# ---------------------------------------------------------------------------

class TestDetectCPUFallback:
    """When all GPU probes fail/unavailable, detect() must return a CPU HALProfile.

    The sub-detectors are imported locally inside detect(), so we patch them at
    their source module paths (dojo.hal.nvidia.detect_nvidia, etc.).
    """

    def test_returns_hal_profile(self):
        """detect() always returns a HALProfile, even with no GPU."""
        with (
            patch("dojo.hal.nvidia.detect_nvidia", return_value=None),
            patch("dojo.hal.apple.detect_apple", return_value=None),
            patch("dojo.hal.amd.detect_amd", return_value=None),
            patch("dojo.hal.intel.detect_intel", return_value=None),
        ):
            profile = detect()
        assert isinstance(profile, HALProfile)

    def test_falls_back_to_cpu_tier(self):
        """When pynvml is unavailable and no other GPU found, tier must be TIER_CPU."""
        with (
            patch("dojo.hal.nvidia.detect_nvidia", return_value=None),
            patch("dojo.hal.apple.detect_apple", return_value=None),
            patch("dojo.hal.amd.detect_amd", return_value=None),
            patch("dojo.hal.intel.detect_intel", return_value=None),
        ):
            profile = detect()
        assert profile.tier == HardwareTier.TIER_CPU

    def test_cpu_profile_has_no_fp8(self):
        with (
            patch("dojo.hal.nvidia.detect_nvidia", return_value=None),
            patch("dojo.hal.apple.detect_apple", return_value=None),
            patch("dojo.hal.amd.detect_amd", return_value=None),
            patch("dojo.hal.intel.detect_intel", return_value=None),
        ):
            profile = detect()
        assert profile.has_native_fp8 is False
        assert profile.has_native_fp4 is False

    def test_cpu_profile_has_sha256(self):
        with (
            patch("dojo.hal.nvidia.detect_nvidia", return_value=None),
            patch("dojo.hal.apple.detect_apple", return_value=None),
            patch("dojo.hal.amd.detect_amd", return_value=None),
            patch("dojo.hal.intel.detect_intel", return_value=None),
        ):
            profile = detect()
        assert len(profile.profile_sha256) == 64  # SHA-256 hex

    def test_nvidia_profile_returned_when_available(self):
        """detect() returns the NVIDIA profile when the nvidia detector succeeds."""
        with (
            patch("dojo.hal.nvidia.detect_nvidia", return_value=_AMPERE_PROFILE),
            patch("dojo.hal.apple.detect_apple", return_value=None),
            patch("dojo.hal.amd.detect_amd", return_value=None),
            patch("dojo.hal.intel.detect_intel", return_value=None),
        ):
            profile = detect()
        assert profile.tier == HardwareTier.TIER_3090

    def test_detector_swallows_exceptions_from_sub_detectors(self):
        """detect() warns and continues if a sub-detector raises unexpectedly."""
        with (
            patch("dojo.hal.nvidia.detect_nvidia", side_effect=RuntimeError("boom")),
            patch("dojo.hal.apple.detect_apple", return_value=None),
            patch("dojo.hal.amd.detect_amd", return_value=None),
            patch("dojo.hal.intel.detect_intel", return_value=None),
        ):
            profile = detect()
        assert profile.tier == HardwareTier.TIER_CPU


# ---------------------------------------------------------------------------
# HALProfile — hardware capability flags
# ---------------------------------------------------------------------------

class TestHALProfileFlags:
    def test_ampere_has_no_native_fp8(self):
        assert _AMPERE_PROFILE.has_native_fp8 is False

    def test_ampere_compute_capability(self):
        assert _AMPERE_PROFILE.compute_capability == "8.6"

    def test_ada_has_native_fp8(self):
        assert _ADA_PROFILE.has_native_fp8 is True

    def test_ada_compute_capability(self):
        assert _ADA_PROFILE.compute_capability == "8.9"

    def test_cpu_has_zero_vram(self):
        assert _CPU_PROFILE.vram_bytes == 0


# ---------------------------------------------------------------------------
# Profile signing
# ---------------------------------------------------------------------------

class TestProfileSigning:
    def test_sign_profile_returns_64_char_hex(self):
        sha = sign_profile(_AMPERE_PROFILE)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_sign_profile_is_deterministic(self):
        sha1 = sign_profile(_AMPERE_PROFILE)
        sha2 = sign_profile(_AMPERE_PROFILE)
        assert sha1 == sha2

    def test_verify_profile_signature_self_consistent(self):
        # Profile built via _make_profile has matching sha — should verify True.
        # Note: verify_profile_signature checks stored sha == recomputed sha,
        # where recomputed sha is sign_profile over the dict INCLUDING profile_sha256.
        # So a profile is only self-consistent if it was built by re-signing after
        # setting profile_sha256 = sign_profile(first_pass).
        # _make_profile does this correctly, so verify must return True.
        assert verify_profile_signature(_AMPERE_PROFILE) is True

    def test_tampered_sha_fails_verification(self):
        tampered = HALProfile(
            tier=_AMPERE_PROFILE.tier,
            gpu_name=_AMPERE_PROFILE.gpu_name,
            vram_bytes=_AMPERE_PROFILE.vram_bytes,
            compute_capability=_AMPERE_PROFILE.compute_capability,
            has_native_fp8=_AMPERE_PROFILE.has_native_fp8,
            has_native_fp4=_AMPERE_PROFILE.has_native_fp4,
            has_nvlink=_AMPERE_PROFILE.has_nvlink,
            pcie_gen=_AMPERE_PROFILE.pcie_gen,
            interconnect_bandwidth_gb_s=_AMPERE_PROFILE.interconnect_bandwidth_gb_s,
            memory_bandwidth_gb_s=_AMPERE_PROFILE.memory_bandwidth_gb_s,
            driver_version=_AMPERE_PROFILE.driver_version,
            profile_sha256="0" * 64,  # wrong sha
        )
        assert verify_profile_signature(tampered) is False


# ---------------------------------------------------------------------------
# FP8Handler
# ---------------------------------------------------------------------------

class TestFP8Handler:
    def test_ensure_supported_raises_on_ampere(self):
        with pytest.raises(PrecisionUnsupported):
            FP8Handler.ensure_supported(_AMPERE_PROFILE)

    def test_ensure_supported_passes_on_ada(self):
        # Must not raise
        FP8Handler.ensure_supported(_ADA_PROFILE)

    def test_is_fp8_supported_false_for_ampere(self):
        assert FP8Handler.is_fp8_supported(_AMPERE_PROFILE) is False

    def test_is_fp8_supported_true_for_ada(self):
        assert FP8Handler.is_fp8_supported(_ADA_PROFILE) is True

    def test_module_level_helper(self):
        assert is_fp8_supported(_ADA_PROFILE) is True
        assert is_fp8_supported(_AMPERE_PROFILE) is False


# ---------------------------------------------------------------------------
# Turbo policy
# ---------------------------------------------------------------------------

class TestTurboPolicy:
    def test_ampere_gets_bf16_not_fp8(self):
        policies = get_policies_for_profile(_AMPERE_PROFILE)
        assert PolicyType.BF16_DEFAULT in policies
        assert PolicyType.FP8_QUANT not in policies

    def test_ampere_gets_w8a16_marlin(self):
        policies = get_policies_for_profile(_AMPERE_PROFILE)
        assert PolicyType.W8A16_MARLIN_SERVING in policies

    def test_ada_gets_fp8(self):
        policies = get_policies_for_profile(_ADA_PROFILE)
        assert PolicyType.FP8_QUANT in policies

    def test_ada_does_not_get_bf16_default(self):
        policies = get_policies_for_profile(_ADA_PROFILE)
        assert PolicyType.BF16_DEFAULT not in policies

    def test_cpu_gets_int4(self):
        policies = get_policies_for_profile(_CPU_PROFILE)
        assert PolicyType.INT4_QUANT in policies

    def test_hopper_profile_gets_fp8(self):
        hopper = _make_profile(
            tier=HardwareTier.TIER_H100,
            gpu_name="NVIDIA H100",
            compute_capability="9.0",
            has_native_fp8=True,
        )
        policies = get_policies_for_profile(hopper)
        assert PolicyType.FP8_QUANT in policies
        assert PolicyType.BF16_DEFAULT not in policies

    def test_policies_list_is_non_empty(self):
        for tier in HardwareTier:
            profile = HALProfile(
                tier=tier,
                gpu_name="test",
                vram_bytes=0,
                compute_capability="0.0",
                has_native_fp8=False,
                has_native_fp4=False,
                has_nvlink=False,
                pcie_gen=0,
                interconnect_bandwidth_gb_s=0.0,
                memory_bandwidth_gb_s=0.0,
                driver_version="none",
                profile_sha256="a" * 64,
            )
            policies = get_policies_for_profile(profile)
            assert len(policies) > 0, f"Expected non-empty policies for {tier}"
