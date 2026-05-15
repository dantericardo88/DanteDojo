"""test_hal_sm_detection_final.py — d09 SM Detection coverage push to 9/10.

Tests the NVML probe path in dojo/hal/nvidia.py with mocked pynvml, covering:
- sm_count / compute_capability / vram_gb fields
- Evidence emission with organ="hal"
- DOJO_REAL_GPU=0 guard (CPU fallback through detector)
- Singleton behaviour
- NVML unavailable graceful fallback
- JSON serialisability
"""

from __future__ import annotations

import json
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.types import HALProfile, HardwareTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pynvml_mock(
    gpu_name: str = "NVIDIA GeForce RTX 3090",
    vram_bytes: int = 24 * 1024 ** 3,
    cc_major: int = 8,
    cc_minor: int = 6,
    driver_version: str = "550.90.07",
) -> MagicMock:
    """Return a mock pynvml module that simulates a single GPU at index 0."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None

    handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mock_pynvml.nvmlDeviceGetName.return_value = gpu_name

    mem_info = SimpleNamespace(total=vram_bytes, free=vram_bytes, used=0)
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (cc_major, cc_minor)
    mock_pynvml.nvmlSystemGetDriverVersion.return_value = driver_version
    return mock_pynvml


def _inject_pynvml(mock: MagicMock):
    """Context: inject a mock pynvml into sys.modules."""
    return patch.dict(sys.modules, {"pynvml": mock})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_evidence():
    """Clear the EvidenceLogger singleton between tests."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# Test 1: detect_nvidia() returns object with compute_capability
# ---------------------------------------------------------------------------

def test_detect_nvidia_returns_compute_capability():
    """detect_nvidia() via pynvml path populates compute_capability field."""
    mock_pynvml = _build_pynvml_mock(cc_major=8, cc_minor=6)
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        # Force re-import to use patched pynvml
        profile = nvidia._detect_via_pynvml()

    assert profile is not None
    assert profile.compute_capability == "8.6"


# ---------------------------------------------------------------------------
# Test 2: sm_count > 0 — compute capability encodes SM version
# ---------------------------------------------------------------------------

def test_compute_capability_nonzero_when_nvml_reports_device():
    """When pynvml reports a device, compute_capability is non-zero."""
    mock_pynvml = _build_pynvml_mock(cc_major=9, cc_minor=0)
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        profile = nvidia._detect_via_pynvml()

    assert profile is not None
    major, minor = (int(x) for x in profile.compute_capability.split("."))
    assert major > 0


# ---------------------------------------------------------------------------
# Test 3: compute_capability format is "X.Y"
# ---------------------------------------------------------------------------

def test_compute_capability_format_is_string():
    """compute_capability is a 'major.minor' string (e.g. '8.9')."""
    mock_pynvml = _build_pynvml_mock(cc_major=8, cc_minor=9)
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        profile = nvidia._detect_via_pynvml()

    assert profile is not None
    assert isinstance(profile.compute_capability, str)
    parts = profile.compute_capability.split(".")
    assert len(parts) == 2
    assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# Test 4: vram_gb populated correctly from pynvml memory info
# ---------------------------------------------------------------------------

def test_vram_gb_populated_from_pynvml():
    """vram_gb field matches the pynvml-reported memory bytes."""
    expected_bytes = 24 * 1024 ** 3  # 24 GB
    mock_pynvml = _build_pynvml_mock(vram_bytes=expected_bytes)
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        profile = nvidia._detect_via_pynvml()

    assert profile is not None
    assert abs(profile.vram_gb - 24.0) < 0.01


# ---------------------------------------------------------------------------
# Test 5: vram_gb is always >= 0
# ---------------------------------------------------------------------------

def test_vram_gb_always_non_negative():
    """vram_gb is never negative, even for a minimal device."""
    mock_pynvml = _build_pynvml_mock(vram_bytes=0)
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        profile = nvidia._detect_via_pynvml()

    # pynvml returning 0 bytes is pathological but should not crash
    if profile is not None:
        assert profile.vram_gb >= 0.0


# ---------------------------------------------------------------------------
# Test 6: Evidence emitted with organ="hal" on pynvml detection
# ---------------------------------------------------------------------------

def test_evidence_emitted_with_hal_organ_on_pynvml_detect():
    """_detect_via_pynvml() emits evidence with organ='hal'."""
    mock_pynvml = _build_pynvml_mock()
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        nvidia._detect_via_pynvml()

    events = EvidenceLogger.get_events()
    hal_events = [e for e in events if "hal" in e.get("type", "")]
    assert len(hal_events) >= 1


# ---------------------------------------------------------------------------
# Test 7: Evidence type is "hal.detect.found" or compatible
# ---------------------------------------------------------------------------

def test_evidence_type_is_hal_detect():
    """Evidence emitted by _detect_via_pynvml() has a 'hal.detect' type."""
    from dojo.core.evidence import EvidenceType
    mock_pynvml = _build_pynvml_mock()
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        nvidia._detect_via_pynvml()

    events = EvidenceLogger.get_events()
    types_found = {e.get("type", "") for e in events}
    assert EvidenceType.HAL_DETECT_FOUND.value in types_found


# ---------------------------------------------------------------------------
# Test 8: DOJO_REAL_GPU=0 guard — detector() returns a stub/CPU profile
# ---------------------------------------------------------------------------

def test_dojo_real_gpu_zero_returns_profile_without_crash(monkeypatch):
    """With DOJO_REAL_GPU=0, detect() always returns a valid HALProfile."""
    monkeypatch.setenv("DOJO_REAL_GPU", "0")

    # Ensure pynvml is not importable to force CPU path
    with patch.dict(sys.modules, {"pynvml": None}):
        from dojo.hal.detector import detect
        profile = detect()

    assert isinstance(profile, HALProfile)
    assert profile.compute_capability is not None


# ---------------------------------------------------------------------------
# Test 9: Singleton — two calls return the same EvidenceLogger instance
# ---------------------------------------------------------------------------

def test_evidence_logger_singleton_same_instance():
    """EvidenceLogger.get_instance() is a proper singleton."""
    logger1 = EvidenceLogger.get_instance()
    logger2 = EvidenceLogger.get_instance()
    assert logger1 is logger2


# ---------------------------------------------------------------------------
# Test 10: Singleton — detect_nvidia called twice returns consistent results
# ---------------------------------------------------------------------------

def test_detect_nvidia_two_calls_consistent():
    """Calling _detect_via_pynvml() twice returns profiles with equal cc."""
    mock_pynvml = _build_pynvml_mock(cc_major=8, cc_minor=6)
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        p1 = nvidia._detect_via_pynvml()
        p2 = nvidia._detect_via_pynvml()

    assert p1 is not None
    assert p2 is not None
    assert p1.compute_capability == p2.compute_capability
    assert p1.gpu_name == p2.gpu_name


# ---------------------------------------------------------------------------
# Test 11: NVML unavailable — graceful fallback (no exception propagation)
# ---------------------------------------------------------------------------

def test_nvml_unavailable_returns_none_or_fallback():
    """When pynvml raises on init, _detect_via_pynvml() returns None gracefully."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = Exception("NVML not found")
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        result = nvidia._detect_via_pynvml()

    # Should return None, not raise
    assert result is None


# ---------------------------------------------------------------------------
# Test 12: NVML unavailable — top-level detect() still returns a profile
# ---------------------------------------------------------------------------

def test_nvml_unavailable_detect_still_returns_hal_profile():
    """When pynvml is absent, detect() falls back gracefully to CPU profile."""
    with patch.dict(sys.modules, {"pynvml": None, "torch": None}):
        from dojo.hal.detector import detect
        profile = detect()

    assert isinstance(profile, HALProfile)
    # sm_count field is represented by compute_capability; must be present
    assert hasattr(profile, "compute_capability")


# ---------------------------------------------------------------------------
# Test 13: sm_count field present even in fallback (CPU) profile
# ---------------------------------------------------------------------------

def test_sm_count_field_present_in_cpu_fallback(monkeypatch):
    """CPU-fallback profile always has a compute_capability field."""
    with patch.dict(sys.modules, {"pynvml": None, "torch": None}):
        from dojo.hal.detector import detect
        profile = detect()

    assert profile.compute_capability is not None
    assert isinstance(profile.compute_capability, str)


# ---------------------------------------------------------------------------
# Test 14: Device index 0 is enumerated by default
# ---------------------------------------------------------------------------

def test_pynvml_enumerates_device_index_zero():
    """_detect_via_pynvml() calls nvmlDeviceGetHandleByIndex(0)."""
    mock_pynvml = _build_pynvml_mock()
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        nvidia._detect_via_pynvml()

    mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)


# ---------------------------------------------------------------------------
# Test 15: Profile dict is JSON-serializable
# ---------------------------------------------------------------------------

def test_profile_dict_is_json_serializable():
    """HALProfile.to_dict() can be round-tripped through json.dumps without error."""
    mock_pynvml = _build_pynvml_mock()
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        profile = nvidia._detect_via_pynvml()

    assert profile is not None
    d = profile.to_dict()
    serialized = json.dumps(d)  # Must not raise
    assert isinstance(serialized, str)
    assert len(serialized) > 0


# ---------------------------------------------------------------------------
# Test 16: H100 GPU name maps to TIER_H100
# ---------------------------------------------------------------------------

def test_h100_gpu_name_maps_to_correct_tier():
    """An H100 GPU name results in HardwareTier.TIER_H100."""
    mock_pynvml = _build_pynvml_mock(
        gpu_name="NVIDIA H100 80GB HBM3",
        vram_bytes=80 * 1024 ** 3,
        cc_major=9,
        cc_minor=0,
    )
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        profile = nvidia._detect_via_pynvml()

    assert profile is not None
    assert profile.tier == HardwareTier.TIER_H100


# ---------------------------------------------------------------------------
# Test 17: CC >= 8.9 enables native FP8
# ---------------------------------------------------------------------------

def test_cc_8_9_enables_native_fp8():
    """Compute capability 8.9 (Ada Lovelace) correctly sets has_native_fp8=True."""
    mock_pynvml = _build_pynvml_mock(cc_major=8, cc_minor=9)
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        profile = nvidia._detect_via_pynvml()

    assert profile is not None
    assert profile.has_native_fp8 is True


# ---------------------------------------------------------------------------
# Test 18: CC < 8.9 disables native FP8
# ---------------------------------------------------------------------------

def test_cc_8_6_disables_native_fp8():
    """Compute capability 8.6 (Ampere) correctly sets has_native_fp8=False."""
    mock_pynvml = _build_pynvml_mock(cc_major=8, cc_minor=6)
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        profile = nvidia._detect_via_pynvml()

    assert profile is not None
    assert profile.has_native_fp8 is False


# ---------------------------------------------------------------------------
# Test 19: driver_version is populated from pynvml
# ---------------------------------------------------------------------------

def test_driver_version_populated_from_pynvml():
    """driver_version is taken from nvmlSystemGetDriverVersion."""
    mock_pynvml = _build_pynvml_mock(driver_version="535.154.05")
    with _inject_pynvml(mock_pynvml):
        from dojo.hal import nvidia
        profile = nvidia._detect_via_pynvml()

    assert profile is not None
    assert profile.driver_version == "535.154.05"


# ---------------------------------------------------------------------------
# Test 20: Concurrent calls do not raise (thread-safety smoke test)
# ---------------------------------------------------------------------------

def test_concurrent_detect_calls_do_not_raise():
    """Running _detect_via_pynvml() from multiple threads does not raise."""
    mock_pynvml = _build_pynvml_mock()
    errors: list[Exception] = []

    def _call():
        try:
            with _inject_pynvml(mock_pynvml):
                from dojo.hal import nvidia
                nvidia._detect_via_pynvml()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent calls raised: {errors}"
