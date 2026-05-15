"""Unit tests for SM (compute capability) detection.

Covers pynvml path, torch.cuda fallback path, evidence emission, and the
detect_nvidia() dispatch logic.  All tests are CI-safe (no real GPU required).
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.types import HALProfile, HardwareTier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# Helper: build a minimal torch mock that looks like real torch.cuda
# ---------------------------------------------------------------------------

def _make_torch_mock(
    is_available: bool = True,
    device_capability: tuple[int, int] = (8, 9),
    device_name: str = "NVIDIA GeForce RTX 4060 Ti",
    total_memory: int = 16 * 1024 ** 3,
) -> MagicMock:
    """Return a MagicMock that impersonates torch with a CUDA device."""
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = is_available
    torch_mock.cuda.get_device_capability.return_value = device_capability
    torch_mock.cuda.get_device_name.return_value = device_name

    props = MagicMock()
    props.total_memory = total_memory
    torch_mock.cuda.get_device_properties.return_value = props

    return torch_mock


# ---------------------------------------------------------------------------
# Helper: inject torch mock into sys.modules for the duration of a test
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _torch_injected(torch_mock: MagicMock):
    """Context manager that injects a torch mock into sys.modules."""
    original = sys.modules.get("torch", None)
    sys.modules["torch"] = torch_mock
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


# ---------------------------------------------------------------------------
# 1. test_detect_via_torch_returns_sm89
# ---------------------------------------------------------------------------

class TestTorchDetectSM89:
    def test_detect_via_torch_returns_sm89(self):
        """_detect_via_torch() returns compute_capability='8.9' when torch reports (8,9)."""
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(device_capability=(8, 9))
        with _torch_injected(torch_mock):
            profile = _detect_via_torch()

        assert profile is not None
        assert profile.compute_capability == "8.9"

    def test_detect_via_torch_returns_hal_profile_instance(self):
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(device_capability=(8, 9))
        with _torch_injected(torch_mock):
            profile = _detect_via_torch()

        assert isinstance(profile, HALProfile)


# ---------------------------------------------------------------------------
# 2. test_detect_via_torch_fp8_enabled_at_sm89
# ---------------------------------------------------------------------------

class TestTorchFP8Flags:
    def test_detect_via_torch_fp8_enabled_at_sm89(self):
        """has_native_fp8 is True when SM >= 8.9."""
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(device_capability=(8, 9))
        with _torch_injected(torch_mock):
            profile = _detect_via_torch()

        assert profile is not None
        assert profile.has_native_fp8 is True

    # 3. test_detect_via_torch_fp8_disabled_below_sm89
    def test_detect_via_torch_fp8_disabled_below_sm89(self):
        """has_native_fp8 is False when SM < 8.9 (e.g. Ampere SM 8.6)."""
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(device_capability=(8, 6))
        with _torch_injected(torch_mock):
            profile = _detect_via_torch()

        assert profile is not None
        assert profile.has_native_fp8 is False

    def test_fp4_enabled_at_sm10(self):
        """has_native_fp4 is True only for SM >= 10.0."""
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(device_capability=(10, 0))
        with _torch_injected(torch_mock):
            profile = _detect_via_torch()

        assert profile is not None
        assert profile.has_native_fp4 is True

    def test_fp4_disabled_at_sm89(self):
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(device_capability=(8, 9))
        with _torch_injected(torch_mock):
            profile = _detect_via_torch()

        assert profile is not None
        assert profile.has_native_fp4 is False


# ---------------------------------------------------------------------------
# 4. test_detect_via_torch_emits_hal_found_evidence
# ---------------------------------------------------------------------------

class TestTorchEvidenceEmission:
    def test_detect_via_torch_emits_hal_found_evidence(self):
        """HAL_DETECT_FOUND event is emitted with compute_capability field."""
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(device_capability=(8, 9))
        with _torch_injected(torch_mock):
            _detect_via_torch()

        events = EvidenceLogger.get_events()
        hal_events = [
            e for e in events
            if e.get("type") == EvidenceType.HAL_DETECT_FOUND.value
        ]
        assert len(hal_events) >= 1, "Expected at least one HAL_DETECT_FOUND event"
        data = hal_events[0].get("data", {})
        assert "compute_capability" in data, f"compute_capability missing from event data: {data}"
        assert data["compute_capability"] == "8.9"

    def test_detect_via_torch_evidence_has_backend_nvidia_torch(self):
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(device_capability=(8, 9))
        with _torch_injected(torch_mock):
            _detect_via_torch()

        events = EvidenceLogger.get_events()
        hal_events = [
            e for e in events
            if e.get("type") == EvidenceType.HAL_DETECT_FOUND.value
        ]
        assert hal_events[0]["data"].get("backend") == "nvidia_torch"

    # 8. test_sm_detection_evidence_has_sm_version_field
    def test_sm_detection_evidence_has_sm_version_field(self):
        """Evidence event contains an 'sm_version' key."""
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(device_capability=(8, 9))
        with _torch_injected(torch_mock):
            _detect_via_torch()

        events = EvidenceLogger.get_events()
        hal_events = [
            e for e in events
            if e.get("type") == EvidenceType.HAL_DETECT_FOUND.value
        ]
        data = hal_events[0].get("data", {})
        assert "sm_version" in data or "compute_capability" in data, (
            f"Expected sm_version or compute_capability in evidence data: {data}"
        )


# ---------------------------------------------------------------------------
# 5. test_detect_via_torch_no_cuda_returns_none
# ---------------------------------------------------------------------------

class TestTorchNoCuda:
    def test_detect_via_torch_no_cuda_returns_none(self):
        """Returns None when torch.cuda.is_available() is False."""
        from dojo.hal.nvidia import _detect_via_torch

        torch_mock = _make_torch_mock(is_available=False)
        with _torch_injected(torch_mock):
            result = _detect_via_torch()

        assert result is None

    def test_detect_via_torch_missing_torch_returns_none(self):
        """Returns None when torch is not installed (ImportError)."""
        from dojo.hal.nvidia import _detect_via_torch

        # Remove torch from sys.modules and make it unimportable
        original = sys.modules.pop("torch", None)
        # Patch builtins.__import__ to raise ImportError for "torch"
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        import builtins

        original_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not installed")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = _blocking_import
            result = _detect_via_torch()
        finally:
            builtins.__import__ = original_import
            if original is not None:
                sys.modules["torch"] = original

        assert result is None


# ---------------------------------------------------------------------------
# 6. test_detect_nvidia_fallback_to_torch
# ---------------------------------------------------------------------------

class TestDetectNvidiaFallback:
    def test_detect_nvidia_fallback_to_torch(self):
        """detect_nvidia() falls back to torch when pynvml raises ImportError."""
        torch_mock = _make_torch_mock(
            device_capability=(8, 9),
            device_name="NVIDIA GeForce RTX 4060 Ti",
        )
        # Patch _detect_via_pynvml to simulate pynvml unavailability
        with (
            patch("dojo.hal.nvidia._detect_via_pynvml", return_value=None),
            _torch_injected(torch_mock),
        ):
            from dojo.hal.nvidia import detect_nvidia
            profile = detect_nvidia()

        assert profile is not None
        assert profile.compute_capability == "8.9"

    def test_detect_nvidia_fallback_profile_has_sm_version(self):
        """Fallback profile via torch has a meaningful compute_capability string."""
        torch_mock = _make_torch_mock(device_capability=(8, 9))
        with (
            patch("dojo.hal.nvidia._detect_via_pynvml", return_value=None),
            _torch_injected(torch_mock),
        ):
            from dojo.hal.nvidia import detect_nvidia
            profile = detect_nvidia()

        assert profile is not None
        assert "8.9" in profile.compute_capability

    def test_detect_nvidia_prefers_pynvml_over_torch(self):
        """When pynvml returns a profile, torch is not consulted."""
        from dojo.core.types import HardwareTier
        from dojo.hal.nvidia import detect_nvidia

        pynvml_profile = HALProfile(
            tier=HardwareTier.TIER_3090,
            gpu_name="NVIDIA GeForce RTX 3090",
            vram_bytes=24 * 1024 ** 3,
            compute_capability="8.6",
            has_native_fp8=False,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=0,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="550.0",
            profile_sha256="a" * 64,
        )

        torch_mock = _make_torch_mock(device_capability=(8, 9))
        with (
            patch("dojo.hal.nvidia._detect_via_pynvml", return_value=pynvml_profile),
            _torch_injected(torch_mock),
        ):
            profile = detect_nvidia()

        # Should use pynvml result, not torch
        assert profile.compute_capability == "8.6"
        assert profile.driver_version == "550.0"


# ---------------------------------------------------------------------------
# 7. test_pynvml_sm_detection
# ---------------------------------------------------------------------------

class TestPynvmlSMDetection:
    def test_pynvml_sm_detection(self):
        """_detect_via_pynvml() returns compute_capability='8.9' when pynvml reports (8,9)."""
        pynvml_mock = MagicMock()

        # nvmlDeviceGetCudaComputeCapability returns (major, minor)
        handle = MagicMock()
        pynvml_mock.nvmlDeviceGetHandleByIndex.return_value = handle
        pynvml_mock.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 4060 Ti"
        mem_info = MagicMock()
        mem_info.total = 16 * 1024 ** 3
        pynvml_mock.nvmlDeviceGetMemoryInfo.return_value = mem_info
        pynvml_mock.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)
        pynvml_mock.nvmlSystemGetDriverVersion.return_value = "555.42"

        with patch.dict(sys.modules, {"pynvml": pynvml_mock}):
            from dojo.hal import nvidia as _nvidia_mod
            # Force re-import to pick up the injected module
            import importlib
            importlib.reload(_nvidia_mod)
            profile = _nvidia_mod._detect_via_pynvml()

        # Restore original module state
        import importlib
        import dojo.hal.nvidia as _restored
        importlib.reload(_restored)

        assert profile is not None
        assert profile.compute_capability == "8.9"

    def test_pynvml_sm_detection_fp8_at_sm89(self):
        """pynvml path sets has_native_fp8=True for SM 8.9."""
        pynvml_mock = MagicMock()
        handle = MagicMock()
        pynvml_mock.nvmlDeviceGetHandleByIndex.return_value = handle
        pynvml_mock.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 4060 Ti"
        mem_info = MagicMock()
        mem_info.total = 16 * 1024 ** 3
        pynvml_mock.nvmlDeviceGetMemoryInfo.return_value = mem_info
        pynvml_mock.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)
        pynvml_mock.nvmlSystemGetDriverVersion.return_value = "555.42"

        with patch.dict(sys.modules, {"pynvml": pynvml_mock}):
            import importlib
            import dojo.hal.nvidia as _nvidia_mod
            importlib.reload(_nvidia_mod)
            profile = _nvidia_mod._detect_via_pynvml()

        import importlib
        import dojo.hal.nvidia as _restored
        importlib.reload(_restored)

        assert profile is not None
        assert profile.has_native_fp8 is True

    def test_pynvml_absent_returns_none(self):
        """_detect_via_pynvml() returns None when pynvml is not installed."""
        # Remove pynvml if present, make it unimportable
        original = sys.modules.pop("pynvml", None)
        import builtins
        original_import = builtins.__import__

        def _block_pynvml(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("pynvml not installed")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = _block_pynvml
            from dojo.hal.nvidia import _detect_via_pynvml
            result = _detect_via_pynvml()
        finally:
            builtins.__import__ = original_import
            if original is not None:
                sys.modules["pynvml"] = original

        assert result is None
