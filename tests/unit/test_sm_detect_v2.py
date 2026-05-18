"""Tests for d09: nvidia-ml-py preferred SM detection."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from dojo.hal.sm_detect import (
    SmInfo,
    detect_sm,
    sm_supports_fp4,
    sm_supports_fp8,
)


class TestSmInfo:
    def test_sm_float_combines_major_minor(self):
        info = SmInfo(available=True, sm_major=8, sm_minor=9,
                     gpu_name="L40", source="nvidia-ml-py")
        assert info.sm_float == 8.9

    def test_sm_float_zero_when_unavailable(self):
        info = SmInfo(available=False, sm_major=0, sm_minor=0,
                     gpu_name="cpu", source="none")
        assert info.sm_float == 0.0


class TestNvidiaMlPyPath:
    def test_uses_pynvml_when_available(self):
        fake = MagicMock()
        fake.__version__ = "12.535.0"
        fake.nvmlInit = MagicMock()
        fake.nvmlShutdown = MagicMock()
        fake.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
        fake.nvmlDeviceGetCudaComputeCapability = MagicMock(return_value=(8, 9))
        fake.nvmlDeviceGetName = MagicMock(return_value="L40-Test")
        with patch.dict(sys.modules, {"pynvml": fake}):
            info = detect_sm()
        assert info.sm_float == 8.9
        assert info.source == "nvidia-ml-py"
        assert "L40" in info.gpu_name

    def test_handles_bytes_gpu_name(self):
        fake = MagicMock()
        fake.__version__ = "12.0.0"
        fake.nvmlInit = MagicMock()
        fake.nvmlShutdown = MagicMock()
        fake.nvmlDeviceGetHandleByIndex = MagicMock(return_value="h")
        fake.nvmlDeviceGetCudaComputeCapability = MagicMock(return_value=(9, 0))
        fake.nvmlDeviceGetName = MagicMock(return_value=b"H100 SXM5")
        with patch.dict(sys.modules, {"pynvml": fake}):
            info = detect_sm()
        assert "H100" in info.gpu_name


class TestTorchCudaFallback:
    def test_torch_path_used_when_pynvml_missing(self, monkeypatch):
        # Make pynvml import fail
        monkeypatch.setitem(sys.modules, "pynvml", None)
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(8, 6)):
                with patch("torch.cuda.get_device_name", return_value="RTX 3090"):
                    info = detect_sm()
        assert info.sm_float == 8.6
        assert info.source == "torch.cuda"


class TestEnvVarFallback:
    def test_env_var_used_when_no_gpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        monkeypatch.setenv("DOJO_MOCK_SM", "9.0")
        with patch("torch.cuda.is_available", return_value=False):
            info = detect_sm()
        assert info.sm_float == 9.0
        assert info.source == "env"

    def test_malformed_env_var_skipped(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        monkeypatch.setenv("DOJO_MOCK_SM", "garbage")
        with patch("torch.cuda.is_available", return_value=False):
            info = detect_sm()
        assert info.source == "none"


class TestNoGpuFallback:
    def test_returns_unavailable_when_nothing_works(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        monkeypatch.delenv("DOJO_MOCK_SM", raising=False)
        with patch("torch.cuda.is_available", return_value=False):
            info = detect_sm()
        assert info.available is False
        assert info.source == "none"


class TestCapabilityGates:
    def test_fp8_requires_sm_89(self):
        info = SmInfo(available=True, sm_major=8, sm_minor=9,
                     gpu_name="L40", source="nvidia-ml-py")
        assert sm_supports_fp8(info)

    def test_fp8_not_on_ampere(self):
        info = SmInfo(available=True, sm_major=8, sm_minor=6,
                     gpu_name="3090", source="nvidia-ml-py")
        assert not sm_supports_fp8(info)

    def test_fp4_requires_sm_100(self):
        info = SmInfo(available=True, sm_major=10, sm_minor=0,
                     gpu_name="B200", source="nvidia-ml-py")
        assert sm_supports_fp4(info)

    def test_fp4_not_on_hopper(self):
        info = SmInfo(available=True, sm_major=9, sm_minor=0,
                     gpu_name="H100", source="nvidia-ml-py")
        assert not sm_supports_fp4(info)

    def test_fp8_false_when_unavailable(self):
        info = SmInfo(available=False, sm_major=0, sm_minor=0,
                     gpu_name="cpu", source="none")
        assert not sm_supports_fp8(info)


class TestEvidence:
    def test_emits_hal_sm_detect_event(self, monkeypatch):
        from dojo.core.evidence import EvidenceLogger
        monkeypatch.setitem(sys.modules, "pynvml", None)
        with patch("torch.cuda.is_available", return_value=False):
            EvidenceLogger.clear()
            detect_sm()
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "hal.sm_detect" for e in events)
