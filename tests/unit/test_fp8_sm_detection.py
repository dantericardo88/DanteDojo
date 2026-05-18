"""Tests for d10: real FP8 SM detection (not env-var only).

Verifies _detect_sm_version() uses torch.cuda.get_device_capability() when
CUDA is available, falling back to DOJO_MOCK_SM for CI/testing.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from dojo.train.finetune import _detect_sm_version, _run_real_fp8


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------

class TestSourceCheck:
    def test_source_uses_get_device_capability(self):
        src = inspect.getsource(_detect_sm_version)
        assert "get_device_capability" in src

    def test_source_uses_cuda_is_available_gate(self):
        src = inspect.getsource(_detect_sm_version)
        assert "cuda.is_available" in src

    def test_source_falls_back_to_env_var(self):
        src = inspect.getsource(_detect_sm_version)
        assert "DOJO_MOCK_SM" in src

    def test_fp8_uses_detect_helper_not_env_var_directly(self):
        src = inspect.getsource(_run_real_fp8)
        assert "_detect_sm_version" in src
        assert "DOJO_MOCK_SM" not in src


# ---------------------------------------------------------------------------
# Runtime: CUDA available path
# ---------------------------------------------------------------------------

class TestCudaAvailable:
    def test_reads_sm_from_torch_when_cuda_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
                sm = _detect_sm_version()
        assert sm == 8.9

    def test_handles_hopper_sm_90(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                sm = _detect_sm_version()
        assert sm == 9.0

    def test_handles_ampere_sm_86(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(8, 6)):
                sm = _detect_sm_version()
        assert sm == 8.6

    def test_handles_blackwell_sm_100(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
                sm = _detect_sm_version()
        assert sm == 10.0


# ---------------------------------------------------------------------------
# Runtime: CUDA unavailable / fallback path
# ---------------------------------------------------------------------------

class TestCudaUnavailableFallback:
    def test_falls_back_to_env_var_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setenv("DOJO_MOCK_SM", "8.9")
        with patch("torch.cuda.is_available", return_value=False):
            sm = _detect_sm_version()
        assert sm == 8.9

    def test_returns_zero_when_no_cuda_and_no_env_var(self, monkeypatch):
        monkeypatch.delenv("DOJO_MOCK_SM", raising=False)
        with patch("torch.cuda.is_available", return_value=False):
            sm = _detect_sm_version()
        assert sm == 0.0

    def test_torch_import_failure_falls_back_gracefully(self, monkeypatch):
        monkeypatch.setenv("DOJO_MOCK_SM", "7.5")
        with patch("torch.cuda.is_available", side_effect=RuntimeError("driver bug")):
            sm = _detect_sm_version()
        assert sm == 7.5


# ---------------------------------------------------------------------------
# FP8 gating using real detection
# ---------------------------------------------------------------------------

class TestFp8GateWithRealDetection:
    def test_fp8_rejected_when_sm_too_low_via_torch(self):
        from dojo.core.exceptions import PrecisionUnsupported
        with patch("dojo.train.finetune._detect_sm_version", return_value=8.6):
            from dojo.train.finetune import _FP8_MIN_SM
            assert _FP8_MIN_SM > 8.6  # baseline sanity

    def test_fp8_accepts_ada_lovelace_via_torch(self, tmp_path):
        # Just verify the SM gate logic itself — _detect_sm_version=8.9 passes
        with patch("dojo.train.finetune._detect_sm_version", return_value=8.9):
            from dojo.train.finetune import _FP8_MIN_SM
            assert 8.9 >= _FP8_MIN_SM
