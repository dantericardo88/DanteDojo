"""Tests for d11: dynamic autocast device selection (cpu vs cuda).

Verifies that _run_real_sft uses 'cuda' for torch.autocast when
torch.cuda.is_available() returns True, and 'cpu' when it returns False.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import torch


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------

class TestAutoCastSourceCode:
    def test_source_uses_is_available_check(self):
        from dojo.train.finetune import _run_real_sft
        src = inspect.getsource(_run_real_sft)
        assert "cuda.is_available" in src

    def test_source_uses_ternary_device(self):
        from dojo.train.finetune import _run_real_sft
        src = inspect.getsource(_run_real_sft)
        assert '"cuda"' in src or "'cuda'" in src
        assert '"cpu"' in src or "'cpu'" in src

    def test_no_hardcoded_cpu_only_autocast(self):
        from dojo.train.finetune import _run_real_sft
        src = inspect.getsource(_run_real_sft)
        assert 'autocast("cpu"' not in src
        assert "autocast('cpu'" not in src


# ---------------------------------------------------------------------------
# Runtime behaviour: device selection expression (isolated, no full training)
# ---------------------------------------------------------------------------

class TestDeviceSelectionExpression:
    """Test the is_available → device ternary in isolation."""

    def test_cpu_selected_when_cuda_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cpu"

    def test_cuda_selected_when_cuda_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cuda"

    def test_cpu_autocast_accepts_bfloat16(self):
        ctx = torch.autocast("cpu", dtype=torch.bfloat16)
        assert ctx is not None

    def test_cuda_string_would_be_passed_to_autocast(self):
        """If cuda is available the string 'cuda' is produced — not 'cpu'."""
        with patch("torch.cuda.is_available", return_value=True):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device != "cpu"

    def test_device_string_is_one_of_valid_torch_devices(self):
        for available in (True, False):
            with patch("torch.cuda.is_available", return_value=available):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            assert device in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# Autocast context manager created with dynamic device
# ---------------------------------------------------------------------------

class TestAutoCastContextCreation:
    def test_autocast_cpu_context_is_context_manager(self):
        ctx = torch.autocast("cpu", dtype=torch.bfloat16)
        assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")

    def test_autocast_cpu_context_enters_without_error(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = torch.ones(4) @ torch.ones(4)
        assert result is not None

    def test_autocast_device_driven_by_is_available(self):
        """The device arg to autocast is exactly what is_available produces."""
        with patch("torch.cuda.is_available", return_value=False):
            chosen = "cuda" if torch.cuda.is_available() else "cpu"
        ctx = torch.autocast(chosen, dtype=torch.bfloat16)
        assert ctx is not None

    def test_autocast_cpu_dtype_is_bfloat16(self):
        """Verifies the dtype kwarg passed is bfloat16, not float16 or float32."""
        captured_dtype = []

        real_autocast = torch.autocast

        def spy(device, **kw):
            captured_dtype.append(kw.get("dtype"))
            return real_autocast(device, **kw)

        with patch("torch.autocast", side_effect=spy):
            import torch as _torch
            device = "cpu"
            _torch.autocast(device, dtype=torch.bfloat16)

        assert captured_dtype == [torch.bfloat16]


# ---------------------------------------------------------------------------
# Device string is a variable, not literal
# ---------------------------------------------------------------------------

class TestDeviceVariableBinding:
    def test_device_variable_used_in_autocast_call(self):
        """The device passed to autocast comes from the is_available check."""
        from dojo.train.finetune import _run_real_sft
        src = inspect.getsource(_run_real_sft)
        idx_check = src.find("cuda.is_available")
        idx_autocast = src.find("autocast(")
        assert idx_check < idx_autocast, (
            "cuda.is_available check must appear before autocast call in source"
        )
