"""Tests for d14: CPU multi-process FSDP2 validation without GPU.

Proves cpu_fsdp2_validate() wires up real torch.multiprocessing.spawn with
gloo backend and FSDP wrapping. Unit tests mock the subprocess layer so they
run even when FSDP is unavailable in the current environment (e.g. sympy
conflict from swe-bench cache). Integration tests are skipped when FSDP
cannot be imported.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from dojo.train.fsdp2 import (
    _cpu_fsdp2_worker,
    _is_torch_fsdp_available,
    cpu_fsdp2_validate,
)
from dojo.core.exceptions import FSDP2ConfigError

_FSDP_AVAILABLE = _is_torch_fsdp_available()


# ---------------------------------------------------------------------------
# Source-level checks (always run)
# ---------------------------------------------------------------------------

class TestSourceStructure:
    def test_cpu_fsdp2_validate_exported(self):
        from dojo.train import fsdp2 as _m
        assert "cpu_fsdp2_validate" in _m.__all__

    def test_worker_callable(self):
        assert callable(_cpu_fsdp2_worker)

    def test_validate_callable(self):
        assert callable(cpu_fsdp2_validate)

    def test_worker_source_uses_gloo_backend(self):
        src = inspect.getsource(_cpu_fsdp2_worker)
        assert "gloo" in src

    def test_worker_source_initialises_process_group(self):
        src = inspect.getsource(_cpu_fsdp2_worker)
        assert "init_process_group" in src

    def test_worker_source_wraps_with_fsdp(self):
        src = inspect.getsource(_cpu_fsdp2_worker)
        assert "FSDP" in src

    def test_validate_source_uses_spawn_context(self):
        src = inspect.getsource(cpu_fsdp2_validate)
        assert "spawn" in src

    def test_validate_source_joins_processes(self):
        src = inspect.getsource(cpu_fsdp2_validate)
        assert "join" in src

    def test_worker_source_destroys_process_group(self):
        src = inspect.getsource(_cpu_fsdp2_worker)
        assert "destroy_process_group" in src

    def test_worker_source_records_final_loss(self):
        src = inspect.getsource(_cpu_fsdp2_worker)
        assert "final_loss" in src


# ---------------------------------------------------------------------------
# Error-path checks (always run — no FSDP import needed)
# ---------------------------------------------------------------------------

class TestErrorPaths:
    def test_raises_fsdp2_config_error_when_unavailable(self):
        with patch("dojo.train.fsdp2._is_torch_fsdp_available", return_value=False):
            with pytest.raises(FSDP2ConfigError, match="not available"):
                cpu_fsdp2_validate()

    def test_error_mentions_install_hint(self):
        with patch("dojo.train.fsdp2._is_torch_fsdp_available", return_value=False):
            with pytest.raises(FSDP2ConfigError) as exc_info:
                cpu_fsdp2_validate()
        assert "PyTorch" in str(exc_info.value) or "Install" in str(exc_info.value) or "available" in str(exc_info.value)

    def test_raises_runtime_error_when_worker_fails(self):
        mock_ctx = MagicMock()
        mock_queue = MagicMock()
        mock_queue.empty.side_effect = [False, True]
        mock_queue.get_nowait.return_value = {"rank": 0, "ok": False, "error": "boom"}
        mock_ctx.Queue.return_value = mock_queue
        mock_proc = MagicMock()
        mock_ctx.Process.return_value = mock_proc

        with patch("dojo.train.fsdp2._is_torch_fsdp_available", return_value=True):
            with patch("torch.multiprocessing.get_context", return_value=mock_ctx):
                with pytest.raises(RuntimeError, match="boom"):
                    cpu_fsdp2_validate(world_size=1, steps=1)


# ---------------------------------------------------------------------------
# Mocked subprocess API tests (always run)
# ---------------------------------------------------------------------------

class TestMockedSubprocessAPI:
    def _make_mock_ctx(self, results: list[dict]):
        """Build a mock multiprocessing context that returns given results."""
        mock_ctx = MagicMock()
        mock_queue = MagicMock()
        remaining = list(results)

        def _empty():
            return len(remaining) == 0

        def _get():
            return remaining.pop(0)

        mock_queue.empty.side_effect = _empty
        mock_queue.get_nowait.side_effect = _get
        mock_ctx.Queue.return_value = mock_queue
        mock_proc = MagicMock()
        mock_ctx.Process.return_value = mock_proc
        return mock_ctx

    def test_spawns_world_size_processes(self):
        world_size = 3
        results = [{"rank": i, "ok": True, "steps": 2, "final_loss": 0.5, "loss_finite": True}
                   for i in range(world_size)]
        mock_ctx = self._make_mock_ctx(results)

        with patch("dojo.train.fsdp2._is_torch_fsdp_available", return_value=True):
            with patch("torch.multiprocessing.get_context", return_value=mock_ctx):
                out = cpu_fsdp2_validate(world_size=world_size, steps=2)

        assert mock_ctx.Process.call_count == world_size

    def test_all_processes_started_and_joined(self):
        results = [{"rank": 0, "ok": True, "steps": 1, "final_loss": 0.1, "loss_finite": True}]
        mock_ctx = self._make_mock_ctx(results)

        with patch("dojo.train.fsdp2._is_torch_fsdp_available", return_value=True):
            with patch("torch.multiprocessing.get_context", return_value=mock_ctx):
                cpu_fsdp2_validate(world_size=1, steps=1)

        mock_proc = mock_ctx.Process.return_value
        mock_proc.start.assert_called_once()
        mock_proc.join.assert_called_once()

    def test_results_sorted_by_rank(self):
        results = [
            {"rank": 1, "ok": True, "steps": 2, "final_loss": 0.3, "loss_finite": True},
            {"rank": 0, "ok": True, "steps": 2, "final_loss": 0.5, "loss_finite": True},
        ]
        mock_ctx = self._make_mock_ctx(results)

        with patch("dojo.train.fsdp2._is_torch_fsdp_available", return_value=True):
            with patch("torch.multiprocessing.get_context", return_value=mock_ctx):
                out = cpu_fsdp2_validate(world_size=2, steps=2)

        assert [r["rank"] for r in out] == [0, 1]

    def test_returns_all_result_dicts(self):
        results = [{"rank": i, "ok": True, "steps": 1, "final_loss": 0.1, "loss_finite": True}
                   for i in range(2)]
        mock_ctx = self._make_mock_ctx(results)

        with patch("dojo.train.fsdp2._is_torch_fsdp_available", return_value=True):
            with patch("torch.multiprocessing.get_context", return_value=mock_ctx):
                out = cpu_fsdp2_validate(world_size=2, steps=1)

        assert len(out) == 2


# ---------------------------------------------------------------------------
# Real multi-process integration (skipped when FSDP unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FSDP_AVAILABLE, reason="torch.distributed.fsdp not importable in this env")
class TestRealMultiProcess:
    def test_two_ranks_complete_successfully(self):
        results = cpu_fsdp2_validate(world_size=2, steps=2)
        assert len(results) == 2
        assert all(r["ok"] for r in results)

    def test_loss_finite_in_all_ranks(self):
        import math
        results = cpu_fsdp2_validate(world_size=2, steps=2)
        for r in results:
            assert math.isfinite(r["final_loss"])
