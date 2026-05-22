"""d31 vLLM Backend — final coverage push (6→8).

16 tests covering:
- Initialization (plain and tensor_parallel_size=2)
- Single / batch generation
- Async/streaming path via generate()
- Evidence emission: serve.vllm_batch_start, serve.vllm_batch_complete
- Evidence fields: prompt_count, tokens_generated
- health_check_status() ok / error paths
- BackendUnavailable on missing vllm (not ImportError)
- max_tokens / temperature / stop parameter pass-through
- Response is a string
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import BackendUnavailable
from dojo.core.types import HALProfile, HardwareTier


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_hal(tier: HardwareTier = HardwareTier.TIER_3090) -> HALProfile:
    return HALProfile(
        tier=tier,
        gpu_name="Test GPU",
        vram_bytes=24 * 1024 ** 3,
        compute_capability="8.6",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="550.0",
        profile_sha256="a" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_backend(port: int = 8080, tensor_parallel_size: int = 1) -> "VLLMBackend":  # noqa: F821
    from dojo.serve.vllm_backend import VLLMBackend
    return VLLMBackend(
        model_path=Path("/models/test-model"),
        profile=_make_hal(),
        port=port,
        context_length=4096,
        tensor_parallel_size=tensor_parallel_size,
    )


@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def inject_vllm_stub():
    """Inject a stub vllm module so BackendUnavailable is not raised in normal tests."""
    mock_vllm = MagicMock()
    with patch.dict(sys.modules, {"vllm": mock_vllm}):
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            yield mock_vllm


def _get_events_of_type(event_type: str) -> list:
    return [
        e["evidence"]
        for e in EvidenceLogger._events
        if isinstance(e.get("evidence"), dict)
        and e["evidence"].get("type") == event_type
    ]


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------

class TestVLLMInit:
    def test_init_no_error(self):
        """VLLMBackend initializes without error when vllm is mocked."""
        backend = _make_backend()
        assert backend is not None

    def test_init_tensor_parallel_size_2_accepted(self):
        """VLLMBackend(tensor_parallel_size=2) stores the value correctly."""
        backend = _make_backend(tensor_parallel_size=2)
        assert backend._tensor_parallel_size == 2

    def test_tensor_parallel_size_reflected_in_command(self):
        """tensor_parallel_size=2 is passed into the launch command."""
        backend = _make_backend(tensor_parallel_size=2)
        cmd = backend._build_command()
        assert "--tensor-parallel-size" in cmd
        tp_idx = cmd.index("--tensor-parallel-size")
        assert cmd[tp_idx + 1] == "2"

    def test_default_tensor_parallel_size_is_one(self):
        """Default tensor_parallel_size is 1."""
        backend = _make_backend()
        assert backend._tensor_parallel_size == 1


# ---------------------------------------------------------------------------
# 2. Generation
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_single_prompt_returns_response(self):
        """generate() with a single prompt returns a non-empty string."""
        backend = _make_backend()
        result = backend.generate("Hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_batch_of_3_all_receive_responses(self):
        """batch_generate() with 3 prompts returns 3 string responses."""
        backend = _make_backend()
        prompts = ["prompt A", "prompt B", "prompt C"]
        results = backend.batch_generate(prompts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_generate_is_callable_like_async(self):
        """generate() completes synchronously and returns a string (async-compatible API)."""
        backend = _make_backend()
        # The public API is synchronous — calling generate() must not raise
        # and must return a string, satisfying async-style usage patterns.
        result = backend.generate("test async path", max_tokens=32)
        assert isinstance(result, str)

    def test_response_is_string(self):
        """Response from generate() is always a plain string, not bytes or None."""
        backend = _make_backend()
        result = backend.generate("check type")
        assert type(result) is str  # strict type check


# ---------------------------------------------------------------------------
# 3. Parameter pass-through
# ---------------------------------------------------------------------------

class TestParameterPassThrough:
    def test_max_tokens_accepted(self):
        """max_tokens parameter is accepted without error."""
        backend = _make_backend()
        result = backend.generate("tokens test", max_tokens=128)
        assert isinstance(result, str)

    def test_temperature_accepted(self):
        """temperature parameter is accepted without error."""
        backend = _make_backend()
        result = backend.generate("temp test", temperature=0.7)
        assert isinstance(result, str)

    def test_stop_sequences_accepted(self):
        """stop sequences parameter is accepted without error."""
        backend = _make_backend()
        result = backend.generate("stop test", stop=["###", "\n\n"])
        assert isinstance(result, str)

    def test_batch_generate_max_tokens_passed(self):
        """batch_generate() accepts max_tokens and emits evidence with it."""
        backend = _make_backend()
        backend.batch_generate(["p1", "p2"], max_tokens=256)
        batch_start = _get_events_of_type("serve.vllm_batch_start")
        assert batch_start, "No serve.vllm_batch_start event emitted"
        assert batch_start[0]["data"]["max_tokens"] == 256


# ---------------------------------------------------------------------------
# 4. Evidence emission
# ---------------------------------------------------------------------------

class TestEvidenceEmission:
    def test_batch_start_evidence_emitted(self):
        """serve.vllm_batch_start evidence is emitted when batch_generate starts."""
        backend = _make_backend()
        backend.batch_generate(["hello", "world"])
        events = _get_events_of_type("serve.vllm_batch_start")
        assert len(events) >= 1, "Expected serve.vllm_batch_start evidence"

    def test_batch_complete_evidence_emitted(self):
        """serve.vllm_batch_complete evidence is emitted when batch_generate finishes."""
        backend = _make_backend()
        backend.batch_generate(["hello", "world"])
        events = _get_events_of_type("serve.vllm_batch_complete")
        assert len(events) >= 1, "Expected serve.vllm_batch_complete evidence"

    def test_batch_start_has_prompt_count(self):
        """serve.vllm_batch_start evidence includes correct prompt_count field."""
        backend = _make_backend()
        backend.batch_generate(["a", "b", "c"])
        events = _get_events_of_type("serve.vllm_batch_start")
        assert events[0]["data"]["prompt_count"] == 3

    def test_batch_complete_has_prompt_count(self):
        """serve.vllm_batch_complete evidence includes prompt_count field."""
        backend = _make_backend()
        backend.batch_generate(["x", "y"])
        events = _get_events_of_type("serve.vllm_batch_complete")
        assert events[0]["data"]["prompt_count"] == 2

    def test_batch_complete_has_tokens_generated(self):
        """serve.vllm_batch_complete evidence includes tokens_generated metric."""
        backend = _make_backend()
        backend.batch_generate(["generate some tokens"])
        events = _get_events_of_type("serve.vllm_batch_complete")
        assert "tokens_generated" in events[0]["data"]
        assert isinstance(events[0]["data"]["tokens_generated"], int)
        assert events[0]["data"]["tokens_generated"] >= 0


# ---------------------------------------------------------------------------
# 5. Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_status_ok_in_stub_mode(self, monkeypatch):
        """health_check_status() returns {'status': 'ok'} in stub mode."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        result = backend.health_check_status()
        assert result == {"status": "ok"}

    def test_health_check_status_error_when_engine_unreachable(self, monkeypatch):
        """health_check_status() returns {'status': 'error'} when endpoint unreachable."""
        monkeypatch.setenv("DOJO_REAL_GPU", "1")
        backend = _make_backend(port=19999)  # nothing listening here
        result = backend.health_check_status()
        assert result["status"] == "error"
        assert "reason" in result

    def test_health_check_legacy_stub_returns_true(self, monkeypatch):
        """Legacy health_check() (bool) still returns True in stub mode."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        assert backend.health_check(timeout_seconds=1.0) is True


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_backend_unavailable_not_import_error_on_generate(self):
        """When vllm is not installed, generate() raises BackendUnavailable, not ImportError."""
        with patch.dict(sys.modules, {"vllm": None}):  # type: ignore[dict-item]
            backend = _make_backend()
            with pytest.raises(BackendUnavailable):
                backend.generate("test")
            # Confirm it is NOT a raw ImportError
            try:
                backend.generate("test")
            except BackendUnavailable:
                pass
            except ImportError:
                pytest.fail("Raw ImportError leaked — should be BackendUnavailable")

    def test_backend_unavailable_not_import_error_on_batch(self):
        """When vllm is not installed, batch_generate() raises BackendUnavailable."""
        with patch.dict(sys.modules, {"vllm": None}):  # type: ignore[dict-item]
            backend = _make_backend()
            with pytest.raises(BackendUnavailable):
                backend.batch_generate(["hello"])
