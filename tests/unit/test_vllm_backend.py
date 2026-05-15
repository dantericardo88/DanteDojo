"""Unit tests for dojo/serve/vllm_backend.py — d31 vLLM paged-attention serving.

Covers:
1.  start() in stub mode returns non-zero PID
2.  health_check() in stub mode returns True immediately
3.  _build_command() includes --port N
4.  serve_concurrent(3 requests) returns list of 3 results
5.  each result has a "response" key
6.  serve_concurrent emits SERVE_READY evidence with throughput
7.  throughput_tokens_per_s > 0
8.  vllm missing → BackendUnavailable
9.  endpoint_url is http://localhost:{port}
10. result count matches input count
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import BackendUnavailable
from dojo.core.types import HALProfile, HardwareTier


# ---------------------------------------------------------------------------
# Helpers
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


def _make_backend(port: int = 8080) -> "VLLMBackend":
    from dojo.serve.vllm_backend import VLLMBackend
    return VLLMBackend(
        model_path=Path("/models/test-model"),
        profile=_make_hal(),
        port=port,
        context_length=4096,
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
        yield mock_vllm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVLLMBackendStart:
    def test_start_stub_mode_returns_nonzero_pid(self):
        """start() in stub mode must return a non-zero PID."""
        backend = _make_backend()
        pid = backend.start()
        assert isinstance(pid, int)
        assert pid > 0

    def test_health_check_stub_mode_returns_true(self):
        """health_check() in stub mode returns True immediately without polling."""
        backend = _make_backend()
        result = backend.health_check(timeout_seconds=1.0)
        assert result is True

    def test_build_command_includes_port(self):
        """_build_command() must include --port <N>."""
        backend = _make_backend(port=9123)
        cmd = backend._build_command()
        assert "--port" in cmd
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == "9123"


class TestServeConcurrent:
    def test_serve_concurrent_stub_returns_list(self):
        """serve_concurrent with 3 requests returns a list."""
        backend = _make_backend()
        reqs = [{"prompt": f"hello {i}"} for i in range(3)]
        results = backend.serve_concurrent(reqs)
        assert isinstance(results, list)

    def test_serve_concurrent_all_have_response_field(self):
        """Every result dict must contain a 'response' key."""
        backend = _make_backend()
        reqs = [{"prompt": f"prompt {i}"} for i in range(3)]
        results = backend.serve_concurrent(reqs)
        for r in results:
            assert "response" in r, f"Missing 'response' key in result: {r}"

    def test_serve_concurrent_result_count_matches_input(self):
        """3 requests in → 3 results out."""
        backend = _make_backend()
        reqs = [{"prompt": f"q{i}"} for i in range(3)]
        results = backend.serve_concurrent(reqs)
        assert len(results) == 3

    def test_serve_concurrent_emits_serve_ready_evidence(self):
        """serve_concurrent must emit a SERVE_READY evidence event."""
        backend = _make_backend()
        reqs = [{"prompt": "test prompt"} for _ in range(3)]
        backend.serve_concurrent(reqs)

        events = EvidenceLogger._events
        evidence_types = [
            e["evidence"].get("type")
            for e in events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "serve.ready" in evidence_types, (
            f"Expected 'serve.ready' in emitted types, got: {evidence_types}"
        )

    def test_serve_concurrent_throughput_positive(self):
        """SERVE_READY evidence must carry throughput_tokens_per_s > 0."""
        backend = _make_backend()
        reqs = [{"prompt": f"p{i}"} for i in range(3)]
        backend.serve_concurrent(reqs)

        events = EvidenceLogger._events
        serve_ready = next(
            (
                e["evidence"]["data"]
                for e in events
                if isinstance(e.get("evidence"), dict)
                and e["evidence"].get("type") == "serve.ready"
            ),
            None,
        )
        assert serve_ready is not None, "No serve.ready event found"
        assert serve_ready["throughput_tokens_per_s"] > 0, (
            f"throughput must be > 0, got {serve_ready['throughput_tokens_per_s']}"
        )

    def test_serve_concurrent_ready_has_concurrent_requests_field(self):
        """SERVE_READY data must include concurrent_requests count."""
        backend = _make_backend()
        reqs = [{"prompt": "x"} for _ in range(3)]
        backend.serve_concurrent(reqs)

        events = EvidenceLogger._events
        serve_ready_data = next(
            (
                e["evidence"]["data"]
                for e in events
                if isinstance(e.get("evidence"), dict)
                and e["evidence"].get("type") == "serve.ready"
            ),
            None,
        )
        assert serve_ready_data is not None
        assert serve_ready_data["concurrent_requests"] == 3


class TestBackendUnavailable:
    def test_vllm_missing_raises_backend_unavailable_on_start(self):
        """BackendUnavailable raised when vllm is not installed (start)."""
        # Remove vllm from sys.modules to simulate missing package
        with patch.dict(sys.modules, {"vllm": None}):  # type: ignore[dict-item]
            backend = _make_backend()
            with pytest.raises(BackendUnavailable):
                backend.start()

    def test_vllm_missing_raises_backend_unavailable_on_serve(self):
        """BackendUnavailable raised when vllm is not installed (serve_concurrent)."""
        with patch.dict(sys.modules, {"vllm": None}):  # type: ignore[dict-item]
            backend = _make_backend()
            with pytest.raises(BackendUnavailable):
                backend.serve_concurrent([{"prompt": "hello"}])


class TestEndpointUrl:
    def test_endpoint_url_format(self):
        """endpoint_url must be http://localhost:{port}."""
        backend = _make_backend(port=7777)
        assert backend.endpoint_url == "http://localhost:7777"

    def test_endpoint_url_reflects_port(self):
        """Different ports produce different endpoint URLs."""
        b1 = _make_backend(port=8080)
        b2 = _make_backend(port=9090)
        assert b1.endpoint_url != b2.endpoint_url
        assert "8080" in b1.endpoint_url
        assert "9090" in b2.endpoint_url


# ---------------------------------------------------------------------------
# New test classes (d31 coverage expansion)
# ---------------------------------------------------------------------------


class TestVLLMBackendStartStop:
    def test_start_returns_pid(self, monkeypatch):
        """start() returns a positive integer PID in stub mode."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        pid = backend.start()
        assert isinstance(pid, int)
        assert pid > 0

    def test_start_emits_serve_start_evidence(self, monkeypatch):
        """serve.start evidence is emitted when serve_concurrent is called after start()."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        backend.start()
        backend.serve_concurrent([{"prompt": "hello"}])

        evidence_types = [
            e["evidence"].get("type")
            for e in EvidenceLogger._events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "serve.start" in evidence_types, (
            f"Expected 'serve.start' in emitted types, got: {evidence_types}"
        )

    def test_stop_no_error_when_not_started(self, monkeypatch):
        """stop() on a fresh (never-started) instance must not raise."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        # Should complete without exception
        backend.stop()

    def test_start_twice_idempotent(self, monkeypatch):
        """Calling start() twice in stub mode must not crash."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        pid1 = backend.start()
        pid2 = backend.start()
        assert pid1 > 0
        assert pid2 > 0


class TestVLLMBuildCommand:
    def test_command_includes_model_path(self, monkeypatch):
        """_build_command() must include the model path string."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        from dojo.serve.vllm_backend import VLLMBackend
        backend = VLLMBackend(
            model_path=Path("/models/my-model"),
            profile=_make_hal(),
            port=8080,
            context_length=4096,
        )
        cmd = backend._build_command()
        assert "/models/my-model" in cmd or str(Path("/models/my-model")) in cmd

    def test_command_includes_port(self, monkeypatch):
        """_build_command() must include the port number as a string."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend(port=9876)
        cmd = backend._build_command()
        assert "9876" in cmd

    def test_command_includes_context_length(self, monkeypatch):
        """_build_command() must include the context length."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        from dojo.serve.vllm_backend import VLLMBackend
        backend = VLLMBackend(
            model_path=Path("/models/test"),
            profile=_make_hal(),
            port=8080,
            context_length=8192,
        )
        cmd = backend._build_command()
        assert "8192" in cmd

    def test_command_starts_with_vllm_or_python(self, monkeypatch):
        """_build_command() first token must be 'python' or contain 'vllm'."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        cmd = backend._build_command()
        assert len(cmd) >= 1
        # Either starts with python (python -m vllm...) or directly with vllm
        first = cmd[0].lower()
        joined = " ".join(cmd).lower()
        assert "python" in first or "vllm" in joined, (
            f"Command does not appear to reference python or vllm: {cmd}"
        )


class TestVLLMServeConcurrentStub:
    def test_result_has_elapsed_s(self, monkeypatch):
        """Every result dict must contain an 'elapsed_s' key."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        results = backend.serve_concurrent([{"prompt": "a"}, {"prompt": "b"}])
        for r in results:
            assert "elapsed_s" in r, f"Missing 'elapsed_s' in result: {r}"

    def test_result_elapsed_s_non_negative(self, monkeypatch):
        """elapsed_s in every result must be >= 0."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        results = backend.serve_concurrent([{"prompt": "x"}, {"prompt": "y"}, {"prompt": "z"}])
        for r in results:
            assert isinstance(r["elapsed_s"], float)
            assert r["elapsed_s"] >= 0.0, f"elapsed_s must be non-negative, got {r['elapsed_s']}"

    def test_empty_requests_returns_empty(self, monkeypatch):
        """serve_concurrent([]) must return an empty list."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        results = backend.serve_concurrent([])
        assert results == []

    def test_result_tokens_is_int(self, monkeypatch):
        """'tokens' field in every result must be a positive integer."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        results = backend.serve_concurrent([{"prompt": "count tokens"}, {"prompt": "also count"}])
        for r in results:
            assert isinstance(r["tokens"], int), f"tokens must be int, got {type(r['tokens'])}"
            assert r["tokens"] > 0, f"tokens must be > 0, got {r['tokens']}"

    def test_evidence_has_throughput(self, monkeypatch):
        """SERVE_READY evidence must carry throughput_tokens_per_s > 0 for non-empty requests."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        backend.serve_concurrent([{"prompt": "p1"}, {"prompt": "p2"}, {"prompt": "p3"}])

        serve_ready_data = next(
            (
                e["evidence"]["data"]
                for e in EvidenceLogger._events
                if isinstance(e.get("evidence"), dict)
                and e["evidence"].get("type") == "serve.ready"
            ),
            None,
        )
        assert serve_ready_data is not None, "No serve.ready event found"
        assert serve_ready_data["throughput_tokens_per_s"] > 0, (
            f"throughput_tokens_per_s must be > 0, got {serve_ready_data['throughput_tokens_per_s']}"
        )


class TestVLLMMissingPackage:
    def test_missing_vllm_raises_backend_unavailable(self, monkeypatch):
        """BackendUnavailable raised when vllm is set to None in sys.modules."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        with patch.dict(sys.modules, {"vllm": None}):  # type: ignore[dict-item]
            backend = _make_backend()
            with pytest.raises(BackendUnavailable):
                backend.start()
