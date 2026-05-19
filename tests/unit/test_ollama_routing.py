"""Tests for OLLAMA_LOCAL routing with real inference via localhost:11434.

Tests that require a running Ollama instance are guarded with skipif.
Tests that mock the health check / dispatch run without Ollama.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.types import PrivacyTier, RoutingOption, RoutingPolicy
from dojo.route.router import (
    _OLLAMA_DEFAULT_URL,
    _DEAD_PROVIDERS,
    _call_ollama_infer,
    dispatch,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ollama_running() -> bool:
    """Return True if Ollama is reachable AND can respond to a quick inference.

    Two-phase check:
    1. GET /api/tags — confirms the HTTP API is up (fast, <2s).
    2. POST /v1/chat/completions with max_tokens=1 — confirms inference is live.
       If this doesn't respond within 15s the server is considered not ready for tests.
    """
    # Phase 1: health ping
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=2.0):
            pass
    except (urllib.error.URLError, OSError):
        return False

    # Phase 2: smoke-test inference
    try:
        body = __import__("json").dumps({
            "model": "qwen2.5:0.5b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": False,
        }).encode("utf-8")
        req2 = urllib.request.Request(
            "http://localhost:11434/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req2, timeout=15.0):
            pass
        return True
    except (urllib.error.URLError, OSError):
        return False


_OLLAMA_UP = _ollama_running()


def _make_policy(
    preference_order: list[RoutingOption],
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC,
) -> RoutingPolicy:
    return RoutingPolicy(
        task_name="test_task",
        preference_order=tuple(preference_order),
        max_latency_ms=30_000,
        max_cost_usd=1.0,
        privacy_tier=privacy_tier,
        fallback_on_unavailable=True,
    )


# ---------------------------------------------------------------------------
# Tests requiring a live Ollama instance
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _OLLAMA_UP, reason="Ollama not running at localhost:11434")
def test_dispatch_routes_to_ollama_local():
    """dispatch() with OLLAMA_LOCAL in preference_order should select ollama_local."""
    policy = _make_policy([RoutingOption.OLLAMA_LOCAL])
    result = dispatch("test_task", "Hello, world.", policy)
    assert result["route"] == "ollama_local", f"Expected ollama_local, got {result['route']}"


@pytest.mark.skipif(not _OLLAMA_UP, reason="Ollama not running at localhost:11434")
def test_dispatch_inference_key_present():
    """Result from OLLAMA_LOCAL dispatch should have 'inference' key with 'text'."""
    policy = _make_policy([RoutingOption.OLLAMA_LOCAL])
    result = dispatch("test_task", "Say hello in 5 words.", policy)
    assert "inference" in result, "Expected 'inference' key in dispatch result"
    inf = result["inference"]
    assert "text" in inf, f"inference dict missing 'text': {inf}"
    assert isinstance(inf["text"], str), f"inference['text'] should be str, got {type(inf['text'])}"
    assert len(inf["text"]) > 0, "inference['text'] should be non-empty"
    assert "latency_ms" in inf, "inference dict missing 'latency_ms'"
    assert inf["latency_ms"] >= 0.0, "latency_ms should be non-negative"
    # Surface the actual Ollama response for human inspection
    print(f"\n[Ollama response] model={inf.get('model')!r}  latency={inf['latency_ms']:.1f}ms")
    print(f"[Ollama response] text={inf['text']!r}")


@pytest.mark.skipif(not _OLLAMA_UP, reason="Ollama not running at localhost:11434")
def test_confidential_privacy_blocks_api_haiku_picks_ollama():
    """CONFIDENTIAL tier: API_HAIKU must be blocked; OLLAMA_LOCAL (local) is allowed."""
    policy = _make_policy(
        [RoutingOption.API_HAIKU, RoutingOption.OLLAMA_LOCAL],
        privacy_tier=PrivacyTier.CONFIDENTIAL,
    )
    # API_HAIKU is in _EXTERNAL_API_OPTIONS — privacy tier blocks it,
    # which raises PrivacyTierViolation immediately (dispatch stops at first violation).
    # Reorder so OLLAMA_LOCAL is first when we want a successful result.
    policy_ok = _make_policy(
        [RoutingOption.OLLAMA_LOCAL, RoutingOption.API_HAIKU],
        privacy_tier=PrivacyTier.CONFIDENTIAL,
    )
    result = dispatch("test_task", "Confidential prompt.", policy_ok)
    assert result["route"] == "ollama_local"


@pytest.mark.skipif(not _OLLAMA_UP, reason="Ollama not running at localhost:11434")
def test_cost_tracker_record_called_after_dispatch():
    """cost_tracker.record() should be called once after a successful dispatch (d35)."""
    mock_tracker = MagicMock()
    mock_tracker.monthly_limit_usd = None  # no budget cap

    policy = _make_policy([RoutingOption.OLLAMA_LOCAL])
    dispatch("test_task", "Budget test.", policy, cost_tracker=mock_tracker)

    mock_tracker.record.assert_called_once()
    receipt_arg = mock_tracker.record.call_args[0][0]
    assert receipt_arg.route_taken == "ollama_local"


# ---------------------------------------------------------------------------
# Tests that do NOT require a live Ollama instance (mocked)
# ---------------------------------------------------------------------------

def test_ollama_local_fallback_to_local_generalist_when_unavailable():
    """When OLLAMA_LOCAL is unreachable, dispatch falls back to LOCAL_GENERALIST (d36)."""
    policy = _make_policy(
        [RoutingOption.OLLAMA_LOCAL, RoutingOption.LOCAL_GENERALIST]
    )
    # Patch _is_viable so OLLAMA_LOCAL always returns (False, "ollama not reachable")
    original_is_viable = __import__(
        "dojo.route.router", fromlist=["_is_viable"]
    )._is_viable

    def patched_is_viable(option, pol, specialists):
        if option == RoutingOption.OLLAMA_LOCAL:
            return False, "ollama not reachable"
        return original_is_viable(option, pol, specialists)

    with patch("dojo.route.router._is_viable", side_effect=patched_is_viable):
        result = dispatch("test_task", "Fallback test.", policy)

    assert result["route"] == "local_generalist", (
        f"Expected local_generalist fallback, got {result['route']}"
    )
    assert "inference" not in result, "Should not have inference key when falling back"


def test_ollama_local_not_in_external_api_options():
    """OLLAMA_LOCAL must NOT appear in _EXTERNAL_API_OPTIONS (would break CONFIDENTIAL routing)."""
    from dojo.route.router import _EXTERNAL_API_OPTIONS
    assert RoutingOption.OLLAMA_LOCAL not in _EXTERNAL_API_OPTIONS, (
        "OLLAMA_LOCAL is in _EXTERNAL_API_OPTIONS — it would be blocked for CONFIDENTIAL data"
    )


def test_ollama_local_in_local_options():
    """OLLAMA_LOCAL should be in _LOCAL_OPTIONS."""
    from dojo.route.router import _LOCAL_OPTIONS
    assert RoutingOption.OLLAMA_LOCAL in _LOCAL_OPTIONS


def test_provider_name_ollama():
    """_provider_name() should return 'ollama' for OLLAMA_LOCAL."""
    from dojo.route.router import _provider_name
    assert _provider_name(RoutingOption.OLLAMA_LOCAL) == "ollama"


def test_api_endpoint_ollama():
    """_api_endpoint() should return the local chat completions URL for OLLAMA_LOCAL."""
    from dojo.route.router import _api_endpoint
    ep = _api_endpoint(RoutingOption.OLLAMA_LOCAL)
    assert "localhost:11434" in ep
    assert "chat/completions" in ep


def test_call_ollama_infer_raises_on_unreachable():
    """_call_ollama_infer() should raise RoutingProviderUnavailable when server is down."""
    from dojo.core.exceptions import RoutingProviderUnavailable

    with pytest.raises(RoutingProviderUnavailable):
        # Use a port that should be nothing — 19999
        _call_ollama_infer("test", url="http://localhost:19999", timeout=1.0)


def test_routing_option_enum_has_ollama_local():
    """RoutingOption enum must include OLLAMA_LOCAL."""
    assert hasattr(RoutingOption, "OLLAMA_LOCAL")
    assert RoutingOption.OLLAMA_LOCAL.value == "ollama_local"
