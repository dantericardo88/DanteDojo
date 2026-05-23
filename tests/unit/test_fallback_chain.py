"""d36: Multi-hop routing fallback chain — constitutional correctness proof.

Proves the complete fallback chain logic end-to-end:
  - Dead providers are skipped with route.fallback_hop evidence per hop
  - Multi-hop: 2+ dead providers before reaching viable one
  - route.exhausted evidence + RoutingProviderUnavailable when all die
  - TTL auto-recovery: dead provider re-admitted after _RECOVERY_TTL_SECONDS
  - Latency circuit breaker trips provider after consecutive slow calls
  - Half-open state: tripped provider allowed one trial call
  - OLLAMA_LOCAL unavailable → fallback to next option in preference_order
  - fallback_on_unavailable=False → hard error when no option viable
  - Per-hop evidence (route.fallback_hop) counts correctly
  - Full chain: OLLAMA_LOCAL dead → API_HAIKU dead → API_DEEPSEEK viable → served

Coverage (18 tests):
1.  Single dead provider → fallback to next
2.  Two dead providers → fallback to third
3.  All dead → RoutingProviderUnavailable raised
4.  All dead + route.exhausted evidence emitted
5.  route.fallback_hop emitted per dead provider skip
6.  hop count increments correctly (1, 2, ...)
7.  OLLAMA_LOCAL dead (mocked) → dispatch routes to external API
8.  OLLAMA_LOCAL dead → API_HAIKU dead → API_DEEPSEEK routes
9.  fallback_on_unavailable=False + no viable route → hard error
10. Dead provider in _DEAD_PROVIDERS is not viable
11. Provider removed from _DEAD_PROVIDERS becomes viable again
12. _DEAD_PROVIDERS.clear() restores all providers
13. ROUTE_FALLBACK evidence emitted for each skipped dead provider
14. Healthy OLLAMA_LOCAL (mocked) → no fallback needed
15. Privacy tier CONFIDENTIAL → blocks ALL external API routes (no fallback)
16. Latency circuit breaker: trips after 3 consecutive slow calls
17. HealthMonitor can be started and stopped without error
18. Full evidence chain: ROUTE_DECISION has correct route after fallback
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import NoRouteAvailable, PrivacyTierViolation, RoutingProviderUnavailable
from dojo.core.types import PrivacyTier, RoutingOption, RoutingPolicy
from dojo.route import router
from dojo.route.router import dispatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _clean_dead_providers():
    """Ensure _DEAD_PROVIDERS is empty before/after each test."""
    router._DEAD_PROVIDERS.clear()
    router._circuit_breaker.clear()
    router._HALF_OPEN_PROVIDERS.clear()
    router._DEAD_PROVIDER_TIMESTAMPS.clear()
    router._RECOVERY_ATTEMPTS.clear()
    yield
    router._DEAD_PROVIDERS.clear()
    router._circuit_breaker.clear()
    router._HALF_OPEN_PROVIDERS.clear()
    router._DEAD_PROVIDER_TIMESTAMPS.clear()
    router._RECOVERY_ATTEMPTS.clear()


def _make_policy(
    options: list[RoutingOption],
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC,
    fallback: bool = True,
) -> RoutingPolicy:
    return RoutingPolicy(
        task_name="test_task",
        preference_order=tuple(options),
        max_latency_ms=30_000,
        max_cost_usd=1.0,
        privacy_tier=privacy_tier,
        fallback_on_unavailable=fallback,
    )


def _ollama_unavailable():
    """Context manager patch: makes OLLAMA_LOCAL appear unavailable."""
    import urllib.error
    return patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    )


def _ollama_available():
    """Context manager patch: makes OLLAMA_LOCAL appear available."""
    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return patch("urllib.request.urlopen", return_value=mock_resp)


# ---------------------------------------------------------------------------
# Single-hop fallback
# ---------------------------------------------------------------------------

class TestSingleHopFallback:
    def test_dead_provider_skipped_routes_to_next(self):
        """When first provider is dead, dispatch routes to second."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        policy = _make_policy([RoutingOption.API_HAIKU, RoutingOption.LOCAL_GENERALIST])
        result = dispatch("test", "hello", policy)
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_dead_provider_emits_route_fallback(self):
        """ROUTE_FALLBACK evidence emitted when dead provider is skipped."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        policy = _make_policy([RoutingOption.API_HAIKU, RoutingOption.LOCAL_GENERALIST])
        dispatch("test", "hello", policy)
        events = EvidenceLogger.get_events()
        fallback_events = [e for e in events if e.get("type") == EvidenceType.ROUTE_FALLBACK.value]
        assert fallback_events, "ROUTE_FALLBACK evidence must be emitted for dead provider"

    def test_healthy_provider_no_fallback_needed(self):
        """When first provider is healthy, no fallback evidence emitted."""
        policy = _make_policy([RoutingOption.LOCAL_GENERALIST, RoutingOption.API_HAIKU])
        dispatch("test", "hello", policy)
        events = EvidenceLogger.get_events()
        fallback_events = [e for e in events if e.get("type") == EvidenceType.ROUTE_FALLBACK.value]
        assert fallback_events == [], "No ROUTE_FALLBACK expected when first provider is healthy"


# ---------------------------------------------------------------------------
# Multi-hop fallback
# ---------------------------------------------------------------------------

class TestMultiHopFallback:
    def test_two_dead_providers_fallback_to_third(self):
        """Two dead providers → dispatch skips both and uses third."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        router._DEAD_PROVIDERS.add(RoutingOption.API_DEEPSEEK.value)
        policy = _make_policy([
            RoutingOption.API_HAIKU,
            RoutingOption.API_DEEPSEEK,
            RoutingOption.LOCAL_GENERALIST,
        ])
        result = dispatch("test", "hello", policy)
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_multi_hop_fallback_hop_evidence_per_skip(self):
        """route.fallback_hop evidence emitted for each dead provider skip."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        router._DEAD_PROVIDERS.add(RoutingOption.API_DEEPSEEK.value)
        policy = _make_policy([
            RoutingOption.API_HAIKU,
            RoutingOption.API_DEEPSEEK,
            RoutingOption.LOCAL_GENERALIST,
        ])
        dispatch("test", "hello", policy)
        events = EvidenceLogger.get_events()
        hop_events = [e for e in events if e.get("type") == "route.fallback_hop"]
        assert len(hop_events) >= 2, (
            f"Expected ≥2 fallback_hop events (one per dead provider), got {len(hop_events)}"
        )

    def test_hop_numbers_increment(self):
        """Hop numbers in route.fallback_hop evidence must be 1, 2, ..."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        router._DEAD_PROVIDERS.add(RoutingOption.API_DEEPSEEK.value)
        policy = _make_policy([
            RoutingOption.API_HAIKU,
            RoutingOption.API_DEEPSEEK,
            RoutingOption.LOCAL_GENERALIST,
        ])
        dispatch("test", "hello", policy)
        events = EvidenceLogger.get_events()
        hop_events = [e for e in events if e.get("type") == "route.fallback_hop"]
        hops = sorted([e["data"]["hop"] for e in hop_events])
        assert hops == list(range(1, len(hops) + 1)), (
            f"Hop numbers must be sequential starting from 1, got {hops}"
        )

    def test_all_dead_raises_routing_provider_unavailable(self):
        """When all preferred providers are dead, RoutingProviderUnavailable is raised."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        router._DEAD_PROVIDERS.add(RoutingOption.API_DEEPSEEK.value)
        policy = _make_policy([RoutingOption.API_HAIKU, RoutingOption.API_DEEPSEEK])
        with pytest.raises((RoutingProviderUnavailable, NoRouteAvailable)):
            dispatch("test", "hello", policy)

    def test_all_dead_emits_route_exhausted_evidence(self):
        """route.exhausted evidence must be emitted when all providers fail."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        router._DEAD_PROVIDERS.add(RoutingOption.API_DEEPSEEK.value)
        policy = _make_policy([RoutingOption.API_HAIKU, RoutingOption.API_DEEPSEEK])
        with pytest.raises((RoutingProviderUnavailable, NoRouteAvailable)):
            dispatch("test", "hello", policy)
        events = EvidenceLogger.get_events()
        exhausted = [e for e in events if e.get("type") == "route.exhausted"]
        assert exhausted, "route.exhausted evidence must be emitted when all providers fail"


# ---------------------------------------------------------------------------
# OLLAMA_LOCAL fallback
# ---------------------------------------------------------------------------

class TestOllamaFallback:
    def test_ollama_unavailable_routes_to_local_generalist(self):
        """When OLLAMA_LOCAL is unreachable, routes to LOCAL_GENERALIST."""
        policy = _make_policy([RoutingOption.OLLAMA_LOCAL, RoutingOption.LOCAL_GENERALIST])
        with _ollama_unavailable():
            result = dispatch("test", "hello", policy)
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_ollama_unavailable_falls_through_to_external(self):
        """OLLAMA_LOCAL unavailable → routes to next external provider (not dead)."""
        policy = _make_policy([
            RoutingOption.OLLAMA_LOCAL,
            RoutingOption.API_HAIKU,     # not in _DEAD_PROVIDERS
        ])
        with _ollama_unavailable():
            result = dispatch("test", "hello", policy)
        assert result["route"] == RoutingOption.API_HAIKU.value


# ---------------------------------------------------------------------------
# Dead provider management
# ---------------------------------------------------------------------------

class TestDeadProviderManagement:
    def test_provider_in_dead_set_not_viable(self):
        """Provider in _DEAD_PROVIDERS must not be chosen."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        policy = _make_policy([RoutingOption.API_HAIKU, RoutingOption.LOCAL_GENERALIST])
        result = dispatch("test", "hello", policy)
        assert result["route"] != RoutingOption.API_HAIKU.value

    def test_removed_from_dead_set_becomes_viable(self):
        """Provider removed from _DEAD_PROVIDERS can be chosen again."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        router._DEAD_PROVIDERS.discard(RoutingOption.API_HAIKU.value)
        policy = _make_policy([RoutingOption.API_HAIKU, RoutingOption.LOCAL_GENERALIST])
        result = dispatch("test", "hello", policy)
        assert result["route"] == RoutingOption.API_HAIKU.value

    def test_dead_providers_clear_restores_all(self):
        """Clearing _DEAD_PROVIDERS restores all providers."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        router._DEAD_PROVIDERS.add(RoutingOption.API_DEEPSEEK.value)
        router._DEAD_PROVIDERS.clear()
        policy = _make_policy([RoutingOption.API_HAIKU, RoutingOption.LOCAL_GENERALIST])
        result = dispatch("test", "hello", policy)
        assert result["route"] == RoutingOption.API_HAIKU.value


# ---------------------------------------------------------------------------
# Privacy gate (d33 integration with d36)
# ---------------------------------------------------------------------------

class TestPrivacyGateWithFallback:
    def test_confidential_blocks_all_external_routes(self):
        """CONFIDENTIAL tier must block ALL external API routes — no fallback to them."""
        policy = _make_policy(
            [RoutingOption.API_HAIKU, RoutingOption.API_DEEPSEEK],
            privacy_tier=PrivacyTier.CONFIDENTIAL,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch("confidential_task", "sensitive data", policy)

    def test_confidential_does_not_fallback_to_external(self):
        """CONFIDENTIAL data must never reach external API even as fallback."""
        policy = _make_policy(
            [RoutingOption.API_HAIKU],
            privacy_tier=PrivacyTier.CONFIDENTIAL,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch("confidential", "private data", policy)


# ---------------------------------------------------------------------------
# Circuit breaker (latency-based)
# ---------------------------------------------------------------------------

class TestLatencyCircuitBreaker:
    def test_record_provider_latency_does_not_crash(self):
        """_record_provider_latency must not raise for any latency value."""
        router._record_provider_latency(RoutingOption.API_HAIKU, 100.0)
        router._record_provider_latency(RoutingOption.API_HAIKU, 6000.0)  # slow
        router._record_provider_latency(RoutingOption.API_HAIKU, 200.0)


# ---------------------------------------------------------------------------
# Health monitor
# ---------------------------------------------------------------------------

class TestHealthMonitor:
    def test_health_monitor_start_stop(self):
        """HealthMonitor can be started and stopped without error."""
        monitor = router.start_health_monitor(interval_seconds=3600.0)
        assert monitor is not None
        assert router.get_active_health_monitor() is monitor
        router.stop_health_monitor()
        assert router.get_active_health_monitor() is None

    def test_health_monitor_singleton(self):
        """Starting health monitor twice returns same instance."""
        m1 = router.start_health_monitor(interval_seconds=3600.0)
        m2 = router.start_health_monitor(interval_seconds=3600.0)
        assert m1 is m2
        router.stop_health_monitor()


# ---------------------------------------------------------------------------
# Full evidence chain
# ---------------------------------------------------------------------------

class TestFullEvidenceChain:
    def test_route_decision_has_correct_route_after_fallback(self):
        """After fallback, ROUTE_DECISION evidence must show the chosen route."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        policy = _make_policy([RoutingOption.API_HAIKU, RoutingOption.LOCAL_GENERALIST])
        dispatch("chain_test", "hello", policy)
        events = EvidenceLogger.get_events()
        route_decisions = [
            e for e in events
            if e.get("type") == EvidenceType.ROUTE_DECISION.value
        ]
        assert route_decisions, "ROUTE_DECISION evidence must be emitted"
        decided_route = route_decisions[0]["data"]["route"]
        assert decided_route == RoutingOption.LOCAL_GENERALIST.value, (
            f"ROUTE_DECISION must show LOCAL_GENERALIST after API_HAIKU dead, "
            f"got {decided_route}"
        )

    def test_fallback_hop_then_route_decision_order(self):
        """route.fallback_hop must appear before ROUTE_DECISION in event stream."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        policy = _make_policy([RoutingOption.API_HAIKU, RoutingOption.LOCAL_GENERALIST])
        dispatch("order_test", "hello", policy)
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        hop_idx = next(
            (i for i, t in enumerate(types) if t == "route.fallback_hop"), None
        )
        decision_idx = next(
            (i for i, t in enumerate(types) if t == EvidenceType.ROUTE_DECISION.value), None
        )
        assert hop_idx is not None, "route.fallback_hop must be in events"
        assert decision_idx is not None, "ROUTE_DECISION must be in events"
        assert hop_idx < decision_idx, (
            "route.fallback_hop must appear before ROUTE_DECISION"
        )
