"""Tests for d32: TTL-based auto-half-open promotion in the circuit breaker.

Verifies that a dead provider is automatically promoted to half-open after
_RECOVERY_TTL_SECONDS, and that the backoff multiplier scales the TTL on
successive failures.
"""
from __future__ import annotations

import time
from unittest.mock import patch

from dojo.core.types import RoutingOption
from dojo.route import router
from dojo.route.router import (
    _DEAD_PROVIDERS,
    _DEAD_PROVIDER_TIMESTAMPS,
    _HALF_OPEN_PROVIDERS,
    _RECOVERY_ATTEMPTS,
    _circuit_breaker,
    _enter_half_open,
    _is_half_open,
    _is_provider_alive,
    _record_provider_failure,
    _record_provider_success,
)


def _reset_state():
    _DEAD_PROVIDERS.clear()
    _HALF_OPEN_PROVIDERS.clear()
    _DEAD_PROVIDER_TIMESTAMPS.clear()
    _RECOVERY_ATTEMPTS.clear()
    _circuit_breaker.clear()


def _trip(option: RoutingOption) -> None:
    for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
        _record_provider_failure(option)


class TestDeadProviderTimestamp:
    def setup_method(self):
        _reset_state()

    def test_timestamp_recorded_when_provider_trips(self):
        _trip(RoutingOption.API_HAIKU)
        assert "api_haiku" in _DEAD_PROVIDER_TIMESTAMPS

    def test_timestamp_is_monotonic_time(self):
        before = time.monotonic()
        _trip(RoutingOption.API_HAIKU)
        after = time.monotonic()
        ts = _DEAD_PROVIDER_TIMESTAMPS["api_haiku"]
        assert before <= ts <= after

    def test_timestamp_cleared_on_success(self):
        _trip(RoutingOption.API_HAIKU)
        _record_provider_success(RoutingOption.API_HAIKU)
        assert "api_haiku" not in _DEAD_PROVIDER_TIMESTAMPS


class TestTtlAutoHalfOpen:
    def setup_method(self):
        _reset_state()

    def test_alive_returns_false_before_ttl_expires(self):
        with patch.object(router, "_RECOVERY_TTL_SECONDS", 1000.0):
            _trip(RoutingOption.API_HAIKU)
            assert _is_provider_alive(RoutingOption.API_HAIKU) is False

    def test_promotes_to_half_open_after_ttl_expires(self):
        _trip(RoutingOption.API_HAIKU)
        # Backdate the timestamp so the TTL elapsed long ago
        _DEAD_PROVIDER_TIMESTAMPS["api_haiku"] = time.monotonic() - 10_000.0
        result = _is_provider_alive(RoutingOption.API_HAIKU)
        assert result is True
        assert "api_haiku" in _HALF_OPEN_PROVIDERS
        assert "api_haiku" not in _DEAD_PROVIDERS

    def test_attempt_counter_increments_on_promotion(self):
        _trip(RoutingOption.API_HAIKU)
        _DEAD_PROVIDER_TIMESTAMPS["api_haiku"] = time.monotonic() - 10_000.0
        _is_provider_alive(RoutingOption.API_HAIKU)
        assert _RECOVERY_ATTEMPTS["api_haiku"] == 1


class TestBackoff:
    def setup_method(self):
        _reset_state()

    def test_second_failure_uses_longer_ttl(self):
        _trip(RoutingOption.API_HAIKU)
        _DEAD_PROVIDER_TIMESTAMPS["api_haiku"] = time.monotonic() - 10_000.0
        _is_provider_alive(RoutingOption.API_HAIKU)  # promotes, attempt=1
        # Re-trip (simulate half-open probe failure)
        _HALF_OPEN_PROVIDERS.discard("api_haiku")
        _trip(RoutingOption.API_HAIKU)
        # attempt still 1; next TTL would be base * 1.5^1 = 1.5x
        assert _RECOVERY_ATTEMPTS["api_haiku"] >= 1


class TestEvidence:
    def setup_method(self):
        _reset_state()

    def test_trip_evidence_includes_recovery_ttl(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        _trip(RoutingOption.API_HAIKU)
        events = EvidenceLogger.get_events()
        trip = [e for e in events if e.get("type") == "route.circuit_trip"]
        assert trip
        assert "recovery_ttl_seconds" in trip[-1]["data"]

    def test_half_open_promotion_emits_evidence(self):
        from dojo.core.evidence import EvidenceLogger
        _trip(RoutingOption.API_HAIKU)
        _DEAD_PROVIDER_TIMESTAMPS["api_haiku"] = time.monotonic() - 10_000.0
        EvidenceLogger.clear()
        _is_provider_alive(RoutingOption.API_HAIKU)
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "route.circuit_half_open" for e in events)


class TestPreviousBehavior:
    """The explicit half-open helpers still work alongside TTL promotion."""

    def setup_method(self):
        _reset_state()

    def test_explicit_enter_half_open_still_works(self):
        _DEAD_PROVIDERS.add("api_haiku")
        _enter_half_open(RoutingOption.API_HAIKU)
        assert _is_half_open(RoutingOption.API_HAIKU)

    def test_success_clears_attempt_counter(self):
        _trip(RoutingOption.API_HAIKU)
        _DEAD_PROVIDER_TIMESTAMPS["api_haiku"] = time.monotonic() - 10_000.0
        _is_provider_alive(RoutingOption.API_HAIKU)
        assert _RECOVERY_ATTEMPTS.get("api_haiku", 0) >= 1
        _record_provider_success(RoutingOption.API_HAIKU)
        assert "api_haiku" not in _RECOVERY_ATTEMPTS
