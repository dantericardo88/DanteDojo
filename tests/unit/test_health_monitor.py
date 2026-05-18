"""Tests for d36: HealthMonitor auto-recovery of dead providers.

Validates that HealthMonitor.probe_once() and start()/stop() work, that
recovered providers are removed from _DEAD_PROVIDERS, and that the thread
is daemon + idempotent.
"""
from __future__ import annotations

import time
from unittest.mock import patch

from dojo.core.types import RoutingOption
from dojo.route.router import (
    HealthMonitor,
    _DEAD_PROVIDERS,
    _DEAD_PROVIDER_TIMESTAMPS,
    _HALF_OPEN_PROVIDERS,
    _RECOVERY_ATTEMPTS,
    _circuit_breaker,
)


def _reset_state():
    _DEAD_PROVIDERS.clear()
    _HALF_OPEN_PROVIDERS.clear()
    _DEAD_PROVIDER_TIMESTAMPS.clear()
    _RECOVERY_ATTEMPTS.clear()
    _circuit_breaker.clear()


class TestProbeOnce:
    def setup_method(self):
        _reset_state()

    def test_no_dead_providers_returns_zero(self):
        m = HealthMonitor()
        assert m.probe_once() == 0

    def test_recovers_dead_provider_when_healthy(self):
        _DEAD_PROVIDERS.add("api_haiku")
        m = HealthMonitor(
            endpoint_resolver=lambda _: "http://example/health",
        )
        with patch("dojo.route.router._health_check_with_timeout", return_value=True):
            recovered = m.probe_once()
        assert recovered == 1
        assert "api_haiku" not in _DEAD_PROVIDERS

    def test_leaves_dead_provider_dead_when_still_failing(self):
        _DEAD_PROVIDERS.add("api_haiku")
        m = HealthMonitor(
            endpoint_resolver=lambda _: "http://example/health",
        )
        with patch("dojo.route.router._health_check_with_timeout", return_value=False):
            recovered = m.probe_once()
        assert recovered == 0
        assert "api_haiku" in _DEAD_PROVIDERS

    def test_skips_provider_with_no_endpoint(self):
        _DEAD_PROVIDERS.add("unknown_provider_xyz")
        m = HealthMonitor(endpoint_resolver=lambda _: "")
        recovered = m.probe_once()
        assert recovered == 0

    def test_iteration_counter_increments(self):
        m = HealthMonitor()
        before = m.iterations
        m.probe_once()
        assert m.iterations == before + 1

    def test_total_recoveries_accumulates(self):
        _DEAD_PROVIDERS.add("api_haiku")
        m = HealthMonitor(endpoint_resolver=lambda _: "http://example/health")
        with patch("dojo.route.router._health_check_with_timeout", return_value=True):
            m.probe_once()
        assert m.total_recoveries == 1


class TestRecoveryClearsState:
    def setup_method(self):
        _reset_state()

    def test_recovery_clears_dead_timestamp(self):
        _DEAD_PROVIDERS.add("api_haiku")
        _DEAD_PROVIDER_TIMESTAMPS["api_haiku"] = time.monotonic()
        m = HealthMonitor(endpoint_resolver=lambda _: "http://example/health")
        with patch("dojo.route.router._health_check_with_timeout", return_value=True):
            m.probe_once()
        assert "api_haiku" not in _DEAD_PROVIDER_TIMESTAMPS

    def test_recovery_clears_attempt_counter(self):
        _DEAD_PROVIDERS.add("api_haiku")
        _RECOVERY_ATTEMPTS["api_haiku"] = 3
        m = HealthMonitor(endpoint_resolver=lambda _: "http://example/health")
        with patch("dojo.route.router._health_check_with_timeout", return_value=True):
            m.probe_once()
        assert "api_haiku" not in _RECOVERY_ATTEMPTS

    def test_recovery_resets_circuit_breaker(self):
        _DEAD_PROVIDERS.add("api_haiku")
        _circuit_breaker["api_haiku"] = 2
        m = HealthMonitor(endpoint_resolver=lambda _: "http://example/health")
        with patch("dojo.route.router._health_check_with_timeout", return_value=True):
            m.probe_once()
        assert _circuit_breaker["api_haiku"] == 0


class TestEvidence:
    def setup_method(self):
        _reset_state()

    def test_recovery_emits_evidence(self):
        from dojo.core.evidence import EvidenceLogger
        _DEAD_PROVIDERS.add("api_haiku")
        EvidenceLogger.clear()
        m = HealthMonitor(endpoint_resolver=lambda _: "http://example/health")
        with patch("dojo.route.router._health_check_with_timeout", return_value=True):
            m.probe_once()
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "route.health_monitor_recovery" for e in events)


class TestThreadLifecycle:
    def setup_method(self):
        _reset_state()

    def test_start_creates_thread(self):
        m = HealthMonitor(interval_seconds=0.05)
        m.start()
        try:
            assert m._thread is not None
            assert m._thread.is_alive()
        finally:
            m.stop(timeout_seconds=1.0)

    def test_thread_is_daemon(self):
        m = HealthMonitor(interval_seconds=0.05)
        m.start()
        try:
            assert m._thread.daemon is True
        finally:
            m.stop(timeout_seconds=1.0)

    def test_stop_terminates_thread(self):
        m = HealthMonitor(interval_seconds=0.05)
        m.start()
        m.stop(timeout_seconds=2.0)
        assert m._thread is None

    def test_start_is_idempotent(self):
        m = HealthMonitor(interval_seconds=0.05)
        m.start()
        first = m._thread
        m.start()
        try:
            assert m._thread is first
        finally:
            m.stop(timeout_seconds=1.0)

    def test_stop_is_idempotent(self):
        m = HealthMonitor()
        m.stop()
        m.stop()  # no exception

    def test_thread_recovers_provider_in_background(self):
        _DEAD_PROVIDERS.add("api_haiku")
        m = HealthMonitor(
            interval_seconds=0.02,
            endpoint_resolver=lambda _: "http://example/health",
        )
        with patch("dojo.route.router._health_check_with_timeout", return_value=True):
            m.start()
            # Wait briefly for at least one probe
            deadline = time.monotonic() + 2.0
            while "api_haiku" in _DEAD_PROVIDERS and time.monotonic() < deadline:
                time.sleep(0.05)
            m.stop(timeout_seconds=1.0)
        assert "api_haiku" not in _DEAD_PROVIDERS
