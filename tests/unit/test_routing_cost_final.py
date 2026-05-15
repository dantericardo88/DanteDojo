"""Final routing + cost tests — d32 (circuit breaker), d35 (budget soft-warn), d36 (multi-hop).

Covers:
- d32: latency-based circuit breaker, half-open state, adversarial HTTP-200+error body
- d35: soft-warn at 80%, BudgetExceeded at 101%, soft-warn evidence, reset_monthly clears state
- d36: chaos multi-hop (A→B→C), all-providers-exhausted, payload preservation, per-hop evidence
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import BudgetExceeded, NoRouteAvailable, RoutingProviderUnavailable
from dojo.core.types import (
    CostReceipt,
    PrivacyTier,
    RoutingOption,
    RoutingPolicy,
)
from dojo.route import router
from dojo.route.cost_tracker import BudgetStatus, CostTracker
from dojo.route.router import dispatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(
    options: list[RoutingOption],
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC,
    task: str = "test",
) -> RoutingPolicy:
    return RoutingPolicy(
        task_name=task,
        preference_order=tuple(options),
        max_latency_ms=30_000,
        max_cost_usd=1.0,
        privacy_tier=privacy_tier,
        fallback_on_unavailable=True,
    )


def _make_receipt(
    route: RoutingOption = RoutingOption.API_HAIKU,
    task: str = "test",
    cost: float = 0.0,
    month: int = 5,
    year: int = 2026,
) -> CostReceipt:
    return CostReceipt(
        task=task,
        route_taken=route.value,
        tokens_in=100,
        tokens_out=50,
        cost_usd=cost,
        latency_ms=120.0,
        local_displaced_estimate_usd=0.0,
        recorded_at=datetime(year, month, 15, tzinfo=timezone.utc),
    )


def _events_of_type(event_type: str) -> list[dict]:
    return [
        e["evidence"]
        for e in EvidenceLogger._events
        if isinstance(e.get("evidence"), dict)
        and e["evidence"].get("type") == event_type
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def reset_router_state():
    """Reset all circuit-breaker and dead-provider state between tests."""
    router._DEAD_PROVIDERS.clear()
    router._circuit_breaker.clear()
    router._HALF_OPEN_PROVIDERS.clear()
    router._latency_window.clear()
    yield
    router._DEAD_PROVIDERS.clear()
    router._circuit_breaker.clear()
    router._HALF_OPEN_PROVIDERS.clear()
    router._latency_window.clear()


@pytest.fixture(autouse=True)
def no_real_gpu(monkeypatch):
    monkeypatch.setenv("DOJO_REAL_GPU", "0")
    monkeypatch.delenv("DOJO_HEALTH_CHECK", raising=False)


# ===========================================================================
# d32 — Circuit-breaker routing
# ===========================================================================

class TestD32CircuitBreaker:
    """d32: latency-based breaker, half-open state, adversarial HTTP-200+error."""

    # --- Latency-based circuit breaker ---

    def test_latency_breaker_trips_after_consecutive_slow_calls(self):
        """Three consecutive calls above threshold trip the latency-based breaker."""
        option = RoutingOption.API_HAIKU
        threshold_ms = router._LATENCY_BREAKER_THRESHOLD_MS
        window = router._LATENCY_BREAKER_WINDOW

        # Record window-1 slow calls — should NOT trip yet
        for _ in range(window - 1):
            router._record_provider_latency(option, threshold_ms + 1.0)
        assert option.value not in router._DEAD_PROVIDERS

        # Record the final slow call — must trip
        router._record_provider_latency(option, threshold_ms + 1.0)
        assert option.value in router._DEAD_PROVIDERS

    def test_latency_breaker_does_not_trip_on_fast_calls(self):
        """Fast calls (below threshold) never trip the latency breaker."""
        option = RoutingOption.API_DEEPSEEK
        fast_ms = router._LATENCY_BREAKER_THRESHOLD_MS - 1.0
        for _ in range(router._LATENCY_BREAKER_WINDOW * 2):
            router._record_provider_latency(option, fast_ms)
        assert option.value not in router._DEAD_PROVIDERS

    def test_latency_breaker_emits_circuit_trip_evidence(self):
        """route.circuit_trip evidence is emitted when latency breaker trips."""
        option = RoutingOption.API_SONNET
        threshold_ms = router._LATENCY_BREAKER_THRESHOLD_MS
        for _ in range(router._LATENCY_BREAKER_WINDOW):
            router._record_provider_latency(option, threshold_ms + 500.0)

        trip_events = _events_of_type("route.circuit_trip")
        assert len(trip_events) >= 1
        assert any(
            e["data"]["provider"] == option.value
            and e["data"]["reason"] == "consecutive_high_latency"
            for e in trip_events
        )

    def test_latency_breaker_resets_window_after_trip(self):
        """After tripping, the latency window is cleared so a new run can recover."""
        option = RoutingOption.API_GPT5_MINI
        threshold_ms = router._LATENCY_BREAKER_THRESHOLD_MS
        for _ in range(router._LATENCY_BREAKER_WINDOW):
            router._record_provider_latency(option, threshold_ms + 1.0)

        assert option.value in router._DEAD_PROVIDERS
        # Window should be empty after trip
        assert len(router._latency_window.get(option.value, [])) == 0

    # --- Half-open state ---

    def test_half_open_enter_removes_from_dead(self):
        """_enter_half_open() moves the provider from dead to half-open."""
        option = RoutingOption.API_HAIKU
        router._DEAD_PROVIDERS.add(option.value)

        router._enter_half_open(option)

        assert option.value not in router._DEAD_PROVIDERS
        assert option.value in router._HALF_OPEN_PROVIDERS

    def test_half_open_success_closes_breaker(self):
        """A successful probe in half-open state closes the breaker (removes from half-open)."""
        option = RoutingOption.API_HAIKU
        router._DEAD_PROVIDERS.add(option.value)
        router._enter_half_open(option)

        # Simulate success: _record_provider_success closes it
        router._record_provider_success(option)

        assert option.value not in router._HALF_OPEN_PROVIDERS
        assert option.value not in router._DEAD_PROVIDERS

    def test_half_open_failure_returns_to_dead(self):
        """A failure during half-open probe re-trips the breaker (back to dead)."""
        option = RoutingOption.API_HAIKU
        router._DEAD_PROVIDERS.add(option.value)
        router._enter_half_open(option)

        # Simulate threshold-many failures to re-trip
        for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
            router._record_provider_failure(option)

        assert option.value in router._DEAD_PROVIDERS
        assert option.value not in router._HALF_OPEN_PROVIDERS

    def test_half_open_success_emits_circuit_reset_evidence(self):
        """route.circuit_reset evidence is emitted when half-open probe succeeds."""
        option = RoutingOption.API_DEEPSEEK
        router._DEAD_PROVIDERS.add(option.value)
        router._enter_half_open(option)
        router._record_provider_success(option)

        reset_events = _events_of_type("route.circuit_reset")
        assert any(e["data"]["provider"] == option.value for e in reset_events)

    # --- Adversarial: HTTP 200 with {"error": "..."} body ---

    def test_failure_circuit_trip_emits_evidence(self):
        """route.circuit_trip evidence is emitted after threshold error-rate failures."""
        option = RoutingOption.API_SONNET
        for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
            router._record_provider_failure(option)

        trip_events = _events_of_type("route.circuit_trip")
        assert len(trip_events) >= 1
        assert any(
            e["data"]["provider"] == option.value
            and e["data"]["reason"] == "consecutive_failures"
            for e in trip_events
        )

    def test_adversarial_http200_error_body_counted_as_failure(self):
        """An HTTP-200 response with {'error': '...'} in the body counts as a failure.

        The router's circuit breaker must trip when the caller records the
        failure (simulating detection of an error-body in 200 OK responses).
        """
        option = RoutingOption.API_GPT5_MINI

        # Simulate detecting error bodies in 200 OK responses and recording them
        adversarial_responses = [
            {"status_code": 200, "body": json.dumps({"error": "model_overloaded"})},
            {"status_code": 200, "body": json.dumps({"error": "rate_limit_exceeded"})},
            {"status_code": 200, "body": json.dumps({"error": "context_length_exceeded"})},
        ]

        for resp in adversarial_responses:
            body = json.loads(resp["body"])
            # Detection logic: HTTP 200 but body contains "error" key → record failure
            if resp["status_code"] == 200 and "error" in body:
                router._record_provider_failure(option)

        # After 3 adversarial-200 failures, breaker must trip
        assert option.value in router._DEAD_PROVIDERS

    def test_adversarial_http200_error_body_trip_emits_evidence(self):
        """route.circuit_trip evidence emitted for adversarial HTTP-200+error bodies."""
        option = RoutingOption.API_HAIKU
        adversarial_bodies = [{"error": "upstream_error"}] * router._CIRCUIT_BREAKER_THRESHOLD

        for body in adversarial_bodies:
            if "error" in body:
                router._record_provider_failure(option)

        trip_events = _events_of_type("route.circuit_trip")
        assert any(e["data"]["provider"] == option.value for e in trip_events)


# ===========================================================================
# d35 — Cost tracking with budget soft-warn
# ===========================================================================

class TestD35BudgetSoftWarn:
    """d35: soft-warn at 80%, exceeded at 101%, evidence emission, reset clears state."""

    def test_soft_warn_returned_at_85_percent(self, tmp_path: Path):
        """check_budget() returns SOFT_WARN when projected usage is 85% of limit."""
        limit = 1.0
        tracker = CostTracker(storage_path=tmp_path / "r.jsonl", monthly_limit_usd=limit)

        # Record 80% of limit as existing spend in current month
        existing_spend = 0.80
        tracker.record(_make_receipt(cost=existing_spend))

        # New request of 5% → projected = 85%
        result = tracker.check_budget(0.05)
        assert result == BudgetStatus.SOFT_WARN

    def test_soft_warn_does_not_raise(self, tmp_path: Path):
        """check_budget() must NOT raise when spend is in the soft-warn zone."""
        limit = 1.0
        tracker = CostTracker(storage_path=tmp_path / "r.jsonl", monthly_limit_usd=limit)
        tracker.record(_make_receipt(cost=0.80))

        # Should return SOFT_WARN without raising
        try:
            result = tracker.check_budget(0.05)
        except BudgetExceeded:
            pytest.fail("check_budget raised BudgetExceeded in soft-warn zone")
        assert result == BudgetStatus.SOFT_WARN

    def test_budget_exceeded_raised_at_101_percent(self, tmp_path: Path):
        """BudgetExceeded is raised when projected spend exceeds 100% of limit."""
        limit = 1.0
        tracker = CostTracker(storage_path=tmp_path / "r.jsonl", monthly_limit_usd=limit)
        tracker.record(_make_receipt(cost=0.95))

        with pytest.raises(BudgetExceeded):
            tracker.check_budget(0.10)  # 95% + 10% = 105% > 100%

    def test_ok_returned_below_soft_warn(self, tmp_path: Path):
        """BudgetStatus.OK returned when projected spend is below 80% of limit."""
        tracker = CostTracker(storage_path=tmp_path / "r.jsonl", monthly_limit_usd=1.0)
        result = tracker.check_budget(0.50)
        assert result == BudgetStatus.OK

    def test_soft_warn_evidence_emitted_at_crossing(self, tmp_path: Path):
        """cost.budget_soft_warn evidence is emitted on the first soft-warn crossing."""
        limit = 1.0
        tracker = CostTracker(storage_path=tmp_path / "r.jsonl", monthly_limit_usd=limit)
        tracker.record(_make_receipt(cost=0.79))

        # This call pushes projected to 0.79 + 0.05 = 0.84 → soft-warn zone
        tracker.check_budget(0.05)

        soft_warn_events = _events_of_type("cost.budget_soft_warn")
        assert len(soft_warn_events) >= 1
        ev_data = soft_warn_events[0]["data"]
        assert ev_data["monthly_limit_usd"] == limit
        assert ev_data["usage_fraction"] >= 0.80

    def test_soft_warn_evidence_emitted_only_once(self, tmp_path: Path):
        """cost.budget_soft_warn evidence is NOT re-emitted on repeated check_budget calls."""
        limit = 1.0
        tracker = CostTracker(storage_path=tmp_path / "r.jsonl", monthly_limit_usd=limit)
        tracker.record(_make_receipt(cost=0.80))

        tracker.check_budget(0.05)
        tracker.check_budget(0.05)
        tracker.check_budget(0.05)

        soft_warn_events = _events_of_type("cost.budget_soft_warn")
        assert len(soft_warn_events) == 1, (
            f"Expected exactly 1 soft-warn event, got {len(soft_warn_events)}"
        )

    def test_reset_monthly_clears_soft_warn_state(self, tmp_path: Path):
        """reset_monthly() resets the soft-warn flag so it fires again next month."""
        limit = 1.0
        tracker = CostTracker(storage_path=tmp_path / "r.jsonl", monthly_limit_usd=limit)
        tracker.record(_make_receipt(cost=0.80))
        tracker.check_budget(0.05)

        # First soft-warn was emitted
        assert len(_events_of_type("cost.budget_soft_warn")) == 1

        # Reset (new month)
        tracker.reset_monthly()
        assert tracker._soft_warn_emitted is False

        # Now a second soft-warn call fires again
        tracker.check_budget(0.05)
        assert len(_events_of_type("cost.budget_soft_warn")) == 2


# ===========================================================================
# d36 — Multi-hop fallback
# ===========================================================================

class TestD36MultiHopFallback:
    """d36: chaos multi-hop, all-fail exhaustion, payload preservation, evidence."""

    def test_chaos_multihop_a_fails_b_fails_c_succeeds(self):
        """Provider A fails → Provider B fails → Provider C (local) succeeds."""
        # Kill A (API_HAIKU) and B (API_DEEPSEEK)
        for opt in [RoutingOption.API_HAIKU, RoutingOption.API_DEEPSEEK]:
            for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
                router._record_provider_failure(opt)

        policy = _make_policy([
            RoutingOption.API_HAIKU,
            RoutingOption.API_DEEPSEEK,
            RoutingOption.LOCAL_GENERALIST,
        ])
        result = dispatch(task="chaos_test", input_text="hello world", policy=policy)

        # C (LOCAL_GENERALIST) must have served the request
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_chaos_multihop_three_api_providers_all_fail_local_wins(self):
        """Three API providers fail in order; local generalist wins."""
        dead_opts = [
            RoutingOption.API_HAIKU,
            RoutingOption.API_SONNET,
            RoutingOption.API_GPT5_MINI,
        ]
        for opt in dead_opts:
            for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
                router._record_provider_failure(opt)

        policy = _make_policy(
            dead_opts + [RoutingOption.LOCAL_GENERALIST]
        )
        result = dispatch(task="chaos_3hop", input_text="test", policy=policy)
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_all_providers_fail_raises_routing_exhausted(self):
        """When every provider fails, RoutingProviderUnavailable is raised."""
        all_api = [
            RoutingOption.API_HAIKU,
            RoutingOption.API_SONNET,
            RoutingOption.API_DEEPSEEK,
            RoutingOption.API_GPT5_MINI,
        ]
        for opt in all_api:
            for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
                router._record_provider_failure(opt)

        policy = _make_policy(all_api)
        with pytest.raises(RoutingProviderUnavailable):
            dispatch(task="exhausted", input_text="fail all", policy=policy)

    def test_all_providers_fail_emits_route_exhausted_evidence(self):
        """route.exhausted evidence is emitted when all providers are unavailable."""
        all_api = [
            RoutingOption.API_HAIKU,
            RoutingOption.API_SONNET,
        ]
        for opt in all_api:
            for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
                router._record_provider_failure(opt)

        policy = _make_policy(all_api, task="exhausted_task")
        with pytest.raises(RoutingProviderUnavailable):
            dispatch(task="exhausted_task", input_text="x", policy=policy)

        exhausted_events = _events_of_type("route.exhausted")
        assert len(exhausted_events) >= 1
        assert any(
            e["data"]["task"] == "exhausted_task" for e in exhausted_events
        )

    def test_fallback_preserves_original_request_payload(self):
        """Fallback routing preserves the exact input_text in the cost receipt."""
        original_payload = "This is the exact original request payload — must not be truncated."

        # Kill first provider so we fall back
        for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
            router._record_provider_failure(RoutingOption.API_HAIKU)

        policy = _make_policy([
            RoutingOption.API_HAIKU,
            RoutingOption.LOCAL_GENERALIST,
        ])
        result = dispatch(task="payload_test", input_text=original_payload, policy=policy)

        # The receipt task must be preserved
        receipt: CostReceipt = result["cost_receipt"]
        assert receipt.task == "payload_test"
        # The token count is derived from the original (untruncated) payload
        expected_tokens = max(1, len(original_payload.split()))
        assert receipt.tokens_in == expected_tokens

    def test_fallback_hop_evidence_emitted_per_failed_provider(self):
        """route.fallback_hop evidence is emitted for each failed provider hop."""
        failed = [RoutingOption.API_HAIKU, RoutingOption.API_DEEPSEEK]
        for opt in failed:
            for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
                router._record_provider_failure(opt)

        policy = _make_policy(failed + [RoutingOption.LOCAL_GENERALIST])
        dispatch(task="hop_test", input_text="hop hop hop", policy=policy)

        hop_events = _events_of_type("route.fallback_hop")
        failed_values = {opt.value for opt in failed}
        emitted_providers = {e["data"]["failed_provider"] for e in hop_events}
        assert failed_values.issubset(emitted_providers), (
            f"Expected hop events for {failed_values}, got {emitted_providers}"
        )

    def test_fallback_hop_evidence_has_hop_counter(self):
        """route.fallback_hop evidence includes an incrementing hop counter."""
        failed = [
            RoutingOption.API_HAIKU,
            RoutingOption.API_SONNET,
            RoutingOption.API_DEEPSEEK,
        ]
        for opt in failed:
            for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
                router._record_provider_failure(opt)

        policy = _make_policy(failed + [RoutingOption.LOCAL_GENERALIST])
        dispatch(task="hop_counter", input_text="three hops", policy=policy)

        hop_events = sorted(
            _events_of_type("route.fallback_hop"),
            key=lambda e: e["data"]["hop"],
        )
        # Should have at least 3 hop events with hop numbers 1, 2, 3
        hop_numbers = [e["data"]["hop"] for e in hop_events]
        assert 1 in hop_numbers
        assert 2 in hop_numbers
        assert 3 in hop_numbers

    def test_route_exhausted_evidence_lists_all_failed_providers(self):
        """route.exhausted evidence data includes all failed providers."""
        failed = [RoutingOption.API_HAIKU, RoutingOption.API_SONNET]
        for opt in failed:
            for _ in range(router._CIRCUIT_BREAKER_THRESHOLD):
                router._record_provider_failure(opt)

        policy = _make_policy(failed, task="list_failed")
        with pytest.raises(RoutingProviderUnavailable):
            dispatch(task="list_failed", input_text="x", policy=policy)

        exhausted_events = _events_of_type("route.exhausted")
        assert len(exhausted_events) >= 1
        ev_data = exhausted_events[0]["data"]
        for opt in failed:
            assert opt.value in ev_data["failed_providers"]
