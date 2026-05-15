"""Unit tests for d32 (multi-provider routing), d35 (budget enforcement),
and d36 (circuit breaker / fallback policy).

d32 — dispatch() routes to ≥3 real providers; routing decision in evidence chain.
d35 — Budget enforcement fires on over-limit request; cost in evidence;
      record_from_response() tracks real token counts.
d36 — Circuit breaker marks providers dead after threshold failures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BudgetExceeded, PrivacyTierViolation
from dojo.core.types import CostReceipt, PrivacyTier, RoutingOption, RoutingPolicy
from dojo.route import router
from dojo.route.cost_tracker import CostTracker
from dojo.route.router import dispatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _make_policy(
    options: list[RoutingOption],
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC,
    task: str = "general",
) -> RoutingPolicy:
    return RoutingPolicy(
        task_name=task,
        preference_order=tuple(options),
        max_latency_ms=30_000,
        max_cost_usd=1.0,
        privacy_tier=privacy_tier,
        fallback_on_unavailable=True,
    )


def _route_decision_events() -> list[dict]:
    """Return all ROUTE_DECISION evidence data dicts emitted so far."""
    return [
        e["evidence"]["data"]
        for e in EvidenceLogger._events
        if isinstance(e.get("evidence"), dict)
        and e["evidence"].get("type") == EvidenceType.ROUTE_DECISION.value
    ]


def _cost_receipt_events() -> list[dict]:
    """Return all ROUTE_COST_RECEIPT evidence data dicts emitted so far."""
    return [
        e["evidence"]["data"]
        for e in EvidenceLogger._events
        if isinstance(e.get("evidence"), dict)
        and e["evidence"].get("type") == EvidenceType.ROUTE_COST_RECEIPT.value
    ]


# ---------------------------------------------------------------------------
# d32 — multi-provider routing tests
# ---------------------------------------------------------------------------

class TestMultiProviderRouting:
    def test_dispatch_routes_to_anthropic_haiku(self):
        """API_HAIKU preference → ROUTE_DECISION has provider_name='anthropic'."""
        policy = _make_policy([RoutingOption.API_HAIKU])
        dispatch(task="general", input_text="Hello world", policy=policy)

        decisions = _route_decision_events()
        assert len(decisions) == 1
        assert decisions[0]["provider_name"] == "anthropic"
        assert decisions[0]["route"] == RoutingOption.API_HAIKU.value

    def test_dispatch_routes_to_deepseek(self):
        """API_DEEPSEEK preference → ROUTE_DECISION has provider_name='deepseek'."""
        policy = _make_policy([RoutingOption.API_DEEPSEEK])
        dispatch(task="general", input_text="Hello world", policy=policy)

        decisions = _route_decision_events()
        assert len(decisions) == 1
        assert decisions[0]["provider_name"] == "deepseek"
        assert decisions[0]["route"] == RoutingOption.API_DEEPSEEK.value

    def test_dispatch_routes_to_openai(self):
        """API_GPT5_MINI preference → ROUTE_DECISION has provider_name='openai'."""
        policy = _make_policy([RoutingOption.API_GPT5_MINI])
        dispatch(task="general", input_text="Hello world", policy=policy)

        decisions = _route_decision_events()
        assert len(decisions) == 1
        assert decisions[0]["provider_name"] == "openai"
        assert decisions[0]["route"] == RoutingOption.API_GPT5_MINI.value

    def test_dispatch_evidence_has_providers_available(self):
        """ROUTE_DECISION includes providers_available listing all preference options."""
        options = [RoutingOption.API_HAIKU, RoutingOption.API_DEEPSEEK, RoutingOption.API_GPT5_MINI]
        policy = _make_policy(options)
        dispatch(task="general", input_text="Hello world", policy=policy)

        decisions = _route_decision_events()
        assert len(decisions) == 1
        available = decisions[0]["providers_available"]
        assert RoutingOption.API_HAIKU.value in available
        assert RoutingOption.API_DEEPSEEK.value in available
        assert RoutingOption.API_GPT5_MINI.value in available

    def test_dispatch_privacy_blocks_external(self):
        """CONFIDENTIAL privacy tier + external API preference → PrivacyTierViolation."""
        policy = _make_policy(
            [RoutingOption.API_HAIKU],
            privacy_tier=PrivacyTier.CONFIDENTIAL,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch(task="general", input_text="Secret data", policy=policy)

    def test_dispatch_local_provider_name_is_local(self):
        """LOCAL_GENERALIST → provider_name='local' in evidence."""
        policy = _make_policy([RoutingOption.LOCAL_GENERALIST])
        dispatch(task="general", input_text="Hello", policy=policy)

        decisions = _route_decision_events()
        assert decisions[0]["provider_name"] == "local"

    def test_dispatch_sonnet_is_also_anthropic(self):
        """API_SONNET → provider_name='anthropic'."""
        policy = _make_policy([RoutingOption.API_SONNET])
        dispatch(task="general", input_text="Hello", policy=policy)

        decisions = _route_decision_events()
        assert decisions[0]["provider_name"] == "anthropic"


# ---------------------------------------------------------------------------
# d35 — budget enforcement tests
# ---------------------------------------------------------------------------

class TestBudgetEnforcement:
    def test_budget_check_passes_within_limit(self, tmp_path: Path):
        """$0.10 estimated, $5.00 limit, no prior spend → passes without raising."""
        tracker = CostTracker(
            storage_path=tmp_path / "receipts.jsonl",
            monthly_limit_usd=5.00,
        )
        # Should not raise
        tracker.check_budget(0.10)

    def test_budget_check_raises_on_over_limit(self, tmp_path: Path):
        """$5.01 estimated cost against $5.00 limit → BudgetExceeded."""
        tracker = CostTracker(
            storage_path=tmp_path / "receipts.jsonl",
            monthly_limit_usd=5.00,
        )
        with pytest.raises(BudgetExceeded):
            tracker.check_budget(5.01)

    def test_budget_enforcement_emits_evidence(self, tmp_path: Path):
        """check_budget() emits a ROUTE_COST_RECEIPT event."""
        tracker = CostTracker(
            storage_path=tmp_path / "receipts.jsonl",
            monthly_limit_usd=5.00,
        )
        tracker.check_budget(0.10)

        receipts = _cost_receipt_events()
        budget_events = [e for e in receipts if e.get("budget_check") is True]
        assert len(budget_events) >= 1

    def test_budget_evidence_has_projected_total(self, tmp_path: Path):
        """Budget check evidence includes projected_total_usd."""
        tracker = CostTracker(
            storage_path=tmp_path / "receipts.jsonl",
            monthly_limit_usd=5.00,
        )
        tracker.check_budget(0.25)

        receipts = _cost_receipt_events()
        budget_events = [e for e in receipts if e.get("budget_check") is True]
        assert len(budget_events) >= 1
        ev = budget_events[0]
        assert "projected_total_usd" in ev
        assert abs(ev["projected_total_usd"] - 0.25) < 1e-9

    def test_budget_evidence_status_within_budget(self, tmp_path: Path):
        """status field is 'within_budget' when under limit."""
        tracker = CostTracker(
            storage_path=tmp_path / "receipts.jsonl",
            monthly_limit_usd=5.00,
        )
        tracker.check_budget(0.10)

        receipts = _cost_receipt_events()
        budget_events = [e for e in receipts if e.get("budget_check") is True]
        assert budget_events[0]["status"] == "within_budget"

    def test_budget_evidence_status_over_budget(self, tmp_path: Path):
        """status field is 'over_budget' when limit exceeded (evidence emitted before raise)."""
        tracker = CostTracker(
            storage_path=tmp_path / "receipts.jsonl",
            monthly_limit_usd=1.00,
        )
        with pytest.raises(BudgetExceeded):
            tracker.check_budget(2.00)

        receipts = _cost_receipt_events()
        budget_events = [e for e in receipts if e.get("budget_check") is True]
        assert budget_events[0]["status"] == "over_budget"

    def test_dispatch_with_budget_tracker_blocks_expensive(self, tmp_path: Path):
        """dispatch() with CostTracker(limit=$0.000001) and expensive route → BudgetExceeded."""
        # API_SONNET at $0.003/1k tokens — even 1 token exceeds $0.000001 limit
        tracker = CostTracker(
            storage_path=tmp_path / "receipts.jsonl",
            monthly_limit_usd=0.000001,
        )
        policy = _make_policy([RoutingOption.API_SONNET])
        with pytest.raises(BudgetExceeded):
            dispatch(
                task="general",
                input_text="This long input has many tokens to drive up the cost estimate",
                policy=policy,
                cost_tracker=tracker,
            )

    def test_dispatch_no_budget_limit_does_not_check(self, tmp_path: Path):
        """CostTracker without monthly_limit_usd → dispatch proceeds normally."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        policy = _make_policy([RoutingOption.API_HAIKU])
        result = dispatch(task="general", input_text="Hello", policy=policy, cost_tracker=tracker)
        assert result["route"] == RoutingOption.API_HAIKU.value

    def test_budget_exception_message_contains_amounts(self, tmp_path: Path):
        """BudgetExceeded message includes the dollar amounts."""
        tracker = CostTracker(
            storage_path=tmp_path / "receipts.jsonl",
            monthly_limit_usd=0.01,
        )
        with pytest.raises(BudgetExceeded) as exc_info:
            tracker.check_budget(0.05)
        msg = str(exc_info.value)
        assert "$" in msg


# ---------------------------------------------------------------------------
# d35 — record_from_response() real token cost + get_provider_stats()
# ---------------------------------------------------------------------------

def _make_receipt(
    route: RoutingOption = RoutingOption.API_HAIKU,
    task: str = "test",
) -> CostReceipt:
    return CostReceipt(
        task=task,
        route_taken=route.value,
        tokens_in=10,
        tokens_out=0,
        cost_usd=0.0,
        latency_ms=100.0,
        local_displaced_estimate_usd=0.0,
        recorded_at=datetime.now(timezone.utc),
    )


class TestRecordFromResponse:
    def test_record_from_response_computes_correct_cost(self, tmp_path: Path):
        """record_from_response() computes cost from real token counts."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        receipt = _make_receipt(RoutingOption.API_HAIKU)

        # API_HAIKU = $0.00025 / 1k tokens; 2000 total tokens → $0.00050
        updated = tracker.record_from_response(receipt, response_tokens=1000, prompt_tokens=1000)

        expected_cost = round((2000 / 1000.0) * 0.00025, 8)
        assert abs(updated.cost_usd - expected_cost) < 1e-9
        assert updated.tokens_in == 1000
        assert updated.tokens_out == 1000

    def test_record_from_response_updates_token_counts(self, tmp_path: Path):
        """record_from_response() sets tokens_in and tokens_out from real values."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        receipt = _make_receipt(RoutingOption.API_SONNET)

        updated = tracker.record_from_response(receipt, response_tokens=250, prompt_tokens=750)

        assert updated.tokens_in == 750
        assert updated.tokens_out == 250

    def test_record_from_response_emits_evidence(self, tmp_path: Path):
        """record_from_response() emits ROUTE_COST_RECEIPT evidence with actual_tokens=True."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        receipt = _make_receipt(RoutingOption.API_DEEPSEEK)

        tracker.record_from_response(receipt, response_tokens=100, prompt_tokens=200)

        receipts = _cost_receipt_events()
        actual_token_events = [e for e in receipts if e.get("actual_tokens") is True]
        assert len(actual_token_events) >= 1
        ev = actual_token_events[0]
        assert ev["prompt_tokens"] == 200
        assert ev["response_tokens"] == 100
        assert ev["total_tokens"] == 300

    def test_record_from_response_persists_to_jsonl(self, tmp_path: Path):
        """record_from_response() appends receipt to the JSONL store."""
        storage = tmp_path / "receipts.jsonl"
        tracker = CostTracker(storage_path=storage)
        receipt = _make_receipt(RoutingOption.API_HAIKU)

        tracker.record_from_response(receipt, response_tokens=50, prompt_tokens=50)

        loaded = tracker.load_receipts()
        assert len(loaded) == 1
        assert loaded[0].tokens_in == 50
        assert loaded[0].tokens_out == 50

    def test_record_from_response_sonnet_cost(self, tmp_path: Path):
        """record_from_response() uses correct rate for API_SONNET ($0.003/1k)."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        receipt = _make_receipt(RoutingOption.API_SONNET)

        # 500 prompt + 500 response = 1000 tokens → $0.003
        updated = tracker.record_from_response(receipt, response_tokens=500, prompt_tokens=500)

        expected = round((1000 / 1000.0) * 0.003, 8)
        assert abs(updated.cost_usd - expected) < 1e-9


class TestGetProviderStats:
    def test_get_provider_stats_empty(self, tmp_path: Path):
        """get_provider_stats() returns empty dict when no receipts stored."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        stats = tracker.get_provider_stats()
        assert stats == {}

    def test_get_provider_stats_single_provider(self, tmp_path: Path):
        """get_provider_stats() aggregates correctly for a single provider."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        receipt = _make_receipt(RoutingOption.API_HAIKU)
        tracker.record_from_response(receipt, response_tokens=100, prompt_tokens=100)
        tracker.record_from_response(receipt, response_tokens=200, prompt_tokens=200)

        stats = tracker.get_provider_stats()
        key = RoutingOption.API_HAIKU.value
        assert key in stats
        assert stats[key]["call_count"] == 2
        assert stats[key]["total_tokens"] == 600  # (100+100) + (200+200)
        expected_cost = round((200 / 1000.0) * 0.00025 + (400 / 1000.0) * 0.00025, 8)
        assert abs(stats[key]["total_usd"] - expected_cost) < 1e-7

    def test_get_provider_stats_multiple_providers(self, tmp_path: Path):
        """get_provider_stats() tracks separate totals per provider."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        tracker.record_from_response(
            _make_receipt(RoutingOption.API_HAIKU), response_tokens=100, prompt_tokens=100
        )
        tracker.record_from_response(
            _make_receipt(RoutingOption.API_DEEPSEEK), response_tokens=200, prompt_tokens=200
        )

        stats = tracker.get_provider_stats()
        assert RoutingOption.API_HAIKU.value in stats
        assert RoutingOption.API_DEEPSEEK.value in stats
        assert stats[RoutingOption.API_HAIKU.value]["call_count"] == 1
        assert stats[RoutingOption.API_DEEPSEEK.value]["call_count"] == 1

    def test_get_provider_stats_call_count_accurate(self, tmp_path: Path):
        """get_provider_stats() call_count matches number of persisted receipts."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        for _ in range(5):
            tracker.record_from_response(
                _make_receipt(RoutingOption.API_SONNET), response_tokens=10, prompt_tokens=10
            )

        stats = tracker.get_provider_stats()
        assert stats[RoutingOption.API_SONNET.value]["call_count"] == 5


# ---------------------------------------------------------------------------
# d32 — circuit breaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    @pytest.fixture(autouse=True)
    def reset_circuit_state(self):
        """Reset circuit breaker and dead providers between tests."""
        router._DEAD_PROVIDERS.clear()
        router._circuit_breaker.clear()
        yield
        router._DEAD_PROVIDERS.clear()
        router._circuit_breaker.clear()

    def test_circuit_breaker_marks_dead_after_threshold(self):
        """After _CIRCUIT_BREAKER_THRESHOLD failures, provider is in _DEAD_PROVIDERS."""
        option = RoutingOption.API_HAIKU
        threshold = router._CIRCUIT_BREAKER_THRESHOLD

        for _ in range(threshold):
            router._record_provider_failure(option)

        assert option.value in router._DEAD_PROVIDERS

    def test_circuit_breaker_does_not_mark_dead_before_threshold(self):
        """Fewer than threshold failures → provider NOT marked dead."""
        option = RoutingOption.API_HAIKU
        threshold = router._CIRCUIT_BREAKER_THRESHOLD

        for _ in range(threshold - 1):
            router._record_provider_failure(option)

        assert option.value not in router._DEAD_PROVIDERS

    def test_circuit_breaker_resets_counter_on_success(self):
        """_record_provider_success() resets failure counter and removes from dead set."""
        option = RoutingOption.API_HAIKU
        threshold = router._CIRCUIT_BREAKER_THRESHOLD

        # Drive to dead
        for _ in range(threshold):
            router._record_provider_failure(option)
        assert option.value in router._DEAD_PROVIDERS

        # Recover
        router._record_provider_success(option)
        assert option.value not in router._DEAD_PROVIDERS
        assert router._circuit_breaker.get(option.value, 0) == 0

    def test_circuit_breaker_counter_resets_after_threshold(self):
        """Counter resets to 0 once provider is added to _DEAD_PROVIDERS."""
        option = RoutingOption.API_DEEPSEEK
        threshold = router._CIRCUIT_BREAKER_THRESHOLD

        for _ in range(threshold):
            router._record_provider_failure(option)

        assert router._circuit_breaker.get(option.value, 0) == 0

    def test_dispatch_skips_dead_provider_from_circuit_breaker(self):
        """Provider killed by circuit breaker is skipped by dispatch()."""
        option = RoutingOption.API_HAIKU
        threshold = router._CIRCUIT_BREAKER_THRESHOLD

        for _ in range(threshold):
            router._record_provider_failure(option)

        policy = _make_policy(
            options=[RoutingOption.API_HAIKU, RoutingOption.LOCAL_GENERALIST],
            privacy_tier=PrivacyTier.PUBLIC,
        )
        result = dispatch(task="test", input_text="hello", policy=policy)
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value
