"""Unit tests for d32 (multi-provider routing) and d35 (budget enforcement).

d32 — dispatch() routes to ≥3 real providers; routing decision in evidence chain.
d35 — Budget enforcement fires on over-limit request; cost in evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BudgetExceeded, PrivacyTierViolation
from dojo.core.types import PrivacyTier, RoutingOption, RoutingPolicy
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
