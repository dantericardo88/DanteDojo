"""Unit tests for dojo/route/ — Phase 3C inference routing layer.

Covers:
- dispatch with LOCAL_SPECIALIST available → returns route=local_specialist
- dispatch with CONFIDENTIAL + API_HAIKU preference → raises PrivacyTierViolation
- dispatch with SECRETS + API_SONNET preference → raises PrivacyTierViolation
- dispatch with no available routes → raises NoRouteAvailable
- dispatch emits ROUTE_PRIVACY_BLOCKED evidence on privacy violation
- CostTracker.record persists receipt to JSONL (tmp_path)
- CostTracker.monthly_summary returns correct totals
- default_policy_for_task("bookkeeping") → first preference is LOCAL_SPECIALIST
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import NoRouteAvailable, PrivacyTierViolation
from dojo.core.types import (
    CostReceipt,
    PrivacyTier,
    RoutingOption,
    RoutingPolicy,
)
from dojo.route.cost_tracker import CostTracker
from dojo.route.policy import TASK_DEFAULTS, default_policy_for_task
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
    task: str = "bookkeeping",
    options: list[RoutingOption] = None,
    privacy_tier: PrivacyTier = PrivacyTier.INTERNAL,
    fallback: bool = True,
) -> RoutingPolicy:
    if options is None:
        options = [RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST]
    return RoutingPolicy(
        task_name=task,
        preference_order=tuple(options),
        max_latency_ms=30_000,
        max_cost_usd=0.01,
        privacy_tier=privacy_tier,
        fallback_on_unavailable=fallback,
    )


def _make_receipt(
    task: str = "bookkeeping",
    route: str = "local_specialist",
    cost: float = 0.001,
    month: int = 3,
    year: int = 2026,
) -> CostReceipt:
    return CostReceipt(
        task=task,
        route_taken=route,
        tokens_in=100,
        tokens_out=50,
        cost_usd=cost,
        latency_ms=100.0,
        local_displaced_estimate_usd=0.0025,
        recorded_at=datetime(year, month, 15, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# dispatch tests
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_local_specialist_available_returns_local(self):
        policy = _make_policy(
            options=[RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST]
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Process this invoice",
            policy=policy,
            available_specialists=["qwen-7b-bookkeeping"],
        )
        assert result["route"] == RoutingOption.LOCAL_SPECIALIST.value

    def test_local_specialist_unavailable_falls_back(self):
        """No specialist deployed → fall back to local generalist."""
        policy = _make_policy(
            options=[RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST]
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Process this invoice",
            policy=policy,
            available_specialists=[],
        )
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_confidential_plus_api_haiku_raises_privacy_violation(self):
        policy = _make_policy(
            options=[RoutingOption.API_HAIKU],
            privacy_tier=PrivacyTier.CONFIDENTIAL,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch(
                task="bookkeeping",
                input_text="Confidential financial data",
                policy=policy,
            )

    def test_secrets_plus_api_sonnet_raises_privacy_violation(self):
        policy = _make_policy(
            options=[RoutingOption.API_SONNET],
            privacy_tier=PrivacyTier.SECRETS,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch(
                task="code_review",
                input_text="Secret source code",
                policy=policy,
            )

    def test_confidential_api_haiku_emits_privacy_blocked_evidence(self):
        policy = _make_policy(
            options=[RoutingOption.API_HAIKU],
            privacy_tier=PrivacyTier.CONFIDENTIAL,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch(
                task="bookkeeping",
                input_text="Confidential data",
                policy=policy,
            )

        events = EvidenceLogger._events
        evidence_types = [
            e["evidence"].get("type")
            for e in events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "route.privacy.blocked" in evidence_types

    def test_no_routes_available_raises_no_route(self):
        """All options are unavailable → NoRouteAvailable."""
        # LOCAL_SPECIALIST with no specialists deployed, no fallback
        policy = RoutingPolicy(
            task_name="bookkeeping",
            preference_order=(RoutingOption.LOCAL_SPECIALIST,),
            max_latency_ms=30_000,
            max_cost_usd=0.01,
            privacy_tier=PrivacyTier.INTERNAL,
            fallback_on_unavailable=False,
        )
        with pytest.raises(NoRouteAvailable):
            dispatch(
                task="bookkeeping",
                input_text="Invoice data",
                policy=policy,
                available_specialists=[],
            )

    def test_dispatch_returns_cost_receipt(self):
        policy = _make_policy(
            options=[RoutingOption.LOCAL_GENERALIST]
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Some text",
            policy=policy,
        )
        assert "cost_receipt" in result
        assert isinstance(result["cost_receipt"], CostReceipt)

    def test_local_route_endpoint_is_none(self):
        policy = _make_policy(
            options=[RoutingOption.LOCAL_GENERALIST]
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Some text",
            policy=policy,
        )
        assert result["endpoint_url"] is None

    def test_api_route_endpoint_is_not_none(self):
        policy = _make_policy(
            options=[RoutingOption.API_HAIKU],
            privacy_tier=PrivacyTier.PUBLIC,
        )
        result = dispatch(
            task="general",
            input_text="Some text",
            policy=policy,
        )
        assert result["endpoint_url"] is not None
        assert "http" in result["endpoint_url"]

    def test_dispatch_records_cost_when_tracker_given(self, tmp_path):
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        policy = _make_policy(
            options=[RoutingOption.LOCAL_GENERALIST]
        )
        dispatch(
            task="bookkeeping",
            input_text="Some text",
            policy=policy,
            cost_tracker=tracker,
        )
        receipts = tracker.load_receipts()
        assert len(receipts) == 1

    def test_secrets_with_local_option_succeeds(self):
        """SECRETS tier with LOCAL_GENERALIST should work fine."""
        policy = _make_policy(
            options=[RoutingOption.LOCAL_GENERALIST],
            privacy_tier=PrivacyTier.SECRETS,
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Top secret data",
            policy=policy,
        )
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value


# ---------------------------------------------------------------------------
# CostTracker tests
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_record_persists_to_jsonl(self, tmp_path):
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        receipt = _make_receipt(cost=0.005)
        tracker.record(receipt)

        storage_file = tmp_path / "receipts.jsonl"
        assert storage_file.exists()
        lines = storage_file.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_record_multiple_appends(self, tmp_path):
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        for i in range(3):
            tracker.record(_make_receipt(cost=float(i) * 0.001))

        receipts = tracker.load_receipts()
        assert len(receipts) == 3

    def test_record_emits_cost_receipt_evidence(self, tmp_path):
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        tracker.record(_make_receipt())

        events = EvidenceLogger._events
        evidence_types = [
            e["evidence"].get("type")
            for e in events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "route.cost_receipt" in evidence_types

    def test_monthly_summary_correct_totals(self, tmp_path):
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")

        # March 2026: 2 receipts
        tracker.record(_make_receipt(cost=0.010, month=3, year=2026))
        tracker.record(_make_receipt(cost=0.005, month=3, year=2026))
        # April 2026: 1 receipt (different month — should NOT be counted)
        tracker.record(_make_receipt(cost=0.020, month=4, year=2026))

        summary = tracker.monthly_summary(2026, 3)
        assert summary["receipts"] == 2
        assert abs(summary["total_cost_usd"] - 0.015) < 1e-6

    def test_monthly_summary_empty_month(self, tmp_path):
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        summary = tracker.monthly_summary(2026, 1)
        assert summary["receipts"] == 0
        assert summary["total_cost_usd"] == 0.0

    def test_load_receipts_empty_when_no_file(self, tmp_path):
        tracker = CostTracker(storage_path=tmp_path / "nonexistent.jsonl")
        assert tracker.load_receipts() == []

    def test_monthly_summary_local_displaced(self, tmp_path):
        """local_displaced_usd should sum local_displaced_estimate_usd."""
        tracker = CostTracker(storage_path=tmp_path / "receipts.jsonl")
        r = CostReceipt(
            task="bookkeeping",
            route_taken="local_specialist",
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.0,
            latency_ms=50.0,
            local_displaced_estimate_usd=0.0025,
            recorded_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        )
        tracker.record(r)
        summary = tracker.monthly_summary(2026, 3)
        assert abs(summary["local_displaced_usd"] - 0.0025) < 1e-7


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------

class TestPolicy:
    def test_default_policy_bookkeeping_first_is_local_specialist(self):
        policy = default_policy_for_task("bookkeeping")
        assert policy.preference_order[0] == RoutingOption.LOCAL_SPECIALIST

    def test_default_policy_unknown_task_falls_back_to_general(self):
        policy = default_policy_for_task("totally_unknown_task")
        assert len(policy.preference_order) > 0

    def test_default_policy_privacy_tier_passed_through(self):
        policy = default_policy_for_task("bookkeeping", PrivacyTier.CONFIDENTIAL)
        assert policy.privacy_tier == PrivacyTier.CONFIDENTIAL

    def test_task_defaults_contains_expected_tasks(self):
        for task in ("bookkeeping", "b2b_email", "code_review", "ocr_cleanup", "sop_extraction", "general"):
            assert task in TASK_DEFAULTS

    def test_bookkeeping_options_are_all_local(self):
        options = TASK_DEFAULTS["bookkeeping"]
        for opt in options:
            assert opt in (RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST)

    def test_general_task_uses_api_haiku(self):
        assert RoutingOption.API_HAIKU in TASK_DEFAULTS["general"]

    def test_b2b_email_has_specialist_preference(self):
        assert TASK_DEFAULTS["b2b_email"][0] == RoutingOption.LOCAL_SPECIALIST
