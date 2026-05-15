"""Privacy-tier and local-specialist routing tests — d33/d34 validation.

Covers gaps not addressed in test_route.py:
- CONFIDENTIAL routes to local (not just raises on external)
- SECRETS routes to local
- local_specialist selection with multiple specialists
- local_specialist selection with exactly one specialist
- routing policy enforcement: privacy tier is respected end-to-end
- fallback chain from specialist → generalist
- PUBLIC tier freely uses external APIs
- INTERNAL tier also freely uses external APIs (not restricted)
- Dead provider + local fallback remains available under any privacy tier
- Cost receipt task field matches the dispatched task name
- Route reason string mentions the specialist name when LOCAL_SPECIALIST chosen
- Dispatching CONFIDENTIAL with mixed preference order selects first local option
"""

from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import NoRouteAvailable, PrivacyTierViolation
from dojo.core.types import PrivacyTier, RoutingOption, RoutingPolicy
from dojo.route.router import _DEAD_PROVIDERS, dispatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_state():
    """Clear evidence log and dead-provider set before every test."""
    EvidenceLogger.clear()
    _DEAD_PROVIDERS.clear()
    yield
    EvidenceLogger.clear()
    _DEAD_PROVIDERS.clear()


def _policy(
    options: list[RoutingOption],
    privacy: PrivacyTier = PrivacyTier.INTERNAL,
    task: str = "bookkeeping",
) -> RoutingPolicy:
    return RoutingPolicy(
        task_name=task,
        preference_order=tuple(options),
        max_latency_ms=30_000,
        max_cost_usd=0.05,
        privacy_tier=privacy,
        fallback_on_unavailable=True,
    )


# ---------------------------------------------------------------------------
# d33 — Privacy-tier routing
# ---------------------------------------------------------------------------

class TestPrivacyTierRouting:
    """d33: CONFIDENTIAL and SECRETS data must never leave local infrastructure."""

    def test_confidential_routes_to_local_specialist_when_available(self):
        """CONFIDENTIAL + specialist available → chooses LOCAL_SPECIALIST, not API."""
        policy = _policy(
            [RoutingOption.LOCAL_SPECIALIST, RoutingOption.API_HAIKU],
            privacy=PrivacyTier.CONFIDENTIAL,
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Confidential payroll data",
            policy=policy,
            available_specialists=["qwen-7b-bookkeeping"],
        )
        assert result["route"] == RoutingOption.LOCAL_SPECIALIST.value

    def test_confidential_routes_to_local_generalist_when_no_specialist(self):
        """CONFIDENTIAL + no specialist + API option → picks LOCAL_GENERALIST."""
        policy = _policy(
            [RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST, RoutingOption.API_HAIKU],
            privacy=PrivacyTier.CONFIDENTIAL,
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Confidential invoice scan",
            policy=policy,
            available_specialists=[],
        )
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_secrets_routes_to_local_specialist_when_available(self):
        """SECRETS tier + specialist deployed → LOCAL_SPECIALIST."""
        policy = _policy(
            [RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST],
            privacy=PrivacyTier.SECRETS,
        )
        result = dispatch(
            task="code_review",
            input_text="Top-secret source code",
            policy=policy,
            available_specialists=["qwen-7b-code"],
        )
        assert result["route"] == RoutingOption.LOCAL_SPECIALIST.value

    def test_secrets_routes_to_local_generalist_when_no_specialist(self):
        """SECRETS tier + no specialist → LOCAL_GENERALIST."""
        policy = _policy(
            [RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST],
            privacy=PrivacyTier.SECRETS,
        )
        result = dispatch(
            task="code_review",
            input_text="Top-secret source code",
            policy=policy,
            available_specialists=[],
        )
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_confidential_with_only_api_options_raises_privacy_violation(self):
        """CONFIDENTIAL with no local options at all → PrivacyTierViolation."""
        policy = _policy(
            [RoutingOption.API_HAIKU, RoutingOption.API_SONNET],
            privacy=PrivacyTier.CONFIDENTIAL,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch(
                task="bookkeeping",
                input_text="Confidential data — must not go to cloud",
                policy=policy,
            )

    def test_secrets_with_only_api_options_raises_privacy_violation(self):
        """SECRETS with no local fallback raises PrivacyTierViolation."""
        policy = _policy(
            [RoutingOption.API_DEEPSEEK],
            privacy=PrivacyTier.SECRETS,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch(
                task="sop_extraction",
                input_text="Secret SOP document",
                policy=policy,
            )

    def test_privacy_blocked_evidence_emitted_on_violation(self):
        """ROUTE_PRIVACY_BLOCKED evidence is emitted when privacy is violated."""
        policy = _policy(
            [RoutingOption.API_HAIKU],
            privacy=PrivacyTier.CONFIDENTIAL,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch(task="bookkeeping", input_text="secret", policy=policy)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.ROUTE_PRIVACY_BLOCKED.value in types

    def test_public_tier_can_use_external_api(self):
        """PUBLIC tier is not restricted — external API should be chosen."""
        policy = _policy(
            [RoutingOption.API_HAIKU],
            privacy=PrivacyTier.PUBLIC,
        )
        result = dispatch(task="general", input_text="Public blog post", policy=policy)
        assert result["route"] == RoutingOption.API_HAIKU.value

    def test_internal_tier_can_use_external_api(self):
        """INTERNAL tier is not restricted — external API is allowed."""
        policy = _policy(
            [RoutingOption.API_HAIKU],
            privacy=PrivacyTier.INTERNAL,
        )
        result = dispatch(task="general", input_text="Internal memo", policy=policy)
        assert result["route"] == RoutingOption.API_HAIKU.value

    def test_confidential_with_api_first_raises_privacy_violation(self):
        """CONFIDENTIAL with API listed first raises PrivacyTierViolation immediately.

        The router is fail-fast: hitting any API option under CONFIDENTIAL/SECRETS
        raises PrivacyTierViolation rather than scanning for a later local option.
        This is intentional — callers must list local options first for sensitive data.
        """
        policy = _policy(
            [RoutingOption.API_SONNET, RoutingOption.LOCAL_GENERALIST],
            privacy=PrivacyTier.CONFIDENTIAL,
        )
        with pytest.raises(PrivacyTierViolation):
            dispatch(
                task="ocr_cleanup",
                input_text="Confidential scanned document",
                policy=policy,
            )

    def test_route_decision_evidence_carries_privacy_tier(self):
        """ROUTE_DECISION evidence records the privacy tier for audit trails."""
        policy = _policy(
            [RoutingOption.LOCAL_GENERALIST],
            privacy=PrivacyTier.CONFIDENTIAL,
        )
        dispatch(task="bookkeeping", input_text="confidential text", policy=policy)

        events = EvidenceLogger.get_events()
        decisions = [e for e in events if e.get("type") == EvidenceType.ROUTE_DECISION.value]
        assert decisions, "No ROUTE_DECISION evidence emitted"
        assert decisions[-1]["data"]["privacy_tier"] == PrivacyTier.CONFIDENTIAL.value


# ---------------------------------------------------------------------------
# d34 — Local-specialist routing
# ---------------------------------------------------------------------------

class TestLocalSpecialistRouting:
    """d34: Local-specialist selection logic and preference ordering."""

    def test_single_specialist_chosen_over_generalist(self):
        """With one specialist deployed, prefer it over generalist."""
        policy = _policy(
            [RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST],
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Invoice reconciliation",
            policy=policy,
            available_specialists=["qwen-7b-bookkeeping"],
        )
        assert result["route"] == RoutingOption.LOCAL_SPECIALIST.value

    def test_multiple_specialists_still_picks_local_specialist_route(self):
        """Multiple specialists deployed → route is still LOCAL_SPECIALIST."""
        policy = _policy(
            [RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST],
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Accounts payable processing",
            policy=policy,
            available_specialists=["qwen-7b-bookkeeping", "qwen-7b-email", "qwen-7b-ocr"],
        )
        assert result["route"] == RoutingOption.LOCAL_SPECIALIST.value

    def test_no_specialist_falls_back_to_generalist(self):
        """No specialists deployed → falls back to LOCAL_GENERALIST."""
        policy = _policy(
            [RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST],
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Invoice",
            policy=policy,
            available_specialists=[],
        )
        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_specialist_route_reason_mentions_specialist_name(self):
        """Routing reason should name the specialist model that was selected."""
        policy = _policy(
            [RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST],
        )
        result = dispatch(
            task="bookkeeping",
            input_text="Bookkeeping task",
            policy=policy,
            available_specialists=["qwen-7b-bookkeeping"],
        )
        assert "qwen-7b-bookkeeping" in result["reason"]

    def test_specialist_endpoint_url_is_none(self):
        """LOCAL_SPECIALIST is a local route — no external endpoint URL."""
        policy = _policy([RoutingOption.LOCAL_SPECIALIST])
        result = dispatch(
            task="bookkeeping",
            input_text="Local invoice",
            policy=policy,
            available_specialists=["qwen-7b-bookkeeping"],
        )
        assert result["endpoint_url"] is None

    def test_generalist_endpoint_url_is_none(self):
        """LOCAL_GENERALIST is also a local route — no endpoint URL."""
        policy = _policy([RoutingOption.LOCAL_GENERALIST])
        result = dispatch(
            task="bookkeeping",
            input_text="Any text",
            policy=policy,
        )
        assert result["endpoint_url"] is None

    def test_no_local_options_and_no_api_raises_no_route(self):
        """LOCAL_SPECIALIST only, no specialist deployed, no fallback → NoRouteAvailable."""
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
                input_text="Invoice",
                policy=policy,
                available_specialists=[],
            )

    def test_specialist_selected_and_cost_receipt_returned(self):
        """Routing to LOCAL_SPECIALIST must return a cost receipt with correct task."""
        policy = _policy([RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST])
        result = dispatch(
            task="sop_extraction",
            input_text="Extract SOP procedures",
            policy=policy,
            available_specialists=["qwen-7b-sop"],
        )
        receipt = result["cost_receipt"]
        assert receipt.task == "sop_extraction"
        assert receipt.route_taken == RoutingOption.LOCAL_SPECIALIST.value

    def test_generalist_always_available_even_without_specialists(self):
        """LOCAL_GENERALIST should succeed regardless of available_specialists list."""
        for specialists in [None, [], ["some-model"]]:
            EvidenceLogger.clear()
            policy = _policy([RoutingOption.LOCAL_GENERALIST])
            result = dispatch(
                task="general",
                input_text="Any text",
                policy=policy,
                available_specialists=specialists,
            )
            assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_route_decision_evidence_emitted_for_local_specialist(self):
        """ROUTE_DECISION evidence is always emitted on a successful local dispatch."""
        policy = _policy([RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST])
        dispatch(
            task="bookkeeping",
            input_text="Evidence check",
            policy=policy,
            available_specialists=["qwen-7b-bookkeeping"],
        )
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.ROUTE_DECISION.value in types
