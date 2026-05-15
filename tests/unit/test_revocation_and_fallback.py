"""Unit tests for d08 (sovereign revocation live check) and d36 (dead provider fallback).

d08 — check_revocation() in dojo/promote/revocation.py
     — revocation gate wired into promote() in dojo/promote/gate.py
d36 — dead provider fallback in dojo/route/router.py
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PassportRevoked, PromotionBlocked, RoutingProviderUnavailable
from dojo.core.types import ConstitutionalPassport, ModelArtifact, PrivacyTier, RoutingOption, RoutingPolicy
from dojo.promote.gate import GateState, promote
from dojo.promote.revocation import _cache_clear, check_revocation
from dojo.route import router
from dojo.route.router import dispatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence_and_dead_providers():
    """Reset evidence logger, dead providers set, and revocation cache between tests."""
    EvidenceLogger.clear()
    router._DEAD_PROVIDERS.clear()
    _cache_clear()
    yield
    EvidenceLogger.clear()
    router._DEAD_PROVIDERS.clear()
    _cache_clear()


def _make_passport(
    passport_id: str = "passport-test-001",
    revocation_url: str = "http://127.0.0.1:8002/v1/passport/revoke",
) -> ConstitutionalPassport:
    now = datetime.now(timezone.utc)
    return ConstitutionalPassport(
        passport_id=passport_id,
        model_artifact_id="dojo://test-model/v1",
        harvest_receipt_id="receipt-001",
        harvest_check_receipt=b"harvest-sig",
        dojo_check_receipt=b"dojo-sig",
        forge_check_receipt=b"forge-sig",
        merkle_root="deadbeef" * 8,
        issued_at=now,
        expires_at=now + timedelta(days=90),
        revocation_url=revocation_url,
    )


def _make_http_response(body: dict[str, Any], status: int = 200):
    """Build a mock urllib response context manager."""
    encoded = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = encoded
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_policy(
    options: list[RoutingOption],
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC,
    fallback: bool = True,
) -> RoutingPolicy:
    return RoutingPolicy(
        task_name="test_task",
        preference_order=tuple(options),
        max_latency_ms=30_000,
        max_cost_usd=0.10,
        privacy_tier=privacy_tier,
        fallback_on_unavailable=fallback,
    )


# ---------------------------------------------------------------------------
# d08 — revocation check tests
# ---------------------------------------------------------------------------

class TestRevocationCheckValid:
    def test_revocation_check_valid_passport(self):
        """HTTP returns {revoked: false} → no exception, returns False."""
        passport = _make_passport()
        mock_resp = _make_http_response({"revoked": False})

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = check_revocation(passport)

        assert result is False

    def test_revocation_check_revoked_raises(self):
        """HTTP returns {revoked: true} → PassportRevoked raised."""
        passport = _make_passport()
        mock_resp = _make_http_response({"revoked": True})

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(PassportRevoked) as exc_info:
                check_revocation(passport)

        assert "passport-test-001" in str(exc_info.value)

    def test_revocation_check_network_failure_allows(self):
        """URLError → fail-open, no exception, returns False."""
        passport = _make_passport()

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = check_revocation(passport)

        assert result is False

    def test_revocation_check_emits_evidence(self):
        """REVOCATION_CHECKED evidence is emitted during the check."""
        passport = _make_passport()
        mock_resp = _make_http_response({"revoked": False})

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        evidence_types = [e["type"] for e in events]
        assert EvidenceType.REVOCATION_CHECKED.value in evidence_types

    def test_revocation_check_evidence_has_passport_id(self):
        """Emitted evidence contains the passport_id."""
        passport = _make_passport(passport_id="passport-xyz-999")
        mock_resp = _make_http_response({"revoked": False})

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        revocation_events = [
            e for e in events
            if e.get("type") == EvidenceType.REVOCATION_CHECKED.value
        ]
        assert any(
            e.get("data", {}).get("passport_id") == "passport-xyz-999"
            for e in revocation_events
        ), f"passport_id not found in revocation events: {revocation_events}"

    def test_revocation_check_revoked_evidence_status(self):
        """When revoked, the final evidence event has status='revoked'."""
        passport = _make_passport()
        mock_resp = _make_http_response({"revoked": True})

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(PassportRevoked):
                check_revocation(passport)

        events = EvidenceLogger.get_events()
        revocation_events = [
            e for e in events
            if e.get("type") == EvidenceType.REVOCATION_CHECKED.value
        ]
        statuses = [e.get("data", {}).get("status") for e in revocation_events]
        assert "revoked" in statuses, f"Expected status='revoked' in {statuses}"


# ---------------------------------------------------------------------------
# d36 — dead provider fallback tests
# ---------------------------------------------------------------------------

class TestDeadProviderFallback:
    def test_dead_provider_skipped_in_dispatch(self):
        """API_HAIKU marked dead; preference [HAIKU, DEEPSEEK] → routes to DEEPSEEK."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        policy = _make_policy(
            options=[RoutingOption.API_HAIKU, RoutingOption.API_DEEPSEEK],
            privacy_tier=PrivacyTier.PUBLIC,
        )

        result = dispatch(task="test", input_text="hello", policy=policy)

        assert result["route"] == RoutingOption.API_DEEPSEEK.value

    def test_all_providers_dead_raises_unavailable(self):
        """All external providers dead, no local fallback → RoutingProviderUnavailable."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        router._DEAD_PROVIDERS.add(RoutingOption.API_DEEPSEEK.value)
        policy = _make_policy(
            options=[RoutingOption.API_HAIKU, RoutingOption.API_DEEPSEEK],
            privacy_tier=PrivacyTier.PUBLIC,
        )

        with pytest.raises(RoutingProviderUnavailable):
            dispatch(task="test", input_text="hello", policy=policy)

    def test_dead_provider_fallback_to_local(self):
        """All external providers dead; LOCAL_GENERALIST in preference → routes to local."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        router._DEAD_PROVIDERS.add(RoutingOption.API_SONNET.value)
        policy = _make_policy(
            options=[
                RoutingOption.API_HAIKU,
                RoutingOption.API_SONNET,
                RoutingOption.LOCAL_GENERALIST,
            ],
            privacy_tier=PrivacyTier.PUBLIC,
        )

        result = dispatch(task="test", input_text="hello", policy=policy)

        assert result["route"] == RoutingOption.LOCAL_GENERALIST.value

    def test_dead_provider_skipped_emits_fallback_evidence(self):
        """ROUTE_FALLBACK evidence is emitted when a dead provider is skipped."""
        router._DEAD_PROVIDERS.add(RoutingOption.API_HAIKU.value)
        policy = _make_policy(
            options=[RoutingOption.API_HAIKU, RoutingOption.LOCAL_GENERALIST],
            privacy_tier=PrivacyTier.PUBLIC,
        )

        dispatch(task="test", input_text="hello", policy=policy)

        events = EvidenceLogger.get_events()
        evidence_types = [e["type"] for e in events]
        assert EvidenceType.ROUTE_FALLBACK.value in evidence_types, (
            f"Expected {EvidenceType.ROUTE_FALLBACK.value!r} in {evidence_types}"
        )


# ---------------------------------------------------------------------------
# d08 — revocation gate wired into promote()
# ---------------------------------------------------------------------------

def _make_artifact(revocation_url: str = "") -> ModelArtifact:
    return ModelArtifact(
        artifact_id="dojo://test-model/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="abc123",
        passport=None,
        revocation_url=revocation_url,
    )


class TestPromoteRevocationGate:
    def test_promote_skips_revocation_when_url_empty(self):
        """promote() with revocation_url='' skips revocation check — no HTTP calls."""
        artifact = _make_artifact(revocation_url="")

        # If check_revocation were called it would try real network; we assert it's NOT called.
        with patch("dojo.promote.revocation.check_revocation") as mock_check:
            passport = promote(
                artifact,
                "receipt-valid-001",
                "deadbeef" * 8,
                baseline_f1=0.70,
                candidate_f1=0.80,
            )

        mock_check.assert_not_called()
        assert passport is not None

    def test_promote_calls_revocation_when_url_set(self):
        """promote() with revocation_url set calls check_revocation once."""
        artifact = _make_artifact(revocation_url="http://127.0.0.1:8002/v1/passport/revoke")

        with patch("dojo.promote.revocation.check_revocation", return_value=False) as mock_check:
            passport = promote(
                artifact,
                "receipt-valid-001",
                "deadbeef" * 8,
                baseline_f1=0.70,
                candidate_f1=0.80,
            )

        mock_check.assert_called_once()
        assert passport is not None

    def test_promote_raises_promotion_blocked_on_revocation_failure(self):
        """promote() raises PromotionBlocked(gate='revocation_check') when revocation raises."""
        artifact = _make_artifact(revocation_url="http://127.0.0.1:8002/v1/passport/revoke")

        with patch(
            "dojo.promote.revocation.check_revocation",
            side_effect=PassportRevoked("Passport revoked", reason="revoked"),
        ):
            with pytest.raises(PromotionBlocked) as exc_info:
                promote(
                    artifact,
                    "receipt-valid-001",
                    "deadbeef" * 8,
                    baseline_f1=0.70,
                    candidate_f1=0.80,
                )

        err = exc_info.value
        assert err.gate == "revocation_check"
        assert len(err.reasons) > 0

    def test_promote_fsm_history_includes_revocation_check_state(self):
        """FSM transitions include REVOCATION_CHECK when revocation_url is set."""
        artifact = _make_artifact(revocation_url="http://127.0.0.1:8002/v1/passport/revoke")

        with patch("dojo.promote.revocation.check_revocation", return_value=False):
            promote(
                artifact,
                "receipt-valid-001",
                "deadbeef" * 8,
                baseline_f1=0.70,
                candidate_f1=0.80,
            )

        events = EvidenceLogger.get_events()
        passport_events = [
            e for e in events
            if e.get("type") == EvidenceType.PROMOTE_PASSPORT_ISSUED.value
        ]
        assert len(passport_events) == 1
        history = passport_events[0]["data"]["fsm_history"]
        state_values = [s for pair in history for s in pair]
        assert GateState.REVOCATION_CHECK.value in state_values, (
            f"Expected {GateState.REVOCATION_CHECK.value!r} in FSM history: {history}"
        )

    def test_promote_passport_issued_evidence_has_revocation_url(self):
        """PROMOTE_PASSPORT_ISSUED evidence includes revocation_url field."""
        url = "http://127.0.0.1:8002/v1/passport/revoke"
        artifact = _make_artifact(revocation_url=url)

        with patch("dojo.promote.revocation.check_revocation", return_value=False):
            promote(
                artifact,
                "receipt-valid-001",
                "deadbeef" * 8,
                baseline_f1=0.70,
                candidate_f1=0.80,
            )

        events = EvidenceLogger.get_events()
        passport_events = [
            e for e in events
            if e.get("type") == EvidenceType.PROMOTE_PASSPORT_ISSUED.value
        ]
        assert len(passport_events) == 1
        assert passport_events[0]["data"]["revocation_url"] == url

    def test_promote_no_url_passport_issued_evidence_revocation_url_is_none(self):
        """PROMOTE_PASSPORT_ISSUED evidence has revocation_url=None when no URL set."""
        artifact = _make_artifact(revocation_url="")

        passport = promote(
            artifact,
            "receipt-valid-001",
            "deadbeef" * 8,
            baseline_f1=0.70,
            candidate_f1=0.80,
        )

        events = EvidenceLogger.get_events()
        passport_events = [
            e for e in events
            if e.get("type") == EvidenceType.PROMOTE_PASSPORT_ISSUED.value
        ]
        assert len(passport_events) == 1
        assert passport_events[0]["data"]["revocation_url"] is None
