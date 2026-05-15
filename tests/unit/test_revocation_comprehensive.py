"""Comprehensive unit tests for d08 — sovereign revocation check.

Tests cover check_revocation() directly and its integration with promote().
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PassportRevoked
from dojo.core.types import ConstitutionalPassport, ModelArtifact
from dojo.promote.gate import promote
from dojo.promote.revocation import check_revocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_passport(
    passport_id: str = "passport-comprehensive-001",
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


def _make_http_response(body: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a mock urllib response context manager."""
    encoded = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = encoded
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence() -> Generator[None, None, None]:
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# TestRevocationURLCheck — 8 tests
# ---------------------------------------------------------------------------

class TestRevocationURLCheck:

    def test_valid_passport_not_revoked(self):
        """HTTP returns {revoked: false} → no exception raised, returns False."""
        passport = _make_passport()
        mock_resp = _make_http_response({"revoked": False})

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = check_revocation(passport)

        assert result is False

    def test_revoked_passport_raises(self):
        """HTTP returns {revoked: true, reason: '...'} → PassportRevoked raised."""
        passport = _make_passport(passport_id="passport-revoked-999")
        mock_resp = _make_http_response({
            "revoked": True,
            "reason": "policy violation",
        })

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(PassportRevoked) as exc_info:
                check_revocation(passport)

        assert "passport-revoked-999" in str(exc_info.value)

    def test_revocation_url_empty_skips_check(self):
        """promote() with empty revocation_url → check_revocation never called, no network.

        The one-door gate (promote()) guards the call: if revocation_url is
        empty the function is not invoked at all.  We verify this by asserting
        that check_revocation is not patched-into and no network calls occur.
        """
        artifact = _make_artifact(revocation_url="")

        with patch("urllib.request.urlopen") as mock_urlopen:
            promote(
                artifact,
                "receipt-valid-001",
                "deadbeef" * 8,
                baseline_f1=0.70,
                candidate_f1=0.80,
            )

        # urlopen must NOT have been called at any point
        mock_urlopen.assert_not_called()

    def test_network_timeout_falls_back_gracefully(self):
        """URLError/timeout → no exception; fail-open returns False."""
        passport = _make_passport()

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timed out"),
        ):
            result = check_revocation(passport)

        assert result is False

    def test_revocation_emits_evidence(self):
        """'promote.revocation.checked' evidence emitted with result."""
        passport = _make_passport()
        mock_resp = _make_http_response({"revoked": False})

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        event_types = [e["type"] for e in events]
        assert EvidenceType.REVOCATION_CHECKED.value in event_types

    def test_revoked_evidence_has_reason(self):
        """Evidence includes revocation reason when passport is revoked."""
        passport = _make_passport()
        mock_resp = _make_http_response({"revoked": True})

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(PassportRevoked):
                check_revocation(passport)

        events = EvidenceLogger.get_events()
        # The final REVOCATION_CHECKED event for a revoked passport has status="revoked"
        revocation_events = [
            e for e in events
            if e.get("type") == EvidenceType.REVOCATION_CHECKED.value
        ]
        # At least one event must carry status="revoked" or revoked=True
        statuses = [e.get("data", {}).get("status") for e in revocation_events]
        revoked_flags = [e.get("data", {}).get("revoked") for e in revocation_events]
        assert "revoked" in statuses or True in revoked_flags, (
            f"No revoked indicator found in events: {revocation_events}"
        )

    def test_check_revocation_called_in_promote_gate(self):
        """patch check_revocation → verify promote() calls it when revocation_url is set."""
        artifact = _make_artifact(revocation_url="http://127.0.0.1:8002/v1/passport/revoke")

        with patch(
            "dojo.promote.revocation.check_revocation",
            return_value=False,
        ) as mock_check:
            passport = promote(
                artifact,
                "receipt-valid-001",
                "deadbeef" * 8,
                baseline_f1=0.70,
                candidate_f1=0.80,
            )

        mock_check.assert_called_once()
        assert passport is not None

    def test_no_revocation_call_without_url(self):
        """patch check_revocation → verify NOT called when revocation_url is empty."""
        artifact = _make_artifact(revocation_url="")

        with patch(
            "dojo.promote.revocation.check_revocation",
        ) as mock_check:
            passport = promote(
                artifact,
                "receipt-valid-001",
                "deadbeef" * 8,
                baseline_f1=0.70,
                candidate_f1=0.80,
            )

        mock_check.assert_not_called()
        assert passport is not None
