"""Tests for d08: fail-closed strict revocation gate."""
from __future__ import annotations

import urllib.error
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from dojo.core.revocation_strict import (
    PassportRevokedStrict,
    RevocationResult,
    StrictRevocationCheck,
)


def _mock_response(status: int = 200):
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestHappyPath:
    def test_http_200_means_valid(self):
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            urlopen=MagicMock(return_value=_mock_response(200)),
        )
        r = chk.check("pp-1")
        assert r.revoked is False

    def test_gate_passes_when_valid(self):
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            urlopen=MagicMock(return_value=_mock_response(200)),
        )
        chk.gate("pp-1")  # no raise


class TestRevoked:
    def test_http_404_means_revoked(self):
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            urlopen=MagicMock(return_value=_mock_response(404)),
        )
        r = chk.check("pp-1")
        assert r.revoked is True
        assert r.reason == "not_found"

    def test_gate_raises_when_revoked(self):
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            urlopen=MagicMock(return_value=_mock_response(404)),
        )
        with pytest.raises(PassportRevokedStrict, match="revoked"):
            chk.gate("pp-1")


class TestFailClosed:
    def test_network_error_revokes_when_fail_closed(self):
        urlopen = MagicMock(side_effect=urllib.error.URLError("no route"))
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            fail_closed=True,
            urlopen=urlopen,
        )
        r = chk.check("pp-1")
        assert r.revoked is True
        assert r.network_error is True

    def test_network_error_allows_when_fail_open(self):
        urlopen = MagicMock(side_effect=urllib.error.URLError("no route"))
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            fail_closed=False,
            urlopen=urlopen,
        )
        r = chk.check("pp-1")
        assert r.revoked is False
        assert r.network_error is True

    def test_unexpected_exception_revokes_when_fail_closed(self):
        urlopen = MagicMock(side_effect=RuntimeError("boom"))
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            fail_closed=True,
            urlopen=urlopen,
        )
        r = chk.check("pp-1")
        assert r.revoked is True


class TestHttp5xx:
    def test_500_revokes_when_fail_closed(self):
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            fail_closed=True,
            urlopen=MagicMock(return_value=_mock_response(503)),
        )
        r = chk.check("pp-1")
        assert r.revoked is True
        assert "503" in r.reason

    def test_500_allows_when_fail_open(self):
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            fail_closed=False,
            urlopen=MagicMock(return_value=_mock_response(503)),
        )
        r = chk.check("pp-1")
        assert r.revoked is False


class TestEvidence:
    def test_evidence_emitted_on_check(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            urlopen=MagicMock(return_value=_mock_response(200)),
        )
        chk.check("pp-1")
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "passport.revocation_check_strict" for e in events)

    def test_evidence_includes_network_error_flag(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        chk = StrictRevocationCheck(
            endpoint_url="http://test/revoke",
            urlopen=MagicMock(side_effect=urllib.error.URLError("x")),
        )
        chk.check("pp-1")
        events = EvidenceLogger.get_events()
        evs = [e for e in events if e.get("type") == "passport.revocation_check_strict"]
        assert evs[0]["data"]["network_error"] is True
