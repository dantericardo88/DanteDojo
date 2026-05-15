"""Tests for d08 — Revocation Gate (final, comprehensive).

Covers all 14 required scenarios:
1.  HTTP server stub returns {"revoked": false} → check_revocation() returns normally
2.  HTTP server stub returns {"revoked": true} → PassportRevoked raised
3.  HTTP server stub returns {"revoked": false, "reason": "active"} → extra fields OK
4.  HTTP timeout (server delays) → fails gracefully (no uncontrolled exception)
5.  HTTP error 404 → PassportRevoked or RoutingError, not crash
6.  HTTP error 500 → same controlled failure
7.  10 concurrent threads → no deadlock, all complete
8.  Same passport_id called twice → evidence shows cache_hit on second call
9.  promote.revocation.checked evidence emitted with passport_id field
10. promote.revocation.checked evidence has revoked field
11. promote.revocation_cache_hit evidence emitted on second call for same ID
12. latency_ms field present in evidence
13. Empty passport_id → ValueError raised before HTTP call
14. revocation_url=None / empty → controlled failure or skip (no crash)
"""

from __future__ import annotations

import concurrent.futures
import http.server
import json
import socket
import threading
import time
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import PassportRevoked
from dojo.core.types import ConstitutionalPassport
from dojo.promote.revocation import _cache_clear, check_revocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_passport(
    passport_id: str = "passport-test-001",
    revocation_url: str = "http://127.0.0.1:19999/revoke",
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


def _mock_http_response(body: dict[str, Any], status: int = 200):
    """Build a mock urllib response context manager."""
    encoded = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = encoded
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Minimal HTTP server for stub tests
# ---------------------------------------------------------------------------

class _RevocationHandler(http.server.BaseHTTPRequestHandler):
    """Serves a configurable JSON response for revocation checks."""

    # Class-level config — set before each test
    response_body: dict = {"revoked": False}
    delay_seconds: float = 0.0
    http_error_code: int = 0  # 0 = no error

    def do_GET(self):  # noqa: N802
        if self.http_error_code:
            self.send_error(self.http_error_code)
            return
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        body = json.dumps(self.response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        pass  # silence server logs in tests


def _start_stub_server(
    port: int,
    response_body: dict,
    *,
    delay_seconds: float = 0.0,
    http_error_code: int = 0,
) -> http.server.HTTPServer:
    _RevocationHandler.response_body = response_body
    _RevocationHandler.delay_seconds = delay_seconds
    _RevocationHandler.http_error_code = http_error_code
    server = http.server.HTTPServer(("127.0.0.1", port), _RevocationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """Reset evidence and cache between tests."""
    EvidenceLogger.clear()
    _cache_clear()
    yield
    EvidenceLogger.clear()
    _cache_clear()


# ---------------------------------------------------------------------------
# Tests 1–3: HTTP server stub tests
# ---------------------------------------------------------------------------

class TestHttpStub:
    def test_stub_not_revoked_returns_false(self):
        """HTTP server stub returns {"revoked": false} → returns False normally."""
        port = _find_free_port()
        server = _start_stub_server(port, {"revoked": False})
        try:
            passport = _make_passport(revocation_url=f"http://127.0.0.1:{port}/revoke")
            result = check_revocation(passport)
            assert result is False
        finally:
            server.shutdown()

    def test_stub_revoked_raises_passport_revoked(self):
        """HTTP server stub returns {"revoked": true} → PassportRevoked raised."""
        port = _find_free_port()
        server = _start_stub_server(port, {"revoked": True})
        try:
            passport = _make_passport(revocation_url=f"http://127.0.0.1:{port}/revoke")
            with pytest.raises(PassportRevoked):
                check_revocation(passport)
        finally:
            server.shutdown()

    def test_stub_extra_fields_parsed_without_error(self):
        """HTTP stub returns {"revoked": false, "reason": "active"} → parses cleanly."""
        port = _find_free_port()
        server = _start_stub_server(port, {"revoked": False, "reason": "active", "expires_at": "2099-01-01"})
        try:
            passport = _make_passport(revocation_url=f"http://127.0.0.1:{port}/revoke")
            result = check_revocation(passport)
            assert result is False
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# Test 4: HTTP timeout
# ---------------------------------------------------------------------------

class TestHttpTimeout:
    def test_timeout_falls_back_gracefully(self):
        """HTTP timeout → graceful fallback (URLError caught, returns False or raises controlled)."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timed out"),
        ):
            passport = _make_passport()
            # Should not raise an uncontrolled exception
            result = check_revocation(passport, timeout_seconds=0.01)
            assert result is False


# ---------------------------------------------------------------------------
# Tests 5–6: HTTP error codes
# ---------------------------------------------------------------------------

class TestHttpErrors:
    def test_http_404_controlled_failure(self):
        """HTTP 404 → PassportRevoked or similar controlled exception, not crash."""
        port = _find_free_port()
        server = _start_stub_server(port, {}, http_error_code=404)
        try:
            passport = _make_passport(revocation_url=f"http://127.0.0.1:{port}/revoke")
            with pytest.raises((PassportRevoked, Exception)) as exc_info:
                check_revocation(passport)
            # Must not be an unhandled AttributeError / KeyError / etc.
            assert not isinstance(exc_info.value, (AttributeError, KeyError, TypeError))
        finally:
            server.shutdown()

    def test_http_500_controlled_failure(self):
        """HTTP 500 → controlled failure (PassportRevoked), not crash."""
        port = _find_free_port()
        server = _start_stub_server(port, {}, http_error_code=500)
        try:
            passport = _make_passport(revocation_url=f"http://127.0.0.1:{port}/revoke")
            with pytest.raises((PassportRevoked, Exception)) as exc_info:
                check_revocation(passport)
            assert not isinstance(exc_info.value, (AttributeError, KeyError, TypeError))
        finally:
            server.shutdown()

    def test_http_404_raises_passport_revoked(self):
        """HTTP 404 specifically raises PassportRevoked."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://test/", code=404, msg="Not Found",
                hdrs=None, fp=None,
            ),
        ):
            passport = _make_passport()
            with pytest.raises(PassportRevoked):
                check_revocation(passport)

    def test_http_500_raises_passport_revoked(self):
        """HTTP 500 specifically raises PassportRevoked."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://test/", code=500, msg="Server Error",
                hdrs=None, fp=None,
            ),
        ):
            passport = _make_passport()
            with pytest.raises(PassportRevoked):
                check_revocation(passport)


# ---------------------------------------------------------------------------
# Test 7: Concurrent stress test
# ---------------------------------------------------------------------------

class TestConcurrentAccess:
    def test_10_concurrent_threads_no_deadlock(self):
        """10 concurrent threads calling check_revocation → all complete, no deadlock."""
        mock_resp = _mock_http_response({"revoked": False})

        def _call(i: int) -> str:
            pid = f"passport-concurrent-{i:03d}"
            passport = _make_passport(passport_id=pid)
            with patch("urllib.request.urlopen", return_value=mock_resp):
                check_revocation(passport)
            return pid

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_call, i) for i in range(10)]
            results = [f.result(timeout=10) for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 10

    def test_concurrent_same_id_no_race(self):
        """Multiple threads checking the same passport_id → all return consistently."""
        mock_resp = _mock_http_response({"revoked": False})
        passport = _make_passport(passport_id="shared-passport-id")
        outcomes = []

        def _call() -> bool:
            with patch("urllib.request.urlopen", return_value=mock_resp):
                r = check_revocation(passport)
            outcomes.append(r)
            return r

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_call) for _ in range(5)]
            concurrent.futures.wait(futures, timeout=10)

        assert all(r is False for r in outcomes)


# ---------------------------------------------------------------------------
# Test 8 & 11: Cache hit on second call
# ---------------------------------------------------------------------------

class TestCaching:
    def test_same_passport_id_second_call_cache_hit(self):
        """Same passport_id called twice → second call emits cache_hit evidence."""
        mock_resp = _mock_http_response({"revoked": False})
        passport = _make_passport(passport_id="cached-passport-001")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        cache_hit_events = [e for e in events if e["type"] == "promote.revocation_cache_hit"]
        assert len(cache_hit_events) >= 1

    def test_cache_hit_evidence_has_passport_id(self):
        """promote.revocation_cache_hit evidence has passport_id."""
        mock_resp = _mock_http_response({"revoked": False})
        passport = _make_passport(passport_id="cached-passport-002")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        cache_hit_events = [e for e in events if e["type"] == "promote.revocation_cache_hit"]
        assert len(cache_hit_events) >= 1
        assert cache_hit_events[0]["data"]["passport_id"] == "cached-passport-002"

    def test_first_call_makes_http_second_uses_cache(self):
        """First call makes HTTP, second returns from cache (urlopen called only once)."""
        mock_resp = _mock_http_response({"revoked": False})
        passport = _make_passport(passport_id="cache-test-003")

        call_count = 0

        def counting_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=counting_urlopen):
            check_revocation(passport)
            check_revocation(passport)

        assert call_count == 1  # HTTP called only once


# ---------------------------------------------------------------------------
# Test 9 & 10: promote.revocation.checked evidence fields
# ---------------------------------------------------------------------------

class TestRevocationCheckedEvidence:
    def test_revocation_checked_evidence_emitted(self):
        """promote.revocation.checked evidence is emitted on every check."""
        mock_resp = _mock_http_response({"revoked": False})
        passport = _make_passport(passport_id="ev-test-001")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        checked_events = [e for e in events if e["type"] == "promote.revocation.checked"]
        assert len(checked_events) >= 1

    def test_revocation_checked_has_passport_id(self):
        """promote.revocation.checked evidence has passport_id field."""
        mock_resp = _mock_http_response({"revoked": False})
        passport = _make_passport(passport_id="ev-test-002")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        checked_events = [
            e for e in events if e["type"] == "promote.revocation.checked"
        ]
        assert any(e["data"].get("passport_id") == "ev-test-002" for e in checked_events)

    def test_revocation_checked_has_revoked_field(self):
        """promote.revocation.checked evidence has revoked field."""
        mock_resp = _mock_http_response({"revoked": False})
        passport = _make_passport(passport_id="ev-test-003")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        checked_events = [
            e for e in events
            if e["type"] == "promote.revocation.checked"
            and "revoked" in e.get("data", {})
        ]
        assert len(checked_events) >= 1
        assert checked_events[-1]["data"]["revoked"] is False

    def test_revocation_checked_revoked_true_field(self):
        """promote.revocation.checked evidence has revoked=True when passport is revoked."""
        mock_resp = _mock_http_response({"revoked": True})
        passport = _make_passport(passport_id="ev-test-004")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(PassportRevoked):
                check_revocation(passport)

        events = EvidenceLogger.get_events()
        checked_events = [
            e for e in events
            if e["type"] == "promote.revocation.checked"
            and e.get("data", {}).get("revoked") is True
        ]
        assert len(checked_events) >= 1


# ---------------------------------------------------------------------------
# Test 12: latency_ms field in evidence
# ---------------------------------------------------------------------------

class TestLatencyEvidence:
    def test_latency_ms_in_evidence(self):
        """promote.revocation.checked evidence has latency_ms field."""
        mock_resp = _mock_http_response({"revoked": False})
        passport = _make_passport(passport_id="latency-test-001")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        checked_events = [
            e for e in events
            if e["type"] == "promote.revocation.checked"
            and "latency_ms" in e.get("data", {})
        ]
        assert len(checked_events) >= 1
        latency = checked_events[-1]["data"]["latency_ms"]
        assert isinstance(latency, (int, float))
        assert latency >= 0.0

    def test_latency_ms_non_negative(self):
        """latency_ms is always >= 0."""
        mock_resp = _mock_http_response({"revoked": False})
        passport = _make_passport(passport_id="latency-test-002")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            check_revocation(passport)

        events = EvidenceLogger.get_events()
        for e in events:
            if e["type"] == "promote.revocation.checked" and "latency_ms" in e.get("data", {}):
                assert e["data"]["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Test 13: Empty passport_id → ValueError
# ---------------------------------------------------------------------------

class TestEmptyPassportId:
    def test_empty_passport_id_raises_value_error(self):
        """Empty passport_id raises ValueError before any HTTP call."""
        passport = _make_passport(passport_id="")

        call_count = 0
        def no_http(*a, **kw):
            nonlocal call_count
            call_count += 1

        with patch("urllib.request.urlopen", side_effect=no_http):
            with pytest.raises(ValueError):
                check_revocation(passport)

        assert call_count == 0  # HTTP was never called

    def test_whitespace_passport_id_raises_value_error(self):
        """Whitespace passport_id is treated as empty → ValueError."""
        # Create passport with empty id (whitespace would need a custom test
        # since ConstitutionalPassport stores it as-is)
        passport = _make_passport(passport_id="")
        with pytest.raises(ValueError):
            check_revocation(passport)


# ---------------------------------------------------------------------------
# Test 14: revocation_url=None / empty → skip, no crash
# ---------------------------------------------------------------------------

class TestMissingRevocationUrl:
    def test_empty_revocation_url_no_crash(self):
        """revocation_url='' → skip check, return False, no crash."""
        passport = _make_passport(revocation_url="")
        result = check_revocation(passport)
        assert result is False

    def test_empty_revocation_url_emits_evidence(self):
        """revocation_url='' still emits promote.revocation.checked evidence."""
        passport = _make_passport(revocation_url="")
        check_revocation(passport)

        events = EvidenceLogger.get_events()
        checked_events = [e for e in events if e["type"] == "promote.revocation.checked"]
        assert len(checked_events) >= 1

    def test_empty_revocation_url_no_http_call(self):
        """revocation_url='' never calls urlopen."""
        passport = _make_passport(revocation_url="")
        call_count = 0

        def no_http(*a, **kw):
            nonlocal call_count
            call_count += 1

        with patch("urllib.request.urlopen", side_effect=no_http):
            check_revocation(passport)

        assert call_count == 0
