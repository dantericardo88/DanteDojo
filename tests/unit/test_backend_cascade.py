"""Tests for d47: multi-backend serving cascade with auto-fallback."""
from __future__ import annotations

import pytest

from dojo.serve.backend_cascade import (
    BackendCascade,
    BackendCascadeExhausted,
    first_successful,
)


def _ok(name: str):
    return lambda prompt, **kw: {"backend": name, "text": f"{name}:{prompt}"}


def _fail(name: str):
    def _raise(prompt, **kw):
        raise RuntimeError(f"{name}_failed")
    return _raise


class TestConstruction:
    def test_empty_backends_raises(self):
        with pytest.raises(BackendCascadeExhausted):
            BackendCascade(backends=[])


class TestCascadeHappyPath:
    def test_first_backend_used_when_succeeds(self):
        cascade = BackendCascade(backends=[("a", _ok("a")), ("b", _ok("b"))])
        result = cascade.generate("hi")
        assert result["backend"] == "a"

    def test_only_one_attempt_when_first_succeeds(self):
        cascade = BackendCascade(backends=[("a", _ok("a")), ("b", _ok("b"))])
        cascade.generate("hi")
        assert len(cascade.attempt_log) == 1

    def test_attempt_log_includes_duration(self):
        cascade = BackendCascade(backends=[("a", _ok("a"))])
        cascade.generate("hi")
        assert cascade.attempt_log[0].duration_ms >= 0


class TestCascadeFallback:
    def test_skips_failing_backend(self):
        cascade = BackendCascade(backends=[
            ("primary", _fail("primary")),
            ("secondary", _ok("secondary")),
        ])
        result = cascade.generate("hi")
        assert result["backend"] == "secondary"

    def test_two_attempts_when_fallback(self):
        cascade = BackendCascade(backends=[
            ("primary", _fail("primary")),
            ("secondary", _ok("secondary")),
        ])
        cascade.generate("hi")
        assert len(cascade.attempt_log) == 2

    def test_first_attempt_marked_failed(self):
        cascade = BackendCascade(backends=[
            ("primary", _fail("primary")),
            ("secondary", _ok("secondary")),
        ])
        cascade.generate("hi")
        assert cascade.attempt_log[0].succeeded is False
        assert cascade.attempt_log[1].succeeded is True

    def test_failed_attempt_records_error(self):
        cascade = BackendCascade(backends=[
            ("primary", _fail("primary")),
            ("secondary", _ok("secondary")),
        ])
        cascade.generate("hi")
        assert "primary_failed" in cascade.attempt_log[0].error


class TestCascadeExhausted:
    def test_all_failing_raises(self):
        cascade = BackendCascade(backends=[
            ("a", _fail("a")),
            ("b", _fail("b")),
        ])
        with pytest.raises(BackendCascadeExhausted, match="all 2"):
            cascade.generate("hi")

    def test_exhausted_error_lists_all_failures(self):
        cascade = BackendCascade(backends=[
            ("a", _fail("a")),
            ("b", _fail("b")),
        ])
        with pytest.raises(BackendCascadeExhausted) as exc_info:
            cascade.generate("hi")
        assert "a" in str(exc_info.value)
        assert "b" in str(exc_info.value)


class TestKwargsForwarding:
    def test_kwargs_passed_to_backend(self):
        captured = {}

        def backend(prompt, **kw):
            captured.update(kw)
            return {"ok": True}

        cascade = BackendCascade(backends=[("x", backend)])
        cascade.generate("hi", max_tokens=42, temperature=0.5)
        assert captured["max_tokens"] == 42
        assert captured["temperature"] == 0.5


class TestFirstSuccessful:
    def test_returns_winning_name_and_result(self):
        name, result = first_successful(
            [("primary", _fail("primary")), ("secondary", _ok("secondary"))],
            "hi",
        )
        assert name == "secondary"
        assert result["backend"] == "secondary"


class TestEvidence:
    def test_per_attempt_evidence(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        cascade = BackendCascade(backends=[("a", _fail("a")), ("b", _ok("b"))])
        cascade.generate("hi")
        events = EvidenceLogger.get_events()
        attempts = [e for e in events if e.get("type") == "serve.cascade_attempt"]
        assert len(attempts) == 2

    def test_summary_event_includes_winner(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        cascade = BackendCascade(backends=[("a", _ok("a"))])
        cascade.generate("hi")
        events = EvidenceLogger.get_events()
        summary = [e for e in events if e.get("type") == "serve.cascade_complete"]
        assert summary
        assert summary[0]["data"]["winning_backend"] == "a"

    def test_summary_event_on_failure(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        cascade = BackendCascade(backends=[("a", _fail("a"))])
        with pytest.raises(BackendCascadeExhausted):
            cascade.generate("hi")
        events = EvidenceLogger.get_events()
        summary = [e for e in events if e.get("type") == "serve.cascade_complete"]
        assert summary[0]["data"]["success"] is False
