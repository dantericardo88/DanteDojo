"""Tests for d35: signed cost receipts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dojo.route.cost_receipts import (
    CostReceiptError,
    append_signed_receipt,
    build_cost_receipt,
    sign_cost_receipt,
    verify_cost_receipt,
    verify_receipt_log,
)


_SECRET = b"cost-secret-32-bytes-aaaaaaaaaaaa"


def _sample() -> dict:
    return build_cost_receipt(
        request_id="req-001",
        provider="api_haiku",
        prompt_tokens=120,
        completion_tokens=64,
        usd_cost=0.0021,
        privacy_tier="shared",
    )


class TestBuild:
    def test_kind_is_cost_receipt(self):
        assert _sample()["kind"] == "cost_receipt"

    def test_includes_provider(self):
        assert _sample()["provider"] == "api_haiku"

    def test_includes_token_counts(self):
        r = _sample()
        assert r["prompt_tokens"] == 120
        assert r["completion_tokens"] == 64

    def test_negative_tokens_rejected(self):
        with pytest.raises(CostReceiptError):
            build_cost_receipt("r", "p", -1, 0, 0.0)

    def test_negative_cost_rejected(self):
        with pytest.raises(CostReceiptError):
            build_cost_receipt("r", "p", 0, 0, -0.1)


class TestSign:
    def test_signature_added(self):
        signed = sign_cost_receipt(_sample(), secret=_SECRET)
        assert "signature" in signed

    def test_signature_version_set(self):
        signed = sign_cost_receipt(_sample(), secret=_SECRET)
        assert signed["signature_version"] == "v1"

    def test_rejects_wrong_kind(self):
        bad = _sample()
        bad["kind"] = "other"
        with pytest.raises(CostReceiptError):
            sign_cost_receipt(bad, secret=_SECRET)


class TestVerify:
    def test_verifies_freshly_signed(self):
        signed = sign_cost_receipt(_sample(), secret=_SECRET)
        assert verify_cost_receipt(signed, secret=_SECRET) is True

    def test_rejects_wrong_secret(self):
        signed = sign_cost_receipt(_sample(), secret=_SECRET)
        assert verify_cost_receipt(signed, secret=b"other-key") is False

    def test_rejects_tampered_cost(self):
        signed = sign_cost_receipt(_sample(), secret=_SECRET)
        signed["usd_cost"] = 999.0
        assert verify_cost_receipt(signed, secret=_SECRET) is False

    def test_rejects_wrong_kind(self):
        signed = sign_cost_receipt(_sample(), secret=_SECRET)
        signed["kind"] = "other"
        assert verify_cost_receipt(signed, secret=_SECRET) is False

    def test_env_secret_resolves(self, monkeypatch):
        monkeypatch.setenv("DOJO_COST_RECEIPT_SECRET", "env-secret-32-bytes-aaaaaaaaaa")
        signed = sign_cost_receipt(_sample())
        assert verify_cost_receipt(signed)


class TestJsonlLog:
    def test_appends_one_line_per_receipt(self, tmp_path: Path):
        log = tmp_path / "cost.jsonl"
        append_signed_receipt(_sample(), log, secret=_SECRET)
        append_signed_receipt(_sample(), log, secret=_SECRET)
        lines = log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_each_line_is_parseable_json(self, tmp_path: Path):
        log = tmp_path / "cost.jsonl"
        append_signed_receipt(_sample(), log, secret=_SECRET)
        record = json.loads(log.read_text(encoding="utf-8").strip())
        assert record["kind"] == "cost_receipt"
        assert "signature" in record

    def test_creates_parent_dir(self, tmp_path: Path):
        log = tmp_path / "sub" / "deeper" / "cost.jsonl"
        append_signed_receipt(_sample(), log, secret=_SECRET)
        assert log.exists()


class TestVerifyLog:
    def test_verify_log_all_valid(self, tmp_path: Path):
        log = tmp_path / "cost.jsonl"
        for _ in range(3):
            append_signed_receipt(_sample(), log, secret=_SECRET)
        results = verify_receipt_log(log, secret=_SECRET)
        assert len(results) == 3
        assert all(r["valid"] for r in results)

    def test_verify_log_detects_tampered_line(self, tmp_path: Path):
        log = tmp_path / "cost.jsonl"
        append_signed_receipt(_sample(), log, secret=_SECRET)
        # Tamper: rewrite the line with a different usd_cost
        contents = log.read_text(encoding="utf-8").strip()
        tampered = json.loads(contents)
        tampered["usd_cost"] = 999.99
        log.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
        results = verify_receipt_log(log, secret=_SECRET)
        assert results[0]["valid"] is False

    def test_verify_missing_log_returns_empty(self, tmp_path: Path):
        results = verify_receipt_log(tmp_path / "missing.jsonl", secret=_SECRET)
        assert results == []

    def test_verify_log_malformed_line(self, tmp_path: Path):
        log = tmp_path / "cost.jsonl"
        log.write_text("not json\n", encoding="utf-8")
        results = verify_receipt_log(log, secret=_SECRET)
        assert results[0]["valid"] is False
        assert results[0].get("error") == "json"


class TestEvidence:
    def test_evidence_emitted_on_sign(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        sign_cost_receipt(_sample(), secret=_SECRET)
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "route.cost_receipt_signed" for e in events)
