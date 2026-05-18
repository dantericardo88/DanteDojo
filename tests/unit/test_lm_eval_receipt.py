"""Tests for d37: signed lm-eval receipts.

Verifies build_lm_eval_receipt, sign_lm_eval_receipt, verify_lm_eval_receipt
including env-var secret resolution and tampering detection.
"""
from __future__ import annotations

import pytest

from dojo.eval.lm_eval_receipt import (
    LmEvalReceiptError,
    build_lm_eval_receipt,
    sign_lm_eval_receipt,
    verify_lm_eval_receipt,
)


def _sample_receipt() -> dict:
    return build_lm_eval_receipt(
        model_artifact_id="qwen-1.5b-v2",
        tasks=["arc_easy", "hellaswag"],
        mean_scores={"arc_easy": 0.61, "hellaswag": 0.78},
        merkle_hash="b" * 64,
        num_fewshot=5,
        primary_metric_key="acc_norm,none",
        evaluated_at="2026-05-18T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# build_lm_eval_receipt
# ---------------------------------------------------------------------------

class TestBuild:
    def test_kind_is_lm_eval(self):
        r = _sample_receipt()
        assert r["kind"] == "lm_eval"

    def test_includes_tasks(self):
        r = _sample_receipt()
        assert r["tasks"] == ["arc_easy", "hellaswag"]

    def test_includes_mean_scores(self):
        r = _sample_receipt()
        assert r["mean_scores"]["arc_easy"] == 0.61

    def test_includes_merkle_hash(self):
        r = _sample_receipt()
        assert r["merkle_hash"] == "b" * 64

    def test_extra_field_when_provided(self):
        r = build_lm_eval_receipt(
            model_artifact_id="m",
            tasks=["t1"],
            mean_scores={"t1": 0.5},
            merkle_hash="x" * 64,
            extra={"note": "test"},
        )
        assert r["extra"] == {"note": "test"}

    def test_no_extra_when_omitted(self):
        r = _sample_receipt()
        assert "extra" not in r


# ---------------------------------------------------------------------------
# sign_lm_eval_receipt
# ---------------------------------------------------------------------------

class TestSign:
    def test_signature_added(self):
        signed = sign_lm_eval_receipt(_sample_receipt(), secret_key=b"abc")
        assert "signature" in signed

    def test_signature_version_set(self):
        signed = sign_lm_eval_receipt(_sample_receipt(), secret_key=b"abc")
        assert signed["signature_version"] == "v1"

    def test_rejects_non_lm_eval_kind(self):
        bad = _sample_receipt()
        bad["kind"] = "wrong"
        with pytest.raises(LmEvalReceiptError):
            sign_lm_eval_receipt(bad, secret_key=b"abc")

    def test_signature_resolves_env_secret(self, monkeypatch):
        monkeypatch.setenv("DOJO_LM_EVAL_SECRET", "env-secret-32bytes-ok-aaaaaaaaaa")
        signed = sign_lm_eval_receipt(_sample_receipt())
        assert verify_lm_eval_receipt(signed)

    def test_signature_uses_ci_fallback_when_no_secret(self, monkeypatch):
        monkeypatch.delenv("DOJO_LM_EVAL_SECRET", raising=False)
        signed = sign_lm_eval_receipt(_sample_receipt())
        # CI fallback is deterministic — verify with same env state
        assert verify_lm_eval_receipt(signed)


# ---------------------------------------------------------------------------
# verify_lm_eval_receipt
# ---------------------------------------------------------------------------

class TestVerify:
    def test_verifies_freshly_signed_receipt(self):
        signed = sign_lm_eval_receipt(_sample_receipt(), secret_key=b"abc")
        assert verify_lm_eval_receipt(signed, secret_key=b"abc") is True

    def test_rejects_wrong_secret(self):
        signed = sign_lm_eval_receipt(_sample_receipt(), secret_key=b"abc")
        assert verify_lm_eval_receipt(signed, secret_key=b"xyz") is False

    def test_rejects_non_lm_eval_kind(self):
        signed = sign_lm_eval_receipt(_sample_receipt(), secret_key=b"abc")
        signed["kind"] = "other"
        assert verify_lm_eval_receipt(signed, secret_key=b"abc") is False

    def test_rejects_tampered_mean_scores(self):
        signed = sign_lm_eval_receipt(_sample_receipt(), secret_key=b"abc")
        signed["mean_scores"] = {"arc_easy": 0.99}
        assert verify_lm_eval_receipt(signed, secret_key=b"abc") is False

    def test_rejects_tampered_tasks(self):
        signed = sign_lm_eval_receipt(_sample_receipt(), secret_key=b"abc")
        signed["tasks"] = ["different_task"]
        assert verify_lm_eval_receipt(signed, secret_key=b"abc") is False


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_evidence_emitted_on_sign(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        sign_lm_eval_receipt(_sample_receipt(), secret_key=b"abc")
        events = EvidenceLogger.get_events()
        signed = [e for e in events if e.get("data", {}).get("phase") == "lm_eval_signed"]
        assert len(signed) >= 1

    def test_evidence_includes_fingerprint(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        sign_lm_eval_receipt(_sample_receipt(), secret_key=b"abc")
        events = EvidenceLogger.get_events()
        signed_events = [e for e in events
                         if e.get("data", {}).get("phase") == "lm_eval_signed"]
        assert "signature_fingerprint" in signed_events[0]["data"]
