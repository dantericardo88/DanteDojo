"""Tests for d40: HMAC-SHA256 signing of EvalReceipt.

Verifies sign_eval_receipt() produces deterministic signatures, that
verify_eval_receipt() catches tampering, and that signatures are stable
under key reordering and field reordering.
"""
from __future__ import annotations

import pytest

from dojo.eval.receipt_signer import (
    receipt_signature_fingerprint,
    sign_eval_receipt,
    verify_eval_receipt,
)


_SECRET = b"super-secret-32-bytes-of-entropy"
_OTHER_SECRET = b"different-secret-for-testing-aaa"


def _sample_receipt() -> dict:
    return {
        "model_artifact_id": "qwen-1.5b-v2",
        "suite_type": "dante_v1",
        "tasks": ["task1", "task2"],
        "mean_scores": {"task1": 0.92, "task2": 0.88},
        "merkle_hash": "a" * 64,
        "evaluated_at": "2026-05-18T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

class TestSigning:
    def test_signature_field_added(self):
        signed = sign_eval_receipt(_sample_receipt(), _SECRET)
        assert "signature" in signed

    def test_signature_version_added(self):
        signed = sign_eval_receipt(_sample_receipt(), _SECRET)
        assert signed["signature_version"] == "v1"

    def test_signature_is_hex_string(self):
        signed = sign_eval_receipt(_sample_receipt(), _SECRET)
        sig = signed["signature"]
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex digest length
        int(sig, 16)  # must parse as hex

    def test_empty_secret_raises(self):
        with pytest.raises(ValueError):
            sign_eval_receipt(_sample_receipt(), b"")

    def test_original_receipt_unchanged(self):
        receipt = _sample_receipt()
        snapshot = dict(receipt)
        sign_eval_receipt(receipt, _SECRET)
        assert receipt == snapshot


class TestDeterminism:
    def test_same_input_produces_same_signature(self):
        s1 = sign_eval_receipt(_sample_receipt(), _SECRET)
        s2 = sign_eval_receipt(_sample_receipt(), _SECRET)
        assert s1["signature"] == s2["signature"]

    def test_different_keys_produce_different_signatures(self):
        s1 = sign_eval_receipt(_sample_receipt(), _SECRET)
        s2 = sign_eval_receipt(_sample_receipt(), _OTHER_SECRET)
        assert s1["signature"] != s2["signature"]

    def test_field_order_does_not_affect_signature(self):
        r1 = {"a": 1, "b": 2, "c": 3}
        r2 = {"c": 3, "a": 1, "b": 2}
        s1 = sign_eval_receipt(r1, _SECRET)
        s2 = sign_eval_receipt(r2, _SECRET)
        assert s1["signature"] == s2["signature"]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class TestVerification:
    def test_verifies_valid_signature(self):
        signed = sign_eval_receipt(_sample_receipt(), _SECRET)
        assert verify_eval_receipt(signed, _SECRET) is True

    def test_rejects_wrong_secret(self):
        signed = sign_eval_receipt(_sample_receipt(), _SECRET)
        assert verify_eval_receipt(signed, _OTHER_SECRET) is False

    def test_rejects_tampered_field(self):
        signed = sign_eval_receipt(_sample_receipt(), _SECRET)
        signed["mean_scores"] = {"task1": 0.99, "task2": 0.99}
        assert verify_eval_receipt(signed, _SECRET) is False

    def test_rejects_tampered_signature(self):
        signed = sign_eval_receipt(_sample_receipt(), _SECRET)
        signed["signature"] = "0" * 64
        assert verify_eval_receipt(signed, _SECRET) is False

    def test_rejects_missing_signature(self):
        receipt = _sample_receipt()
        assert verify_eval_receipt(receipt, _SECRET) is False

    def test_rejects_empty_secret(self):
        signed = sign_eval_receipt(_sample_receipt(), _SECRET)
        assert verify_eval_receipt(signed, b"") is False

    def test_rejects_wrong_signature_version(self):
        signed = sign_eval_receipt(_sample_receipt(), _SECRET)
        signed["signature_version"] = "v999"
        assert verify_eval_receipt(signed, _SECRET) is False


# ---------------------------------------------------------------------------
# Resign behaviour (canonicalization drops existing signature)
# ---------------------------------------------------------------------------

class TestResign:
    def test_can_resign_already_signed_receipt(self):
        signed1 = sign_eval_receipt(_sample_receipt(), _SECRET)
        signed2 = sign_eval_receipt(signed1, _SECRET)
        # Resigning the same dict yields the same signature (canonicalization
        # strips the existing signature before hashing).
        assert signed1["signature"] == signed2["signature"]


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_default_fingerprint_length_eight(self):
        sig = "abcdef1234567890" * 4
        assert receipt_signature_fingerprint(sig) == "abcdef12"

    def test_custom_fingerprint_length(self):
        sig = "abcdef1234567890"
        assert receipt_signature_fingerprint(sig, length=4) == "abcd"

    def test_empty_signature_returns_empty(self):
        assert receipt_signature_fingerprint("") == ""
