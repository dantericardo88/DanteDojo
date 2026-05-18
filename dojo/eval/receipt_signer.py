"""HMAC-SHA256 signing for EvalReceipt (d40).

Provides cryptographic signing of evaluation receipts so downstream consumers
can verify the receipt was produced by a holder of the secret key.  Uses
HMAC-SHA256 from the standard library — no extra dependencies.

For asymmetric signing (Ed25519) wire up a separate key store; this module
covers the symmetric case which is sufficient for internal pipeline use.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Dict, Optional


_SIGNATURE_VERSION = "v1"


def _canonicalize_receipt(receipt_dict: Dict[str, Any]) -> bytes:
    """Produce a stable canonical serialization for signing.

    Drops any pre-existing ``signature`` / ``signature_version`` fields so a
    receipt can be re-signed without recursion.  Uses sorted keys + compact
    separators so byte-for-byte reproducibility is guaranteed.
    """
    cleaned = {k: v for k, v in receipt_dict.items()
               if k not in ("signature", "signature_version")}
    payload = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    return payload.encode("utf-8")


def sign_eval_receipt(
    receipt_dict: Dict[str, Any],
    secret_key: bytes,
) -> Dict[str, Any]:
    """Return a copy of ``receipt_dict`` with HMAC-SHA256 signature attached.

    Args:
        receipt_dict: An EvalReceipt.to_dict() output (or compatible mapping).
        secret_key:   Shared secret used for HMAC.  Use 32+ random bytes.

    Returns:
        A new dict identical to ``receipt_dict`` plus:
        - ``signature``:         hex-encoded HMAC-SHA256 of the canonical receipt.
        - ``signature_version``: signing scheme version string.

    Raises:
        ValueError: if ``secret_key`` is empty.
    """
    if not secret_key:
        raise ValueError("secret_key must be non-empty for signing")

    payload = _canonicalize_receipt(receipt_dict)
    digest = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()

    signed = dict(receipt_dict)
    signed["signature"] = digest
    signed["signature_version"] = _SIGNATURE_VERSION
    return signed


def verify_eval_receipt(
    receipt_dict: Dict[str, Any],
    secret_key: bytes,
) -> bool:
    """Verify a signed receipt's HMAC-SHA256 signature.

    Args:
        receipt_dict: Receipt as produced by ``sign_eval_receipt``.
        secret_key:   The same secret used to sign.

    Returns:
        True iff the signature matches and the version is supported.
    """
    if not secret_key:
        return False

    signature: Optional[str] = receipt_dict.get("signature")
    version: Optional[str] = receipt_dict.get("signature_version")
    if not signature or version != _SIGNATURE_VERSION:
        return False

    payload = _canonicalize_receipt(receipt_dict)
    expected = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def receipt_signature_fingerprint(signature_hex: str, length: int = 8) -> str:
    """Return a short, human-readable fingerprint for a signature hex string."""
    return signature_hex[:length] if signature_hex else ""


__all__ = [
    "sign_eval_receipt",
    "verify_eval_receipt",
    "receipt_signature_fingerprint",
]
