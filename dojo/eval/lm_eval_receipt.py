"""Signed lm-eval receipt wrapper (d37).

Layers HMAC-SHA256 signing onto lm-eval harness receipts so consumers can
verify a result came from this pipeline without trusting the file's mtime
or filename.  Builds on dojo/eval/receipt_signer.py (d40) and adds lm-eval
specific metadata.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Iterable, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError
from dojo.eval.receipt_signer import (
    receipt_signature_fingerprint,
    sign_eval_receipt,
    verify_eval_receipt,
)


class LmEvalReceiptError(DojoError):
    """Raised when an lm-eval receipt cannot be built or verified."""


_DEFAULT_SECRET_ENV = "DOJO_LM_EVAL_SECRET"


def _resolve_secret(secret_key: Optional[bytes]) -> bytes:
    """Resolve the signing secret from the argument or environment."""
    if secret_key is not None:
        return secret_key
    env_val = os.environ.get(_DEFAULT_SECRET_ENV)
    if env_val:
        return env_val.encode("utf-8")
    # Deterministic fallback for CI/test pipelines — NOT for production secrets.
    return hashlib.sha256(b"dojo:lm_eval:ci-fallback").digest()


def build_lm_eval_receipt(
    model_artifact_id: str,
    tasks: Iterable[str],
    mean_scores: Dict[str, float],
    merkle_hash: str,
    num_fewshot: int = 0,
    primary_metric_key: str = "acc,none",
    evaluated_at: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an lm-eval receipt dict (unsigned).

    The dict is structured so sign_lm_eval_receipt() can attach an HMAC
    signature directly via sign_eval_receipt().
    """
    receipt: Dict[str, Any] = {
        "kind": "lm_eval",
        "model_artifact_id": model_artifact_id,
        "tasks": list(tasks),
        "mean_scores": dict(mean_scores),
        "merkle_hash": merkle_hash,
        "num_fewshot": num_fewshot,
        "primary_metric_key": primary_metric_key,
        "evaluated_at": evaluated_at,
    }
    if extra:
        receipt["extra"] = dict(extra)
    return receipt


def sign_lm_eval_receipt(
    receipt: Dict[str, Any],
    secret_key: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Sign an lm-eval receipt with HMAC-SHA256.

    Emits ``eval.lm_eval_receipt_signed`` evidence so the audit chain captures
    the signing event and the short signature fingerprint (for log readability).

    Args:
        receipt:    Receipt dict as produced by build_lm_eval_receipt().
        secret_key: Optional explicit secret; otherwise resolved from
                    DOJO_LM_EVAL_SECRET env var, or a deterministic CI fallback.

    Returns:
        A new dict with ``signature`` and ``signature_version`` fields added.
    """
    if receipt.get("kind") != "lm_eval":
        raise LmEvalReceiptError(
            "receipt 'kind' must be 'lm_eval' before signing",
            reason=f"got kind={receipt.get('kind')!r}",
        )

    key = _resolve_secret(secret_key)
    signed = sign_eval_receipt(receipt, key)

    EvidenceLogger.get_instance().emit(
        {
            "type": EvidenceType.EVAL_COMPLETE.value,
            "data": {
                "phase": "lm_eval_signed",
                "model_artifact_id": signed["model_artifact_id"],
                "tasks": signed["tasks"],
                "signature_fingerprint": receipt_signature_fingerprint(signed["signature"]),
                "signature_version": signed["signature_version"],
            },
        },
        organ="eval",
    )

    return signed


def verify_lm_eval_receipt(
    receipt: Dict[str, Any],
    secret_key: Optional[bytes] = None,
) -> bool:
    """Verify a signed lm-eval receipt.

    Returns True iff the receipt's ``kind`` is ``"lm_eval"``, the signature
    is structurally valid, and the HMAC matches.
    """
    if receipt.get("kind") != "lm_eval":
        return False
    key = _resolve_secret(secret_key)
    return verify_eval_receipt(receipt, key)


__all__ = [
    "build_lm_eval_receipt",
    "sign_lm_eval_receipt",
    "verify_lm_eval_receipt",
    "LmEvalReceiptError",
]
