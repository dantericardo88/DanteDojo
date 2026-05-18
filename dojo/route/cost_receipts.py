"""Signed cost receipts for routing decisions (d35).

The inspector flagged d35 with "JSONL real" — receipts were written but
unsigned, so downstream auditors had to trust the writer.  This module
adds:
  - structured CostReceipt dict builder
  - HMAC-SHA256 signing reusing dojo.eval.receipt_signer
  - verification + tamper detection
  - append-only JSONL log with signature column
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError
from dojo.eval.receipt_signer import sign_eval_receipt, verify_eval_receipt


class CostReceiptError(DojoError):
    """Raised when a cost receipt cannot be built, signed, or verified."""


_DEFAULT_SECRET_ENV = "DOJO_COST_RECEIPT_SECRET"


def _resolve_secret(secret: Optional[bytes]) -> bytes:
    if secret is not None:
        return secret
    env = os.environ.get(_DEFAULT_SECRET_ENV)
    if env:
        return env.encode("utf-8")
    import hashlib
    return hashlib.sha256(b"dojo:cost:ci-fallback").digest()


@dataclass
class CostReceipt:
    """Immutable record of a single inference request's cost.

    Constructed by the router; signed before being written to JSONL.
    """

    request_id: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    usd_cost: float
    privacy_tier: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["kind"] = "cost_receipt"
        return d


def build_cost_receipt(
    request_id: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    usd_cost: float,
    privacy_tier: str = "shared",
) -> Dict:
    """Build an unsigned cost receipt dict."""
    if prompt_tokens < 0 or completion_tokens < 0:
        raise CostReceiptError(
            "token counts must be non-negative",
            reason=f"prompt={prompt_tokens}, completion={completion_tokens}",
        )
    if usd_cost < 0:
        raise CostReceiptError(
            "usd_cost must be non-negative",
            reason=f"usd_cost={usd_cost}",
        )
    receipt = CostReceipt(
        request_id=request_id,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usd_cost=usd_cost,
        privacy_tier=privacy_tier,
    )
    return receipt.to_dict()


def sign_cost_receipt(receipt: Dict, secret: Optional[bytes] = None) -> Dict:
    """HMAC-SHA256 sign a cost receipt, returning a new dict with signature."""
    if receipt.get("kind") != "cost_receipt":
        raise CostReceiptError(
            "receipt 'kind' must be 'cost_receipt'",
            reason=f"got {receipt.get('kind')!r}",
        )
    key = _resolve_secret(secret)
    signed = sign_eval_receipt(receipt, key)

    EvidenceLogger.get_instance().emit(
        {
            "type": "route.cost_receipt_signed",
            "data": {
                "request_id": signed.get("request_id"),
                "provider": signed.get("provider"),
                "usd_cost": signed.get("usd_cost"),
                "signature_version": signed.get("signature_version"),
            },
        },
        organ="route",
    )

    return signed


def verify_cost_receipt(receipt: Dict, secret: Optional[bytes] = None) -> bool:
    """Verify a signed cost receipt; returns True iff signature checks out."""
    if receipt.get("kind") != "cost_receipt":
        return False
    key = _resolve_secret(secret)
    return verify_eval_receipt(receipt, key)


def append_signed_receipt(
    receipt: Dict,
    log_path: Path | str,
    secret: Optional[bytes] = None,
) -> Dict:
    """Sign a receipt and append it as a single JSONL line.

    Creates the file (and parent directory) if needed.  Returns the signed
    receipt so callers can also stash it in memory.
    """
    signed = sign_cost_receipt(receipt, secret=secret)
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(signed, separators=(",", ":")) + "\n")
    return signed


def verify_receipt_log(
    log_path: Path | str,
    secret: Optional[bytes] = None,
) -> List[Dict]:
    """Verify every receipt in a JSONL log file.

    Returns:
        List of dicts: ``[{"line": int, "valid": bool, "receipt": dict}, ...]``
    """
    p = Path(log_path)
    if not p.exists():
        return []
    results: List[Dict] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            receipt = json.loads(line)
        except json.JSONDecodeError:
            results.append({"line": i, "valid": False, "receipt": None, "error": "json"})
            continue
        valid = verify_cost_receipt(receipt, secret=secret)
        results.append({"line": i, "valid": valid, "receipt": receipt})
    return results


__all__ = [
    "CostReceipt",
    "CostReceiptError",
    "build_cost_receipt",
    "sign_cost_receipt",
    "verify_cost_receipt",
    "append_signed_receipt",
    "verify_receipt_log",
]
