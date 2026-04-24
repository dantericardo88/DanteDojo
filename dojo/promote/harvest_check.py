"""DanteDojo promotion gate — Harvest provenance check.

Verifies that a DanteHarvest receipt is valid, not revoked, and that the
associated pack license allows derived fine-tuned model redistribution.

Fail-closed: any unexpected error → BLOCKED, never GREEN.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

from dojo.core.evidence import EvidenceLogger, EvidenceType

if TYPE_CHECKING:
    from dojo.promote.gate import GateCheckResult


def _make_receipt(gate: str, passed: bool, reasons: list[str]) -> bytes:
    payload = json.dumps(
        {"gate": gate, "passed": passed, "reasons": reasons, "ts": time.time()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).digest()


def check_harvest(
    harvest_receipt_id: str,
    pack_license_allows_ft: bool = True,
    *,
    harvest_client: Any = None,
) -> "GateCheckResult":
    """Verify Harvest provenance receipt.

    Checks:
    - Receipt not revoked (mock: revoked if receipt_id starts with "revoked_")
    - License allows derived fine-tuned model redistribution
    - Pack signature still valid (mock: always valid unless revoked)

    Returns GateCheckResult. Emits PROMOTE_HARVEST_GREEN or PROMOTE_HARVEST_BLOCKED.
    """
    # Deferred import to avoid circular dependency with gate.py
    from dojo.promote.gate import GateCheckResult  # noqa: PLC0415

    logger = EvidenceLogger.get_instance()
    reasons: list[str] = []
    gate = "harvest_check"

    try:
        # --- Revocation check ---
        revoked = False
        if harvest_client is not None:
            try:
                revoked = harvest_client.check_revocation(harvest_receipt_id)
            except Exception as exc:
                reasons.append(f"Harvest client error during revocation check: {exc}")
                revoked = True  # fail-closed
        else:
            # Mock: revoked if receipt_id starts with "revoked_"
            revoked = harvest_receipt_id.startswith("revoked_")

        if revoked:
            reasons.append(f"Receipt '{harvest_receipt_id}' has been revoked")

        # --- License check ---
        if not pack_license_allows_ft:
            reasons.append("Pack license does not allow derived fine-tuned model redistribution")

        # --- Pack signature check (mock: valid unless revoked) ---
        if harvest_client is not None and not revoked:
            try:
                receipt_data = harvest_client.verify_receipt(harvest_receipt_id)
                if not receipt_data.get("valid", False):
                    reasons.append("Pack signature is no longer valid per Harvest receipt")
            except Exception as exc:
                reasons.append(f"Pack signature verification failed: {exc}")
        # Mock: signature valid unless revoked (already handled above)

    except Exception as exc:
        reasons.append(f"Unexpected error in harvest_check: {exc}")

    passed = len(reasons) == 0
    receipt = _make_receipt(gate, passed, reasons)

    result = GateCheckResult(gate=gate, passed=passed, reasons=reasons, receipt=receipt)

    ev_type = EvidenceType.PROMOTE_HARVEST_GREEN if passed else EvidenceType.PROMOTE_HARVEST_BLOCKED
    logger.emit(
        {
            "type": ev_type.value,
            "data": {
                "gate": gate,
                "passed": passed,
                "harvest_receipt_id": harvest_receipt_id,
                "reasons": reasons,
            },
        },
        organ="promote",
    )

    return result
