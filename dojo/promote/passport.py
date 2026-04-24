"""DanteDojo promotion gate — ConstitutionalPassport builder.

Assembles a ConstitutionalPassport from the three gate receipts.
merkle_root = seal_passport() over all receipts + artifact fingerprint.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from dojo.core.types import ConstitutionalPassport

if TYPE_CHECKING:
    from dojo.promote.gate import GateCheckResult


def build_passport(
    candidate: "object",  # ModelArtifact — avoid circular import at module level
    harvest_receipt: "GateCheckResult",
    dojo_receipt: "GateCheckResult",
    forge_receipt: "GateCheckResult",
    harvest_receipt_id: str,
) -> ConstitutionalPassport:
    """Build a ConstitutionalPassport from the three gate receipts.

    merkle_root = SHA-256 Merkle tree of (harvest_receipt + dojo_receipt +
                  forge_receipt + artifact fingerprint).
    expires_at = issued_at + 90 days.
    revocation_url = "http://127.0.0.1:8002/v1/passport/revoke".
    """
    from dojo.core.types import ModelArtifact  # noqa: PLC0415
    from dojo.promote.sealer import seal_passport  # noqa: PLC0415

    assert isinstance(candidate, ModelArtifact), "candidate must be a ModelArtifact"

    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(days=90)
    passport_id = str(uuid.uuid4())

    # Build a preliminary passport to compute the seal (merkle_root).
    # We use a placeholder merkle_root first, then recompute.
    preliminary = ConstitutionalPassport(
        passport_id=passport_id,
        model_artifact_id=candidate.artifact_id,
        harvest_receipt_id=harvest_receipt_id,
        harvest_check_receipt=harvest_receipt.receipt,
        dojo_check_receipt=dojo_receipt.receipt,
        forge_check_receipt=forge_receipt.receipt,
        merkle_root="",  # placeholder
        issued_at=issued_at,
        expires_at=expires_at,
        revocation_url="http://127.0.0.1:8002/v1/passport/revoke",
    )

    # Compute the real seal
    merkle_root = seal_passport(preliminary)

    # Rebuild with the real merkle_root
    passport = ConstitutionalPassport(
        passport_id=passport_id,
        model_artifact_id=candidate.artifact_id,
        harvest_receipt_id=harvest_receipt_id,
        harvest_check_receipt=harvest_receipt.receipt,
        dojo_check_receipt=dojo_receipt.receipt,
        forge_check_receipt=forge_receipt.receipt,
        merkle_root=merkle_root,
        issued_at=issued_at,
        expires_at=expires_at,
        revocation_url="http://127.0.0.1:8002/v1/passport/revoke",
    )

    return passport
