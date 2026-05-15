"""DanteDojo pack license and provenance enforcement.

Provides three integrity/compliance utilities:

- verify_data_integrity(pack, data_bytes) — SHA-256 tamper detection
- verify_manifest_integrity(pack) — manifest_sha256 format check
- build_provenance_summary(pack) — extract ProvenanceManifest fields
- check_license_allows_training(pack) — enforce allows_derived_ft gate
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

from dojo.core.evidence import EvidenceType, emit_evidence
from dojo.core.exceptions import PackLicenseRejected, PackSignatureInvalid
from dojo.pack.schema import SpecializationPack


# ---------------------------------------------------------------------------
# d47 — Data integrity verification
# ---------------------------------------------------------------------------

_HEX64_RE = re.compile(r'^[0-9a-fA-F]{64}$')


def verify_data_integrity(pack: SpecializationPack, data_bytes: bytes) -> bool:
    """Compute SHA-256 of *data_bytes* and compare to pack.data_parquet_sha256.

    Raises PackSignatureInvalid if the digest does not match.
    Emits PACK_SIGNATURE_VERIFIED evidence with check_type="data_sha256".

    Returns True on success.
    """
    actual_digest = hashlib.sha256(data_bytes).hexdigest()
    expected = pack.data_parquet_sha256.lower()

    emit_evidence(
        {
            "type": EvidenceType.PACK_SIGNATURE_VERIFIED.value,
            "data": {
                "pack_id": pack.pack_id,
                "check_type": "data_sha256",
                "expected": expected,
                "actual": actual_digest,
                "match": actual_digest == expected,
            },
        },
        organ="pack",
    )

    if actual_digest != expected:
        raise PackSignatureInvalid(
            f"Data integrity check failed for pack '{pack.pack_id}': "
            f"expected {expected[:16]}… got {actual_digest[:16]}…",
            reason="data_parquet_sha256 mismatch",
        )

    return True


def verify_manifest_integrity(pack: SpecializationPack) -> bool:
    """Verify pack.manifest_sha256 is a well-formed 64-char hex string.

    Raises PackSignatureInvalid if the value is empty or not valid hex.
    Emits PACK_SIGNATURE_VERIFIED evidence with check_type="manifest_sha256".

    Returns True on success.
    """
    value = pack.manifest_sha256

    is_valid = bool(value) and bool(_HEX64_RE.match(value))

    emit_evidence(
        {
            "type": EvidenceType.PACK_SIGNATURE_VERIFIED.value,
            "data": {
                "pack_id": pack.pack_id,
                "check_type": "manifest_sha256",
                "manifest_sha256": value,
                "valid": is_valid,
            },
        },
        organ="pack",
    )

    if not is_valid:
        raise PackSignatureInvalid(
            f"manifest_sha256 is invalid for pack '{pack.pack_id}': "
            f"expected a 64-char hex string, got '{value}'",
            reason="manifest_sha256_invalid",
        )

    return True


# ---------------------------------------------------------------------------
# d47 — Provenance summary extraction
# ---------------------------------------------------------------------------

def build_provenance_summary(pack: SpecializationPack) -> Dict[str, Any]:
    """Extract key provenance fields from pack.provenance as a plain dict.

    Returns a dict with keys:
        harvest_organ_version, sources, collection_date_utc,
        dedup_method, human_reviewed.

    This dict is intended for embedding in PACK_INDEXED evidence.
    """
    prov = pack.provenance
    return {
        "harvest_organ_version": prov.harvest_organ_version,
        "sources": list(prov.sources),
        "collection_date_utc": prov.collection_date_utc,
        "dedup_method": prov.dedup_method,
        "human_reviewed": prov.human_reviewed,
    }


# ---------------------------------------------------------------------------
# d48 — License enforcement
# ---------------------------------------------------------------------------

def check_license_allows_training(pack: SpecializationPack) -> None:
    """Gate fine-tuning on pack.license.allows_derived_ft.

    Emits PACK_LICENSE_CHECKED evidence regardless of outcome.
    Raises PackLicenseRejected if allows_derived_ft is False.
    """
    allows = pack.license.allows_derived_ft
    spdx_id = pack.license.spdx_id

    emit_evidence(
        {
            "type": EvidenceType.PACK_LICENSE_CHECKED.value,
            "data": {
                "pack_id": pack.pack_id,
                "license_check": "training",
                "allows_derived_ft": allows,
                "spdx_id": spdx_id,
            },
        },
        organ="pack",
    )

    if not allows:
        raise PackLicenseRejected(
            f"Pack '{pack.pack_id}' license '{spdx_id}' does not allow "
            "derived fine-tuned model redistribution (allows_derived_ft=False)",
            reason=f"spdx_id={spdx_id}",
        )


__all__ = [
    "build_provenance_summary",
    "check_license_allows_training",
    "verify_data_integrity",
    "verify_manifest_integrity",
]
