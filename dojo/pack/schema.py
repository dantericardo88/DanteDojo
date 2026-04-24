"""DanteDojo pack schema — SpecializationPack dataclass and validation.

Defines the canonical schema for a DanteHarvest specialization pack manifest,
validation logic, and the pack_from_dict constructor. Fail-closed: any missing
required field raises PackSchemaInvalid immediately with the field name.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dojo.core.exceptions import PackSchemaInvalid
from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest


REQUIRED_FIELDS: frozenset[str] = frozenset({
    "pack_id",
    "version",
    "schema_version",
    "sample_count",
    "format",
    "splits",
    "harvest_receipt",
    "license",
    "provenance",
    "signature",
    "manifest_sha256",
    "data_parquet_sha256",
})


@dataclass(frozen=True)
class SpecializationPack:
    """Immutable descriptor for an ingested DanteHarvest specialization pack.

    All fields are validated at construction time via pack_from_dict or
    validate_schema.  data_path is set by the importer after the pack is
    written to local storage.
    """

    pack_id: str                   # e.g. "empanada-bookkeeping"
    version: str                   # semver string, e.g. "1.0.0"
    schema_version: str            # pack schema version, e.g. "1.0"
    sample_count: int
    format: PackFormat
    splits: Dict[str, int]         # {"train": N, "val": M, "held_out_gold": K}
    harvest_receipt: str           # Merkle root from DanteHarvest
    license: LicenseSpec
    provenance: ProvenanceManifest
    signature: bytes               # Ed25519 signature by Harvest key
    manifest_sha256: str
    data_parquet_sha256: str
    data_path: Optional[Path] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "version": self.version,
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "format": self.format.value,
            "splits": dict(self.splits),
            "harvest_receipt": self.harvest_receipt,
            "license": self.license.to_dict(),
            "provenance": self.provenance.to_dict(),
            "signature": base64.b64encode(self.signature).decode("ascii"),
            "manifest_sha256": self.manifest_sha256,
            "data_parquet_sha256": self.data_parquet_sha256,
            "data_path": str(self.data_path) if self.data_path else None,
        }


def validate_schema(data: dict) -> None:  # type: ignore[type-arg]
    """Raise PackSchemaInvalid if any REQUIRED_FIELDS are absent from *data*.

    The exception message names the first missing field so callers and tests
    can assert on a specific field name.
    """
    for field_name in sorted(REQUIRED_FIELDS):
        if field_name not in data:
            raise PackSchemaInvalid(
                f"Pack manifest missing required field: '{field_name}'",
                reason=field_name,
            )


def pack_from_dict(data: dict) -> SpecializationPack:  # type: ignore[type-arg]
    """Parse *data* (from a manifest JSON) and return a SpecializationPack.

    Calls validate_schema first, then constructs the typed dataclass.
    """
    validate_schema(data)

    raw_license = data["license"]
    license_spec = LicenseSpec(
        spdx_id=raw_license["spdx_id"],
        allows_derived_ft=bool(raw_license["allows_derived_ft"]),
        allows_commercial_use=bool(raw_license["allows_commercial_use"]),
        requires_attribution=bool(raw_license["requires_attribution"]),
    )

    raw_prov = data["provenance"]
    provenance = ProvenanceManifest(
        harvest_organ_version=raw_prov["harvest_organ_version"],
        sources=tuple(raw_prov["sources"]),
        collection_date_utc=raw_prov["collection_date_utc"],
        dedup_method=raw_prov["dedup_method"],
        human_reviewed=bool(raw_prov["human_reviewed"]),
    )

    raw_sig = data["signature"]
    if isinstance(raw_sig, bytes):
        signature = raw_sig
    elif isinstance(raw_sig, str):
        # Accept both hex and base64 encoded signatures from JSON
        try:
            signature = bytes.fromhex(raw_sig)
        except ValueError:
            signature = base64.b64decode(raw_sig)
    else:
        signature = bytes(raw_sig)

    raw_format = data["format"]
    pack_format = PackFormat(raw_format) if isinstance(raw_format, str) else raw_format

    data_path_raw = data.get("data_path")
    data_path = Path(data_path_raw) if data_path_raw else None

    return SpecializationPack(
        pack_id=str(data["pack_id"]),
        version=str(data["version"]),
        schema_version=str(data["schema_version"]),
        sample_count=int(data["sample_count"]),
        format=pack_format,
        splits=dict(data["splits"]),
        harvest_receipt=str(data["harvest_receipt"]),
        license=license_spec,
        provenance=provenance,
        signature=signature,
        manifest_sha256=str(data["manifest_sha256"]),
        data_parquet_sha256=str(data["data_parquet_sha256"]),
        data_path=data_path,
    )


__all__ = [
    "REQUIRED_FIELDS",
    "SpecializationPack",
    "pack_from_dict",
    "validate_schema",
]
