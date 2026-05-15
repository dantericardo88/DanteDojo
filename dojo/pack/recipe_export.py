"""DanteDojo recipe export — serialise a SpecializationPack as a portable recipe.

A "recipe" is a compact, portable JSON descriptor that captures everything
needed to reproduce a training run: pack identity, schema, provenance, license,
split counts, and config hash.  It does NOT include raw data — only metadata.

Provenance is embedded verbatim so downstream tooling can verify lineage without
accessing the original DanteHarvest instance.

Functions:
    export_recipe(pack, config_hash, output_path) -> Path
    load_recipe(path) -> dict
    validate_recipe(data) -> None
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.pack.schema import SpecializationPack


# ---------------------------------------------------------------------------
# Required recipe schema fields
# ---------------------------------------------------------------------------

RECIPE_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "recipe_version",
    "pack_id",
    "pack_version",
    "schema_version",
    "sample_count",
    "format",
    "splits",
    "harvest_receipt",
    "license",
    "provenance",
    "manifest_sha256",
    "data_parquet_sha256",
    "recipe_sha256",
})

RECIPE_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recipe_sha256(data: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 over the recipe body (excluding recipe_sha256)."""
    body = {k: v for k, v in data.items() if k != "recipe_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_recipe(
    pack: SpecializationPack,
    output_path: Path,
    config_hash: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialise *pack* as a portable JSON recipe file.

    The recipe captures all pack metadata (identity, schema, provenance,
    license, splits) plus an optional ``config_hash`` linking the recipe to
    the FinetuneConfig that consumed it.  Raw data is never included.

    A ``recipe_sha256`` field is appended as a tamper-evident seal.

    Args:
        pack:        Source SpecializationPack.
        output_path: Destination file path; parent dirs are created if absent.
        config_hash: Optional FinetuneConfig.config_hash() for traceability.
        metadata:    Optional extra key/value pairs embedded under ``_meta``.

    Returns:
        Resolved output_path.
    """
    recipe: Dict[str, Any] = {
        "recipe_version": RECIPE_VERSION,
        "pack_id": pack.pack_id,
        "pack_version": pack.version,
        "schema_version": pack.schema_version,
        "sample_count": pack.sample_count,
        "format": pack.format.value,
        "splits": dict(pack.splits),
        "harvest_receipt": pack.harvest_receipt,
        "license": pack.license.to_dict(),
        "provenance": pack.provenance.to_dict(),
        "manifest_sha256": pack.manifest_sha256,
        "data_parquet_sha256": pack.data_parquet_sha256,
    }

    if config_hash:
        recipe["config_hash"] = config_hash

    if metadata:
        recipe["_meta"] = metadata

    recipe["recipe_sha256"] = _recipe_sha256(recipe)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(recipe, fh, indent=2, sort_keys=True)

    EvidenceLogger.get_instance().emit(
        {
            "type": "pack.recipe.exported",
            "data": {
                "pack_id": pack.pack_id,
                "pack_version": pack.version,
                "recipe_path": str(output_path),
                "recipe_sha256": recipe["recipe_sha256"],
                "config_hash": config_hash or "",
            },
        },
        organ="pack",
    )

    return output_path


def load_recipe(path: Path) -> Dict[str, Any]:
    """Load a recipe JSON file and return the raw dict (no validation).

    Raises:
        FileNotFoundError: if path does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_recipe(data: Dict[str, Any]) -> None:
    """Validate a recipe dict against RECIPE_REQUIRED_FIELDS and verify recipe_sha256.

    Raises:
        ValueError: if required fields are missing or recipe_sha256 mismatches.
    """
    missing = RECIPE_REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(
            f"Recipe missing required fields: {sorted(missing)}"
        )

    expected_sha = _recipe_sha256(data)
    if data.get("recipe_sha256") != expected_sha:
        raise ValueError(
            f"recipe_sha256 mismatch: stored {data.get('recipe_sha256')!r} "
            f"!= computed {expected_sha!r}"
        )


__all__ = [
    "RECIPE_REQUIRED_FIELDS",
    "RECIPE_VERSION",
    "export_recipe",
    "load_recipe",
    "validate_recipe",
]
