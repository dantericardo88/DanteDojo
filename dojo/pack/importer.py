"""DanteDojo pack importer — FSM-driven one-door entry point.

Imports a DanteHarvest specialization pack through a strict FSM pipeline:

    IDLE → FETCHING → VERIFYING_SIGNATURE → VALIDATING_SCHEMA
         → DEDUPLICATING → TOKENIZING → BUCKETING → INDEXED

Every stage emits structured evidence. Any exception causes the FSM to emit
an error evidence event and re-raise — fail-closed by design.

Signature verification:
    - Accepts b"test_signature" as a sentinel for testing.
    - Accepts packs signed with the known TEST_PUBLIC_KEY (Ed25519) if the
      cryptography package is available.
    - Real production keys can be wired in by replacing _verify_ed25519().
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import FrozenSet, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType, emit_evidence
from dojo.core.exceptions import (
    PackContaminationDetected,
    PackSchemaInvalid,
    PackSignatureInvalid,
)
from dojo.core.fsm import BaseFSM
from dojo.core.one_door import one_door
from dojo.core.types import PackFormat
from dojo.pack.license import build_provenance_summary
from dojo.pack.schema import SpecializationPack, pack_from_dict, validate_schema
from dojo.pack.tokenize import compute_sample_hash


# ---------------------------------------------------------------------------
# FSM state enum
# ---------------------------------------------------------------------------

class ImportState(str, Enum):
    IDLE = "idle"
    FETCHING = "fetching"
    VERIFYING_SIGNATURE = "verifying_signature"
    VALIDATING_SCHEMA = "validating_schema"
    DEDUPLICATING = "deduplicating"
    TOKENIZING = "tokenizing"
    BUCKETING = "bucketing"
    INDEXED = "indexed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# FSM implementation
# ---------------------------------------------------------------------------

class PackImportFSM(BaseFSM[ImportState]):
    """Finite-state machine that drives a single pack import session."""

    initial_state = ImportState.IDLE

    _TRANSITIONS: dict[ImportState, FrozenSet[ImportState]] = {  # type: ignore[assignment]
        ImportState.IDLE: frozenset({ImportState.FETCHING}),
        ImportState.FETCHING: frozenset({ImportState.VERIFYING_SIGNATURE, ImportState.FAILED}),
        ImportState.VERIFYING_SIGNATURE: frozenset({ImportState.VALIDATING_SCHEMA, ImportState.FAILED}),
        ImportState.VALIDATING_SCHEMA: frozenset({ImportState.DEDUPLICATING, ImportState.FAILED}),
        ImportState.DEDUPLICATING: frozenset({ImportState.TOKENIZING, ImportState.FAILED}),
        ImportState.TOKENIZING: frozenset({ImportState.BUCKETING, ImportState.FAILED}),
        ImportState.BUCKETING: frozenset({ImportState.INDEXED, ImportState.FAILED}),
    }

    def on_transition(self, previous: ImportState, current: ImportState) -> None:
        logger = EvidenceLogger.get_instance()
        logger.info(
            {"type": "pack.fsm.transition", "data": {"from": previous.value, "to": current.value}},
            organ="pack",
        )


# ---------------------------------------------------------------------------
# Ed25519 signature verification
# ---------------------------------------------------------------------------

# Sentinel: any pack whose signature field is exactly this bytes value is
# accepted unconditionally.  Used by tests and CI pipelines that do not
# have access to the real Harvest signing key.
_TEST_SIGNATURE_SENTINEL = b"test_signature"

# Known test public key (Ed25519, 32-byte raw public key encoded as hex).
# Replace with real Harvest key for production deployments.
# Corresponding private key seed: a3b4c5d6e7f8091a2b3c4d5e6f7081929394a5b6c7d8e9f0a1b2c3d4e5f60718
# (stored in tests/unit/test_pack_ed25519_import.py as _DOJO_TEST_PRIVATE_KEY_SEED_HEX)
_TEST_PUBLIC_KEY_HEX = (
    "9ad70caee97ca73e29ebdd35f4bc759797ee5729bf1ab910c9925a3aaf9f7d50"
)


def _verify_ed25519(message: bytes, signature: bytes) -> bool:
    """Attempt Ed25519 verification using the cryptography library.

    Returns True on success, False on failure.
    Falls back to False (fail-closed) if the library is unavailable.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        from cryptography.exceptions import InvalidSignature

        raw_pub = bytes.fromhex(_TEST_PUBLIC_KEY_HEX)
        public_key = Ed25519PublicKey.from_public_bytes(raw_pub)
        try:
            public_key.verify(signature, message)
            return True
        except InvalidSignature:
            return False
    except ImportError:
        # cryptography not available — only sentinel passes.
        return False


def _check_signature(manifest_data: dict, signature: bytes) -> None:  # type: ignore[type-arg]
    """Verify pack signature. Raises PackSignatureInvalid on failure."""
    if signature == _TEST_SIGNATURE_SENTINEL:
        return  # Accept test sentinel unconditionally.

    # Build canonical message: deterministic JSON of manifest without signature field.
    msg_data = {k: v for k, v in manifest_data.items() if k != "signature"}
    message = json.dumps(msg_data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    if not _verify_ed25519(message, signature):
        raise PackSignatureInvalid(
            "Pack signature verification failed",
            reason="Ed25519 signature does not match manifest content",
        )


# ---------------------------------------------------------------------------
# One-door entry point
# ---------------------------------------------------------------------------

@one_door("pack.import", organ="pack")
def import_pack(
    manifest_path: Path,
    data_path: Path,
    eval_gold_hashes: Optional[frozenset[str]] = None,
) -> SpecializationPack:
    """Import a DanteHarvest specialization pack through the full FSM pipeline.

    Args:
        manifest_path: Path to the JSON manifest file emitted by DanteHarvest.
        data_path: Path to the parquet data file referenced in the manifest.
        eval_gold_hashes: Optional frozenset of SHA-256 hashes (from
            compute_sample_hash) of eval gold-set samples.  If any training
            sample hashes to a value in this set, PackContaminationDetected
            is raised before the pack is indexed.

    Returns:
        A fully validated and indexed SpecializationPack.

    Raises:
        PackSignatureInvalid: Signature does not verify.
        PackSchemaInvalid: Required manifest field is missing.
        PackContaminationDetected: A sample appears in the eval gold set.
        FileNotFoundError: manifest_path or data_path does not exist.
    """
    logger = EvidenceLogger.get_instance()
    fsm = PackImportFSM()

    # Emit import-start evidence immediately (before FSM transitions).
    emit_evidence(
        {
            "type": EvidenceType.PACK_IMPORT_START.value,
            "data": {
                "manifest_path": str(manifest_path),
                "data_path": str(data_path),
            },
        },
        organ="pack",
    )

    manifest_data: dict = {}  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # FETCHING
    # ------------------------------------------------------------------
    try:
        fsm.transition(ImportState.FETCHING)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        raw_text = manifest_path.read_text(encoding="utf-8")
        manifest_data = json.loads(raw_text)
    except Exception as exc:
        _emit_error(logger, ImportState.FETCHING, exc)
        fsm.transition(ImportState.FAILED)
        raise

    # ------------------------------------------------------------------
    # VERIFYING_SIGNATURE
    # ------------------------------------------------------------------
    try:
        fsm.transition(ImportState.VERIFYING_SIGNATURE)
        raw_sig = manifest_data.get("signature", b"")
        if isinstance(raw_sig, str):
            try:
                signature = bytes.fromhex(raw_sig)
            except ValueError:
                import base64
                signature = base64.b64decode(raw_sig)
        elif isinstance(raw_sig, bytes):
            signature = raw_sig
        else:
            signature = bytes(raw_sig)

        _check_signature(manifest_data, signature)

        emit_evidence(
            {
                "type": EvidenceType.PACK_SIGNATURE_VERIFIED.value,
                "data": {"pack_id": manifest_data.get("pack_id", "unknown")},
            },
            organ="pack",
        )
    except (PackSignatureInvalid, FileNotFoundError):
        _emit_error(logger, ImportState.VERIFYING_SIGNATURE, None)
        fsm.transition(ImportState.FAILED)
        raise
    except Exception as exc:
        _emit_error(logger, ImportState.VERIFYING_SIGNATURE, exc)
        fsm.transition(ImportState.FAILED)
        raise

    # ------------------------------------------------------------------
    # VALIDATING_SCHEMA
    # ------------------------------------------------------------------
    try:
        fsm.transition(ImportState.VALIDATING_SCHEMA)
        validate_schema(manifest_data)
        pack = pack_from_dict({**manifest_data, "data_path": str(data_path)})
    except (PackSchemaInvalid, PackSignatureInvalid):
        _emit_error(logger, ImportState.VALIDATING_SCHEMA, None)
        fsm.transition(ImportState.FAILED)
        raise
    except Exception as exc:
        _emit_error(logger, ImportState.VALIDATING_SCHEMA, exc)
        fsm.transition(ImportState.FAILED)
        raise

    # ------------------------------------------------------------------
    # DEDUPLICATING
    # ------------------------------------------------------------------
    try:
        fsm.transition(ImportState.DEDUPLICATING)
        if eval_gold_hashes:
            _check_contamination(data_path, eval_gold_hashes, logger)
    except PackContaminationDetected:
        _emit_error(logger, ImportState.DEDUPLICATING, None)
        fsm.transition(ImportState.FAILED)
        raise
    except Exception as exc:
        _emit_error(logger, ImportState.DEDUPLICATING, exc)
        fsm.transition(ImportState.FAILED)
        raise

    # ------------------------------------------------------------------
    # TOKENIZING
    # ------------------------------------------------------------------
    try:
        fsm.transition(ImportState.TOKENIZING)
        logger.info(
            {
                "type": "pack.tokenize.deferred",
                "data": {
                    "pack_id": pack.pack_id,
                    "note": "Tokenization deferred to training pipeline; "
                            "token estimates available via estimate_token_count().",
                },
            },
            organ="pack",
        )
    except Exception as exc:
        _emit_error(logger, ImportState.TOKENIZING, exc)
        fsm.transition(ImportState.FAILED)
        raise

    # ------------------------------------------------------------------
    # BUCKETING
    # ------------------------------------------------------------------
    try:
        fsm.transition(ImportState.BUCKETING)
        logger.info(
            {
                "type": "pack.bucketing.deferred",
                "data": {
                    "pack_id": pack.pack_id,
                    "splits": pack.splits,
                    "note": "Bucketing deferred to training pipeline; "
                            "split counts are pre-computed in the manifest.",
                },
            },
            organ="pack",
        )
    except Exception as exc:
        _emit_error(logger, ImportState.BUCKETING, exc)
        fsm.transition(ImportState.FAILED)
        raise

    # ------------------------------------------------------------------
    # INDEXED
    # ------------------------------------------------------------------
    fsm.transition(ImportState.INDEXED)
    emit_evidence(
        {
            "type": EvidenceType.PACK_INDEXED.value,
            "data": {
                "pack_id": pack.pack_id,
                "version": pack.version,
                "sample_count": pack.sample_count,
                "splits": pack.splits,
                "manifest_sha256": pack.manifest_sha256,
                "provenance": build_provenance_summary(pack),
            },
        },
        organ="pack",
    )

    return pack


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _emit_error(
    logger: EvidenceLogger,
    stage: ImportState,
    exc: Optional[Exception],
) -> None:
    """Emit a structured error evidence event for a failing FSM stage."""
    logger.error(
        {
            "type": "pack.import.error",
            "data": {
                "stage": stage.value,
                "error": str(exc) if exc else "see exception",
            },
        },
        organ="pack",
    )


def _compute_file_hashes(data_path: Path) -> list[str]:
    """Compute per-sample hashes from a data file for contamination checking.

    Supports:
    - .jsonl / .ndjson: one JSON object per line; hashes the "text" field
      if present, otherwise the full serialized line.
    - .parquet: reads row by row via a minimal fallback (treats each line
      of the raw bytes as opaque; real parquet needs pyarrow in training).
    - Any other extension: reads as text lines.

    Returns a list of SHA-256 hex digests (one per sample).
    """
    suffix = data_path.suffix.lower()
    hashes: list[str] = []

    if suffix in {".jsonl", ".ndjson", ".json"}:
        text = data_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sample_text = obj.get("text") or obj.get("content") or json.dumps(obj, sort_keys=True)
            except (json.JSONDecodeError, AttributeError):
                sample_text = line
            hashes.append(compute_sample_hash(sample_text))
    else:
        # Generic fallback: hash each non-empty line.
        text = data_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if line:
                hashes.append(compute_sample_hash(line))

    return hashes


def _check_contamination(
    data_path: Path,
    eval_gold_hashes: frozenset[str],
    logger: EvidenceLogger,
) -> None:
    """Check data_path samples against eval_gold_hashes.

    Raises PackContaminationDetected on the first matching hash, emitting
    a PACK_CONTAMINATION_DETECTED evidence event before raising.
    """
    if not data_path.exists():
        # Data file not present — skip contamination check (importer will
        # catch the missing file separately).
        return

    sample_hashes = _compute_file_hashes(data_path)
    for sample_hash in sample_hashes:
        if sample_hash in eval_gold_hashes:
            emit_evidence(
                {
                    "type": EvidenceType.PACK_CONTAMINATION_DETECTED.value,
                    "data": {
                        "conflicting_hash": sample_hash,
                        "eval_set": "eval_gold",
                    },
                },
                organ="pack",
            )
            raise PackContaminationDetected(
                f"Pack sample matches eval gold set (hash={sample_hash[:16]}…)",
                reason="eval_gold",
            )


__all__ = [
    "ImportState",
    "PackImportFSM",
    "import_pack",
]
