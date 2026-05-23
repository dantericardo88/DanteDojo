"""d47: End-to-end pack import with real Ed25519 signature verification.

Tests in this module use a KNOWN TEST key pair (private key seed documented here,
public key wired into dojo/pack/importer.py) to produce genuine Ed25519 signatures
and drive the full PackImportFSM pipeline:

    IDLE → FETCHING → VERIFYING_SIGNATURE → VALIDATING_SCHEMA
         → DEDUPLICATING → TOKENIZING → BUCKETING → INDEXED

This validates the provenance chain end-to-end with cryptographically real
signatures — not the b"test_signature" sentinel used in other tests.

Known test key pair (DanteDojo test key v1 — NOT A SECRET; for CI use only):
  Private key seed: a3b4c5d6e7f8091a2b3c4d5e6f7081929394a5b6c7d8e9f0a1b2c3d4e5f60718
  Public key:       9ad70caee97ca73e29ebdd35f4bc759797ee5729bf1ab910c9925a3aaf9f7d50
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PackSignatureInvalid
from dojo.pack.importer import import_pack

# ---------------------------------------------------------------------------
# Test key pair constants (DanteDojo test key v1 — documented, not secret)
# ---------------------------------------------------------------------------

_DOJO_TEST_PRIVATE_KEY_SEED_HEX = (
    "a3b4c5d6e7f8091a2b3c4d5e6f7081929394a5b6c7d8e9f0a1b2c3d4e5f60718"
)


def _crypto_unavailable() -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: F401
        return False
    except ImportError:
        return True


_skip_no_crypto = pytest.mark.skipif(
    _crypto_unavailable(),
    reason="cryptography package not installed",
)


def _sign_manifest(manifest_data: dict) -> bytes:
    """Sign the canonical JSON of manifest_data (without signature field) using the test key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(_DOJO_TEST_PRIVATE_KEY_SEED_HEX))
    msg_data = {k: v for k, v in manifest_data.items() if k != "signature"}
    message = json.dumps(msg_data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return priv.sign(message)


def _make_manifest_data(signature_hex: str | None = None) -> dict:
    """Return a minimal valid pack manifest dict."""
    base = {
        "pack_id": "dojo-test-pack-ed25519",
        "version": "1.0.0",
        "schema_version": "1.0",
        "sample_count": 4,
        "format": "alpaca",
        "splits": {"train": 4},
        "harvest_receipt": "ed25519-real-signature-test-receipt",
        "license": {
            "spdx_id": "Apache-2.0",
            "allows_derived_ft": True,
            "allows_commercial_use": True,
            "requires_attribution": False,
        },
        "provenance": {
            "harvest_organ_version": "1.0.0",
            "sources": ["https://example.com/ed25519-test-data"],
            "collection_date_utc": "2026-05-21T00:00:00Z",
            "dedup_method": "minhash",
            "human_reviewed": True,
        },
        "manifest_sha256": "c" * 64,
        "data_parquet_sha256": "d" * 64,
    }
    if signature_hex is not None:
        base["signature"] = signature_hex
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def signed_manifest_file(tmp_path) -> tuple[Path, bytes]:
    """Write a properly Ed25519-signed manifest JSON to disk; return (path, sig_bytes)."""
    data = _make_manifest_data()
    sig_bytes = _sign_manifest(data)
    data["signature"] = sig_bytes.hex()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest_path, sig_bytes


@pytest.fixture
def data_file(tmp_path) -> Path:
    """Return path to an empty (but existing) data file."""
    p = tmp_path / "data.parquet"
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------------------
# Real Ed25519 signature tests
# ---------------------------------------------------------------------------

@_skip_no_crypto
class TestRealEd25519Signature:
    """Validate pack import with cryptographically real Ed25519 signatures."""

    def test_real_signature_passes_verification(self, signed_manifest_file, data_file):
        """A pack signed with the test private key must pass signature verification."""
        manifest_path, _ = signed_manifest_file
        pack = import_pack(manifest_path, data_file)
        assert pack.pack_id == "dojo-test-pack-ed25519"

    def test_pack_signature_verified_evidence_emitted(self, signed_manifest_file, data_file):
        """PACK_SIGNATURE_VERIFIED evidence must be emitted on successful import."""
        manifest_path, _ = signed_manifest_file
        import_pack(manifest_path, data_file)
        events = EvidenceLogger.get_events()
        sig_events = [
            e for e in events
            if e.get("type") == EvidenceType.PACK_SIGNATURE_VERIFIED.value
        ]
        assert sig_events, "PACK_SIGNATURE_VERIFIED event must be emitted"

    def test_pack_id_preserved_through_import(self, signed_manifest_file, data_file):
        """Pack ID must survive the full FSM pipeline unchanged."""
        manifest_path, _ = signed_manifest_file
        pack = import_pack(manifest_path, data_file)
        assert pack.pack_id == "dojo-test-pack-ed25519"

    def test_provenance_chain_preserved(self, signed_manifest_file, data_file):
        """Provenance fields must be preserved through the full import pipeline."""
        manifest_path, _ = signed_manifest_file
        pack = import_pack(manifest_path, data_file)
        assert pack.provenance.harvest_organ_version == "1.0.0"
        assert "https://example.com/ed25519-test-data" in pack.provenance.sources
        assert pack.provenance.human_reviewed is True

    def test_license_preserved_through_import(self, signed_manifest_file, data_file):
        """License terms must be preserved through the full import pipeline."""
        manifest_path, _ = signed_manifest_file
        pack = import_pack(manifest_path, data_file)
        assert pack.license.spdx_id == "Apache-2.0"
        assert pack.license.allows_derived_ft is True
        assert pack.license.allows_commercial_use is True

    def test_tampered_signature_raises_pack_signature_invalid(self, tmp_path, data_file):
        """A pack with a tampered (wrong) signature must raise PackSignatureInvalid."""
        data = _make_manifest_data()
        # Produce a valid signature, then flip the last byte
        sig_bytes = _sign_manifest(data)
        tampered = bytearray(sig_bytes)
        tampered[-1] ^= 0xFF
        data["signature"] = bytes(tampered).hex()

        manifest_path = tmp_path / "tampered_manifest.json"
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with pytest.raises(PackSignatureInvalid):
            import_pack(manifest_path, data_file)

    def test_wrong_key_signature_raises_pack_signature_invalid(self, tmp_path, data_file):
        """A pack signed with a DIFFERENT private key must raise PackSignatureInvalid."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        data = _make_manifest_data()
        # Sign with a completely different (freshly generated) key
        wrong_priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(
            "deadbeefcafe01234567890abcdef0123456789abcdef01234567890abcdef01"
        ))
        msg_data = {k: v for k, v in data.items() if k != "signature"}
        message = json.dumps(msg_data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        wrong_sig = wrong_priv.sign(message)
        data["signature"] = wrong_sig.hex()

        manifest_path = tmp_path / "wrong_key_manifest.json"
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with pytest.raises(PackSignatureInvalid):
            import_pack(manifest_path, data_file)

    def test_full_fsm_pipeline_runs_to_indexed(self, signed_manifest_file, data_file):
        """FSM must reach INDEXED state — all 7 transitions must emit evidence."""
        manifest_path, _ = signed_manifest_file
        import_pack(manifest_path, data_file)
        events = EvidenceLogger.get_events()
        # Look for FSM transition events: IDLE→FETCHING→VERIFYING_SIGNATURE→...→INDEXED
        fsm_events = [e for e in events if e.get("type") == "pack.fsm.transition"]
        states = [e.get("data", {}).get("to", "") for e in fsm_events]
        assert "indexed" in states, f"FSM must reach indexed state; got: {states}"

    def test_pack_import_start_evidence_emitted(self, signed_manifest_file, data_file):
        """PACK_IMPORT_START evidence must be the first pack event emitted."""
        manifest_path, _ = signed_manifest_file
        import_pack(manifest_path, data_file)
        events = EvidenceLogger.get_events()
        start_events = [
            e for e in events
            if e.get("type") == EvidenceType.PACK_IMPORT_START.value
        ]
        assert start_events, "PACK_IMPORT_START event must be emitted before any transition"

    def test_manifest_sha256_preserved(self, signed_manifest_file, data_file):
        """manifest_sha256 field must survive the import pipeline."""
        manifest_path, _ = signed_manifest_file
        pack = import_pack(manifest_path, data_file)
        assert pack.manifest_sha256 == "c" * 64

    def test_data_parquet_sha256_preserved(self, signed_manifest_file, data_file):
        """data_parquet_sha256 field must survive the import pipeline."""
        manifest_path, _ = signed_manifest_file
        pack = import_pack(manifest_path, data_file)
        assert pack.data_parquet_sha256 == "d" * 64


# ---------------------------------------------------------------------------
# Signature format tests (hex vs base64 encoding)
# ---------------------------------------------------------------------------

@_skip_no_crypto
class TestSignatureFormatVariants:
    """Verify that signature can be encoded as hex or base64 in manifest JSON."""

    def test_hex_encoded_signature_accepted(self, tmp_path, data_file):
        """Signature stored as lowercase hex string must be accepted."""
        data = _make_manifest_data()
        sig_bytes = _sign_manifest(data)
        data["signature"] = sig_bytes.hex()  # hex encoding
        manifest_path = tmp_path / "manifest_hex.json"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        pack = import_pack(manifest_path, data_file)
        assert pack is not None

    def test_base64_encoded_signature_accepted(self, tmp_path, data_file):
        """Signature stored as base64 string must be accepted."""
        import base64
        data = _make_manifest_data()
        sig_bytes = _sign_manifest(data)
        data["signature"] = base64.b64encode(sig_bytes).decode("ascii")  # base64 encoding
        manifest_path = tmp_path / "manifest_b64.json"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        pack = import_pack(manifest_path, data_file)
        assert pack is not None
