"""Unit tests for dojo/pack/ — Phase 2 pack ingest subsystem.

Covers:
- validate_schema: missing field raises PackSchemaInvalid with field name
- validate_schema: all fields present → no exception
- compute_splits: invariants (sum, ordering, determinism)
- compute_sample_hash: determinism and collision resistance
- estimate_token_count: basic sanity
- import_pack: valid pack → SpecializationPack returned
- import_pack: tampered signature → PackSignatureInvalid
- import_pack: missing field → PackSchemaInvalid
- import_pack: contaminated sample → PackContaminationDetected
- Evidence events emitted during import
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import (
    PackContaminationDetected,
    PackSchemaInvalid,
    PackSignatureInvalid,
)
from dojo.pack.schema import (
    REQUIRED_FIELDS,
    SpecializationPack,
    pack_from_dict,
    validate_schema,
)
from dojo.pack.splits import compute_splits
from dojo.pack.tokenize import (
    compute_sample_hash,
    estimate_token_count,
    normalize_text,
)
from dojo.pack.importer import import_pack


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset EvidenceLogger singleton between every test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _minimal_manifest(*, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a valid manifest dict suitable for testing."""
    base: Dict[str, Any] = {
        "pack_id": "empanada-bookkeeping",
        "version": "1.0.0",
        "schema_version": "1.0",
        "sample_count": 20000,
        "format": "instruction",
        "splits": {"train": 18000, "val": 1000, "held_out_gold": 1000},
        "harvest_receipt": "abc123merkleroot",
        "license": {
            "spdx_id": "CC-BY-4.0",
            "allows_derived_ft": True,
            "allows_commercial_use": True,
            "requires_attribution": True,
        },
        "provenance": {
            "harvest_organ_version": "0.3.0",
            "sources": ["https://example.com/dataset"],
            "collection_date_utc": "2026-04-01T00:00:00Z",
            "dedup_method": "minhash",
            "human_reviewed": True,
        },
        "signature": "74657374_7369676e6174757265",  # hex of b"test_signature"
        "manifest_sha256": "deadbeef" * 8,
        "data_parquet_sha256": "cafebabe" * 8,
    }
    # Store signature as actual bytes string that will be decoded to test_signature
    base["signature"] = b"test_signature".hex()
    if extra:
        base.update(extra)
    return base


def _write_manifest(tmp_path: Path, data: Dict[str, Any]) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _write_data(tmp_path: Path, samples: list[str] | None = None) -> Path:
    """Write a JSONL data file. Defaults to a small benign dataset."""
    if samples is None:
        samples = [f"Sample text number {i} about bookkeeping." for i in range(5)]
    p = tmp_path / "data.jsonl"
    lines = [json.dumps({"text": s}) for s in samples]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_missing_pack_id_raises_with_field_name(self):
        data = _minimal_manifest()
        del data["pack_id"]
        with pytest.raises(PackSchemaInvalid) as exc_info:
            validate_schema(data)
        assert "pack_id" in str(exc_info.value)

    def test_missing_version_raises_with_field_name(self):
        data = _minimal_manifest()
        del data["version"]
        with pytest.raises(PackSchemaInvalid) as exc_info:
            validate_schema(data)
        assert "version" in str(exc_info.value)

    def test_missing_signature_raises(self):
        data = _minimal_manifest()
        del data["signature"]
        with pytest.raises(PackSchemaInvalid) as exc_info:
            validate_schema(data)
        assert "signature" in str(exc_info.value)

    def test_missing_sample_count_raises(self):
        data = _minimal_manifest()
        del data["sample_count"]
        with pytest.raises(PackSchemaInvalid) as exc_info:
            validate_schema(data)
        assert "sample_count" in str(exc_info.value)

    def test_all_required_fields_present_no_exception(self):
        data = _minimal_manifest()
        validate_schema(data)  # must not raise

    def test_extra_fields_are_ignored(self):
        data = _minimal_manifest()
        data["unknown_field"] = "some_value"
        validate_schema(data)  # must not raise

    def test_all_required_fields_covered(self):
        """Every REQUIRED_FIELD, when absent, triggers PackSchemaInvalid."""
        for field_name in REQUIRED_FIELDS:
            data = _minimal_manifest()
            del data[field_name]
            with pytest.raises(PackSchemaInvalid):
                validate_schema(data)


# ---------------------------------------------------------------------------
# pack_from_dict
# ---------------------------------------------------------------------------

class TestPackFromDict:
    def test_returns_specialization_pack(self):
        data = _minimal_manifest()
        pack = pack_from_dict(data)
        assert isinstance(pack, SpecializationPack)

    def test_fields_parsed_correctly(self):
        data = _minimal_manifest()
        pack = pack_from_dict(data)
        assert pack.pack_id == "empanada-bookkeeping"
        assert pack.version == "1.0.0"
        assert pack.sample_count == 20000
        assert pack.splits["train"] == 18000

    def test_missing_field_raises(self):
        data = _minimal_manifest()
        del data["pack_id"]
        with pytest.raises(PackSchemaInvalid):
            pack_from_dict(data)

    def test_to_dict_is_json_serializable(self):
        data = _minimal_manifest()
        pack = pack_from_dict(data)
        d = pack.to_dict()
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# compute_splits
# ---------------------------------------------------------------------------

class TestComputeSplits:
    def test_sum_equals_sample_count(self):
        result = compute_splits(20000)
        assert sum(result.values()) == 20000

    def test_train_greater_than_val(self):
        result = compute_splits(20000)
        assert result["train"] > result["val"]

    def test_val_greater_than_held_out(self):
        """With default equal ratios both splits are the same size — ok."""
        result = compute_splits(20000, val_ratio=0.10, held_out_ratio=0.05)
        assert result["val"] > result["held_out_gold"]

    def test_default_equal_ratios_sum(self):
        """Default val_ratio == held_out_ratio: train dominates."""
        result = compute_splits(20000)
        assert result["train"] > result["val"]
        assert result["train"] > result["held_out_gold"]

    def test_deterministic_same_seed(self):
        r1 = compute_splits(10000, seed=42)
        r2 = compute_splits(10000, seed=42)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        """Determinism check — same count, different seeds produce stable counts.
        (Since we use round(), both seeds produce the same integer counts
         but the function still exercised the seed path.)
        """
        r1 = compute_splits(10000, seed=1)
        r2 = compute_splits(10000, seed=2)
        # Both must still sum to sample_count.
        assert sum(r1.values()) == 10000
        assert sum(r2.values()) == 10000

    def test_small_sample_count(self):
        result = compute_splits(10)
        assert sum(result.values()) == 10

    def test_invalid_sample_count_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_splits(0)

    def test_excess_ratios_raises(self):
        with pytest.raises(ValueError, match="0.50"):
            compute_splits(1000, val_ratio=0.30, held_out_ratio=0.30)

    def test_all_keys_present(self):
        result = compute_splits(1000)
        assert "train" in result
        assert "val" in result
        assert "held_out_gold" in result


# ---------------------------------------------------------------------------
# compute_sample_hash and normalize_text
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_compute_sample_hash_deterministic(self):
        h1 = compute_sample_hash("Hello World")
        h2 = compute_sample_hash("Hello World")
        assert h1 == h2

    def test_different_texts_different_hashes(self):
        h1 = compute_sample_hash("Alpha text here.")
        h2 = compute_sample_hash("Beta text here.")
        assert h1 != h2

    def test_hash_is_hex_string(self):
        h = compute_sample_hash("test text")
        assert len(h) == 64
        int(h, 16)  # must parse as hex

    def test_normalization_collapses_whitespace(self):
        h1 = compute_sample_hash("hello   world")
        h2 = compute_sample_hash("hello world")
        assert h1 == h2  # same after normalization

    def test_normalization_case_insensitive(self):
        h1 = compute_sample_hash("Hello World")
        h2 = compute_sample_hash("hello world")
        assert h1 == h2

    def test_normalize_text_strips(self):
        assert normalize_text("  hello  ") == "hello"

    def test_normalize_text_collapses_spaces(self):
        assert normalize_text("a  b   c") == "a b c"

    def test_normalize_text_lowercases(self):
        assert normalize_text("UPPER") == "upper"

    def test_estimate_token_count_hello_world(self):
        count = estimate_token_count("hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_estimate_token_count_empty_is_zero(self):
        assert estimate_token_count("") == 0

    def test_estimate_token_count_scales_with_length(self):
        short = estimate_token_count("hi")
        long = estimate_token_count("hi " * 100)
        assert long > short

    def test_estimate_token_count_custom_ratio(self):
        count = estimate_token_count("hello", chars_per_token=1.0)
        assert count == 5


# ---------------------------------------------------------------------------
# import_pack — happy path
# ---------------------------------------------------------------------------

class TestImportPackHappyPath:
    def test_valid_pack_returns_specialization_pack(self, tmp_path):
        manifest_data = _minimal_manifest()
        manifest_path = _write_manifest(tmp_path, manifest_data)
        data_path = _write_data(tmp_path)

        result = import_pack(manifest_path, data_path)

        assert isinstance(result, SpecializationPack)
        assert result.pack_id == "empanada-bookkeeping"
        assert result.version == "1.0.0"

    def test_valid_pack_data_path_set(self, tmp_path):
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path)

        result = import_pack(manifest_path, data_path)

        assert result.data_path == data_path

    def test_valid_pack_no_contamination_check(self, tmp_path):
        """Import without eval_gold_hashes should succeed regardless of content."""
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path)

        result = import_pack(manifest_path, data_path, eval_gold_hashes=None)
        assert isinstance(result, SpecializationPack)


# ---------------------------------------------------------------------------
# import_pack — signature failures
# ---------------------------------------------------------------------------

class TestImportPackSignature:
    def test_tampered_signature_raises_pack_signature_invalid(self, tmp_path):
        data = _minimal_manifest()
        # Replace signature with something that is NOT b"test_signature".
        data["signature"] = b"wrong_signature".hex()
        manifest_path = _write_manifest(tmp_path, data)
        data_path = _write_data(tmp_path)

        with pytest.raises(PackSignatureInvalid):
            import_pack(manifest_path, data_path)

    def test_empty_signature_raises(self, tmp_path):
        data = _minimal_manifest()
        data["signature"] = ""
        manifest_path = _write_manifest(tmp_path, data)
        data_path = _write_data(tmp_path)

        with pytest.raises((PackSignatureInvalid, PackSchemaInvalid)):
            import_pack(manifest_path, data_path)


# ---------------------------------------------------------------------------
# import_pack — schema failures
# ---------------------------------------------------------------------------

class TestImportPackSchema:
    def test_missing_pack_id_raises_schema_invalid(self, tmp_path):
        data = _minimal_manifest()
        del data["pack_id"]
        manifest_path = _write_manifest(tmp_path, data)
        data_path = _write_data(tmp_path)

        with pytest.raises(PackSchemaInvalid) as exc_info:
            import_pack(manifest_path, data_path)
        assert "pack_id" in str(exc_info.value)

    def test_missing_version_raises_schema_invalid(self, tmp_path):
        data = _minimal_manifest()
        del data["version"]
        manifest_path = _write_manifest(tmp_path, data)
        data_path = _write_data(tmp_path)

        with pytest.raises(PackSchemaInvalid):
            import_pack(manifest_path, data_path)

    def test_missing_multiple_fields_raises_schema_invalid(self, tmp_path):
        data = _minimal_manifest()
        del data["version"]
        del data["sample_count"]
        manifest_path = _write_manifest(tmp_path, data)
        data_path = _write_data(tmp_path)

        with pytest.raises(PackSchemaInvalid):
            import_pack(manifest_path, data_path)


# ---------------------------------------------------------------------------
# import_pack — contamination detection
# ---------------------------------------------------------------------------

class TestImportPackContamination:
    def test_contaminated_sample_raises(self, tmp_path):
        evil_text = "This exact phrase appears in both training and eval sets."
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path, samples=[evil_text, "benign sample"])

        evil_hash = compute_sample_hash(evil_text)
        eval_gold = frozenset({evil_hash})

        with pytest.raises(PackContaminationDetected):
            import_pack(manifest_path, data_path, eval_gold_hashes=eval_gold)

    def test_clean_samples_pass_contamination_check(self, tmp_path):
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path, samples=["safe sample one", "safe sample two"])

        # Gold hashes of completely different texts.
        eval_gold = frozenset({compute_sample_hash("totally different eval text")})

        result = import_pack(manifest_path, data_path, eval_gold_hashes=eval_gold)
        assert isinstance(result, SpecializationPack)

    def test_empty_eval_gold_never_contaminates(self, tmp_path):
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path, samples=["anything here"])

        result = import_pack(manifest_path, data_path, eval_gold_hashes=frozenset())
        assert isinstance(result, SpecializationPack)


# ---------------------------------------------------------------------------
# Evidence events emitted during import
# ---------------------------------------------------------------------------

class TestImportPackEvidence:
    def test_pack_import_start_emitted(self, tmp_path):
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path)

        import_pack(manifest_path, data_path)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.PACK_IMPORT_START.value in types

    def test_pack_signature_verified_emitted(self, tmp_path):
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path)

        import_pack(manifest_path, data_path)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.PACK_SIGNATURE_VERIFIED.value in types

    def test_pack_indexed_emitted(self, tmp_path):
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path)

        import_pack(manifest_path, data_path)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.PACK_INDEXED.value in types

    def test_contamination_evidence_emitted(self, tmp_path):
        evil_text = "Unique contaminated phrase for evidence test."
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path, samples=[evil_text])
        eval_gold = frozenset({compute_sample_hash(evil_text)})

        with pytest.raises(PackContaminationDetected):
            import_pack(manifest_path, data_path, eval_gold_hashes=eval_gold)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.PACK_CONTAMINATION_DETECTED.value in types

    def test_one_door_entry_emitted(self, tmp_path):
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path)

        import_pack(manifest_path, data_path)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.ONE_DOOR_ENTRY.value in types

    def test_multiple_events_emitted_on_success(self, tmp_path):
        manifest_path = _write_manifest(tmp_path, _minimal_manifest())
        data_path = _write_data(tmp_path)

        import_pack(manifest_path, data_path)

        events = EvidenceLogger.get_events()
        # At minimum: one_door.entry, pack.import.start, pack.signature.verified,
        # pack.indexed + FSM transition events + tokenize/bucketing info events.
        assert len(events) >= 4

    def test_events_cleared_between_tests(self):
        """autouse clear_evidence fixture must reset events to 0."""
        events = EvidenceLogger.get_events()
        assert len(events) == 0
