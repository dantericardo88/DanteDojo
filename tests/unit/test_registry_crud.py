"""Unit tests for registry CRUD — d46: list_artifacts, delete_artifact, integrity_check."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import RegistryError
from dojo.core.types import ModelArtifact
from dojo.registry.registry import (
    delete_artifact,
    list_artifacts,
    register,
    registry_integrity_check,
)
from dojo.registry.store import LocalRegistryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def store(tmp_path) -> LocalRegistryStore:
    return LocalRegistryStore(registry_dir=tmp_path / "registry")


def _artifact(artifact_id: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct") -> ModelArtifact:
    return ModelArtifact(
        artifact_id=artifact_id,
        base_model=base_model,
        lora_weights=None,
        quantization=None,
        training_receipt="receipt-test",
        passport=None,
        created_at=datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc),
    )


# ===========================================================================
# list_artifacts tests
# ===========================================================================

def test_list_artifacts_empty_registry(store):
    result = list_artifacts(store=store)
    assert result == []


def test_list_artifacts_after_register(store):
    a = _artifact("dojo://model/v1")
    register(a, store=store)
    result = list_artifacts(store=store)
    assert len(result) == 1
    assert result[0]["artifact_id"] == "dojo://model/v1"


def test_list_artifacts_pattern_filters(store):
    register(_artifact("dojo://alpha/v1"), store=store)
    register(_artifact("dojo://beta/v1"), store=store)
    register(_artifact("dojo://alpha/v2"), store=store)
    result = list_artifacts(pattern="dojo://alpha/*", store=store)
    ids = [e["artifact_id"] for e in result]
    assert "dojo://alpha/v1" in ids
    assert "dojo://alpha/v2" in ids
    assert "dojo://beta/v1" not in ids


def test_list_artifacts_returns_artifact_id_field(store):
    register(_artifact("dojo://mymodel/v3"), store=store)
    result = list_artifacts(store=store)
    assert "artifact_id" in result[0]
    assert "model_name" in result[0]
    assert "version" in result[0]
    assert "registered_at" in result[0]


def test_list_artifacts_version_parsed(store):
    register(_artifact("dojo://mymodel/v99"), store=store)
    result = list_artifacts(store=store)
    assert result[0]["version"] == "v99"


# ===========================================================================
# delete_artifact tests
# ===========================================================================

def test_delete_artifact_removes_entry(store):
    a = _artifact("dojo://del-me/v1")
    register(a, store=store)
    assert len(list_artifacts(store=store)) == 1
    delete_artifact("dojo://del-me/v1", store=store)
    assert list_artifacts(store=store) == []


def test_delete_artifact_raises_on_missing(store):
    with pytest.raises(RegistryError):
        delete_artifact("dojo://nonexistent/v1", store=store)


def test_delete_artifact_evidence_emitted(store):
    EvidenceLogger.clear()
    a = _artifact("dojo://evidence-test/v1")
    register(a, store=store)
    EvidenceLogger.clear()
    delete_artifact("dojo://evidence-test/v1", store=store)
    events = EvidenceLogger.get_events()
    types = [e["type"] for e in events]
    assert EvidenceType.EVIDENCE_MERKLE_ROOT.value in types


# ===========================================================================
# integrity_check tests
# ===========================================================================

def test_integrity_check_healthy_registry(store):
    register(_artifact("dojo://healthy/v1"), store=store)
    register(_artifact("dojo://healthy/v2"), store=store)
    result = registry_integrity_check(store=store)
    assert result["total"] == 2
    assert result["healthy"] == 2
    assert result["corrupt"] == 0
    assert result["missing"] == 0


def test_integrity_check_returns_total_count(store):
    for i in range(5):
        register(_artifact(f"dojo://batch/v{i}"), store=store)
    result = registry_integrity_check(store=store)
    assert result["total"] == 5


def test_integrity_check_empty_registry(store):
    result = registry_integrity_check(store=store)
    assert result == {"total": 0, "healthy": 0, "corrupt": 0, "missing": 0}


def test_integrity_check_detects_corrupt_manifest(store, tmp_path):
    """A truncated JSON file should be counted as corrupt."""
    corrupt_store = LocalRegistryStore(registry_dir=tmp_path / "corrupt_reg")
    corrupt_store._ensure_dir()
    corrupt_file = corrupt_store._dir / "bad_artifact.json"
    corrupt_file.write_text("{not valid json", encoding="utf-8")
    result = corrupt_store.integrity_check()
    assert result["corrupt"] >= 1


# ===========================================================================
# Round-trip test
# ===========================================================================

def test_registry_roundtrip_register_list_delete(store):
    ids = ["dojo://rt/v1", "dojo://rt/v2", "dojo://rt/v3"]
    for aid in ids:
        register(_artifact(aid), store=store)

    listed = [e["artifact_id"] for e in list_artifacts(store=store)]
    assert sorted(listed) == sorted(ids)

    delete_artifact("dojo://rt/v2", store=store)
    after = [e["artifact_id"] for e in list_artifacts(store=store)]
    assert "dojo://rt/v2" not in after
    assert len(after) == 2
