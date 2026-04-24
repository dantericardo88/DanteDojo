"""Unit tests for dojo/registry/ — Phase 4 model registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import RegistryError
from dojo.core.types import ModelArtifact, QuantizationSpec
from dojo.registry.fingerprint import fingerprint_artifact, fingerprint_weights
from dojo.registry.registry import lookup, register
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
def tmp_registry(tmp_path) -> LocalRegistryStore:
    """LocalRegistryStore backed by a temporary directory."""
    return LocalRegistryStore(registry_dir=tmp_path / "registry")


@pytest.fixture
def artifact_v1() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="dojo://qwen25-7b-empanada/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="receipt-abc123",
        passport=None,
        created_at=datetime(2026, 4, 22, 0, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def artifact_v2() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="dojo://qwen25-7b-empanada/v2",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=QuantizationSpec(
            method="gguf_q4_k_m",
            bits=4,
            group_size=128,
            calibration_dataset="empanada-cal",
        ),
        training_receipt="receipt-def456",
        passport=None,
        created_at=datetime(2026, 4, 22, 1, 0, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# fingerprint_artifact tests
# ---------------------------------------------------------------------------

class TestFingerprintArtifact:
    def test_deterministic_for_same_artifact(self, artifact_v1):
        fp1 = fingerprint_artifact(artifact_v1)
        fp2 = fingerprint_artifact(artifact_v1)
        assert fp1 == fp2

    def test_differs_for_different_artifact_ids(self, artifact_v1, artifact_v2):
        fp1 = fingerprint_artifact(artifact_v1)
        fp2 = fingerprint_artifact(artifact_v2)
        assert fp1 != fp2

    def test_returns_64_hex_chars(self, artifact_v1):
        fp = fingerprint_artifact(artifact_v1)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_quantized_artifact_different_from_base(self, artifact_v1, artifact_v2):
        """v2 has quantization; fingerprint must differ from v1."""
        assert fingerprint_artifact(artifact_v1) != fingerprint_artifact(artifact_v2)


# ---------------------------------------------------------------------------
# fingerprint_weights tests
# ---------------------------------------------------------------------------

class TestFingerprintWeights:
    def test_deterministic(self, tmp_path):
        weights_file = tmp_path / "model.bin"
        weights_file.write_bytes(b"fake weights data" * 100)
        fp1 = fingerprint_weights(weights_file)
        fp2 = fingerprint_weights(weights_file)
        assert fp1 == fp2

    def test_returns_64_hex_chars(self, tmp_path):
        weights_file = tmp_path / "model.bin"
        weights_file.write_bytes(b"X" * 512)
        fp = fingerprint_weights(weights_file)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_differs_for_different_files(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"aaaa" * 256)
        f2.write_bytes(b"bbbb" * 256)
        assert fingerprint_weights(f1) != fingerprint_weights(f2)

    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.bin"
        with pytest.raises(FileNotFoundError):
            fingerprint_weights(missing)


# ---------------------------------------------------------------------------
# LocalRegistryStore tests
# ---------------------------------------------------------------------------

class TestLocalRegistryStore:
    def test_save_writes_json_file(self, tmp_registry, artifact_v1):
        tmp_registry.save(artifact_v1)
        # At least one JSON file should exist
        json_files = list(tmp_registry._dir.glob("*.json"))
        assert len(json_files) == 1

    def test_save_json_file_is_valid(self, tmp_registry, artifact_v1):
        tmp_registry.save(artifact_v1)
        json_files = list(tmp_registry._dir.glob("*.json"))
        with json_files[0].open() as fh:
            manifest = json.load(fh)
        assert manifest["artifact_id"] == artifact_v1.artifact_id

    def test_load_retrieves_artifact(self, tmp_registry, artifact_v1):
        tmp_registry.save(artifact_v1)
        loaded = tmp_registry.load(artifact_v1.artifact_id)
        assert loaded.artifact_id == artifact_v1.artifact_id
        assert loaded.base_model == artifact_v1.base_model
        assert loaded.training_receipt == artifact_v1.training_receipt

    def test_load_preserves_quantization(self, tmp_registry, artifact_v2):
        tmp_registry.save(artifact_v2)
        loaded = tmp_registry.load(artifact_v2.artifact_id)
        assert loaded.quantization is not None
        assert loaded.quantization.method == "gguf_q4_k_m"
        assert loaded.quantization.bits == 4

    def test_list_artifacts_includes_saved(self, tmp_registry, artifact_v1, artifact_v2):
        tmp_registry.save(artifact_v1)
        tmp_registry.save(artifact_v2)
        ids = tmp_registry.list_artifacts()
        assert artifact_v1.artifact_id in ids
        assert artifact_v2.artifact_id in ids

    def test_list_artifacts_empty_when_none_saved(self, tmp_path):
        store = LocalRegistryStore(registry_dir=tmp_path / "empty")
        assert store.list_artifacts() == []

    def test_delete_removes_artifact(self, tmp_registry, artifact_v1):
        tmp_registry.save(artifact_v1)
        assert tmp_registry.exists(artifact_v1.artifact_id)
        tmp_registry.delete(artifact_v1.artifact_id)
        assert not tmp_registry.exists(artifact_v1.artifact_id)

    def test_load_missing_raises_registry_error(self, tmp_registry):
        with pytest.raises(RegistryError):
            tmp_registry.load("dojo://nonexistent/v999")

    def test_delete_missing_raises_registry_error(self, tmp_registry):
        with pytest.raises(RegistryError):
            tmp_registry.delete("dojo://nonexistent/v999")

    def test_exists_returns_false_when_not_saved(self, tmp_registry):
        assert tmp_registry.exists("dojo://ghost/v0") is False

    def test_exists_returns_true_when_saved(self, tmp_registry, artifact_v1):
        tmp_registry.save(artifact_v1)
        assert tmp_registry.exists(artifact_v1.artifact_id) is True

    def test_save_emits_evidence(self, tmp_registry, artifact_v1):
        tmp_registry.save(artifact_v1)
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert "registry.artifact.saved" in types


# ---------------------------------------------------------------------------
# register / lookup one-door tests
# ---------------------------------------------------------------------------

class TestRegisterLookup:
    def test_register_returns_artifact_id(self, tmp_registry, artifact_v1):
        returned_id = register(artifact_v1, store=tmp_registry)
        assert returned_id == artifact_v1.artifact_id

    def test_lookup_returns_artifact(self, tmp_registry, artifact_v1):
        register(artifact_v1, store=tmp_registry)
        result = lookup(artifact_v1.artifact_id, store=tmp_registry)
        assert result.artifact_id == artifact_v1.artifact_id

    def test_register_lookup_round_trip(self, tmp_registry, artifact_v2):
        register(artifact_v2, store=tmp_registry)
        loaded = lookup(artifact_v2.artifact_id, store=tmp_registry)
        assert loaded.artifact_id == artifact_v2.artifact_id
        assert loaded.base_model == artifact_v2.base_model
        assert loaded.training_receipt == artifact_v2.training_receipt

    def test_lookup_missing_raises_registry_error(self, tmp_registry):
        with pytest.raises(RegistryError):
            lookup("dojo://does-not-exist/v0", store=tmp_registry)

    def test_register_emits_evidence(self, tmp_registry, artifact_v1):
        register(artifact_v1, store=tmp_registry)
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert "registry.register" in types
