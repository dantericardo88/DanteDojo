"""Unit tests for d46 — registry version selection (lookup_version, list_versions).

Validates:
- register v1 + v2 → lookup_version(name) returns v2 (latest)
- lookup_version(name, version="v1") returns v1 explicitly
- lookup_version(name, "dojo://model:v2") resolves URI + version
- list_versions returns sorted version list
- delete v1 → list_versions omits v1, lookup v1 raises RegistryError
- RegistryError raised when name not found
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import RegistryError
from dojo.core.types import ModelArtifact
from dojo.registry.registry import (
    delete_artifact,
    list_versions,
    lookup_version,
    register,
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


def _artifact(artifact_id: str, base_model: str = "Qwen/Qwen2.5-7B") -> ModelArtifact:
    return ModelArtifact(
        artifact_id=artifact_id,
        base_model=base_model,
        lora_weights=None,
        quantization=None,
        training_receipt="rcpt-test",
        passport=None,
        created_at=datetime(2026, 5, 19, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLookupVersion:
    def test_lookup_latest_returns_greatest_version(self, store) -> None:
        """With v1 and v2 registered, lookup_version(name) returns v2."""
        register(_artifact("my-model/v1"), store=store)
        register(_artifact("my-model/v2"), store=store)
        art = lookup_version("my-model", store=store)
        assert art.artifact_id == "my-model/v2"

    def test_lookup_explicit_version(self, store) -> None:
        """lookup_version(name, version='v1') must return v1 not v2."""
        register(_artifact("my-model/v1"), store=store)
        register(_artifact("my-model/v2"), store=store)
        art = lookup_version("my-model", version="v1", store=store)
        assert art.artifact_id == "my-model/v1"

    def test_lookup_version_latest_keyword(self, store) -> None:
        """lookup_version(name, version='latest') returns lexicographically greatest."""
        register(_artifact("my-model/v1"), store=store)
        register(_artifact("my-model/v3"), store=store)
        register(_artifact("my-model/v2"), store=store)
        art = lookup_version("my-model", version="latest", store=store)
        assert art.artifact_id == "my-model/v3"

    def test_lookup_version_dojo_uri_with_colon(self, store) -> None:
        """'dojo://my-model:v2' syntax resolves to the correct artifact."""
        register(_artifact("my-model/v1"), store=store)
        register(_artifact("my-model/v2"), store=store)
        art = lookup_version("dojo://my-model:v2", store=store)
        assert art.artifact_id == "my-model/v2"

    def test_lookup_version_dojo_uri_no_version(self, store) -> None:
        """'dojo://my-model' with no version → returns latest."""
        register(_artifact("my-model/v1"), store=store)
        register(_artifact("my-model/v2"), store=store)
        art = lookup_version("dojo://my-model", store=store)
        assert art.artifact_id == "my-model/v2"

    def test_unknown_model_raises_registry_error(self, store) -> None:
        with pytest.raises(RegistryError, match="No artifacts found"):
            lookup_version("nonexistent-model", store=store)

    def test_unknown_version_raises_registry_error(self, store) -> None:
        register(_artifact("my-model/v1"), store=store)
        with pytest.raises(RegistryError, match="not found"):
            lookup_version("my-model", version="v99", store=store)

    def test_evidence_emitted_on_lookup(self, store) -> None:
        register(_artifact("my-model/v1"), store=store)
        lookup_version("my-model", store=store)
        chain = EvidenceLogger.get_instance().list_events()
        types = [e.get("evidence", {}).get("type") for e in chain]
        assert "registry.lookup_version" in types


class TestListVersions:
    def test_empty_registry_returns_empty_list(self, store) -> None:
        assert list_versions("my-model", store=store) == []

    def test_returns_sorted_versions(self, store) -> None:
        register(_artifact("my-model/v3"), store=store)
        register(_artifact("my-model/v1"), store=store)
        register(_artifact("my-model/v2"), store=store)
        versions = list_versions("my-model", store=store)
        assert versions == ["v1", "v2", "v3"]

    def test_excludes_other_models(self, store) -> None:
        register(_artifact("model-a/v1"), store=store)
        register(_artifact("model-b/v1"), store=store)
        register(_artifact("model-b/v2"), store=store)
        assert list_versions("model-a", store=store) == ["v1"]
        assert list_versions("model-b", store=store) == ["v1", "v2"]

    def test_delete_removes_version_from_list(self, store) -> None:
        register(_artifact("my-model/v1"), store=store)
        register(_artifact("my-model/v2"), store=store)
        delete_artifact("my-model/v1", store=store)
        versions = list_versions("my-model", store=store)
        assert "v1" not in versions
        assert "v2" in versions

    def test_delete_then_lookup_v1_raises(self, store) -> None:
        register(_artifact("my-model/v1"), store=store)
        register(_artifact("my-model/v2"), store=store)
        delete_artifact("my-model/v1", store=store)
        with pytest.raises(RegistryError):
            lookup_version("my-model", version="v1", store=store)
