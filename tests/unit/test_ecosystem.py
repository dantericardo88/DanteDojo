"""Unit tests for dojo/ecosystem/ — cross-organ HTTP/artifact contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import ForgeClientError, HarvestClientError
from dojo.core.types import ModelArtifact
from dojo.ecosystem.agents_publisher import AgentsPublisher
from dojo.ecosystem.evidence_bridge import cross_link_evidence, verify_cross_link
from dojo.ecosystem.forge_client import ForgeClient
from dojo.ecosystem.harvest_client import HarvestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence() -> None:
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture()
def sample_artifact() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="dojo://qwen25-7b-empanada/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="abc123deadbeef",
        passport=None,
    )


@pytest.fixture()
def sample_passport(sample_artifact):
    from datetime import datetime, timedelta, timezone

    from dojo.core.types import ConstitutionalPassport

    now = datetime.now(timezone.utc)
    return ConstitutionalPassport(
        passport_id="pp-test-001",
        model_artifact_id=sample_artifact.artifact_id,
        harvest_receipt_id="receipt-valid-001",
        harvest_check_receipt=b"harvest_receipt_bytes",
        dojo_check_receipt=b"dojo_receipt_bytes",
        forge_check_receipt=b"forge_receipt_bytes",
        merkle_root="a" * 64,
        issued_at=now,
        expires_at=now + timedelta(days=90),
        revocation_url="http://127.0.0.1:8002/v1/passport/revoke",
    )


# ---------------------------------------------------------------------------
# HarvestClient
# ---------------------------------------------------------------------------

def test_harvest_client_check_revocation_connection_refused_raises():
    """HarvestClient.check_revocation: connection refused → HarvestClientError raised."""
    import httpx  # noqa: PLC0415

    client = HarvestClient(base_url="http://127.0.0.1:19999")
    with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(HarvestClientError) as exc_info:
            client.check_revocation("some-receipt")
    assert "DANTE_HARVEST_URL" in str(exc_info.value) or "connect" in str(exc_info.value).lower()


def test_harvest_client_verify_receipt_connection_refused_raises():
    """HarvestClient.verify_receipt: connection refused → HarvestClientError raised."""
    import httpx  # noqa: PLC0415

    client = HarvestClient(base_url="http://127.0.0.1:19999")
    with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(HarvestClientError):
            client.verify_receipt("some-receipt")


def test_harvest_client_fetch_pack_connection_refused_raises():
    """HarvestClient.fetch_pack: connection refused → HarvestClientError raised."""
    import httpx  # noqa: PLC0415

    client = HarvestClient(base_url="http://127.0.0.1:19999")
    with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(HarvestClientError):
            client.fetch_pack("empanada-bookkeeping/v3")


# ---------------------------------------------------------------------------
# ForgeClient
# ---------------------------------------------------------------------------

def test_forge_client_verify_policy_connection_refused_raises():
    """ForgeClient.verify_policy: connection refused → ForgeClientError raised."""
    import httpx  # noqa: PLC0415

    client = ForgeClient(base_url="http://127.0.0.1:19999")
    with patch("httpx.post", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(ForgeClientError) as exc_info:
            client.verify_policy({"artifact_id": "test"})
    assert "DANTE_FORGE_URL" in str(exc_info.value) or "connect" in str(exc_info.value).lower()


def test_forge_client_verify_policy_timeout_raises():
    """ForgeClient.verify_policy: timeout → ForgeClientError raised."""
    import httpx  # noqa: PLC0415

    client = ForgeClient(base_url="http://127.0.0.1:19999")
    with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
        with pytest.raises(ForgeClientError) as exc_info:
            client.verify_policy({"artifact_id": "test"}, timeout=1.0)
    assert "timeout" in str(exc_info.value).lower() or "timed" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# AgentsPublisher
# ---------------------------------------------------------------------------

def test_agents_publisher_publish_model_writes_manifest(
    tmp_path, sample_artifact, sample_passport
):
    """AgentsPublisher.publish_model: writes manifest JSON to tmp_path registry_dir."""
    publisher = AgentsPublisher(registry_dir=tmp_path)
    manifest_path = publisher.publish_model(sample_artifact, sample_passport)

    assert manifest_path.exists()
    assert manifest_path.suffix == ".json"

    import json  # noqa: PLC0415

    with manifest_path.open() as fh:
        data = json.load(fh)

    assert data["artifact"]["artifact_id"] == sample_artifact.artifact_id
    assert data["passport"]["passport_id"] == sample_passport.passport_id


def test_agents_publisher_list_published_includes_artifact(
    tmp_path, sample_artifact, sample_passport
):
    """AgentsPublisher.list_published: includes just-published artifact_id."""
    publisher = AgentsPublisher(registry_dir=tmp_path)
    publisher.publish_model(sample_artifact, sample_passport)

    published = publisher.list_published()
    assert sample_artifact.artifact_id in published


def test_agents_publisher_list_published_empty_when_no_registry(tmp_path):
    """AgentsPublisher.list_published: returns [] when registry dir doesn't exist."""
    registry_dir = tmp_path / "nonexistent" / "dir"
    publisher = AgentsPublisher(registry_dir=registry_dir)
    assert publisher.list_published() == []


def test_agents_publisher_publish_emits_evidence(
    tmp_path, sample_artifact, sample_passport
):
    """AgentsPublisher.publish_model: emits evidence event."""
    publisher = AgentsPublisher(registry_dir=tmp_path)
    publisher.publish_model(sample_artifact, sample_passport)

    events = EvidenceLogger.get_events()
    event_types = [e.get("type", e.get("evidence", {}).get("type", "")) for e in events]
    raw_events = EvidenceLogger._events
    found = any(
        e.get("evidence", {}).get("type") == "agents.model.published"
        for e in raw_events
    )
    assert found


# ---------------------------------------------------------------------------
# evidence_bridge
# ---------------------------------------------------------------------------

def test_cross_link_evidence_returns_deterministic_hex():
    """cross_link_evidence: returns deterministic hex string."""
    result1 = cross_link_evidence("local_root_abc", "harvest", "harvest_root_xyz")
    result2 = cross_link_evidence("local_root_abc", "harvest", "harvest_root_xyz")

    # Both calls return the same deterministic hash
    assert result1 == result2
    # Should be a 64-char hex string (SHA-256)
    assert len(result1) == 64
    assert all(c in "0123456789abcdef" for c in result1)


def test_verify_cross_link_correct_inputs_returns_true():
    """verify_cross_link: correct inputs → True."""
    link = cross_link_evidence("local_abc", "forge", "remote_xyz")
    assert verify_cross_link(link, "local_abc", "forge", "remote_xyz") is True


def test_verify_cross_link_tampered_input_returns_false():
    """verify_cross_link: tampered input → False."""
    link = cross_link_evidence("local_abc", "forge", "remote_xyz")
    assert verify_cross_link(link, "local_TAMPERED", "forge", "remote_xyz") is False
    assert verify_cross_link(link, "local_abc", "TAMPERED_organ", "remote_xyz") is False
    assert verify_cross_link(link, "local_abc", "forge", "TAMPERED_xyz") is False


def test_cross_link_evidence_emits_evidence_event():
    """cross_link_evidence emits 'evidence.cross_link' evidence event."""
    EvidenceLogger.clear()
    cross_link_evidence("local_root", "harvest", "harvest_root")

    raw_events = EvidenceLogger._events
    found = any(
        e.get("evidence", {}).get("type") == "evidence.cross_link"
        for e in raw_events
    )
    assert found
