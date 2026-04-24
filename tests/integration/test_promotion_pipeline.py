"""Integration test — end-to-end mock promotion pipeline.

Full pipeline:
  create ModelArtifact
  → run promote() with mock Harvest+Forge
  → get passport
  → verify seal
  → publish to AgentsPublisher
  → list published
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.types import ModelArtifact
from dojo.ecosystem.agents_publisher import AgentsPublisher
from dojo.promote.gate import promote
from dojo.promote.sealer import verify_seal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence() -> None:
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# Full happy-path pipeline
# ---------------------------------------------------------------------------

def test_full_promotion_pipeline(tmp_path: Path) -> None:
    """Full pipeline: create ModelArtifact → promote → verify seal → publish → list."""

    # Step 1: Create a ModelArtifact
    artifact = ModelArtifact(
        artifact_id="dojo://qwen25-7b-empanada-bookkeeping/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="training-receipt-merkle-root-abc123",
        passport=None,
    )

    # Step 2: Run promote() with mock Harvest and Forge (None → mocks)
    passport = promote(
        artifact,
        harvest_receipt_id="receipt-valid-integration-001",
        eval_receipt_merkle="eval-merkle-integration-001",
        harvest_client=None,  # uses mock (no revoked_ prefix → GREEN)
        forge_client=None,    # uses mock (always GREEN)
        baseline_f1=0.70,
        candidate_f1=0.82,   # +12pt improvement — well above the 5pt threshold
    )

    # Step 3: Verify the passport is valid
    assert passport is not None
    assert passport.model_artifact_id == artifact.artifact_id
    assert passport.harvest_receipt_id == "receipt-valid-integration-001"
    assert len(passport.merkle_root) == 64
    assert not passport.is_expired()

    # Step 4: Verify seal
    assert verify_seal(passport) is True

    # Step 5: Publish to AgentsPublisher
    registry_dir = tmp_path / "registry" / "models"
    publisher = AgentsPublisher(registry_dir=registry_dir)
    manifest_path = publisher.publish_model(artifact, passport)

    # Manifest file was written
    assert manifest_path.exists()
    assert manifest_path.is_file()

    # Step 6: List published models
    published = publisher.list_published()
    assert artifact.artifact_id in published

    # Step 7: Verify evidence chain contains PROMOTE_PASSPORT_ISSUED
    events = EvidenceLogger.get_events()
    event_types = [e.get("type", "") for e in events]
    assert EvidenceType.PROMOTE_PASSPORT_ISSUED.value in event_types

    # Step 8: Verify harvest, dojo, and forge GREEN events are all present
    assert EvidenceType.PROMOTE_HARVEST_GREEN.value in event_types
    assert EvidenceType.PROMOTE_DOJO_GREEN.value in event_types
    assert EvidenceType.PROMOTE_FORGE_GREEN.value in event_types


def test_full_pipeline_blocked_at_harvest_does_not_publish(tmp_path: Path) -> None:
    """Pipeline with revoked receipt is blocked before publishing."""
    from dojo.core.exceptions import PromotionBlocked  # noqa: PLC0415

    artifact = ModelArtifact(
        artifact_id="dojo://bad-model/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="bad-receipt",
        passport=None,
    )

    with pytest.raises(PromotionBlocked) as exc_info:
        promote(
            artifact,
            harvest_receipt_id="revoked_receipt_integration_001",
            eval_receipt_merkle="eval-merkle",
            baseline_f1=0.70,
            candidate_f1=0.80,
        )

    err = exc_info.value
    assert err.gate == "harvest_check"

    # Nothing should have been published
    publisher = AgentsPublisher(registry_dir=tmp_path / "registry")
    assert publisher.list_published() == []
