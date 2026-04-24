"""Unit tests for dojo/promote/ — three-way promotion gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PromotionBlocked
from dojo.core.types import ConstitutionalPassport, ModelArtifact
from dojo.promote.dojo_check import check_dojo
from dojo.promote.gate import GateCheckResult, promote
from dojo.promote.harvest_check import check_harvest
from dojo.promote.sealer import seal_passport, verify_seal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence() -> None:
    """Reset evidence logger between tests."""
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
def valid_receipt_id() -> str:
    return "receipt-valid-001"


@pytest.fixture()
def eval_merkle() -> str:
    return "deadbeef" * 8


# ---------------------------------------------------------------------------
# promote() — happy path
# ---------------------------------------------------------------------------

def test_promote_all_green_returns_passport(sample_artifact, valid_receipt_id, eval_merkle):
    """promote() with all-green mocks → returns ConstitutionalPassport."""
    passport = promote(
        sample_artifact,
        valid_receipt_id,
        eval_merkle,
        harvest_client=None,
        forge_client=None,
        baseline_f1=0.70,
        candidate_f1=0.80,
    )
    assert isinstance(passport, ConstitutionalPassport)
    assert passport.model_artifact_id == sample_artifact.artifact_id
    assert passport.harvest_receipt_id == valid_receipt_id
    assert len(passport.merkle_root) == 64  # SHA-256 hex
    assert not passport.is_expired()


def test_promote_emits_passport_issued_evidence(sample_artifact, valid_receipt_id, eval_merkle):
    """promote() emits PROMOTE_PASSPORT_ISSUED evidence."""
    promote(
        sample_artifact,
        valid_receipt_id,
        eval_merkle,
        baseline_f1=0.70,
        candidate_f1=0.80,
    )
    events = EvidenceLogger.get_events()
    event_types = [e["type"] for e in events]
    assert EvidenceType.PROMOTE_PASSPORT_ISSUED.value in event_types


# ---------------------------------------------------------------------------
# promote() — blocked at harvest_check
# ---------------------------------------------------------------------------

def test_promote_revoked_harvest_raises_promotion_blocked(sample_artifact, eval_merkle):
    """promote() with revoked Harvest receipt → PromotionBlocked at harvest_check."""
    with pytest.raises(PromotionBlocked) as exc_info:
        promote(
            sample_artifact,
            "revoked_receipt_001",
            eval_merkle,
            baseline_f1=0.70,
            candidate_f1=0.80,
        )
    err = exc_info.value
    assert err.gate == "harvest_check"
    assert len(err.reasons) > 0


# ---------------------------------------------------------------------------
# promote() — blocked at dojo_check
# ---------------------------------------------------------------------------

def test_promote_insufficient_f1_improvement_raises_promotion_blocked(
    sample_artifact, valid_receipt_id, eval_merkle
):
    """promote() with candidate_f1 not beating baseline+5 → PromotionBlocked at dojo_check."""
    with pytest.raises(PromotionBlocked) as exc_info:
        promote(
            sample_artifact,
            valid_receipt_id,
            eval_merkle,
            baseline_f1=0.70,
            candidate_f1=0.74,  # only 4pt improvement, need 5pt
        )
    err = exc_info.value
    assert err.gate == "dojo_check"
    assert len(err.reasons) > 0


# ---------------------------------------------------------------------------
# promote() — blocked at forge_check
# ---------------------------------------------------------------------------

def test_promote_forge_client_error_raises_promotion_blocked(
    sample_artifact, valid_receipt_id, eval_merkle
):
    """promote() with Forge client returning error → PromotionBlocked at forge_check (fail-closed)."""
    from dojo.core.exceptions import ForgeClientError  # noqa: PLC0415

    failing_forge = MagicMock()
    failing_forge.verify_policy.side_effect = ForgeClientError(
        "Connection refused", reason="connection_refused"
    )

    with pytest.raises(PromotionBlocked) as exc_info:
        promote(
            sample_artifact,
            valid_receipt_id,
            eval_merkle,
            forge_client=failing_forge,
            baseline_f1=0.70,
            candidate_f1=0.80,
        )
    err = exc_info.value
    assert err.gate == "forge_check"


# ---------------------------------------------------------------------------
# check_harvest — unit tests
# ---------------------------------------------------------------------------

def test_check_harvest_revoked_prefix_returns_blocked():
    """check_harvest: revoked_ prefix receipt → blocked."""
    result = check_harvest("revoked_abc123")
    assert result.passed is False
    assert result.gate == "harvest_check"
    assert any("revoked" in r.lower() for r in result.reasons)


def test_check_harvest_valid_receipt_passes():
    """check_harvest: valid receipt_id → passes."""
    result = check_harvest("valid-receipt-999", pack_license_allows_ft=True)
    assert result.passed is True
    assert len(result.reasons) == 0


def test_check_harvest_license_disallows_ft_blocks():
    """check_harvest: license that disallows FT → blocked."""
    result = check_harvest("valid-receipt-001", pack_license_allows_ft=False)
    assert result.passed is False
    assert any("license" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# check_dojo — unit tests
# ---------------------------------------------------------------------------

def test_check_dojo_six_pt_improvement_passes():
    """check_dojo: candidate 0.76 vs baseline 0.70 → passes (6pt > 5pt threshold)."""
    result = check_dojo(
        candidate_f1=0.76,
        baseline_f1=0.70,
        eval_receipt_merkle="deadbeef" * 8,
    )
    assert result.passed is True
    assert len(result.reasons) == 0


def test_check_dojo_four_pt_improvement_blocked():
    """check_dojo: candidate 0.74 vs baseline 0.70 → blocked (4pt < 5pt)."""
    result = check_dojo(
        candidate_f1=0.74,
        baseline_f1=0.70,
        eval_receipt_merkle="deadbeef" * 8,
    )
    assert result.passed is False
    assert len(result.reasons) > 0


def test_check_dojo_exactly_five_pt_passes():
    """check_dojo: exactly 5pt improvement (0.75 vs 0.70) → passes (>= threshold)."""
    result = check_dojo(
        candidate_f1=0.75,
        baseline_f1=0.70,
        eval_receipt_merkle="deadbeef" * 8,
    )
    assert result.passed is True


def test_check_dojo_ci_lower_below_baseline_blocked():
    """check_dojo: ci_lower_bound below baseline → blocked."""
    result = check_dojo(
        candidate_f1=0.80,
        baseline_f1=0.70,
        eval_receipt_merkle="deadbeef" * 8,
        ci_lower_bound=0.68,  # below baseline
    )
    assert result.passed is False
    assert any("CI" in r or "lower bound" in r for r in result.reasons)


def test_check_dojo_ci_lower_meets_baseline_passes():
    """check_dojo: ci_lower_bound >= baseline → passes (when improvement is sufficient)."""
    result = check_dojo(
        candidate_f1=0.80,
        baseline_f1=0.70,
        eval_receipt_merkle="deadbeef" * 8,
        ci_lower_bound=0.71,  # above baseline
    )
    assert result.passed is True


# ---------------------------------------------------------------------------
# seal/verify tests
# ---------------------------------------------------------------------------

def test_verify_seal_built_passport_returns_true(sample_artifact, valid_receipt_id, eval_merkle):
    """verify_seal: built passport → seal verifies True."""
    passport = promote(
        sample_artifact,
        valid_receipt_id,
        eval_merkle,
        baseline_f1=0.70,
        candidate_f1=0.80,
    )
    assert verify_seal(passport) is True


def test_verify_seal_tampered_passport_returns_false(sample_artifact, valid_receipt_id, eval_merkle):
    """verify_seal: tampered passport → False."""
    passport = promote(
        sample_artifact,
        valid_receipt_id,
        eval_merkle,
        baseline_f1=0.70,
        candidate_f1=0.80,
    )
    # Tamper by replacing harvest_check_receipt with garbage
    tampered = ConstitutionalPassport(
        passport_id=passport.passport_id,
        model_artifact_id=passport.model_artifact_id,
        harvest_receipt_id=passport.harvest_receipt_id,
        harvest_check_receipt=b"tampered_garbage",  # tampered!
        dojo_check_receipt=passport.dojo_check_receipt,
        forge_check_receipt=passport.forge_check_receipt,
        merkle_root=passport.merkle_root,  # unchanged — should fail
        issued_at=passport.issued_at,
        expires_at=passport.expires_at,
        revocation_url=passport.revocation_url,
    )
    assert verify_seal(tampered) is False


# ---------------------------------------------------------------------------
# ConstitutionalPassport.is_expired()
# ---------------------------------------------------------------------------

def test_passport_not_expired_with_future_expiry():
    """ConstitutionalPassport.is_expired(): future expiry → False."""
    now = datetime.now(timezone.utc)
    passport = ConstitutionalPassport(
        passport_id="test-id",
        model_artifact_id="dojo://model/v1",
        harvest_receipt_id="receipt-001",
        harvest_check_receipt=b"hr",
        dojo_check_receipt=b"dr",
        forge_check_receipt=b"fr",
        merkle_root="abc",
        issued_at=now,
        expires_at=now + timedelta(days=90),
        revocation_url="http://127.0.0.1:8002/v1/passport/revoke",
    )
    assert passport.is_expired() is False


def test_passport_is_expired_with_past_expiry():
    """ConstitutionalPassport.is_expired(): past expiry → True."""
    past = datetime.now(timezone.utc) - timedelta(days=1)
    passport = ConstitutionalPassport(
        passport_id="test-id",
        model_artifact_id="dojo://model/v1",
        harvest_receipt_id="receipt-001",
        harvest_check_receipt=b"hr",
        dojo_check_receipt=b"dr",
        forge_check_receipt=b"fr",
        merkle_root="abc",
        issued_at=past - timedelta(days=90),
        expires_at=past,
        revocation_url="http://127.0.0.1:8002/v1/passport/revoke",
    )
    assert passport.is_expired() is True
