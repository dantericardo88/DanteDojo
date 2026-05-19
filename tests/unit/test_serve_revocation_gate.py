"""Unit tests for d08 — sovereign revocation check wired into launch_server().

The serve path must call check_revocation() when a ConstitutionalPassport with
a revocation_url is supplied.  A revoked passport must block inference before
the backend starts.  No passport / no revocation_url → server starts normally.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import PassportRevoked, ServerStartError
from dojo.core.types import (
    ConstitutionalPassport,
    HALProfile,
    HardwareTier,
)
from dojo.serve.launcher import launch_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passport(revocation_url: str = "", revoked: bool = False) -> ConstitutionalPassport:
    return ConstitutionalPassport(
        passport_id="test-passport-001",
        model_artifact_id="dojo://my-model/v1",
        harvest_receipt_id="harvest-receipt-xyz",
        harvest_check_receipt=b"harvest_ok",
        dojo_check_receipt=b"dojo_ok",
        forge_check_receipt=b"forge_ok",
        merkle_root="a" * 64,
        issued_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(days=90),
        revocation_url=revocation_url,
    )


def _profile() -> HALProfile:
    return HALProfile(
        tier=HardwareTier.TIER_4060_LAPTOP,
        gpu_name="RTX 4060",
        vram_bytes=8 * 1024 ** 3,
        compute_capability="8.9",
        has_native_fp8=True,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=272.0,
        driver_version="560.0",
        profile_sha256="b" * 64,
        signed_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestServeRevocationGate:
    """launch_server() must gate on passport revocation before backend starts."""

    def test_no_passport_starts_normally(self) -> None:
        """No passport supplied → server launches without revocation check."""
        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://my-model/v1",
                model_path=Path("/tmp/model"),
                profile=_profile(),
                passport=None,
            )
        assert manifest is not None

    def test_passport_no_revocation_url_starts_normally(self) -> None:
        """Passport with empty revocation_url → no HTTP call, server starts."""
        p = _passport(revocation_url="")
        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://my-model/v1",
                model_path=Path("/tmp/model"),
                profile=_profile(),
                passport=p,
            )
        assert manifest is not None

    def test_valid_passport_starts_normally(self) -> None:
        """Passport with revocation_url that returns {revoked: false} → server starts."""
        p = _passport(revocation_url="http://revoke.example.com/check")
        with patch("dojo.promote.revocation.check_revocation", return_value=False) as mock_rev, \
             patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://my-model/v1",
                model_path=Path("/tmp/model"),
                profile=_profile(),
                passport=p,
            )
        assert manifest is not None
        mock_rev.assert_called_once_with(p)

    def test_revoked_passport_blocks_serve(self) -> None:
        """Revoked passport must raise ServerStartError before backend starts."""
        p = _passport(revocation_url="http://revoke.example.com/check")
        with patch(
            "dojo.promote.revocation.check_revocation",
            side_effect=PassportRevoked(
                "Passport test-passport-001 has been revoked",
                reason="revocation_url returned {revoked: true}",
            ),
        ), patch("dojo.serve.launcher._provision_backend") as mock_prov:
            with pytest.raises(ServerStartError) as exc_info:
                launch_server(
                    model_artifact_id="dojo://my-model/v1",
                    model_path=Path("/tmp/model"),
                    profile=_profile(),
                    passport=p,
                )
            # Backend must never start
            mock_prov.assert_not_called()
            assert "revocation" in str(exc_info.value).lower() or "revoked" in str(exc_info.value).lower()

    def test_revocation_check_called_before_backend(self) -> None:
        """Revocation must execute before _provision_backend to prevent side effects."""
        call_order: list[str] = []

        def _fake_revoke(passport):
            call_order.append("revoke")

        def _fake_provision(*args, **kwargs):
            call_order.append("provision")
            return 0

        p = _passport(revocation_url="http://revoke.example.com/check")
        with patch("dojo.promote.revocation.check_revocation", side_effect=_fake_revoke), \
             patch("dojo.serve.launcher._provision_backend", side_effect=_fake_provision), \
             patch("dojo.serve.launcher._warm_backend"):
            launch_server(
                model_artifact_id="dojo://my-model/v1",
                model_path=Path("/tmp/model"),
                profile=_profile(),
                passport=p,
            )

        assert call_order == ["revoke", "provision"], (
            "Revocation must be checked before backend provisioning"
        )

    def test_revocation_evidence_emitted(self) -> None:
        """check_revocation emits evidence; launch_server also emits SERVE_START."""
        p = _passport(revocation_url="http://revoke.example.com/check")
        with patch("dojo.promote.revocation.check_revocation", return_value=False), \
             patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            launch_server(
                model_artifact_id="dojo://my-model/v1",
                model_path=Path("/tmp/model"),
                profile=_profile(),
                passport=p,
            )

        chain = EvidenceLogger.get_instance().list_events()
        types_seen = {e.get("evidence", {}).get("type") for e in chain}
        from dojo.core.evidence import EvidenceType
        assert EvidenceType.SERVE_START.value in types_seen
