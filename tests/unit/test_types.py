"""Unit tests for dojo/core/types.py."""

import json
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dojo.core.types import (
    Backend,
    ConstitutionalPassport,
    CostReceipt,
    DeploymentManifest,
    DojoConfig,
    HALProfile,
    HardwareTier,
    LicenseSpec,
    ModelArtifact,
    PackFormat,
    PackReference,
    PrecisionType,
    PrivacyTier,
    QuantizationSpec,
    RoutingOption,
    RoutingPolicy,
    TrainingBackend,
    TrainingMode,
)


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------

class TestEnums:
    def test_hardware_tiers_cover_all_targets(self):
        required = {"tier_4060_laptop", "tier_3090", "tier_h100", "tier_apple_m4", "tier_cpu"}
        values = {t.value for t in HardwareTier}
        assert required.issubset(values)

    def test_precision_types_include_ampere_safe(self):
        assert PrecisionType.BF16 in PrecisionType
        assert PrecisionType.W8A16 in PrecisionType
        assert PrecisionType.NF4 in PrecisionType

    def test_privacy_tier_confidential_exists(self):
        assert PrivacyTier.CONFIDENTIAL in PrivacyTier
        assert PrivacyTier.SECRETS in PrivacyTier


# ---------------------------------------------------------------------------
# HALProfile
# ---------------------------------------------------------------------------

class TestHALProfile:
    def _make(self, cc: str = "8.6", fp8: bool = False) -> HALProfile:
        return HALProfile(
            tier=HardwareTier.TIER_3090,
            gpu_name="NVIDIA GeForce RTX 3090",
            vram_bytes=24 * 1024 ** 3,
            compute_capability=cc,
            has_native_fp8=fp8,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=4,
            interconnect_bandwidth_gb_s=32.0,
            memory_bandwidth_gb_s=936.0,
            driver_version="550.0",
            profile_sha256="deadbeef" * 8,
        )

    def test_vram_gb_property(self):
        p = self._make()
        assert abs(p.vram_gb - 24.0) < 0.01

    def test_ampere_has_no_native_fp8(self):
        p = self._make(cc="8.6", fp8=False)
        assert p.has_native_fp8 is False

    def test_ada_has_native_fp8(self):
        ada_profile = HALProfile(
            tier=HardwareTier.TIER_4060_LAPTOP,
            gpu_name="NVIDIA GeForce RTX 4060",
            vram_bytes=8 * 1024 ** 3,
            compute_capability="8.9",
            has_native_fp8=True,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=4,
            interconnect_bandwidth_gb_s=16.0,
            memory_bandwidth_gb_s=272.0,
            driver_version="550.0",
            profile_sha256="cafebabe" * 8,
        )
        assert ada_profile.has_native_fp8 is True

    def test_to_dict_is_json_serializable(self):
        p = self._make()
        d = p.to_dict()
        assert json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# DojoConfig
# ---------------------------------------------------------------------------

class TestDojoConfig:
    def test_frozen(self):
        cfg = DojoConfig(
            hardware_tier=HardwareTier.TIER_3090,
            training_mode=TrainingMode.QLORA,
            precision=PrecisionType.NF4,
            backend=TrainingBackend.UNSLOTH,
            seed=42,
            evidence_dir=Path("/tmp/dojo/evidence"),
        )
        with pytest.raises((AttributeError, TypeError)):
            cfg.seed = 99  # type: ignore[misc]

    def test_stable_json_deterministic(self):
        cfg = DojoConfig(
            hardware_tier=HardwareTier.TIER_3090,
            training_mode=TrainingMode.QLORA,
            precision=PrecisionType.NF4,
            backend=TrainingBackend.UNSLOTH,
            seed=42,
            evidence_dir=Path("/tmp/dojo/evidence"),
        )
        assert cfg.stable_json() == cfg.stable_json()


# ---------------------------------------------------------------------------
# Privacy tier enforcement
# ---------------------------------------------------------------------------

class TestRoutingPolicy:
    def test_confidential_blocks_external_api(self):
        p = RoutingPolicy(
            task_name="bookkeeping",
            preference_order=(RoutingOption.LOCAL_SPECIALIST,),
            max_latency_ms=5000,
            max_cost_usd=0.0,
            privacy_tier=PrivacyTier.CONFIDENTIAL,
            fallback_on_unavailable=False,
        )
        assert p.allows_external_api() is False

    def test_secrets_blocks_external_api(self):
        p = RoutingPolicy(
            task_name="tax_docs",
            preference_order=(RoutingOption.LOCAL_SPECIALIST,),
            max_latency_ms=5000,
            max_cost_usd=0.0,
            privacy_tier=PrivacyTier.SECRETS,
            fallback_on_unavailable=False,
        )
        assert p.allows_external_api() is False

    def test_public_allows_external_api(self):
        p = RoutingPolicy(
            task_name="generic",
            preference_order=(RoutingOption.API_HAIKU,),
            max_latency_ms=5000,
            max_cost_usd=1.0,
            privacy_tier=PrivacyTier.PUBLIC,
            fallback_on_unavailable=True,
        )
        assert p.allows_external_api() is True


# ---------------------------------------------------------------------------
# ConstitutionalPassport
# ---------------------------------------------------------------------------

class TestConstitutionalPassport:
    def _make(self, expired: bool = False) -> ConstitutionalPassport:
        now = datetime.now(timezone.utc)
        if expired:
            issued = now - timedelta(days=100)
            expires = now - timedelta(days=10)
        else:
            issued = now
            expires = now + timedelta(days=90)
        return ConstitutionalPassport(
            passport_id="pass-001",
            model_artifact_id="dojo://qwen25-7b/v1",
            harvest_receipt_id="harvest-001",
            harvest_check_receipt=b"harvest_sig",
            dojo_check_receipt=b"dojo_sig",
            forge_check_receipt=b"forge_sig",
            merkle_root="abc123",
            issued_at=issued,
            expires_at=expires,
            revocation_url="http://127.0.0.1:8002/v1/passport/revoke",
        )

    def test_not_expired(self):
        p = self._make(expired=False)
        assert p.is_expired() is False

    def test_expired(self):
        p = self._make(expired=True)
        assert p.is_expired() is True

    def test_to_dict_json_serializable(self):
        p = self._make()
        d = p.to_dict()
        assert json.dumps(d)


# ---------------------------------------------------------------------------
# CostReceipt
# ---------------------------------------------------------------------------

class TestCostReceipt:
    def test_to_dict(self):
        r = CostReceipt(
            task="bookkeeping",
            route_taken="local_specialist",
            tokens_in=500,
            tokens_out=100,
            cost_usd=0.0,
            latency_ms=250.0,
            local_displaced_estimate_usd=0.01,
        )
        d = r.to_dict()
        assert d["route_taken"] == "local_specialist"
        assert json.dumps(d)
