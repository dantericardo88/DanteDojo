"""Unit tests for d47 (provenance chain integrity) and d48 (license enforcement).

d47 — 12 tests covering:
  verify_data_integrity, verify_manifest_integrity, build_provenance_summary,
  and provenance key in PACK_INDEXED evidence.

d48 — 13 tests covering:
  check_license_allows_training, finetune() license gate,
  registry.register() license gate, evidence type and content.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PackLicenseRejected, PackSignatureInvalid
from dojo.core.types import LicenseSpec, ModelArtifact, PackFormat, ProvenanceManifest
from dojo.pack.license import (
    build_provenance_summary,
    check_license_allows_training,
    verify_data_integrity,
    verify_manifest_integrity,
)
from dojo.pack.schema import SpecializationPack


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _make_pack(
    *,
    spdx_id: str = "MIT",
    allows_derived_ft: bool = True,
    allows_commercial_use: bool = True,
    requires_attribution: bool = False,
    manifest_sha256: str = "a" * 64,
    data_parquet_sha256: str = "b" * 64,
    harvest_organ_version: str = "0.4.0",
    sources: tuple = ("https://example.com/data",),
    collection_date_utc: str = "2026-01-01T00:00:00Z",
    dedup_method: str = "minhash",
    human_reviewed: bool = True,
) -> SpecializationPack:
    return SpecializationPack(
        pack_id="test-pack",
        version="1.0.0",
        schema_version="1.0",
        sample_count=100,
        format=PackFormat.INSTRUCTION,
        splits={"train": 80, "val": 20},
        harvest_receipt="abc123",
        license=LicenseSpec(
            spdx_id=spdx_id,
            allows_derived_ft=allows_derived_ft,
            allows_commercial_use=allows_commercial_use,
            requires_attribution=requires_attribution,
        ),
        provenance=ProvenanceManifest(
            harvest_organ_version=harvest_organ_version,
            sources=sources,
            collection_date_utc=collection_date_utc,
            dedup_method=dedup_method,
            human_reviewed=human_reviewed,
        ),
        signature=b"test_signature",
        manifest_sha256=manifest_sha256,
        data_parquet_sha256=data_parquet_sha256,
    )


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_hal(tier=None, vram_bytes: int = 24 * 1024 ** 3):
    from dojo.core.types import HALProfile, HardwareTier
    if tier is None:
        tier = HardwareTier.TIER_3090
    return HALProfile(
        tier=tier,
        gpu_name="Test GPU",
        vram_bytes=vram_bytes,
        compute_capability="8.6",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="550.0",
        profile_sha256="a" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# d47 — verify_data_integrity
# ---------------------------------------------------------------------------

class TestVerifyDataIntegrity:
    def test_matching_hash_passes(self):
        data = b"hello parquet data"
        digest = _sha256_hex(data)
        pack = _make_pack(data_parquet_sha256=digest)
        assert verify_data_integrity(pack, data) is True

    def test_wrong_hash_raises(self):
        data = b"real data"
        pack = _make_pack(data_parquet_sha256="0" * 64)
        with pytest.raises(PackSignatureInvalid):
            verify_data_integrity(pack, data)

    def test_emits_evidence_on_success(self):
        data = b"evidence data"
        digest = _sha256_hex(data)
        pack = _make_pack(data_parquet_sha256=digest)
        verify_data_integrity(pack, data)
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.PACK_SIGNATURE_VERIFIED.value in types

    def test_evidence_has_check_type_data_sha256(self):
        data = b"check type test"
        digest = _sha256_hex(data)
        pack = _make_pack(data_parquet_sha256=digest)
        verify_data_integrity(pack, data)
        events = EvidenceLogger.get_events()
        sig_events = [
            e for e in events
            if e.get("type") == EvidenceType.PACK_SIGNATURE_VERIFIED.value
        ]
        assert any(
            e.get("data", {}).get("check_type") == "data_sha256"
            for e in sig_events
        )

    def test_emits_evidence_even_on_mismatch(self):
        """Evidence is emitted before the exception is raised."""
        data = b"tampered"
        pack = _make_pack(data_parquet_sha256="0" * 64)
        with pytest.raises(PackSignatureInvalid):
            verify_data_integrity(pack, data)
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.PACK_SIGNATURE_VERIFIED.value in types

    def test_wrong_hash_error_message_contains_pack_id(self):
        data = b"some data"
        pack = _make_pack(data_parquet_sha256="0" * 64)
        with pytest.raises(PackSignatureInvalid, match="test-pack"):
            verify_data_integrity(pack, data)


# ---------------------------------------------------------------------------
# d47 — verify_manifest_integrity
# ---------------------------------------------------------------------------

class TestVerifyManifestIntegrity:
    def test_valid_hex_passes(self):
        pack = _make_pack(manifest_sha256="a" * 64)
        assert verify_manifest_integrity(pack) is True

    def test_empty_raises(self):
        pack = _make_pack(manifest_sha256="")
        with pytest.raises(PackSignatureInvalid):
            verify_manifest_integrity(pack)

    def test_non_hex_raises(self):
        pack = _make_pack(manifest_sha256="z" * 64)
        with pytest.raises(PackSignatureInvalid):
            verify_manifest_integrity(pack)

    def test_too_short_raises(self):
        pack = _make_pack(manifest_sha256="abc123")
        with pytest.raises(PackSignatureInvalid):
            verify_manifest_integrity(pack)

    def test_emits_evidence_with_manifest_check_type(self):
        pack = _make_pack(manifest_sha256="f" * 64)
        verify_manifest_integrity(pack)
        events = EvidenceLogger.get_events()
        sig_events = [
            e for e in events
            if e.get("type") == EvidenceType.PACK_SIGNATURE_VERIFIED.value
        ]
        assert any(
            e.get("data", {}).get("check_type") == "manifest_sha256"
            for e in sig_events
        )


# ---------------------------------------------------------------------------
# d47 — build_provenance_summary
# ---------------------------------------------------------------------------

class TestBuildProvenanceSummary:
    def test_has_harvest_organ_version(self):
        pack = _make_pack(harvest_organ_version="1.2.3")
        summary = build_provenance_summary(pack)
        assert summary["harvest_organ_version"] == "1.2.3"

    def test_has_sources(self):
        pack = _make_pack(sources=("https://example.com/a", "https://example.com/b"))
        summary = build_provenance_summary(pack)
        assert "https://example.com/a" in summary["sources"]
        assert "https://example.com/b" in summary["sources"]

    def test_has_collection_date(self):
        pack = _make_pack(collection_date_utc="2025-12-01T00:00:00Z")
        summary = build_provenance_summary(pack)
        assert summary["collection_date_utc"] == "2025-12-01T00:00:00Z"

    def test_has_dedup_method(self):
        pack = _make_pack(dedup_method="simhash")
        summary = build_provenance_summary(pack)
        assert summary["dedup_method"] == "simhash"

    def test_has_human_reviewed(self):
        pack_yes = _make_pack(human_reviewed=True)
        pack_no = _make_pack(human_reviewed=False)
        assert build_provenance_summary(pack_yes)["human_reviewed"] is True
        assert build_provenance_summary(pack_no)["human_reviewed"] is False

    def test_sources_is_list_not_tuple(self):
        pack = _make_pack(sources=("https://example.com",))
        summary = build_provenance_summary(pack)
        assert isinstance(summary["sources"], list)


# ---------------------------------------------------------------------------
# d47 — PACK_INDEXED evidence includes provenance
# ---------------------------------------------------------------------------

class TestPackIndexedIncludesProvenance:
    def test_pack_indexed_evidence_includes_provenance(self, tmp_path):
        """End-to-end: import_pack PACK_INDEXED event must contain provenance key."""
        from dojo.pack.importer import import_pack

        manifest: Dict[str, Any] = {
            "pack_id": "prov-test-pack",
            "version": "1.0.0",
            "schema_version": "1.0",
            "sample_count": 5,
            "format": "instruction",
            "splits": {"train": 4, "val": 1},
            "harvest_receipt": "abc123",
            "license": {
                "spdx_id": "MIT",
                "allows_derived_ft": True,
                "allows_commercial_use": True,
                "requires_attribution": False,
            },
            "provenance": {
                "harvest_organ_version": "0.5.0",
                "sources": ["https://example.com"],
                "collection_date_utc": "2026-01-01T00:00:00Z",
                "dedup_method": "minhash",
                "human_reviewed": True,
            },
            "signature": b"test_signature".hex(),
            "manifest_sha256": "d" * 64,
            "data_parquet_sha256": "e" * 64,
        }
        mpath = tmp_path / "manifest.json"
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
        dpath = tmp_path / "data.jsonl"
        dpath.write_text(json.dumps({"text": "sample"}) + "\n", encoding="utf-8")

        import_pack(mpath, dpath)

        events = EvidenceLogger.get_events()
        indexed = [
            e for e in events
            if e.get("type") == EvidenceType.PACK_INDEXED.value
        ]
        assert indexed, "PACK_INDEXED event not found"
        assert "provenance" in indexed[0].get("data", {}), (
            "PACK_INDEXED data missing 'provenance' key"
        )


# ---------------------------------------------------------------------------
# d48 — check_license_allows_training
# ---------------------------------------------------------------------------

class TestCheckLicenseAllowsTraining:
    def test_mit_passes(self):
        pack = _make_pack(spdx_id="MIT", allows_derived_ft=True)
        check_license_allows_training(pack)  # must not raise

    def test_cc_by_nc_raises(self):
        pack = _make_pack(spdx_id="CC-BY-NC-4.0", allows_derived_ft=False)
        with pytest.raises(PackLicenseRejected):
            check_license_allows_training(pack)

    def test_emits_evidence(self):
        pack = _make_pack(spdx_id="Apache-2.0", allows_derived_ft=True)
        check_license_allows_training(pack)
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.PACK_LICENSE_CHECKED.value in types

    def test_evidence_has_spdx_id(self):
        pack = _make_pack(spdx_id="Apache-2.0", allows_derived_ft=True)
        check_license_allows_training(pack)
        events = EvidenceLogger.get_events()
        lic_events = [
            e for e in events
            if e.get("type") == EvidenceType.PACK_LICENSE_CHECKED.value
        ]
        assert lic_events
        assert lic_events[0]["data"]["spdx_id"] == "Apache-2.0"

    def test_evidence_has_allows_derived_ft(self):
        pack = _make_pack(spdx_id="MIT", allows_derived_ft=True)
        check_license_allows_training(pack)
        events = EvidenceLogger.get_events()
        lic_events = [
            e for e in events
            if e.get("type") == EvidenceType.PACK_LICENSE_CHECKED.value
        ]
        assert lic_events
        assert lic_events[0]["data"]["allows_derived_ft"] is True

    def test_rejection_error_has_spdx_in_message(self):
        pack = _make_pack(spdx_id="CC-BY-NC-4.0", allows_derived_ft=False)
        with pytest.raises(PackLicenseRejected) as exc_info:
            check_license_allows_training(pack)
        assert "CC-BY-NC-4.0" in str(exc_info.value)

    def test_evidence_type_is_pack_license_checked(self):
        pack = _make_pack(spdx_id="MIT", allows_derived_ft=True)
        check_license_allows_training(pack)
        events = EvidenceLogger.get_events()
        assert any(
            e.get("type") == EvidenceType.PACK_LICENSE_CHECKED.value
            for e in events
        )

    def test_emits_evidence_even_on_rejection(self):
        """Evidence is emitted before the exception is raised."""
        pack = _make_pack(spdx_id="CC-BY-NC-4.0", allows_derived_ft=False)
        with pytest.raises(PackLicenseRejected):
            check_license_allows_training(pack)
        events = EvidenceLogger.get_events()
        assert any(
            e.get("type") == EvidenceType.PACK_LICENSE_CHECKED.value
            for e in events
        )


# ---------------------------------------------------------------------------
# d48 — finetune() license gate
# ---------------------------------------------------------------------------

class TestFinetuneWithLicense:
    def test_finetune_blocks_restricted_license(self, tmp_path):
        from dojo.train.finetune import finetune
        from dojo.train.base import FinetuneConfig

        pack = _make_pack(spdx_id="CC-BY-NC-4.0", allows_derived_ft=False)
        cfg = FinetuneConfig(pack_id=pack.pack_id, base_model="qwen/qwen2-7b",
                             output_dir=tmp_path)
        with pytest.raises(PackLicenseRejected):
            finetune(cfg, _make_hal(), pack)

    def test_finetune_allows_permissive_license(self, tmp_path):
        from dojo.train.finetune import finetune
        from dojo.train.base import FinetuneConfig, FinetuneResult

        pack = _make_pack(spdx_id="MIT", allows_derived_ft=True)
        cfg = FinetuneConfig(pack_id=pack.pack_id, base_model="qwen/qwen2-7b",
                             output_dir=tmp_path)
        result = finetune(cfg, _make_hal(), pack)
        assert isinstance(result, FinetuneResult)

    def test_finetune_license_check_before_gpu_work(self, tmp_path):
        """PackLicenseRejected must be raised even when no GPU is present (CPU tier)."""
        from dojo.core.types import HardwareTier
        from dojo.train.finetune import finetune
        from dojo.train.base import FinetuneConfig

        pack = _make_pack(spdx_id="CC-BY-NC-SA-4.0", allows_derived_ft=False)
        cfg = FinetuneConfig(pack_id=pack.pack_id, base_model="qwen/qwen2-7b",
                             output_dir=tmp_path)
        cpu_hal = _make_hal(tier=HardwareTier.TIER_CPU, vram_bytes=0)
        with pytest.raises(PackLicenseRejected):
            finetune(cfg, cpu_hal, pack)

    def test_finetune_license_check_emits_evidence(self, tmp_path):
        from dojo.train.finetune import finetune
        from dojo.train.base import FinetuneConfig

        pack = _make_pack(spdx_id="MIT", allows_derived_ft=True)
        cfg = FinetuneConfig(pack_id=pack.pack_id, base_model="qwen/qwen2-7b",
                             output_dir=tmp_path)
        finetune(cfg, _make_hal(), pack)
        events = EvidenceLogger.get_events()
        assert any(
            e.get("type") == EvidenceType.PACK_LICENSE_CHECKED.value
            for e in events
        )

    def test_license_check_is_first_in_finetune_pipeline(self, tmp_path):
        """License gate fires before TRAINING_START — verify by source inspection.

        We search the function body only (past the closing triple-quote of the
        docstring) so the docstring's mention of TRAINING_START does not
        interfere with the position check.
        """
        from dojo.train.finetune import finetune
        src = inspect.getsource(finetune)
        # Skip past the docstring by finding the closing triple-quote.
        doc_end = src.index('"""', src.index('"""') + 3) + 3
        body = src[doc_end:]
        license_pos = body.index("check_license_allows_training")
        training_start_pos = body.index("TRAINING_START")
        assert license_pos < training_start_pos, (
            "check_license_allows_training must appear before TRAINING_START emission"
        )


# ---------------------------------------------------------------------------
# d48 — registry.register() license gate
# ---------------------------------------------------------------------------

class TestRegistryWithLicense:
    def _make_artifact(self) -> ModelArtifact:
        return ModelArtifact(
            artifact_id="dojo://test-model/v1",
            base_model="qwen/qwen2-7b",
            lora_weights=None,
            quantization=None,
            training_receipt="receipt123",
            passport=None,
        )

    def test_registry_blocks_restricted_license_pack(self, tmp_path):
        from dojo.registry.registry import register
        from dojo.registry.store import LocalRegistryStore

        pack = _make_pack(spdx_id="CC-BY-NC-4.0", allows_derived_ft=False)
        store = LocalRegistryStore(registry_dir=tmp_path / "registry")
        artifact = self._make_artifact()
        with pytest.raises(PackLicenseRejected):
            register(artifact, store=store, pack=pack)

    def test_registry_allows_permissive_license_pack(self, tmp_path):
        from dojo.registry.registry import register
        from dojo.registry.store import LocalRegistryStore

        pack = _make_pack(spdx_id="Apache-2.0", allows_derived_ft=True)
        store = LocalRegistryStore(registry_dir=tmp_path / "registry")
        artifact = self._make_artifact()
        artifact_id = register(artifact, store=store, pack=pack)
        assert artifact_id == "dojo://test-model/v1"

    def test_registry_no_pack_param_still_works(self, tmp_path):
        """Existing callers without pack= must not break."""
        from dojo.registry.registry import register
        from dojo.registry.store import LocalRegistryStore

        store = LocalRegistryStore(registry_dir=tmp_path / "registry")
        artifact = self._make_artifact()
        artifact_id = register(artifact, store=store)
        assert artifact_id == "dojo://test-model/v1"
