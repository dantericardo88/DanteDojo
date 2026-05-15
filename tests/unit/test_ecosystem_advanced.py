"""Advanced ecosystem tests — d46 Axolotl bridge, d47 multi-GPU, d48 recipe export.

12 tests:
    Axolotl bridge (4): config mapping, YAML roundtrip, custom fields, evidence
    Multi-GPU config  (4): TP/PP sizes, FSDP wrapping, device mesh, evidence
    Recipe export     (4): export format, schema validation, round-trip load, provenance
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.types import (
    HALProfile,
    HardwareTier,
    LicenseSpec,
    PackFormat,
    ProvenanceManifest,
)
from dojo.pack.schema import SpecializationPack
from dojo.train.base import FinetuneConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture()
def base_config() -> FinetuneConfig:
    return FinetuneConfig(
        pack_id="test-bookkeeping-pack",
        base_model="meta-llama/Llama-3.2-3B",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        batch_size=1,
        grad_accum_steps=4,
        max_seq_length=2048,
        num_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        seed=42,
    )


@pytest.fixture()
def sample_pack(tmp_path) -> SpecializationPack:
    data_path = tmp_path / "pack_data.parquet"
    data_path.write_bytes(b"fake parquet data")
    return SpecializationPack(
        pack_id="test-bookkeeping-pack",
        version="1.0.0",
        schema_version="1.0",
        sample_count=1000,
        format=PackFormat.ALPACA,
        splits={"train": 900, "val": 50, "held_out_gold": 50},
        harvest_receipt="a" * 64,
        license=LicenseSpec(
            spdx_id="MIT",
            allows_derived_ft=True,
            allows_commercial_use=True,
            requires_attribution=False,
        ),
        provenance=ProvenanceManifest(
            harvest_organ_version="1.2.0",
            sources=("https://example.com/source1",),
            collection_date_utc="2025-01-01T00:00:00Z",
            dedup_method="minhash",
            human_reviewed=True,
        ),
        signature=b"test_signature",
        manifest_sha256="b" * 64,
        data_parquet_sha256="c" * 64,
        data_path=data_path,
    )


@pytest.fixture()
def dual_3090_profile() -> HALProfile:
    return HALProfile(
        tier=HardwareTier.TIER_3090_DUAL,
        gpu_name="RTX 3090",
        vram_bytes=24 * 1024 ** 3,
        compute_capability="8.6",
        has_native_fp8=False,
        has_native_fp4=False,
    )


@pytest.fixture()
def h100_profile() -> HALProfile:
    return HALProfile(
        tier=HardwareTier.TIER_H100,
        gpu_name="H100 SXM5",
        vram_bytes=80 * 1024 ** 3,
        compute_capability="9.0",
        has_native_fp8=True,
        has_native_fp4=False,
    )


# ---------------------------------------------------------------------------
# Axolotl bridge tests (d46)
# ---------------------------------------------------------------------------

class TestAxolotlBridge:
    """4 tests covering axolotl_config_from_finetune and export_axolotl_config."""

    def test_config_mapping_fields(self, base_config, sample_pack, dual_3090_profile):
        """axolotl_config_from_finetune: core DanteDojo fields are mapped to axolotl keys."""
        from dojo.train.axolotl_bridge import axolotl_config_from_finetune

        cfg = axolotl_config_from_finetune(base_config, sample_pack, hal_profile=dual_3090_profile)

        assert cfg["base_model"] == base_config.base_model
        assert cfg["lora_r"] == base_config.lora_r
        assert cfg["lora_alpha"] == base_config.lora_alpha
        assert cfg["num_epochs"] == base_config.num_epochs
        assert cfg["sequence_len"] == base_config.max_seq_length
        assert cfg["micro_batch_size"] == base_config.batch_size
        assert cfg["gradient_accumulation_steps"] == base_config.grad_accum_steps
        assert cfg["learning_rate"] == base_config.learning_rate
        assert cfg["seed"] == base_config.seed
        assert str(base_config.output_dir) == cfg["output_dir"]

    def test_yaml_roundtrip(self, base_config, sample_pack, tmp_path):
        """export_axolotl_config: written file is valid YAML that round-trips to same base_model."""
        pytest.importorskip("yaml", reason="PyYAML required for YAML roundtrip test")
        import yaml

        from dojo.train.axolotl_bridge import export_axolotl_config

        out_path = tmp_path / "axolotl_config.yaml"
        result_path = export_axolotl_config(base_config, sample_pack, out_path)

        assert result_path.exists()
        assert result_path.suffix == ".yaml"

        with result_path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)

        assert loaded["base_model"] == base_config.base_model
        assert loaded["num_epochs"] == base_config.num_epochs

    def test_custom_dojo_fields_embedded(self, base_config, sample_pack):
        """axolotl_config_from_finetune: _dojo block contains pack_id, passport_id, config_hash."""
        from dojo.train.axolotl_bridge import axolotl_config_from_finetune

        passport_id = "pp-test-001"
        cfg = axolotl_config_from_finetune(
            base_config, sample_pack, passport_id=passport_id
        )

        dojo_block = cfg["_dojo"]
        assert dojo_block["pack_id"] == sample_pack.pack_id
        assert dojo_block["pack_version"] == sample_pack.version
        assert dojo_block["harvest_receipt"] == sample_pack.harvest_receipt
        assert dojo_block["dojo_passport_id"] == passport_id
        assert dojo_block["config_hash"] == base_config.config_hash()
        assert dojo_block["sample_count"] == sample_pack.sample_count

    def test_export_emits_training_start_evidence(self, base_config, sample_pack, tmp_path):
        """export_axolotl_config: emits TRAINING_START evidence event with bridge metadata."""
        from dojo.train.axolotl_bridge import export_axolotl_config

        out_path = tmp_path / "axolotl_config.json"
        export_axolotl_config(base_config, sample_pack, out_path, passport_id="pp-ev-test")

        raw_events = EvidenceLogger._events
        found = any(
            e.get("evidence", {}).get("type") == "train.finetune.started"
            and e.get("evidence", {}).get("data", {}).get("bridge") == "axolotl"
            for e in raw_events
        )
        assert found, "Expected TRAINING_START evidence with bridge=axolotl"


# ---------------------------------------------------------------------------
# Multi-GPU config tests (d47)
# ---------------------------------------------------------------------------

class TestMultiGPUConfig:
    """4 tests covering detect_multi_gpu and FSDP strategy selection."""

    def test_fsdp_strategy_sizes_two_3090s(self):
        """detect_multi_gpu(mock): 2×24GB → FULL_SHARD strategy, device_count==2."""
        from dojo.train.multi_gpu import (
            STRATEGY_FULL_SHARD,
            detect_multi_gpu,
        )

        profile = detect_multi_gpu(mock_mode=True)

        assert profile.device_count == 2
        assert profile.recommended_strategy == STRATEGY_FULL_SHARD
        assert profile.per_device_vram_bytes == 24 * 1024 ** 3
        assert profile.total_vram_bytes == 2 * 24 * 1024 ** 3

    def test_fsdp_strategy_single_gpu_no_shard(self):
        """_select_strategy: 1 device → NO_SHARD (DDP-equivalent)."""
        from dojo.train.multi_gpu import (
            STRATEGY_NO_SHARD,
            _select_strategy,
        )

        strategy = _select_strategy(device_count=1, per_device_vram_bytes=24 * 1024 ** 3)
        assert strategy == STRATEGY_NO_SHARD

    def test_fsdp_wrapping_axolotl_config_dual_3090(self, base_config, dual_3090_profile):
        """generate_axolotl_config for TIER_3090_DUAL includes fsdp2=True and world_size=2."""
        from dojo.train.axolotl_backend import generate_axolotl_config

        cfg = generate_axolotl_config(base_config, dual_3090_profile)

        assert cfg["world_size"] == 2
        fsdp_config = cfg.get("fsdp_config") or {}
        assert fsdp_config.get("fsdp2") is True
        # Should be FSDP full_shard
        fsdp_list = cfg.get("fsdp") or []
        assert "full_shard" in fsdp_list

    def test_multi_gpu_profile_emits_fsdp2_evidence(self):
        """detect_multi_gpu(mock): emits FSDP2_PROFILE_DETECTED evidence."""
        from dojo.train.multi_gpu import detect_multi_gpu

        EvidenceLogger.clear()
        detect_multi_gpu(mock_mode=True)

        raw_events = EvidenceLogger._events
        found = any(
            e.get("evidence", {}).get("type") == "train.fsdp2.profile.detected"
            for e in raw_events
        )
        assert found, "Expected FSDP2_PROFILE_DETECTED evidence event"


# ---------------------------------------------------------------------------
# Recipe export tests (d48)
# ---------------------------------------------------------------------------

class TestRecipeExport:
    """4 tests covering export_recipe, load_recipe, validate_recipe, provenance."""

    def test_export_recipe_format(self, sample_pack, tmp_path):
        """export_recipe: writes valid JSON with required top-level keys."""
        from dojo.pack.recipe_export import RECIPE_REQUIRED_FIELDS, export_recipe

        out_path = tmp_path / "recipe.json"
        result = export_recipe(sample_pack, out_path, config_hash="abc123")

        assert result.exists()
        assert result.suffix == ".json"

        with result.open(encoding="utf-8") as fh:
            data = json.load(fh)

        for field in RECIPE_REQUIRED_FIELDS:
            assert field in data, f"Missing required recipe field: {field}"

        assert data["pack_id"] == sample_pack.pack_id
        assert data["pack_version"] == sample_pack.version
        assert data["sample_count"] == sample_pack.sample_count
        assert data["config_hash"] == "abc123"

    def test_validate_recipe_schema_valid(self, sample_pack, tmp_path):
        """validate_recipe: passes on a correctly exported recipe."""
        from dojo.pack.recipe_export import export_recipe, load_recipe, validate_recipe

        out_path = tmp_path / "recipe_valid.json"
        export_recipe(sample_pack, out_path)

        data = load_recipe(out_path)
        # Should not raise
        validate_recipe(data)

    def test_validate_recipe_schema_missing_field_raises(self, sample_pack, tmp_path):
        """validate_recipe: raises ValueError when a required field is missing."""
        from dojo.pack.recipe_export import export_recipe, validate_recipe

        out_path = tmp_path / "recipe_bad.json"
        export_recipe(sample_pack, out_path)

        with out_path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        del data["pack_id"]

        with pytest.raises(ValueError, match="pack_id"):
            validate_recipe(data)

    def test_recipe_roundtrip_provenance(self, sample_pack, tmp_path):
        """export_recipe + load_recipe: provenance block round-trips faithfully."""
        from dojo.pack.recipe_export import export_recipe, load_recipe

        out_path = tmp_path / "recipe_rt.json"
        export_recipe(sample_pack, out_path)

        data = load_recipe(out_path)
        prov = data["provenance"]

        assert prov["harvest_organ_version"] == sample_pack.provenance.harvest_organ_version
        assert prov["sources"] == list(sample_pack.provenance.sources)
        assert prov["collection_date_utc"] == sample_pack.provenance.collection_date_utc
        assert prov["dedup_method"] == sample_pack.provenance.dedup_method
        assert prov["human_reviewed"] == sample_pack.provenance.human_reviewed
