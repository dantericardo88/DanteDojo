"""NF4 integration tests — pushes d25 to 9/10.

Covers wiring of NF4 config into the finetune() mock path and deeper
evidence validation:
1.  get_bnb_nf4_config() returns dict with bnb_4bit_quant_type == "nf4"
2.  bnb_4bit_use_double_quant is True in config
3.  bnb_4bit_compute_dtype is "bfloat16" in config
4.  Evidence type quant.nf4_config emitted after get_bnb_nf4_config()
5.  Evidence has load_in_4bit: True
6.  Evidence has bnb_4bit_use_double_quant: True
7.  Evidence has bnb_4bit_compute_dtype field
8.  Config is deterministic (two calls return same values)
9.  Config dict has exactly the 4 required bnb keys
10. finetune() in mock QLoRA mode emits quant.nf4_config evidence
11. finetune() NF4 evidence appears before training-step evidence
12. NF4 config values are preserved through finetune evidence chain
13. bnb_4bit_quant_type in finetune evidence matches get_bnb_nf4_config() output
14. Memory savings: bnb_4bit_use_double_quant=True is documented in evidence
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch
import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.types import PrecisionType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _make_fake_bnb() -> MagicMock:
    bnb = MagicMock()
    bnb.nn = MagicMock()
    bnb.nn.Linear4bit = MagicMock()
    return bnb


def _inject_bnb(monkeypatch) -> MagicMock:
    fake_bnb = _make_fake_bnb()
    fake_torch = MagicMock()
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_bnb


def _nf4_events() -> List[dict]:
    return [e for e in EvidenceLogger.get_events() if e.get("type") == "quant.nf4_config"]


def _reload_nf4(monkeypatch):
    """Helper: inject bnb and reload nf4 module."""
    _inject_bnb(monkeypatch)
    if "dojo.quant.nf4" in sys.modules:
        del sys.modules["dojo.quant.nf4"]
    from dojo.quant.nf4 import get_bnb_nf4_config
    return get_bnb_nf4_config


# ---------------------------------------------------------------------------
# 1. get_bnb_nf4_config() returns dict with bnb_4bit_quant_type == "nf4"
# ---------------------------------------------------------------------------

class TestNF4QunatType:
    def test_bnb_4bit_quant_type_is_nf4(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        result = get_bnb_nf4_config()
        assert result["bnb_4bit_quant_type"] == "nf4"

    def test_returns_dict_type(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        result = get_bnb_nf4_config()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 2. bnb_4bit_use_double_quant is True in config
# ---------------------------------------------------------------------------

class TestDoubleQuantization:
    def test_bnb_4bit_use_double_quant_is_true(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        result = get_bnb_nf4_config()
        assert result["bnb_4bit_use_double_quant"] is True

    def test_double_quant_is_bool(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        result = get_bnb_nf4_config()
        assert isinstance(result["bnb_4bit_use_double_quant"], bool)


# ---------------------------------------------------------------------------
# 3. bnb_4bit_compute_dtype is "bfloat16" in config
# ---------------------------------------------------------------------------

class TestComputeDtype:
    def test_bnb_4bit_compute_dtype_is_bfloat16(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        result = get_bnb_nf4_config()
        assert result["bnb_4bit_compute_dtype"] == "bfloat16"

    def test_compute_dtype_is_string(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        result = get_bnb_nf4_config()
        assert isinstance(result["bnb_4bit_compute_dtype"], str)


# ---------------------------------------------------------------------------
# 4. Evidence type quant.nf4_config emitted after get_bnb_nf4_config()
# ---------------------------------------------------------------------------

class TestNF4EvidenceEmitted:
    def test_quant_nf4_config_evidence_emitted(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        get_bnb_nf4_config()
        assert len(_nf4_events()) >= 1

    def test_evidence_type_string_matches(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        get_bnb_nf4_config()
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert "quant.nf4_config" in types


# ---------------------------------------------------------------------------
# 5. Evidence has load_in_4bit: True
# ---------------------------------------------------------------------------

class TestEvidenceLoadIn4Bit:
    def test_evidence_has_load_in_4bit_true(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        get_bnb_nf4_config()
        events = _nf4_events()
        assert events[0]["data"]["load_in_4bit"] is True

    def test_evidence_load_in_4bit_is_bool(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        get_bnb_nf4_config()
        events = _nf4_events()
        assert isinstance(events[0]["data"]["load_in_4bit"], bool)


# ---------------------------------------------------------------------------
# 6. Evidence has bnb_4bit_use_double_quant: True
# ---------------------------------------------------------------------------

class TestEvidenceDoubleQuant:
    def test_evidence_has_double_quant_true(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        get_bnb_nf4_config()
        events = _nf4_events()
        assert events[0]["data"]["bnb_4bit_use_double_quant"] is True


# ---------------------------------------------------------------------------
# 7. Evidence has bnb_4bit_compute_dtype field
# ---------------------------------------------------------------------------

class TestEvidenceComputeDtype:
    def test_evidence_has_compute_dtype_field(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        get_bnb_nf4_config()
        events = _nf4_events()
        assert "bnb_4bit_compute_dtype" in events[0].get("data", {})

    def test_evidence_compute_dtype_is_bfloat16(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        get_bnb_nf4_config()
        events = _nf4_events()
        assert events[0]["data"]["bnb_4bit_compute_dtype"] == "bfloat16"


# ---------------------------------------------------------------------------
# 8. Config is deterministic (two calls return same values)
# ---------------------------------------------------------------------------

class TestNF4Deterministic:
    def test_two_calls_return_same_values(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        result1 = get_bnb_nf4_config()
        result2 = get_bnb_nf4_config()
        assert result1 == result2

    def test_config_values_stable_across_calls(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        for _ in range(3):
            r = get_bnb_nf4_config()
            assert r["bnb_4bit_quant_type"] == "nf4"
            assert r["bnb_4bit_use_double_quant"] is True


# ---------------------------------------------------------------------------
# 9. Config dict has exactly the 4 required bnb keys
# ---------------------------------------------------------------------------

class TestNF4RequiredKeys:
    def test_config_has_exactly_four_keys(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        result = get_bnb_nf4_config()
        assert len(result) == 4

    def test_all_four_required_keys_present(self, monkeypatch):
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        result = get_bnb_nf4_config()
        required = {
            "load_in_4bit",
            "bnb_4bit_quant_type",
            "bnb_4bit_use_double_quant",
            "bnb_4bit_compute_dtype",
        }
        assert required == set(result.keys())


# ---------------------------------------------------------------------------
# 10. finetune() in mock QLoRA mode emits quant.nf4_config evidence
# ---------------------------------------------------------------------------

def _make_finetune_deps(tmp_path: Path, training_mode: str = "qlora"):
    """Build minimal FinetuneConfig, HALProfile, and SpecializationPack for tests."""
    from dojo.train.base import FinetuneConfig
    from dojo.core.types import (
        HALProfile, HardwareTier, LicenseSpec, PackFormat, ProvenanceManifest,
    )
    from dojo.pack.schema import SpecializationPack

    config = FinetuneConfig(
        pack_id="nf4-test-pack",
        base_model="test/model-0.5b",
        output_dir=tmp_path / "out",
        num_epochs=1,
        batch_size=1,
        lora_r=8,
        lora_alpha=16,
        max_seq_length=512,
        grad_accum_steps=1,
        training_mode=training_mode,
    )

    profile = HALProfile(
        tier=HardwareTier.TIER_4060_LAPTOP,
        gpu_name="Test GPU",
        vram_bytes=0,
        compute_capability="8.6",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="test",
        profile_sha256="0" * 64,
    )

    license_spec = LicenseSpec(
        spdx_id="proprietary",
        allows_derived_ft=True,
        allows_commercial_use=True,
        requires_attribution=False,
    )
    provenance = ProvenanceManifest(
        harvest_organ_version="0.0.0",
        sources=(),
        collection_date_utc="1970-01-01T00:00:00Z",
        dedup_method="none",
        human_reviewed=False,
    )
    pack = SpecializationPack(
        pack_id="nf4-test-pack",
        version="0.1.0",
        schema_version="1.0",
        sample_count=0,
        format=PackFormat.ALPACA,
        splits={"train": 0, "val": 0},
        harvest_receipt="test",
        license=license_spec,
        provenance=provenance,
        signature=b"\x00" * 64,
        manifest_sha256="0" * 64,
        data_parquet_sha256="0" * 64,
        data_path=None,
    )

    return config, profile, pack


def _run_finetune_mock(tmp_path, monkeypatch, training_mode: str):
    """Helper: run finetune() in mock mode with all slow external deps patched out."""
    monkeypatch.setenv("DOJO_REAL_GPU", "0")
    monkeypatch.setenv("DOJO_MLFLOW_URI", "")  # disable MLflow network call

    _inject_bnb(monkeypatch)
    if "dojo.quant.nf4" in sys.modules:
        del sys.modules["dojo.quant.nf4"]

    # Patch ExperimentTracker at the module level so it never attempts a real MLflow connection.
    # We must import the module object via sys.modules (the @one_door decorator wraps the
    # function so `import dojo.train.finetune` yields the decorated function, not the module).
    import importlib
    from unittest.mock import MagicMock
    from contextlib import contextmanager

    # After @one_door wrapping, the module IS in sys.modules under its dotted name
    importlib.import_module("dojo.train.finetune")
    import sys as _sys
    _ft_mod = _sys.modules["dojo.train.finetune"]

    mock_tracker = MagicMock()

    @contextmanager
    def _noop_run(run_cfg):
        yield

    mock_tracker.run = _noop_run
    mock_tracker.log_params = MagicMock()
    mock_tracker.log_metrics_batch = MagicMock()

    monkeypatch.setattr(_ft_mod, "ExperimentTracker", lambda **kw: mock_tracker)

    from dojo.train.finetune import finetune
    config, profile, pack = _make_finetune_deps(tmp_path, training_mode=training_mode)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return finetune(config, profile, pack)


class TestFinetuneEmitsNF4Evidence:
    def test_finetune_mock_qlora_emits_nf4_config(self, tmp_path, monkeypatch):
        """finetune() with DOJO_REAL_GPU=0 and training_mode=qlora emits quant.nf4_config."""
        _run_finetune_mock(tmp_path, monkeypatch, training_mode="qlora")
        events = _nf4_events()
        assert len(events) >= 1, "Expected quant.nf4_config evidence from finetune() mock path"

    def test_finetune_mock_default_mode_emits_nf4_config(self, tmp_path, monkeypatch):
        """finetune() with training_mode=full_ft (not sft/bf16/fp8) emits quant.nf4_config."""
        _run_finetune_mock(tmp_path, monkeypatch, training_mode="full_ft")
        events = _nf4_events()
        assert len(events) >= 1

    def test_finetune_sft_mode_does_not_emit_nf4_config(self, tmp_path, monkeypatch):
        """finetune() with training_mode=sft bypasses the NF4 path."""
        _run_finetune_mock(tmp_path, monkeypatch, training_mode="sft")
        # sft mode should NOT emit quant.nf4_config
        events = _nf4_events()
        assert len(events) == 0


# ---------------------------------------------------------------------------
# 11. finetune() NF4 evidence appears before training-step evidence
# ---------------------------------------------------------------------------

class TestNF4EvidenceOrdering:
    def test_nf4_evidence_before_training_step(self, tmp_path, monkeypatch):
        """quant.nf4_config must be emitted before the first train.finetune.step."""
        _run_finetune_mock(tmp_path, monkeypatch, training_mode="qlora")

        all_events = EvidenceLogger.get_events()
        types = [e.get("type") for e in all_events]

        nf4_idx = next((i for i, t in enumerate(types) if t == "quant.nf4_config"), None)
        step_idx = next((i for i, t in enumerate(types) if t == EvidenceType.TRAINING_STEP.value), None)

        assert nf4_idx is not None, "quant.nf4_config not found"
        assert step_idx is not None, "training step not found"
        assert nf4_idx < step_idx, "NF4 evidence must precede training step evidence"


# ---------------------------------------------------------------------------
# 12. NF4 config values are preserved through finetune evidence chain
# ---------------------------------------------------------------------------

class TestNF4EvidenceValuesInFinetune:
    def test_nf4_evidence_has_correct_quant_type(self, tmp_path, monkeypatch):
        _run_finetune_mock(tmp_path, monkeypatch, training_mode="qlora")
        events = _nf4_events()
        assert events[0]["data"]["bnb_4bit_quant_type"] == "nf4"

    def test_nf4_evidence_has_load_in_4bit(self, tmp_path, monkeypatch):
        _run_finetune_mock(tmp_path, monkeypatch, training_mode="qlora")
        events = _nf4_events()
        assert events[0]["data"]["load_in_4bit"] is True


# ---------------------------------------------------------------------------
# 13. bnb_4bit_quant_type in finetune evidence matches get_bnb_nf4_config() output
# ---------------------------------------------------------------------------

class TestFinetuneEvidenceMatchesConfig:
    def test_finetune_nf4_quant_type_matches_direct_call(self, tmp_path, monkeypatch):
        # First get the direct config value
        direct_fn = _reload_nf4(monkeypatch)
        direct_config = direct_fn()
        direct_quant_type = direct_config["bnb_4bit_quant_type"]

        # Clear events so finetune events are fresh
        EvidenceLogger.clear()

        # Now run finetune mock (re-uses the already-injected bnb from monkeypatch)
        _run_finetune_mock(tmp_path, monkeypatch, training_mode="qlora")

        ft_nf4_events = _nf4_events()
        assert len(ft_nf4_events) >= 1
        assert ft_nf4_events[0]["data"]["bnb_4bit_quant_type"] == direct_quant_type


# ---------------------------------------------------------------------------
# 14. Memory savings: bnb_4bit_use_double_quant=True is documented in evidence
# ---------------------------------------------------------------------------

class TestDoubleQuantMemorySavings:
    def test_double_quant_documented_in_nf4_evidence(self, monkeypatch):
        """bnb_4bit_use_double_quant=True in evidence signals the memory-saving
        double quantization feature is active for this training run."""
        get_bnb_nf4_config = _reload_nf4(monkeypatch)
        get_bnb_nf4_config()

        events = _nf4_events()
        assert len(events) >= 1
        data = events[0]["data"]
        # Double quantization saves ~0.4 bits/param — True means it is enabled
        assert data["bnb_4bit_use_double_quant"] is True, (
            "bnb_4bit_use_double_quant must be True to enable memory-saving double quantization"
        )

    def test_finetune_evidence_documents_double_quant(self, tmp_path, monkeypatch):
        """finetune() mock evidence chain preserves double quant flag."""
        _run_finetune_mock(tmp_path, monkeypatch, training_mode="qlora")

        events = _nf4_events()
        assert len(events) >= 1
        assert events[0]["data"].get("bnb_4bit_use_double_quant") is True
