"""Wave 20 wire-in integration tests.

Each test exercises the USER-FACING entry point and asserts that the
previously-orphaned module is now invoked by production code. The tests
fail if someone removes the wiring (e.g. by reverting the integration
edits) — that's the whole point.

Maps to:
  d22: gptq_emulation now called from dojo/quant/gptq.py:export_gptq
  d23: awq_emulation now called from dojo/quant/awq.py:export_awq
  d37: sign_lm_eval_receipt now called from dojo/eval/harness.py:run_eval
  d46: validate_axolotl_config now called from
       dojo/train/axolotl_bridge.py:export_axolotl_config
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch  # noqa: F401 — pre-load torch so patch.dict(sys.modules) teardown doesn't trigger double-init


# ---------------------------------------------------------------------------
# d22 GPTQ wire-in
# ---------------------------------------------------------------------------

class TestGptqEmulationWiredIn:
    def test_export_gptq_source_imports_emulation(self):
        from dojo.quant import gptq
        src = inspect.getsource(gptq)
        assert "gptq_emulation" in src, (
            "gptq.py must reference gptq_emulation — it was orphaned before wave 20"
        )

    def test_stub_mode_writes_real_round_trip_metrics(self, tmp_path: Path):
        """The stub path now actually runs the emulation, producing real
        round-trip error metrics rather than a placeholder config."""
        from dojo.quant.gptq import export_gptq

        # Mock gptqmodel so _require_gptqmodel passes; we're testing stub path
        with patch.dict(sys.modules, {"gptqmodel": __import__("types").ModuleType("gptqmodel")}):
            export_gptq(
                model_path=tmp_path / "fake_model",
                output_dir=tmp_path,
                bits=4,
                group_size=128,
            )

        cfg_path = tmp_path / "fake_model-gptq-4bit" / "quantize_config.json"
        assert cfg_path.exists()
        cfg = json.loads(cfg_path.read_text())
        assert cfg["emulation"] == "gptq_emulation.v1"
        assert "round_trip_rmse" in cfg
        assert cfg["round_trip_rmse"] > 0  # real math ran

    def test_higher_bits_yields_lower_round_trip_error(self, tmp_path: Path):
        """Proves the emulation is REAL: more bits → less error."""
        from dojo.quant.gptq import export_gptq

        with patch.dict(sys.modules, {"gptqmodel": __import__("types").ModuleType("gptqmodel")}):
            export_gptq(model_path=tmp_path / "m", output_dir=tmp_path / "out4",
                        bits=4, group_size=128)
            export_gptq(model_path=tmp_path / "m", output_dir=tmp_path / "out8",
                        bits=8, group_size=128)

        cfg4 = json.loads((tmp_path / "out4" / "m-gptq-4bit" / "quantize_config.json").read_text())
        cfg8 = json.loads((tmp_path / "out8" / "m-gptq-8bit" / "quantize_config.json").read_text())
        assert cfg8["round_trip_rmse"] < cfg4["round_trip_rmse"]


# ---------------------------------------------------------------------------
# d23 AWQ wire-in
# ---------------------------------------------------------------------------

class TestAwqEmulationWiredIn:
    def test_export_awq_source_imports_emulation(self):
        from dojo.quant import awq
        src = inspect.getsource(awq)
        assert "awq_emulation" in src, (
            "awq.py must reference awq_emulation — it was orphaned before wave 20"
        )

    def test_stub_mode_writes_real_round_trip_metrics(self, tmp_path: Path):
        from dojo.quant.awq import export_awq

        with patch.dict(sys.modules, {"awq": __import__("types").ModuleType("awq")}):
            with patch("dojo.quant.awq._measure_f1", return_value=(0.90, 0.88)):
                export_awq(
                    model_path=tmp_path / "fake_model",
                    output_dir=tmp_path,
                    bits=4,
                    group_size=128,
                )

        cfg_path = tmp_path / "fake_model-awq-4bit" / "quantize_config.json"
        assert cfg_path.exists()
        cfg = json.loads(cfg_path.read_text())
        assert cfg["emulation"] == "awq_emulation.v1"
        assert cfg["round_trip_rmse"] > 0
        assert "activation_weighted_rmse" in cfg


# ---------------------------------------------------------------------------
# d37 lm-eval receipt signing wire-in
# ---------------------------------------------------------------------------

class TestLmEvalSigningWiredIn:
    def test_run_eval_emits_signed_evidence(self):
        """run_eval() must call sign_lm_eval_receipt — verified by checking
        the EVAL_COMPLETE evidence carries a signature."""
        from dojo.core.evidence import EvidenceLogger
        from dojo.core.types import ModelArtifact
        from dojo.eval.harness import EvalSuite, run_eval

        EvidenceLogger.clear()

        model = ModelArtifact(
            artifact_id="test-artifact",
            base_model="test-base",
            lora_weights=None,
            quantization=None,
            training_receipt="r" * 64,
            passport=None,
        )
        suite = EvalSuite(suite_type="dante_v1", tasks=("toxicity_basic",))
        run_eval(model, suite=suite, seeds=1, allow_synthetic_fallback=True)

        events = EvidenceLogger.get_events()
        eval_complete = [e for e in events
                         if e.get("type") == "eval.complete"]
        assert eval_complete, "run_eval must emit eval.complete"
        assert "signature" in eval_complete[-1]["data"], (
            "EVAL_COMPLETE must include signature — d37 wire-in"
        )
        assert eval_complete[-1]["data"]["signature_version"] == "v1"

    def test_harness_source_imports_receipt_signer(self):
        from dojo.eval import harness
        src = inspect.getsource(harness)
        assert "lm_eval_receipt" in src or "sign_lm_eval_receipt" in src, (
            "harness.py must reference lm_eval_receipt — it was orphaned before wave 20"
        )


# ---------------------------------------------------------------------------
# d46 axolotl schema validation wire-in
# ---------------------------------------------------------------------------

class TestAxolotlValidationWiredIn:
    def test_export_axolotl_source_imports_schema(self):
        from dojo.train import axolotl_bridge
        src = inspect.getsource(axolotl_bridge)
        assert "axolotl_schema" in src or "validate_axolotl_config" in src, (
            "axolotl_bridge.py must reference axolotl_schema — it was orphaned before wave 20"
        )

    def test_export_axolotl_validates_before_write(self, tmp_path: Path):
        """If validate_axolotl_config rejects the cfg, export_axolotl_config
        must raise — proving validation runs before the YAML write."""
        from dojo.train.axolotl_bridge import export_axolotl_config
        from dojo.train.axolotl_schema import AxolotlConfigValidationError

        bad_cfg = {"missing": "everything"}

        with patch(
            "dojo.train.axolotl_bridge.axolotl_config_from_finetune",
            return_value=bad_cfg,
        ):
            with pytest.raises(AxolotlConfigValidationError):
                export_axolotl_config(
                    config=None,  # type: ignore[arg-type]
                    pack=None,    # type: ignore[arg-type]
                    output_path=tmp_path / "out.yaml",
                )

        # Must NOT have written the file — validation should fail-closed
        assert not (tmp_path / "out.yaml").exists()


# ---------------------------------------------------------------------------
# Production-import grep (the orphan check)
# ---------------------------------------------------------------------------

class TestNotOrphaned:
    """For each wired module, assert at least one non-test import exists."""

    DOJO_ROOT = Path(__file__).resolve().parents[2] / "dojo"

    def _count_imports(self, module_path: str) -> int:
        """Count `from <module_path> import` lines in dojo/, excluding tests."""
        count = 0
        for p in self.DOJO_ROOT.rglob("*.py"):
            if "tests" in p.parts:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if f"from {module_path}" in text or f"import {module_path}" in text:
                count += 1
        return count

    def test_gptq_emulation_imported_in_production(self):
        assert self._count_imports("dojo.quant.gptq_emulation") >= 1

    def test_awq_emulation_imported_in_production(self):
        assert self._count_imports("dojo.quant.awq_emulation") >= 1

    def test_lm_eval_receipt_imported_in_production(self):
        assert self._count_imports("dojo.eval.lm_eval_receipt") >= 1

    def test_axolotl_schema_imported_in_production(self):
        assert self._count_imports("dojo.train.axolotl_schema") >= 1
