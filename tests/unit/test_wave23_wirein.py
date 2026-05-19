"""Wave 23 wire-in integration tests — final 5 orphans."""
from __future__ import annotations

import inspect
from pathlib import Path


DOJO_ROOT = Path(__file__).resolve().parents[2] / "dojo"


def _has_import(module_path: str) -> bool:
    for p in DOJO_ROOT.rglob("*.py"):
        if "tests" in p.parts:
            continue
        if module_path in p.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


class TestSmDetectWiredIn:
    def test_finetune_imports_sm_detect(self):
        # Read the file directly — inspect.getsource on a module returns
        # the module body but excludes nested function source in some envs.
        src = (DOJO_ROOT / "train" / "finetune.py").read_text(encoding="utf-8")
        assert "dojo.hal.sm_detect" in src

    def test_detect_sm_version_uses_new_path(self):
        """The legacy _detect_sm_version should now route through dojo.hal.sm_detect first."""
        from dojo.train.finetune import _detect_sm_version
        src = inspect.getsource(_detect_sm_version)
        assert "detect_sm" in src


class TestGgufValidatorWiredIn:
    def test_gguf_imports_validator(self):
        from dojo.quant import gguf
        src = inspect.getsource(gguf)
        assert "validate_gguf_file" in src or "gguf_validator" in src


class TestFp4WiredIn:
    def test_nf4_imports_fp4(self):
        from dojo.quant import nf4
        src = inspect.getsource(nf4)
        assert "from dojo.quant.fp4" in src or "requires_blackwell" in src

    def test_select_4bit_precision_format_default_nf4(self, monkeypatch):
        # Without Blackwell hardware, default to NF4
        from dojo.quant.nf4 import select_4bit_precision_format
        from unittest.mock import patch
        with patch("dojo.hal.sm_detect.detect_sm") as ds:
            ds.return_value.available = False
            assert select_4bit_precision_format() == "nf4"

    def test_select_4bit_precision_format_fp4_on_blackwell(self):
        from dojo.quant.nf4 import select_4bit_precision_format
        from unittest.mock import MagicMock, patch
        info = MagicMock()
        info.available = True
        info.sm_float = 10.0
        with patch("dojo.hal.sm_detect.detect_sm", return_value=info):
            assert select_4bit_precision_format() == "fp4"


class TestMultiJudgeGateWiredIn:
    def test_harness_exposes_run_multi_judge_gate(self):
        from dojo.eval.harness import run_multi_judge_gate
        assert callable(run_multi_judge_gate)

    def test_run_multi_judge_gate_uses_real_class(self):
        from dojo.eval.harness import run_multi_judge_gate

        def good(_resp, _crit):
            return 8.0

        verdict = run_multi_judge_gate(
            response="hello",
            criteria=["clarity"],
            judges=[("a", good), ("b", good)],
        )
        assert verdict["overall_pass"] is True
        assert verdict["agreement"] == 1.0


class TestRecipeDiffWiredIn:
    def test_recipe_export_imports_recipe_diff(self):
        from dojo.pack import recipe_export
        src = inspect.getsource(recipe_export)
        assert "from dojo.train.recipe_diff" in src or "diff_recipes" in src

    def test_diff_recipe_files_returns_structured_diff(self, tmp_path: Path):
        import json
        from dojo.pack.recipe_export import diff_recipe_files
        p1 = tmp_path / "r1.json"
        p2 = tmp_path / "r2.json"
        # Need to satisfy load_recipe's required-field check
        from dojo.pack.recipe_export import RECIPE_REQUIRED_FIELDS, RECIPE_VERSION
        base = {f: (RECIPE_VERSION if f == "recipe_version" else "x")
                for f in RECIPE_REQUIRED_FIELDS}
        p1.write_text(json.dumps(base))
        new = dict(base)
        new["pack_id"] = "different"
        p2.write_text(json.dumps(new))
        diff = diff_recipe_files(p1, p2)
        assert any(c["path"] == "pack_id" for c in diff["changed"])


class TestWave23NotOrphaned:
    def test_sm_detect_imported(self):
        assert _has_import("dojo.hal.sm_detect")

    def test_gguf_validator_imported(self):
        assert _has_import("dojo.quant.gguf_validator")

    def test_fp4_imported(self):
        assert _has_import("dojo.quant.fp4")

    def test_multi_judge_gate_imported(self):
        assert _has_import("dojo.eval.multi_judge_gate")

    def test_recipe_diff_imported(self):
        assert _has_import("dojo.train.recipe_diff")
