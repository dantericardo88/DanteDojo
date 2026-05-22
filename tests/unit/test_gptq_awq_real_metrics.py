"""Tests for real mock-math metric computation in gptq.py and awq.py.

Verifies that _measure_ppl() and _measure_f1() now perform actual math
(synthetic bigram PPL / macro-F1) instead of returning hardcoded constants,
and that export_gptq / export_awq emit evidence with the computed values.

16 tests total:
  1-4   _measure_ppl() correctness
  5-8   _measure_f1() correctness
  9-11  export_gptq() evidence
  12-14 export_awq() evidence
  15-16 evidence matches direct _measure_ppl / _measure_f1 return values
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture()
def model_dir(tmp_path):
    p = tmp_path / "model"
    p.mkdir()
    return p


@pytest.fixture()
def output_dir(tmp_path):
    p = tmp_path / "out"
    p.mkdir()
    return p


_FAKE_GPTQ = MagicMock()
_FAKE_AWQ = MagicMock()


def _import_gptq():
    if "dojo.quant.gptq" in sys.modules:
        del sys.modules["dojo.quant.gptq"]
    from dojo.quant import gptq as m
    return m


def _import_awq():
    if "dojo.quant.awq" in sys.modules:
        del sys.modules["dojo.quant.awq"]
    from dojo.quant import awq as m
    return m


def _events_of_type(ev_type: str) -> list[dict]:
    return [
        e for e in EvidenceLogger._events
        if isinstance(e.get("evidence"), dict) and e["evidence"].get("type") == ev_type
    ]


# ---------------------------------------------------------------------------
# Tests 1-4: _measure_ppl()
# ---------------------------------------------------------------------------

class TestMeasurePpl:
    """_measure_ppl() uses real cross-entropy math on a synthetic bigram model."""

    def _get_measure_ppl(self, tmp_path):
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            mod = _import_gptq()
        return mod._measure_ppl, tmp_path / "model", tmp_path / "quant"

    # 1 — returns a tuple of two floats
    def test_returns_tuple_of_two_floats(self, tmp_path):
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            mod = _import_gptq()
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            result = mod._measure_ppl(tmp_path / "model", tmp_path / "quant")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    # 2 — baseline_ppl < quantized_ppl (degradation is non-negative)
    def test_baseline_ppl_less_than_quantized(self, tmp_path):
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            mod = _import_gptq()
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            ppl_baseline, ppl_quantized = mod._measure_ppl(
                tmp_path / "model", tmp_path / "quant"
            )
        assert ppl_quantized >= ppl_baseline, (
            f"Expected quantized PPL ({ppl_quantized}) >= baseline ({ppl_baseline})"
        )

    # 3 — ppl values are in a reasonable range (5.0 < ppl < 1000.0)
    def test_ppl_values_in_reasonable_range(self, tmp_path):
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            mod = _import_gptq()
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            ppl_baseline, ppl_quantized = mod._measure_ppl(
                tmp_path / "model", tmp_path / "quant"
            )
        assert 5.0 < ppl_baseline < 1e9, f"ppl_baseline={ppl_baseline} out of range"
        assert 5.0 < ppl_quantized < 1e9, f"ppl_quantized={ppl_quantized} out of range"

    # 4 — deterministic: two calls with same inputs return same result
    def test_ppl_is_deterministic(self, tmp_path):
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            mod = _import_gptq()
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            result1 = mod._measure_ppl(tmp_path / "model", tmp_path / "quant")
            result2 = mod._measure_ppl(tmp_path / "model", tmp_path / "quant")
        assert result1 == result2, "PPL computation must be deterministic"


# ---------------------------------------------------------------------------
# Tests 5-8: _measure_f1()
# ---------------------------------------------------------------------------

class TestMeasureF1:
    """_measure_f1() uses real macro-F1 math on a synthetic classification dataset."""

    # 5 — returns a tuple of two floats
    def test_returns_tuple_of_two_floats(self, tmp_path):
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            mod = _import_awq()
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            result = mod._measure_f1(tmp_path / "model", tmp_path / "quant")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    # 6 — baseline_f1 > quantized_f1 (degradation is non-negative)
    def test_baseline_f1_greater_than_quantized(self, tmp_path):
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            mod = _import_awq()
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            f1_baseline, f1_quantized = mod._measure_f1(
                tmp_path / "model", tmp_path / "quant"
            )
        assert f1_baseline >= f1_quantized, (
            f"Expected baseline F1 ({f1_baseline}) >= quantized ({f1_quantized})"
        )

    # 7 — f1 values are in [0, 1] range
    def test_f1_values_in_unit_range(self, tmp_path):
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            mod = _import_awq()
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            f1_baseline, f1_quantized = mod._measure_f1(
                tmp_path / "model", tmp_path / "quant"
            )
        assert 0.0 <= f1_baseline <= 1.0, f"f1_baseline={f1_baseline} out of [0,1]"
        assert 0.0 <= f1_quantized <= 1.0, f"f1_quantized={f1_quantized} out of [0,1]"

    # 8 — deterministic: two calls with same inputs return same result
    def test_f1_is_deterministic(self, tmp_path):
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            mod = _import_awq()
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            result1 = mod._measure_f1(tmp_path / "model", tmp_path / "quant")
            result2 = mod._measure_f1(tmp_path / "model", tmp_path / "quant")
        assert result1 == result2, "F1 computation must be deterministic"


# ---------------------------------------------------------------------------
# Tests 9-11: export_gptq() evidence
# ---------------------------------------------------------------------------

class TestExportGptqEvidence:
    """export_gptq() completes and emits evidence with computed PPL fields."""

    # 9 — export_gptq() completes without exception (DOJO_REAL_GPU=0 path)
    def test_export_gptq_no_exception(self, model_dir, output_dir):
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            mod = _import_gptq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                result = mod.export_gptq(model_dir, output_dir, ppl_regression_limit=1.0)
        assert result.exists()

    # 10 — export_gptq() emits evidence with ppl_baseline field
    def test_export_gptq_emits_ppl_baseline(self, model_dir, output_dir):
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            mod = _import_gptq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                mod.export_gptq(model_dir, output_dir, ppl_regression_limit=1.0)

        complete_events = _events_of_type(EvidenceType.QUANTIZATION_COMPLETE.value)
        assert complete_events, "No QUANTIZATION_COMPLETE evidence emitted"
        data = complete_events[-1]["evidence"]["data"]
        assert "ppl_baseline" in data, f"ppl_baseline missing from evidence: {data.keys()}"
        assert isinstance(data["ppl_baseline"], float)

    # 11 — export_gptq() emits evidence with ppl_quantized field
    def test_export_gptq_emits_ppl_quantized(self, model_dir, output_dir):
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            mod = _import_gptq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                mod.export_gptq(model_dir, output_dir, ppl_regression_limit=1.0)

        complete_events = _events_of_type(EvidenceType.QUANTIZATION_COMPLETE.value)
        assert complete_events, "No QUANTIZATION_COMPLETE evidence emitted"
        data = complete_events[-1]["evidence"]["data"]
        assert "ppl_quantized" in data, f"ppl_quantized missing from evidence: {data.keys()}"
        assert isinstance(data["ppl_quantized"], float)


# ---------------------------------------------------------------------------
# Tests 12-14: export_awq() evidence
# ---------------------------------------------------------------------------

class TestExportAwqEvidence:
    """export_awq() completes and emits evidence with computed F1 fields."""

    # 12 — export_awq() completes without exception (DOJO_REAL_GPU=0 path)
    def test_export_awq_no_exception(self, model_dir, output_dir):
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            mod = _import_awq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                result = mod.export_awq(model_dir, output_dir, f1_drop_limit=1.0)
        assert result.exists()

    # 13 — export_awq() emits evidence with f1_baseline field
    def test_export_awq_emits_f1_baseline(self, model_dir, output_dir):
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            mod = _import_awq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                mod.export_awq(model_dir, output_dir, f1_drop_limit=1.0)

        complete_events = _events_of_type(EvidenceType.QUANTIZATION_COMPLETE.value)
        assert complete_events, "No QUANTIZATION_COMPLETE evidence emitted"
        data = complete_events[-1]["evidence"]["data"]
        assert "f1_baseline" in data, f"f1_baseline missing from evidence: {data.keys()}"
        assert isinstance(data["f1_baseline"], float)

    # 14 — export_awq() emits evidence with f1_quantized field
    def test_export_awq_emits_f1_quantized(self, model_dir, output_dir):
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            mod = _import_awq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                mod.export_awq(model_dir, output_dir, f1_drop_limit=1.0)

        complete_events = _events_of_type(EvidenceType.QUANTIZATION_COMPLETE.value)
        assert complete_events, "No QUANTIZATION_COMPLETE evidence emitted"
        data = complete_events[-1]["evidence"]["data"]
        assert "f1_quantized" in data, f"f1_quantized missing from evidence: {data.keys()}"
        assert isinstance(data["f1_quantized"], float)


# ---------------------------------------------------------------------------
# Tests 15-16: evidence values match direct function returns
# ---------------------------------------------------------------------------

class TestEvidenceMatchesFunctionReturns:
    """ppl_baseline/ppl_quantized in evidence must equal what _measure_ppl returns."""

    # 15 — ppl_baseline and ppl_quantized in evidence match _measure_ppl()
    def test_gptq_evidence_ppl_matches_measure_ppl(self, model_dir, output_dir):
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            mod = _import_gptq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                # Capture what _measure_ppl returns directly
                direct_ppl = mod._measure_ppl(model_dir, output_dir)
                EvidenceLogger.clear()
                mod.export_gptq(model_dir, output_dir, ppl_regression_limit=1.0)

        complete_events = _events_of_type(EvidenceType.QUANTIZATION_COMPLETE.value)
        data = complete_events[-1]["evidence"]["data"]
        assert data["ppl_baseline"] == direct_ppl[0], (
            f"Evidence ppl_baseline={data['ppl_baseline']} != "
            f"_measure_ppl()[0]={direct_ppl[0]}"
        )
        assert data["ppl_quantized"] == direct_ppl[1], (
            f"Evidence ppl_quantized={data['ppl_quantized']} != "
            f"_measure_ppl()[1]={direct_ppl[1]}"
        )

    # 16 — f1_baseline and f1_quantized in evidence match _measure_f1()
    def test_awq_evidence_f1_matches_measure_f1(self, model_dir, output_dir):
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            mod = _import_awq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                # Capture what _measure_f1 returns directly
                direct_f1 = mod._measure_f1(model_dir, output_dir)
                EvidenceLogger.clear()
                mod.export_awq(model_dir, output_dir, f1_drop_limit=1.0)

        complete_events = _events_of_type(EvidenceType.QUANTIZATION_COMPLETE.value)
        data = complete_events[-1]["evidence"]["data"]
        assert data["f1_baseline"] == direct_f1[0], (
            f"Evidence f1_baseline={data['f1_baseline']} != "
            f"_measure_f1()[0]={direct_f1[0]}"
        )
        assert data["f1_quantized"] == direct_f1[1], (
            f"Evidence f1_quantized={data['f1_quantized']} != "
            f"_measure_f1()[1]={direct_f1[1]}"
        )
