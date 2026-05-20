"""Unit tests for dojo/quant/gptq.py — d22 real GPTQ 4-bit quantization.

All tests run without DOJO_REAL_GPU=1 (CI-safe). gptqmodel is injected via
sys.modules so the import guard passes without the library being installed.

Covers:
1. test_export_gptq_stub_mode            — returns output Path, quantize_config.json exists
2. test_export_gptq_emits_start_and_complete — both evidence events emitted
3. test_export_gptq_ppl_regression_raises   — 50% PPL regression → QuantQualityRejected
4. test_export_gptq_ppl_regression_evidence — evidence has ppl_before/after/regression_pct
5. test_export_gptq_missing_gptqmodel       — ImportError → QuantizationError with hint
6. test_export_gptq_bits_in_output_path     — output dir name contains bit count
7. test_export_gptq_group_size_in_config    — quantize_config.json contains correct group_size
8. test_export_gptq_quality_gate_passed_evidence — quality_gate = "passed" in COMPLETE event
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantizationError, QuantQualityRejected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gptqmodel_mock() -> ModuleType:
    """Return a MagicMock module that satisfies the gptqmodel import guard."""
    mock = MagicMock(spec=ModuleType)
    mock.__name__ = "gptqmodel"
    return mock


def _reload_gptq_with_mock(mock_mod: ModuleType):
    """Inject mock into sys.modules and reload dojo.quant.gptq."""
    sys.modules["gptqmodel"] = mock_mod
    import dojo.quant.gptq as gptq_module
    importlib.reload(gptq_module)
    return gptq_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset singleton evidence logger between tests."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def remove_gptqmodel_after():
    """Ensure gptqmodel mock is removed from sys.modules after each test."""
    yield
    sys.modules.pop("gptqmodel", None)
    # Reload the real module (stub state) so later tests start clean
    if "dojo.quant.gptq" in sys.modules:
        try:
            importlib.reload(sys.modules["dojo.quant.gptq"])
        except Exception:
            pass


@pytest.fixture
def gptq_module_with_mock():
    """Provide gptq module with gptqmodel mocked, in stub (non-GPU) mode."""
    mock_mod = _make_gptqmodel_mock()
    sys.modules["gptqmodel"] = mock_mod
    import dojo.quant.gptq as mod
    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Stub mode — returns Path, quantize_config.json exists
# ---------------------------------------------------------------------------

class TestExportGptqStubMode:
    def test_export_gptq_stub_mode(self, tmp_path, gptq_module_with_mock):
        model_path = tmp_path / "my-model"
        model_path.mkdir()
        output_dir = tmp_path / "out"

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            result = gptq_module_with_mock.export_gptq(
                model_path=model_path,
                output_dir=output_dir,
            )

        assert isinstance(result, Path)
        assert result.exists()
        assert (result / "quantize_config.json").exists()


# ---------------------------------------------------------------------------
# 2. Both evidence events emitted
# ---------------------------------------------------------------------------

class TestExportGptqEvidence:
    def test_export_gptq_emits_start_and_complete(self, tmp_path, gptq_module_with_mock):
        model_path = tmp_path / "my-model"
        model_path.mkdir()
        output_dir = tmp_path / "out"

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            gptq_module_with_mock.export_gptq(
                model_path=model_path,
                output_dir=output_dir,
            )

        event_types = [
            e["evidence"].get("type")
            for e in EvidenceLogger._events
            if isinstance(e.get("evidence"), dict)
        ]

        assert EvidenceType.QUANTIZATION_START.value in event_types
        assert EvidenceType.QUANTIZATION_COMPLETE.value in event_types


# ---------------------------------------------------------------------------
# 3. PPL regression above 20% threshold raises QuantQualityRejected
# ---------------------------------------------------------------------------

class TestExportGptqPplRegressionRaises:
    def test_export_gptq_ppl_regression_raises(self, tmp_path, gptq_module_with_mock):
        """Patch _measure_ppl so after is 50% higher than before."""
        model_path = tmp_path / "my-model"
        model_path.mkdir()
        output_dir = tmp_path / "out"

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            with patch.object(gptq_module_with_mock, "_measure_ppl", return_value=(20.0, 30.0)):
                with pytest.raises(QuantQualityRejected) as exc_info:
                    gptq_module_with_mock.export_gptq(
                        model_path=model_path,
                        output_dir=output_dir,
                    )

        assert "ppl" in str(exc_info.value).lower() or "regression" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 4. Evidence has ppl_before, ppl_after, ppl_regression_pct
# ---------------------------------------------------------------------------

class TestExportGptqPplRegressionEvidence:
    def test_export_gptq_ppl_regression_evidence(self, tmp_path, gptq_module_with_mock):
        """Even on rejection, QUANTIZATION_COMPLETE is emitted with PPL data."""
        model_path = tmp_path / "my-model"
        model_path.mkdir()
        output_dir = tmp_path / "out"

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            with patch.object(gptq_module_with_mock, "_measure_ppl", return_value=(20.0, 30.0)):
                with pytest.raises(QuantQualityRejected):
                    gptq_module_with_mock.export_gptq(
                        model_path=model_path,
                        output_dir=output_dir,
                    )

        complete_events = [
            e["evidence"]
            for e in EvidenceLogger._events
            if isinstance(e.get("evidence"), dict)
            and e["evidence"].get("type") == EvidenceType.QUANTIZATION_COMPLETE.value
        ]

        assert len(complete_events) == 1
        data = complete_events[0]["data"]
        assert "ppl_before" in data
        assert "ppl_after" in data
        assert "ppl_regression_pct" in data
        assert data["ppl_before"] == 20.0
        assert data["ppl_after"] == 30.0
        assert data["ppl_regression_pct"] == pytest.approx(50.0, abs=0.01)


# ---------------------------------------------------------------------------
# 5. Missing gptqmodel → QuantizationError with install hint
# ---------------------------------------------------------------------------

class TestExportGptqMissingGptqmodel:
    def test_export_gptq_missing_gptqmodel(self, tmp_path):
        """When gptqmodel is absent in real-GPU mode, raise QuantizationError with pip install hint."""
        # Remove any injected mock so the import actually fails
        sys.modules.pop("gptqmodel", None)

        import dojo.quant.gptq as gptq_module
        importlib.reload(gptq_module)

        with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
            with patch.dict(sys.modules, {"gptqmodel": None}):
                with pytest.raises(QuantizationError) as exc_info:
                    gptq_module.export_gptq(
                        model_path=tmp_path / "model",
                        output_dir=tmp_path / "out",
                    )

        msg = str(exc_info.value)
        assert "gptq-model" in msg or "gptqmodel" in msg.lower()
        assert "pip install" in msg


# ---------------------------------------------------------------------------
# 6. Bit count appears in output directory name
# ---------------------------------------------------------------------------

class TestExportGptqBitsInOutputPath:
    def test_export_gptq_bits_in_output_path(self, tmp_path, gptq_module_with_mock):
        model_path = tmp_path / "some-model"
        model_path.mkdir()
        output_dir = tmp_path / "out"

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            result = gptq_module_with_mock.export_gptq(
                model_path=model_path,
                output_dir=output_dir,
                bits=4,
            )

        assert "4bit" in result.name or "4" in result.name


# ---------------------------------------------------------------------------
# 7. quantize_config.json contains correct group_size
# ---------------------------------------------------------------------------

class TestExportGptqGroupSizeInConfig:
    def test_export_gptq_group_size_in_config(self, tmp_path, gptq_module_with_mock):
        model_path = tmp_path / "my-model"
        model_path.mkdir()
        output_dir = tmp_path / "out"

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            result = gptq_module_with_mock.export_gptq(
                model_path=model_path,
                output_dir=output_dir,
                bits=4,
                group_size=64,
            )

        config_path = result / "quantize_config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["group_size"] == 64
        assert config["bits"] == 4


# ---------------------------------------------------------------------------
# 8. quality_gate = "passed" in COMPLETE evidence on success
# ---------------------------------------------------------------------------

class TestExportGptqQualityGatePassedEvidence:
    def test_export_gptq_quality_gate_passed_evidence(self, tmp_path, gptq_module_with_mock):
        model_path = tmp_path / "my-model"
        model_path.mkdir()
        output_dir = tmp_path / "out"

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            gptq_module_with_mock.export_gptq(
                model_path=model_path,
                output_dir=output_dir,
            )

        complete_events = [
            e["evidence"]
            for e in EvidenceLogger._events
            if isinstance(e.get("evidence"), dict)
            and e["evidence"].get("type") == EvidenceType.QUANTIZATION_COMPLETE.value
        ]

        assert len(complete_events) == 1
        assert complete_events[0]["data"]["quality_gate"] == "passed"


# ---------------------------------------------------------------------------
# Bonus: one_door entry event is emitted
# ---------------------------------------------------------------------------

class TestExportGptqOneDoor:
    def test_one_door_entry_event_emitted(self, tmp_path, gptq_module_with_mock):
        model_path = tmp_path / "my-model"
        model_path.mkdir()
        output_dir = tmp_path / "out"

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            gptq_module_with_mock.export_gptq(
                model_path=model_path,
                output_dir=output_dir,
            )

        event_types = [
            e["evidence"].get("type")
            for e in EvidenceLogger._events
            if isinstance(e.get("evidence"), dict)
        ]
        assert EvidenceType.ONE_DOOR_ENTRY.value in event_types
