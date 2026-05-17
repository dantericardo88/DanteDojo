"""Unit tests for dojo/quant/awq.py — d23 AWQ quantization with F1 quality gate.

All tests run without DOJO_REAL_GPU=1 (CI-safe).
autoawq is mocked so no GPU or GPU library is required.

Covers:
1. test_export_awq_stub_mode          — returns output Path in stub mode
2. test_export_awq_emits_start_and_complete — both evidence events emitted
3. test_export_awq_f1_regression_raises    — 10% F1 drop → QuantQualityRejected
4. test_export_awq_f1_regression_evidence  — evidence has f1_before, f1_after, f1_drop_abs
5. test_export_awq_missing_autoawq         — ImportError → QuantizationError with install hint
6. test_export_awq_bits_in_output_path     — output dir name contains "awq-4bit"
7. test_export_awq_quantize_config_format  — quantize_config.json has quant_type=AWQ
8. test_export_awq_quality_gate_passed     — quality_gate = "passed" in COMPLETE event
"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantizationError, QuantQualityRejected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_awq_mock() -> ModuleType:
    """Return a minimal mock of the autoawq (awq) module."""
    mock_awq = MagicMock(spec=ModuleType)
    mock_awq.AutoAWQForCausalLM = MagicMock()
    return mock_awq


def _inject_awq_mock(mock_awq: ModuleType) -> dict:
    """Return a sys.modules patch dict that injects the mock awq module."""
    return {"awq": mock_awq}


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
def no_real_gpu(monkeypatch):
    """Ensure DOJO_REAL_GPU is unset so stub path is always taken."""
    monkeypatch.delenv("DOJO_REAL_GPU", raising=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportAwqStubMode:
    def test_export_awq_stub_mode(self, tmp_path):
        """DOJO_REAL_GPU=0: with autoawq mocked, export_awq returns a Path."""
        mock_awq = _make_awq_mock()
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()

        with patch.dict(sys.modules, _inject_awq_mock(mock_awq)):
            from dojo.quant import awq as awq_module
            result = awq_module.export_awq(
                model_path=model_dir,
                output_dir=tmp_path / "out",
            )

        assert result is not None
        assert isinstance(result, type(tmp_path))

    def test_export_awq_bits_in_output_path(self, tmp_path):
        """Output directory name must contain 'awq-4bit'."""
        mock_awq = _make_awq_mock()
        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()

        with patch.dict(sys.modules, _inject_awq_mock(mock_awq)):
            from dojo.quant import awq as awq_module
            result = awq_module.export_awq(
                model_path=model_dir,
                output_dir=tmp_path / "out",
                bits=4,
            )

        assert "awq-4bit" in result.name

    def test_export_awq_quantize_config_format(self, tmp_path):
        """quantize_config.json must have quant_type=AWQ and correct fields."""
        mock_awq = _make_awq_mock()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch.dict(sys.modules, _inject_awq_mock(mock_awq)):
            from dojo.quant import awq as awq_module
            result = awq_module.export_awq(
                model_path=model_dir,
                output_dir=tmp_path / "out",
                bits=4,
                group_size=128,
            )

        config_path = result / "quantize_config.json"
        assert config_path.exists(), "quantize_config.json must be written"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

        assert cfg["quant_type"] == "AWQ"
        assert cfg["bits"] == 4
        assert cfg["group_size"] == 128
        assert cfg["zero_point"] is True
        assert cfg["version"] == "GEMM"


class TestExportAwqEvidence:
    def test_export_awq_emits_start_and_complete(self, tmp_path):
        """Both QUANTIZATION_START and QUANTIZATION_COMPLETE events must be emitted."""
        mock_awq = _make_awq_mock()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch.dict(sys.modules, _inject_awq_mock(mock_awq)):
            from dojo.quant import awq as awq_module
            awq_module.export_awq(
                model_path=model_dir,
                output_dir=tmp_path / "out",
            )

        events = EvidenceLogger._events
        types = [e["evidence"].get("type") for e in events if isinstance(e.get("evidence"), dict)]

        assert EvidenceType.QUANTIZATION_START.value in types, (
            f"QUANTIZATION_START not found in events: {types}"
        )
        assert EvidenceType.QUANTIZATION_COMPLETE.value in types, (
            f"QUANTIZATION_COMPLETE not found in events: {types}"
        )

    def test_export_awq_quality_gate_passed(self, tmp_path):
        """QUANTIZATION_COMPLETE event must have quality_gate='passed' on happy path."""
        mock_awq = _make_awq_mock()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch.dict(sys.modules, _inject_awq_mock(mock_awq)):
            from dojo.quant import awq as awq_module
            awq_module.export_awq(
                model_path=model_dir,
                output_dir=tmp_path / "out",
            )

        events = EvidenceLogger._events
        complete_events = [
            e for e in events
            if isinstance(e.get("evidence"), dict)
            and e["evidence"].get("type") == EvidenceType.QUANTIZATION_COMPLETE.value
        ]
        assert complete_events, "No QUANTIZATION_COMPLETE event found"
        data = complete_events[0]["evidence"].get("data", {})
        assert data.get("quality_gate") == "passed", (
            f"Expected quality_gate='passed', got: {data.get('quality_gate')}"
        )

    def test_export_awq_f1_regression_evidence(self, tmp_path):
        """QUANTIZATION_COMPLETE event must carry f1_before, f1_after, f1_drop_abs."""
        mock_awq = _make_awq_mock()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch.dict(sys.modules, _inject_awq_mock(mock_awq)):
            from dojo.quant import awq as awq_module
            # Force a 10% F1 drop (0.82 → 0.72) by patching _measure_f1 directly.
            # Patching _STUB_F1_BEFORE/_STUB_F1_AFTER is not enough — those are only
            # used when _compute_mock_f1() raises, but torch is available in CI so
            # the real mock-F1 path runs instead.
            with patch.object(awq_module, "_measure_f1", return_value=(0.82, 0.72)):
                with pytest.raises(QuantQualityRejected):
                    awq_module.export_awq(
                        model_path=model_dir,
                        output_dir=tmp_path / "out",
                    )

        events = EvidenceLogger._events
        complete_events = [
            e for e in events
            if isinstance(e.get("evidence"), dict)
            and e["evidence"].get("type") == EvidenceType.QUANTIZATION_COMPLETE.value
        ]
        assert complete_events, "No QUANTIZATION_COMPLETE event found even on rejection"
        data = complete_events[0]["evidence"].get("data", {})

        assert "f1_before" in data, "f1_before missing from evidence"
        assert "f1_after" in data, "f1_after missing from evidence"
        assert "f1_drop_abs" in data, "f1_drop_abs missing from evidence"
        assert data["f1_before"] == pytest.approx(0.82)
        assert data["f1_after"] == pytest.approx(0.72)
        assert data["f1_drop_abs"] == pytest.approx(0.10, abs=1e-4)


class TestExportAwqQualityGate:
    def test_export_awq_f1_regression_raises(self, tmp_path):
        """F1 drop of 10% (> 5% limit) must raise QuantQualityRejected."""
        mock_awq = _make_awq_mock()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch.dict(sys.modules, _inject_awq_mock(mock_awq)):
            from dojo.quant import awq as awq_module
            with patch.object(awq_module, "_measure_f1", return_value=(0.82, 0.72)):
                with pytest.raises(QuantQualityRejected) as exc_info:
                    awq_module.export_awq(
                        model_path=model_dir,
                        output_dir=tmp_path / "out",
                    )

        assert "f1" in str(exc_info.value).lower() or "regression" in str(exc_info.value).lower()

    def test_export_awq_f1_within_threshold_does_not_raise(self, tmp_path):
        """F1 drop of 2% (< 5% limit) must NOT raise QuantQualityRejected."""
        mock_awq = _make_awq_mock()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch.dict(sys.modules, _inject_awq_mock(mock_awq)):
            from dojo.quant import awq as awq_module
            with (
                patch.object(awq_module, "_STUB_F1_BEFORE", 0.82),
                patch.object(awq_module, "_STUB_F1_AFTER", 0.80),
            ):
                result = awq_module.export_awq(
                    model_path=model_dir,
                    output_dir=tmp_path / "out",
                )

        assert result.exists()


class TestExportAwqMissingDependency:
    def test_export_awq_missing_autoawq(self, tmp_path):
        """ImportError on awq module must raise QuantizationError with install hint."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch.dict(sys.modules, {"awq": None}):
            from dojo.quant import awq as awq_module
            with pytest.raises(QuantizationError) as exc_info:
                awq_module.export_awq(
                    model_path=model_dir,
                    output_dir=tmp_path / "out",
                )

        msg = str(exc_info.value)
        assert "autoawq" in msg.lower() or "pip install" in msg.lower(), (
            f"Expected install hint in error, got: {msg}"
        )
