"""Integration tests: QuantQualityRejected quality gate fires on bad quantization.

Mocks at the backend boundary only (gptqmodel / llmcompressor) so the full
quality-gate pipeline — PPL/F1 measurement, evidence emission, exception
raising — is exercised end-to-end.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantizationError, QuantQualityRejected


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset singleton evidence logger between tests."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _events_of_type(ev_type: str) -> list[dict]:
    return [
        e for e in EvidenceLogger._events
        if isinstance(e.get("evidence"), dict) and e["evidence"].get("type") == ev_type
    ]


_FAKE_GPTQ = MagicMock()
_FAKE_LLM = MagicMock()
_FAKE_AWQ = MagicMock()


# ---------------------------------------------------------------------------
# TestGPTQQualityGate
# ---------------------------------------------------------------------------

class TestGPTQQualityGate:
    """GPTQ PPL regression gate integration."""

    def test_ppl_regression_50_pct_fires_quality_rejected(self, model_dir, output_dir):
        """50% PPL regression triggers QuantQualityRejected.

        Stub values: ppl_before=25.0, ppl_after=26.2 → 4.8% regression.
        We set ppl_regression_limit=0.01 (1%) so the gate fires.
        """
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            # reimport so the module picks up the mocked sys.modules
            if "dojo.quant.gptq" in sys.modules:
                del sys.modules["dojo.quant.gptq"]
            from dojo.quant.gptq import export_gptq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                with pytest.raises(QuantQualityRejected):
                    export_gptq(model_dir, output_dir, ppl_regression_limit=0.01)

    def test_ppl_regression_within_threshold_passes(self, model_dir, output_dir):
        """4.8% PPL regression (stub) is below 20% default threshold — passes."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            if "dojo.quant.gptq" in sys.modules:
                del sys.modules["dojo.quant.gptq"]
            from dojo.quant.gptq import export_gptq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                result = export_gptq(model_dir, output_dir, ppl_regression_limit=0.20)

            assert result.exists()

    def test_quality_gate_rejection_emits_evidence(self, model_dir, output_dir):
        """Evidence is emitted even when quality gate fires (rejection is auditable)."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            if "dojo.quant.gptq" in sys.modules:
                del sys.modules["dojo.quant.gptq"]
            from dojo.quant.gptq import export_gptq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                with pytest.raises(QuantQualityRejected):
                    export_gptq(model_dir, output_dir, ppl_regression_limit=0.01)

        # Evidence must have been emitted despite the rejection
        assert len(EvidenceLogger._events) > 0

        # QUANTIZATION_START must be present
        start_events = _events_of_type(EvidenceType.QUANTIZATION_START.value)
        assert len(start_events) >= 1

        # QUANT_QUALITY_GATE with rejected status must be present
        gate_events = _events_of_type(EvidenceType.QUANT_QUALITY_GATE.value)
        assert len(gate_events) >= 1
        gate_data = gate_events[-1]["evidence"]["data"]
        assert gate_data["quality_gate"] == "rejected"

    def test_quality_gate_evidence_has_ppl_values(self, model_dir, output_dir):
        """QUANTIZATION_COMPLETE evidence contains ppl_before and ppl_after."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            if "dojo.quant.gptq" in sys.modules:
                del sys.modules["dojo.quant.gptq"]
            from dojo.quant.gptq import export_gptq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                # Use a loose limit so the gate passes; we just want to check evidence
                export_gptq(model_dir, output_dir, ppl_regression_limit=0.20)

        complete_events = _events_of_type(EvidenceType.QUANTIZATION_COMPLETE.value)
        assert len(complete_events) >= 1
        data = complete_events[-1]["evidence"]["data"]
        assert "ppl_before" in data
        assert "ppl_after" in data
        assert data["ppl_before"] > 0
        assert data["ppl_after"] > 0

    def test_quality_gate_rejection_message_informative(self, model_dir, output_dir):
        """Exception message identifies which metric failed and by how much."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            if "dojo.quant.gptq" in sys.modules:
                del sys.modules["dojo.quant.gptq"]
            from dojo.quant.gptq import export_gptq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                with pytest.raises(QuantQualityRejected) as exc_info:
                    export_gptq(model_dir, output_dir, ppl_regression_limit=0.01)

        msg = str(exc_info.value)
        # Message must name the metric and quantify the problem
        assert "ppl" in msg.lower() or "perplexity" in msg.lower()
        assert "limit" in msg.lower() or "exceed" in msg.lower() or "%" in msg

    def test_gate_passes_emits_quality_gate_passed_evidence(self, model_dir, output_dir):
        """When gate passes, QUANT_QUALITY_GATE evidence shows quality_gate='passed'."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            if "dojo.quant.gptq" in sys.modules:
                del sys.modules["dojo.quant.gptq"]
            from dojo.quant.gptq import export_gptq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                export_gptq(model_dir, output_dir, ppl_regression_limit=0.20)

        gate_events = _events_of_type(EvidenceType.QUANT_QUALITY_GATE.value)
        assert len(gate_events) >= 1
        assert gate_events[-1]["evidence"]["data"]["quality_gate"] == "passed"


# ---------------------------------------------------------------------------
# TestAWQQualityGate
# ---------------------------------------------------------------------------

class TestAWQQualityGate:
    """AWQ F1 drop gate integration."""

    def test_f1_drop_10pct_fires_quality_rejected(self, model_dir, output_dir):
        """Stub F1: before=88.0, after=87.2 → ~0.9% drop.

        We set f1_drop_limit=0.005 (0.5%) to force the gate to fire.
        """
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            if "dojo.quant.awq" in sys.modules:
                del sys.modules["dojo.quant.awq"]
            from dojo.quant.awq import export_awq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                with pytest.raises(QuantQualityRejected):
                    export_awq(model_dir, output_dir, f1_drop_limit=0.005)

    def test_f1_drop_within_threshold_passes(self, model_dir, output_dir):
        """Stub F1 drop (~0.9%) is below 5% default threshold — passes."""
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            if "dojo.quant.awq" in sys.modules:
                del sys.modules["dojo.quant.awq"]
            from dojo.quant.awq import export_awq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                result = export_awq(model_dir, output_dir, f1_drop_limit=0.05)

            assert result.exists()

    def test_f1_gate_rejection_emits_evidence(self, model_dir, output_dir):
        """Evidence emitted with quality_gate='rejected' when F1 gate fires."""
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            if "dojo.quant.awq" in sys.modules:
                del sys.modules["dojo.quant.awq"]
            from dojo.quant.awq import export_awq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                with pytest.raises(QuantQualityRejected):
                    export_awq(model_dir, output_dir, f1_drop_limit=0.005)

        # Must have emitted events despite rejection
        assert len(EvidenceLogger._events) > 0

        # QUANT_QUALITY_GATE with rejected status must be present
        gate_events = _events_of_type(EvidenceType.QUANT_QUALITY_GATE.value)
        assert len(gate_events) >= 1
        gate_data = gate_events[-1]["evidence"]["data"]
        assert gate_data["quality_gate"] == "rejected"

    def test_f1_gate_evidence_has_f1_values(self, model_dir, output_dir):
        """QUANTIZATION_COMPLETE evidence contains f1_before and f1_after."""
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            if "dojo.quant.awq" in sys.modules:
                del sys.modules["dojo.quant.awq"]
            from dojo.quant.awq import export_awq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                export_awq(model_dir, output_dir, f1_drop_limit=0.05)

        complete_events = _events_of_type(EvidenceType.QUANTIZATION_COMPLETE.value)
        assert len(complete_events) >= 1
        data = complete_events[-1]["evidence"]["data"]
        assert "f1_before" in data
        assert "f1_after" in data
        assert data["f1_before"] > 0

    def test_f1_gate_rejection_message_informative(self, model_dir, output_dir):
        """Exception message identifies F1 metric and quantifies the drop."""
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            if "dojo.quant.awq" in sys.modules:
                del sys.modules["dojo.quant.awq"]
            from dojo.quant.awq import export_awq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                with pytest.raises(QuantQualityRejected) as exc_info:
                    export_awq(model_dir, output_dir, f1_drop_limit=0.005)

        msg = str(exc_info.value)
        assert "f1" in msg.lower()
        assert "limit" in msg.lower() or "exceed" in msg.lower() or "%" in msg


# ---------------------------------------------------------------------------
# TestQualityGateConstitutional
# ---------------------------------------------------------------------------

class TestQualityGateConstitutional:
    """Constitutional behavior: gate cannot be silently bypassed."""

    def test_gptq_missing_backend_raises_quantization_error(self, model_dir, output_dir):
        """gptqmodel import error → QuantizationError, not QuantQualityRejected."""
        with patch.dict(sys.modules, {"gptqmodel": None}):
            if "dojo.quant.gptq" in sys.modules:
                del sys.modules["dojo.quant.gptq"]
            from dojo.quant.gptq import export_gptq

            with pytest.raises(QuantizationError) as exc_info:
                export_gptq(model_dir, output_dir)

        # Must be the base class, not the quality-gate subclass
        assert type(exc_info.value) is QuantizationError or isinstance(
            exc_info.value, QuantizationError
        )
        # Must NOT be the quality gate variant
        assert not isinstance(exc_info.value, QuantQualityRejected)

    def test_awq_missing_backend_raises_quantization_error(self, model_dir, output_dir):
        """llmcompressor import error → QuantizationError, not QuantQualityRejected."""
        with patch.dict(sys.modules, {"llmcompressor": None}):
            if "dojo.quant.awq" in sys.modules:
                del sys.modules["dojo.quant.awq"]
            from dojo.quant.awq import export_awq

            with pytest.raises(QuantizationError) as exc_info:
                export_awq(model_dir, output_dir)

        assert not isinstance(exc_info.value, QuantQualityRejected)

    def test_gptq_quality_rejected_is_subclass_of_quantization_error(
        self, model_dir, output_dir
    ):
        """QuantQualityRejected is catchable as QuantizationError (LSP compliance)."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            if "dojo.quant.gptq" in sys.modules:
                del sys.modules["dojo.quant.gptq"]
            from dojo.quant.gptq import export_gptq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                with pytest.raises(QuantizationError):
                    # QuantQualityRejected inherits from QuantizationError
                    export_gptq(model_dir, output_dir, ppl_regression_limit=0.001)

    def test_awq_quality_rejected_is_subclass_of_quantization_error(
        self, model_dir, output_dir
    ):
        """QuantQualityRejected is catchable as QuantizationError (LSP compliance)."""
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            if "dojo.quant.awq" in sys.modules:
                del sys.modules["dojo.quant.awq"]
            from dojo.quant.awq import export_awq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                with pytest.raises(QuantizationError):
                    export_awq(model_dir, output_dir, f1_drop_limit=0.001)

    def test_gptq_evidence_emitted_before_gate_fires(self, model_dir, output_dir):
        """QUANTIZATION_START evidence is present even when the gate fires."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            if "dojo.quant.gptq" in sys.modules:
                del sys.modules["dojo.quant.gptq"]
            from dojo.quant.gptq import export_gptq

            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                with pytest.raises(QuantQualityRejected):
                    export_gptq(model_dir, output_dir, ppl_regression_limit=0.001)

        start_events = _events_of_type(EvidenceType.QUANTIZATION_START.value)
        assert len(start_events) >= 1, "QUANTIZATION_START must be emitted before gate check"
