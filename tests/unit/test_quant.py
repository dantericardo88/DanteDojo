"""Unit tests for dojo/quant/ — Phase 3A quantization layer.

Covers:
- QuantValidator.validate: 0% delta → passes
- QuantValidator.validate: 20% perplexity delta → raises QuantQualityRejected
- QuantValidator.validate: 6 point F1 delta → raises QuantQualityRejected
- export_gguf: llama_cpp not installed → QuantizationError with install instructions
- export_gptq: gptqmodel not installed → QuantizationError
- export_awq: llmcompressor not installed → QuantizationError
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import QuantizationError, QuantQualityRejected
from dojo.quant.validator import QuantValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset singleton evidence logger between tests."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# QuantValidator tests
# ---------------------------------------------------------------------------

class TestQuantValidator:
    def test_zero_delta_passes(self):
        validator = QuantValidator()
        passed, reason = validator.validate(
            base_perplexity=10.0,
            quant_perplexity=10.0,
        )
        assert passed is True
        assert "passed" in reason

    def test_no_regression_passes(self):
        """Small improvement (lower perplexity) should pass."""
        validator = QuantValidator()
        passed, reason = validator.validate(
            base_perplexity=10.0,
            quant_perplexity=9.5,
        )
        assert passed is True

    def test_perplexity_20pct_delta_raises(self):
        """20% perplexity regression exceeds 15% threshold."""
        validator = QuantValidator()
        with pytest.raises(QuantQualityRejected) as exc_info:
            validator.validate(
                base_perplexity=10.0,
                quant_perplexity=12.0,   # 20% regression
            )
        assert "perplexity" in str(exc_info.value).lower()

    def test_perplexity_at_threshold_passes(self):
        """Exactly 15% regression is at the limit — spec says >15% rejects, so 15% passes."""
        validator = QuantValidator()
        # 15% regression: 10.0 → 11.5 = exactly 15%
        passed, reason = validator.validate(
            base_perplexity=10.0,
            quant_perplexity=11.5,   # exactly 15%
        )
        assert passed is True

    def test_f1_6pt_delta_raises(self):
        """6 point F1 regression exceeds 5-point threshold."""
        validator = QuantValidator()
        with pytest.raises(QuantQualityRejected) as exc_info:
            validator.validate(
                base_perplexity=10.0,
                quant_perplexity=10.5,   # 5% — below perplexity threshold
                base_f1=90.0,
                quant_f1=84.0,           # 6-point drop
            )
        assert "f1" in str(exc_info.value).lower()

    def test_f1_within_threshold_passes(self):
        """4 point F1 regression is under the 5-point threshold."""
        validator = QuantValidator()
        passed, reason = validator.validate(
            base_perplexity=10.0,
            quant_perplexity=10.0,
            base_f1=90.0,
            quant_f1=86.0,   # 4-point drop — within threshold
        )
        assert passed is True

    def test_emits_quality_gate_evidence(self):
        """validate() should emit QUANT_QUALITY_GATE evidence."""
        validator = QuantValidator()
        validator.validate(base_perplexity=10.0, quant_perplexity=10.0)
        events = EvidenceLogger._events
        types = [e["evidence"].get("type") for e in events if isinstance(e.get("evidence"), dict)]
        assert "quant.quality_gate" in types


# ---------------------------------------------------------------------------
# export_gguf tests
# ---------------------------------------------------------------------------

class TestExportGguf:
    def test_raises_when_llama_cpp_missing(self, tmp_path):
        """When llama_cpp is not importable and stub mode is off, raise QuantizationError."""
        import os
        from dojo.quant import gguf as gguf_module

        with patch.dict(os.environ, {"DOJO_GGUF_STUB": "0"}):
            with patch.dict(sys.modules, {"llama_cpp": None}):
                with pytest.raises(QuantizationError) as exc_info:
                    gguf_module.export_gguf(
                        model_path=tmp_path / "model",
                        output_dir=tmp_path / "out",
                    )
        assert "llama-cpp-python" in str(exc_info.value)

    def test_install_instructions_in_message(self, tmp_path):
        """Error message should contain actionable install instructions."""
        import os
        from dojo.quant import gguf as gguf_module

        with patch.dict(os.environ, {"DOJO_GGUF_STUB": "0"}):
            with patch.dict(sys.modules, {"llama_cpp": None}):
                with pytest.raises(QuantizationError) as exc_info:
                    gguf_module.export_gguf(
                        model_path=tmp_path / "model",
                        output_dir=tmp_path / "out",
                    )
        msg = str(exc_info.value)
        assert "pip install" in msg


# ---------------------------------------------------------------------------
# export_gptq tests
# ---------------------------------------------------------------------------

class TestExportGptq:
    def test_raises_when_gptqmodel_missing(self, tmp_path):
        """When gptqmodel is not importable in real-GPU mode, raise QuantizationError."""
        from dojo.quant import gptq as gptq_module

        with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
            with patch.dict(sys.modules, {"gptqmodel": None}):
                with pytest.raises(QuantizationError) as exc_info:
                    gptq_module.export_gptq(
                        model_path=tmp_path / "model",
                        output_dir=tmp_path / "out",
                    )
        assert "gptq" in str(exc_info.value).lower()

    def test_error_mentions_correct_package(self, tmp_path):
        """Error message should reference gptq-model, not auto-gptq."""
        from dojo.quant import gptq as gptq_module

        with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
            with patch.dict(sys.modules, {"gptqmodel": None}):
                with pytest.raises(QuantizationError) as exc_info:
                    gptq_module.export_gptq(
                        model_path=tmp_path / "model",
                        output_dir=tmp_path / "out",
                    )
        msg = str(exc_info.value)
        assert "gptq-model" in msg or "gptqmodel" in msg.lower()


# ---------------------------------------------------------------------------
# export_awq tests
# ---------------------------------------------------------------------------

class TestExportAwq:
    def test_raises_when_autoawq_missing(self, tmp_path):
        """When autoawq (awq) is not importable in real-GPU mode, raise QuantizationError."""
        from dojo.quant import awq as awq_module

        with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
            with patch.dict(sys.modules, {"awq": None}):
                with pytest.raises(QuantizationError) as exc_info:
                    awq_module.export_awq(
                        model_path=tmp_path / "model",
                        output_dir=tmp_path / "out",
                    )
        assert "autoawq" in str(exc_info.value).lower() or "awq" in str(exc_info.value).lower()

    def test_error_mentions_correct_package(self, tmp_path):
        """Error message should reference autoawq, not llmcompressor."""
        from dojo.quant import awq as awq_module

        with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
            with patch.dict(sys.modules, {"awq": None}):
                with pytest.raises(QuantizationError) as exc_info:
                    awq_module.export_awq(
                        model_path=tmp_path / "model",
                        output_dir=tmp_path / "out",
                    )
        msg = str(exc_info.value)
        assert "autoawq" in msg.lower() or "pip install autoawq" in msg.lower()
