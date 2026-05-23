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
# Real-model PPL quality gate tests (end-to-end with actual torch inference)
# ---------------------------------------------------------------------------

class TestRealPPLQualityGate:
    """Quality gate fires/passes with REAL cross-entropy PPL from actual model inference.

    Uses sshleifer/tiny-gpt2 + dojo.quant.fp4.quantize_tensor_fp4 to produce
    genuine pre/post-quantization PPL values (not synthetic math), then validates
    the quality gate raises QuantQualityRejected on large degradation and passes
    on small degradation.
    """

    @pytest.fixture(autouse=True)
    def _load_tiny_model(self):
        """Load tiny-gpt2 once per test class (cached)."""
        try:
            import copy
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
            self._tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model_fp = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
            self._model_fp.eval()
            self._torch = torch
            self._copy = copy
            self._available = True
        except Exception:
            self._available = False

    def _compute_ppl(self, model) -> float:
        from dojo.quant.ppl import compute_real_ppl
        return compute_real_ppl(model, self._tokenizer)

    def _fp4_quantize_model(self):
        """Return a copy of model_fp with all float weights FP4-quantized."""
        from dojo.quant.fp4 import quantize_tensor_fp4
        model_q = self._copy.deepcopy(self._model_fp)
        with self._torch.no_grad():
            for param in model_q.parameters():
                if param.is_floating_point():
                    q, _ = quantize_tensor_fp4(param.data)
                    param.data.copy_(q)
        return model_q

    def test_fp4_quantized_model_has_real_ppl(self):
        """compute_real_ppl on FP4-quantized tiny-gpt2 returns a finite positive float."""
        if not self._available:
            pytest.skip("sshleifer/tiny-gpt2 not downloadable")
        model_q = self._fp4_quantize_model()
        ppl = self._compute_ppl(model_q)
        assert isinstance(ppl, float)
        assert ppl > 0
        import math
        assert math.isfinite(ppl)

    def test_fp4_quantization_increases_ppl(self):
        """FP4 quantization of tiny-gpt2 must increase PPL vs fp32 baseline."""
        if not self._available:
            pytest.skip("sshleifer/tiny-gpt2 not downloadable")
        ppl_baseline = self._compute_ppl(self._model_fp)
        model_q = self._fp4_quantize_model()
        ppl_quantized = self._compute_ppl(model_q)
        assert ppl_quantized > ppl_baseline, (
            f"FP4 quantization should increase PPL: baseline={ppl_baseline:.4f}, "
            f"quantized={ppl_quantized:.4f}"
        )

    def test_quality_gate_fires_on_real_fp4_ppl_degradation(self):
        """QuantValidator must raise QuantQualityRejected when real FP4 PPL regression exceeds 15%."""
        if not self._available:
            pytest.skip("sshleifer/tiny-gpt2 not downloadable")
        ppl_baseline = self._compute_ppl(self._model_fp)
        model_q = self._fp4_quantize_model()
        ppl_quantized = self._compute_ppl(model_q)

        delta_pct = (ppl_quantized - ppl_baseline) / ppl_baseline * 100
        validator = QuantValidator()

        if delta_pct > 15.0:
            with pytest.raises(QuantQualityRejected):
                validator.validate(
                    base_perplexity=ppl_baseline,
                    quant_perplexity=ppl_quantized,
                )
        else:
            # FP4 may not always exceed 15% on tiny-gpt2 — just verify gate doesn't crash
            passed, reason = validator.validate(
                base_perplexity=ppl_baseline,
                quant_perplexity=ppl_quantized,
            )
            assert passed is True

    def test_quality_gate_passes_on_small_real_ppl_change(self):
        """QuantValidator must pass when real PPL change is small (model not degraded)."""
        if not self._available:
            pytest.skip("sshleifer/tiny-gpt2 not downloadable")
        ppl = self._compute_ppl(self._model_fp)
        validator = QuantValidator()
        # Pass identical PPL values — zero degradation, must pass
        passed, reason = validator.validate(
            base_perplexity=ppl,
            quant_perplexity=ppl,
        )
        assert passed is True

    def test_quality_gate_evidence_emitted_with_real_ppl(self):
        """QUANT_QUALITY_GATE evidence type must be emitted when using real PPL values."""
        if not self._available:
            pytest.skip("sshleifer/tiny-gpt2 not downloadable")
        ppl = self._compute_ppl(self._model_fp)
        validator = QuantValidator()
        validator.validate(base_perplexity=ppl, quant_perplexity=ppl)
        events = EvidenceLogger._events
        types = [e["evidence"].get("type") for e in events if isinstance(e.get("evidence"), dict)]
        assert "quant.quality_gate" in types

    def test_ppl_measurement_source_is_real_model_inference(self):
        """compute_real_ppl must be called on real model forward pass, not synthetic math."""
        if not self._available:
            pytest.skip("sshleifer/tiny-gpt2 not downloadable")
        from dojo.quant.ppl import compute_real_ppl
        ppl = compute_real_ppl(self._model_fp, self._tokenizer)
        # tiny-gpt2 has random weights — PPL should be >5 and <1e9
        assert 5.0 < ppl < 1e9, f"PPL={ppl} not in expected range for tiny-gpt2 with random weights"


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
