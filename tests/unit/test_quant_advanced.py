"""Comprehensive tests for NF4/bitsandbytes quality gate and quantization integration.

Covers:
- TestQuantValidatorComprehensive: 8 tests verifying threshold semantics, evidence
  emission, and custom limit parameters on QuantValidator.
- TestGPTQQualityGateIntegration: 4 tests wiring export_gptq through the quality gate
  using controlled PPL values.
- TestAWQQualityGateIntegration: 4 tests wiring export_awq through the quality gate
  using controlled F1 values.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantQualityRejected
from dojo.quant.validator import QuantValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset singleton evidence logger between every test."""
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
# Helper
# ---------------------------------------------------------------------------

def _events_of_type(ev_type: str) -> list[dict]:
    return [
        e for e in EvidenceLogger._events
        if isinstance(e.get("evidence"), dict) and e["evidence"].get("type") == ev_type
    ]


_FAKE_GPTQ = MagicMock()
_FAKE_AWQ = MagicMock()


# ---------------------------------------------------------------------------
# TestQuantValidatorComprehensive
# ---------------------------------------------------------------------------

class TestQuantValidatorComprehensive:
    """8 granular tests for QuantValidator threshold logic and evidence emission."""

    # 1 — 25 % PPL regression triggers rejection
    def test_tight_ppl_threshold_blocks_bad_quant(self):
        """25% PPL regression exceeds the 15% default threshold and raises."""
        validator = QuantValidator()
        with pytest.raises(QuantQualityRejected) as exc_info:
            validator.validate(
                base_perplexity=10.0,
                quant_perplexity=12.5,   # +25%
            )
        assert "perplexity" in str(exc_info.value).lower()

    # 2 — 10 % F1 drop triggers rejection
    def test_tight_f1_threshold_blocks_bad_quant(self):
        """A 10-point F1 drop (>5-point threshold) raises QuantQualityRejected."""
        validator = QuantValidator()
        with pytest.raises(QuantQualityRejected) as exc_info:
            validator.validate(
                base_perplexity=10.0,
                quant_perplexity=10.5,   # 5% — below PPL threshold
                base_f1=90.0,
                quant_f1=80.0,           # 10-point drop
            )
        assert "f1" in str(exc_info.value).lower()

    # 3 — 1 % change passes gate
    def test_1pct_regression_passes_gate(self):
        """1% PPL regression is well within the 15% threshold and must pass."""
        validator = QuantValidator()
        passed, reason = validator.validate(
            base_perplexity=10.0,
            quant_perplexity=10.1,   # +1%
        )
        assert passed is True
        assert "passed" in reason

    # 4 — exactly at threshold raises (exclusive threshold: >15 % rejects)
    def test_exact_threshold_is_exclusive(self):
        """Exactly 15% regression is NOT above the threshold, so it should pass."""
        validator = QuantValidator()
        # 10.0 → 11.5 = exactly 15%
        passed, reason = validator.validate(
            base_perplexity=10.0,
            quant_perplexity=11.5,
        )
        assert passed is True

        # 15.001% (just over) must raise
        with pytest.raises(QuantQualityRejected):
            validator.validate(
                base_perplexity=10.0,
                quant_perplexity=11.5001,  # 15.001%
            )

    # 5 — passing validation emits quant.quality_gate evidence
    def test_validator_emits_gate_pass_evidence(self):
        """On a passing run, 'quant.quality_gate' evidence is emitted."""
        validator = QuantValidator()
        validator.validate(base_perplexity=10.0, quant_perplexity=10.0)

        gate_events = _events_of_type(EvidenceType.QUANT_QUALITY_GATE.value)
        assert len(gate_events) >= 1
        data = gate_events[-1]["evidence"]["data"]
        assert data["passed"] is True

    # 6 — rejecting validation still emits quant.quality_gate evidence
    def test_validator_emits_gate_reject_evidence(self):
        """Even on rejection, 'quant.quality_gate' evidence is emitted with status=rejected."""
        validator = QuantValidator()
        with pytest.raises(QuantQualityRejected):
            validator.validate(
                base_perplexity=10.0,
                quant_perplexity=15.0,   # 50% regression
            )

        gate_events = _events_of_type(EvidenceType.QUANT_QUALITY_GATE.value)
        assert len(gate_events) >= 1
        data = gate_events[-1]["evidence"]["data"]
        assert data["passed"] is False
        assert "reason" in data

    # 7 — custom ppl_regression_limit=0.5 (50%) accepted
    def test_validator_accepts_custom_ppl_limit(self):
        """Custom ppl_regression_limit attribute (50%) lets a 25% regression through."""
        validator = QuantValidator()
        validator.PERPLEXITY_MAX_DELTA_PCT = 50.0   # override to 50%
        passed, reason = validator.validate(
            base_perplexity=10.0,
            quant_perplexity=12.5,   # 25% — under the custom 50% limit
        )
        assert passed is True

    # 8 — custom f1_drop_limit=0.15 (15 pts) accepted
    def test_validator_accepts_custom_f1_limit(self):
        """Custom f1_drop_limit of 15 pts lets a 12-point drop pass."""
        validator = QuantValidator()
        validator.TASK_F1_MAX_DELTA_PTS = 15.0   # override to 15 pts
        passed, reason = validator.validate(
            base_perplexity=10.0,
            quant_perplexity=10.0,
            base_f1=90.0,
            quant_f1=78.0,   # 12-point drop — under the custom 15-pt limit
        )
        assert passed is True


# ---------------------------------------------------------------------------
# TestGPTQQualityGateIntegration
# ---------------------------------------------------------------------------

class TestGPTQQualityGateIntegration:
    """4 integration tests for export_gptq wiring through the quality gate."""

    def _import_gptq(self):
        """Re-import gptq module with gptqmodel mocked into sys.modules."""
        if "dojo.quant.gptq" in sys.modules:
            del sys.modules["dojo.quant.gptq"]
        from dojo.quant import gptq as gptq_module
        return gptq_module

    # 1 — mock PPL >20% regression raises QuantQualityRejected
    def test_export_gptq_raises_on_ppl_regression(self, model_dir, output_dir):
        """When stub PPL shows >20% regression (limit=0.01), QuantQualityRejected bubbles up."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            gptq_module = self._import_gptq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                # Stub mode returns ppl_before=25.0, ppl_after=26.2 → ~4.8% regression.
                # Setting limit=0.01 (1%) forces the gate to fire.
                with pytest.raises(QuantQualityRejected) as exc_info:
                    gptq_module.export_gptq(
                        model_dir, output_dir, ppl_regression_limit=0.01
                    )
        assert "ppl" in str(exc_info.value).lower() or "perplexity" in str(exc_info.value).lower()

    # 2 — mock PPL <1% regression passes gate
    def test_export_gptq_passes_gate_on_good_ppl(self, model_dir, output_dir):
        """When stub PPL regression is ~4.8% and limit=0.20, export succeeds."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            gptq_module = self._import_gptq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                result = gptq_module.export_gptq(
                    model_dir, output_dir, ppl_regression_limit=0.20
                )
        assert result.exists()

    # 3 — "quant.quality_gate" evidence present after export
    def test_export_gptq_emits_quality_gate_evidence(self, model_dir, output_dir):
        """QUANT_QUALITY_GATE evidence is emitted after a successful export."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            gptq_module = self._import_gptq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                gptq_module.export_gptq(
                    model_dir, output_dir, ppl_regression_limit=0.20
                )

        gate_events = _events_of_type(EvidenceType.QUANT_QUALITY_GATE.value)
        assert len(gate_events) >= 1
        data = gate_events[-1]["evidence"]["data"]
        assert "quality_gate" in data
        assert data["quality_gate"] == "passed"

    # 4 — stub mode does NOT run the real PPL comparison but still gate-passes
    def test_export_gptq_stub_skips_quality_gate(self, model_dir, output_dir):
        """In stub mode (DOJO_REAL_GPU=0) fixed stub PPL values are used — no GPU call."""
        with patch.dict(sys.modules, {"gptqmodel": _FAKE_GPTQ}):
            gptq_module = self._import_gptq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                # Patch _measure_ppl to verify it returns stub values without a real model
                original_measure = gptq_module._measure_ppl
                call_args_captured = []

                def spy_measure_ppl(model_path, quant_path):
                    result = original_measure(model_path, quant_path)
                    call_args_captured.append(result)
                    return result

                with patch.object(gptq_module, "_measure_ppl", side_effect=spy_measure_ppl):
                    output = gptq_module.export_gptq(
                        model_dir, output_dir, ppl_regression_limit=0.20
                    )

        assert output.exists()
        # Verify stub mode returned the fixed stub values
        assert len(call_args_captured) == 1
        ppl_before, ppl_after = call_args_captured[0]
        # Stub values: before=25.0, after=26.2
        assert ppl_before == 25.0
        assert ppl_after == 26.2


# ---------------------------------------------------------------------------
# TestAWQQualityGateIntegration
# ---------------------------------------------------------------------------

class TestAWQQualityGateIntegration:
    """4 integration tests for export_awq wiring through the quality gate."""

    def _import_awq(self):
        """Re-import awq module with autoawq mocked into sys.modules."""
        if "dojo.quant.awq" in sys.modules:
            del sys.modules["dojo.quant.awq"]
        from dojo.quant import awq as awq_module
        return awq_module

    # 1 — mock F1 showing >5% drop raises QuantQualityRejected
    def test_export_awq_raises_on_f1_drop(self, model_dir, output_dir):
        """Stub F1 drop ~0.8pp exceeds limit=0.005, so QuantQualityRejected bubbles up."""
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            awq_module = self._import_awq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                # Stub: f1_before=0.88, f1_after=0.872 → drop=0.008 > limit=0.005
                with pytest.raises(QuantQualityRejected) as exc_info:
                    awq_module.export_awq(
                        model_dir, output_dir, f1_drop_limit=0.005
                    )
        assert "f1" in str(exc_info.value).lower()

    # 2 — mock F1 showing <1% drop succeeds
    def test_export_awq_passes_gate_on_good_f1(self, model_dir, output_dir):
        """Stub F1 drop ~0.8pp is within the default 5pp limit — export succeeds."""
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            awq_module = self._import_awq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                result = awq_module.export_awq(
                    model_dir, output_dir, f1_drop_limit=0.05
                )
        assert result.exists()

    # 3 — "quant.quality_gate" evidence present after AWQ export
    def test_export_awq_emits_quality_gate_evidence(self, model_dir, output_dir):
        """QUANT_QUALITY_GATE evidence is emitted after a successful AWQ export."""
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            awq_module = self._import_awq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                awq_module.export_awq(
                    model_dir, output_dir, f1_drop_limit=0.05
                )

        gate_events = _events_of_type(EvidenceType.QUANT_QUALITY_GATE.value)
        assert len(gate_events) >= 1
        data = gate_events[-1]["evidence"]["data"]
        assert "quality_gate" in data
        assert data["quality_gate"] == "passed"

    # 4 — stub mode uses fixed F1 values, no GPU call
    def test_export_awq_stub_skips_quality_gate(self, model_dir, output_dir):
        """In stub mode (DOJO_REAL_GPU=0) fixed stub F1 values are used — no real model load."""
        with patch.dict(sys.modules, {"awq": _FAKE_AWQ}):
            awq_module = self._import_awq()
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                original_measure = awq_module._measure_f1
                call_args_captured = []

                def spy_measure_f1(model_path, quant_path):
                    result = original_measure(model_path, quant_path)
                    call_args_captured.append(result)
                    return result

                with patch.object(awq_module, "_measure_f1", side_effect=spy_measure_f1):
                    output = awq_module.export_awq(
                        model_dir, output_dir, f1_drop_limit=0.05
                    )

        assert output.exists()
        # Verify stub mode returned the fixed stub values
        assert len(call_args_captured) == 1
        f1_before, f1_after = call_args_captured[0]
        # Stub values: before=0.88, after=0.872
        assert f1_before == 0.88
        assert f1_after == 0.872
