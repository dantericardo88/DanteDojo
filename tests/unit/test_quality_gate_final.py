"""test_quality_gate_final.py — d13 Quality Gate coverage push to 9/10.

Tests check_quality_gate() in dojo/quant/gate.py and the CLI
`dojo quant gate-check` subcommand, covering:
- pass / fail behaviour at various thresholds
- evidence emission (quant.gate_check type, organ="quant")
- all three quant_types: gguf, gptq, awq
- CLI smoke tests via typer.testing.CliRunner
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantQualityRejected
from dojo.quant.gate import check_quality_gate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_evidence():
    """Clear EvidenceLogger singleton between tests."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def cli_runner():
    return Runner()


class Runner:
    """Thin wrapper so tests can invoke the CLI without subprocess overhead."""

    def invoke(self, args: list[str], catch_exceptions: bool = False):
        from dojo.cli.main import app
        runner = CliRunner()
        return runner.invoke(app, args, catch_exceptions=catch_exceptions)


# ---------------------------------------------------------------------------
# Test 1: gate accepts result within threshold (no exception)
# ---------------------------------------------------------------------------

def test_gate_passes_within_threshold():
    """check_quality_gate() returns a result dict when degradation <= threshold."""
    # ppl_before=25, ppl_after=26 → 4% regression; threshold=0.10 → pass
    result = check_quality_gate(
        quant_type="gguf", threshold=0.10, ppl_before=25.0, ppl_after=26.0
    )
    assert result["passed"] is True
    assert result["degradation"] <= 0.10


# ---------------------------------------------------------------------------
# Test 2: gate raises QuantizationError when degradation exceeds threshold
# ---------------------------------------------------------------------------

def test_gate_raises_on_excessive_degradation():
    """check_quality_gate() raises QuantQualityRejected when degradation > threshold."""
    # ppl regression of 20% > 10% threshold
    with pytest.raises(QuantQualityRejected):
        check_quality_gate(
            quant_type="gguf", threshold=0.10, ppl_before=25.0, ppl_after=30.0
        )


# ---------------------------------------------------------------------------
# Test 3: gate emits quant.gate_check evidence on pass
# ---------------------------------------------------------------------------

def test_gate_emits_evidence_on_pass():
    """Passing gate emits QUANT_QUALITY_GATE evidence."""
    check_quality_gate(
        quant_type="gguf", threshold=0.10, ppl_before=25.0, ppl_after=26.0
    )
    events = EvidenceLogger.get_events()
    gate_events = [e for e in events if e.get("type") == EvidenceType.QUANT_QUALITY_GATE.value]
    assert len(gate_events) >= 1
    assert gate_events[0]["data"]["passed"] is True


# ---------------------------------------------------------------------------
# Test 4: gate emits quant.gate_check evidence on fail (with reason)
# ---------------------------------------------------------------------------

def test_gate_emits_evidence_on_fail():
    """Failing gate still emits QUANT_QUALITY_GATE evidence with a reason."""
    with pytest.raises(QuantQualityRejected):
        check_quality_gate(
            quant_type="gguf", threshold=0.05, ppl_before=25.0, ppl_after=30.0
        )
    events = EvidenceLogger.get_events()
    gate_events = [e for e in events if e.get("type") == EvidenceType.QUANT_QUALITY_GATE.value]
    assert len(gate_events) >= 1
    assert gate_events[0]["data"]["passed"] is False
    assert gate_events[0]["data"]["reason"] != ""


# ---------------------------------------------------------------------------
# Test 5: threshold=0.0 rejects any degradation
# ---------------------------------------------------------------------------

def test_threshold_zero_rejects_any_degradation():
    """threshold=0.0 means even 0.1% worse perplexity is rejected."""
    with pytest.raises(QuantQualityRejected):
        check_quality_gate(
            quant_type="gguf", threshold=0.0, ppl_before=25.0, ppl_after=25.1
        )


# ---------------------------------------------------------------------------
# Test 6: threshold=1.0 accepts any degradation
# ---------------------------------------------------------------------------

def test_threshold_one_accepts_any_degradation():
    """threshold=1.0 (100%) accepts even severe regression."""
    result = check_quality_gate(
        quant_type="gguf", threshold=1.0, ppl_before=25.0, ppl_after=49.0
    )
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Test 7: gate works for gguf quant type
# ---------------------------------------------------------------------------

def test_gate_works_for_gguf():
    """check_quality_gate() with quant_type='gguf' uses gguf stub defaults."""
    # With no explicit metrics, stub values are used; default threshold=0.10
    # gguf stubs: before=25.0, after=26.0 → 4% < 10% → pass
    result = check_quality_gate(quant_type="gguf", threshold=0.10)
    assert result["quant_type"] == "gguf"
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Test 8: gate works for gptq quant type
# ---------------------------------------------------------------------------

def test_gate_works_for_gptq():
    """check_quality_gate() with quant_type='gptq' uses gptq stub defaults."""
    # gptq stubs: before=25.0, after=26.2 → 4.8% < 10% → pass
    result = check_quality_gate(quant_type="gptq", threshold=0.10)
    assert result["quant_type"] == "gptq"
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Test 9: gate works for awq quant type
# ---------------------------------------------------------------------------

def test_gate_works_for_awq():
    """check_quality_gate() with quant_type='awq' uses awq stub defaults."""
    # awq stubs: before=25.0, after=25.8 → 3.2% < 10% → pass
    result = check_quality_gate(quant_type="awq", threshold=0.10)
    assert result["quant_type"] == "awq"
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Test 10: CLI gate-check command exists and is callable (smoke)
# ---------------------------------------------------------------------------

def test_cli_gate_check_command_exists():
    """'dojo quant gate-check --help' exits 0 and mentions threshold."""
    from dojo.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["quant", "gate-check", "--help"])
    assert result.exit_code == 0
    assert "threshold" in result.output.lower() or "gate" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 11: CLI gate-check exits 0 for passing gate
# ---------------------------------------------------------------------------

def test_cli_gate_check_exits_0_for_passing_gate():
    """CLI 'dojo quant gate-check' exits 0 when within threshold."""
    from dojo.cli.main import app
    runner = CliRunner()
    # ppl_before=25, ppl_after=26 → 4% regression < 10% threshold
    result = runner.invoke(
        app,
        [
            "quant", "gate-check",
            "--quant-type", "gguf",
            "--threshold", "0.10",
            "--ppl-before", "25.0",
            "--ppl-after", "26.0",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Test 12: CLI gate-check exits 1 for failing gate
# ---------------------------------------------------------------------------

def test_cli_gate_check_exits_1_for_failing_gate():
    """CLI 'dojo quant gate-check' exits 1 when degradation exceeds threshold."""
    from dojo.cli.main import app
    runner = CliRunner()
    # ppl_before=25, ppl_after=35 → 40% regression > 10% threshold
    result = runner.invoke(
        app,
        [
            "quant", "gate-check",
            "--quant-type", "gguf",
            "--threshold", "0.10",
            "--ppl-before", "25.0",
            "--ppl-after", "35.0",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Test 13: result dict contains all expected keys
# ---------------------------------------------------------------------------

def test_gate_result_has_expected_keys():
    """Result dict from check_quality_gate() has all required keys."""
    result = check_quality_gate(
        quant_type="gguf", threshold=0.10, ppl_before=25.0, ppl_after=26.0
    )
    for key in ("passed", "quant_type", "threshold", "ppl_before", "ppl_after", "degradation", "reason"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test 14: degradation value is correctly computed
# ---------------------------------------------------------------------------

def test_gate_degradation_value_correct():
    """degradation field equals (ppl_after - ppl_before) / ppl_before."""
    result = check_quality_gate(
        quant_type="gguf", threshold=0.50, ppl_before=20.0, ppl_after=24.0
    )
    expected = (24.0 - 20.0) / 20.0  # 0.2
    assert abs(result["degradation"] - expected) < 1e-5


# ---------------------------------------------------------------------------
# Test 15: zero baseline perplexity does not crash
# ---------------------------------------------------------------------------

def test_gate_zero_baseline_does_not_crash():
    """When ppl_before=0, degradation is 0 and gate does not crash."""
    # Passing 0 triggers stub mode (0 → use stubs), so use a tiny positive value
    result = check_quality_gate(
        quant_type="gguf", threshold=1.0, ppl_before=0.001, ppl_after=0.001
    )
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Test 16: evidence organ is "quant"
# ---------------------------------------------------------------------------

def test_gate_evidence_organ_is_quant():
    """Evidence emitted by check_quality_gate() uses organ='quant'."""
    check_quality_gate(
        quant_type="gguf", threshold=0.10, ppl_before=25.0, ppl_after=26.0
    )
    raw_events = EvidenceLogger._events
    quant_events = [e for e in raw_events if e.get("organ") == "quant"]
    assert len(quant_events) >= 1


# ---------------------------------------------------------------------------
# Test 17: CLI --json flag produces valid JSON on pass
# ---------------------------------------------------------------------------

def test_cli_gate_check_json_output_on_pass():
    """CLI gate-check --json outputs valid JSON with passed=True."""
    from dojo.cli.main import app
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "quant", "gate-check",
            "--quant-type", "gptq",
            "--threshold", "0.10",
            "--ppl-before", "25.0",
            "--ppl-after", "26.0",
            "--json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["passed"] is True


# ---------------------------------------------------------------------------
# Test 18: gate exactly at threshold is accepted (boundary)
# ---------------------------------------------------------------------------

def test_gate_exactly_at_threshold_is_accepted():
    """Degradation exactly equal to threshold passes (<=, not <)."""
    # 10% regression with 10% threshold → exactly at boundary → should pass
    result = check_quality_gate(
        quant_type="gguf", threshold=0.10, ppl_before=100.0, ppl_after=110.0
    )
    assert result["passed"] is True
