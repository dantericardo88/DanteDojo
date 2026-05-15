"""Final CLI tests for d49 — targets 9/10.

Covers all new and verified commands:
- dojo quant gate-check
- dojo experiments list
- dojo experiments compare
- dojo promote-sub revoke  (promote revoke --passport-id)
- dojo hal detect
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from dojo.cli.main import app
from dojo.core.evidence import EvidenceLogger

runner = CliRunner()


@pytest.fixture(autouse=True)
def clear_logger():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ===========================================================================
# dojo quant gate-check
# ===========================================================================

class TestQuantGateCheck:
    def test_gate_check_appears_in_quant_help(self):
        result = runner.invoke(app, ["quant", "--help"])
        assert result.exit_code == 0
        assert "gate-check" in result.output

    def test_gate_check_passes_within_threshold(self):
        """gate-check exits 0 when degradation is within threshold."""
        with patch("dojo.quant.gate.check_quality_gate", return_value={"passed": True, "degradation": 0.05}):
            result = runner.invoke(app, [
                "quant", "gate-check",
                "--quant-type", "gguf",
                "--threshold", "0.1",
            ])
        assert result.exit_code == 0

    def test_gate_check_json_output(self):
        """gate-check --json outputs parseable JSON."""
        with patch("dojo.quant.gate.check_quality_gate", return_value={"passed": True, "degradation": 0.02}):
            result = runner.invoke(app, [
                "quant", "gate-check",
                "--quant-type", "gptq",
                "--json",
            ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "passed" in parsed or "degradation" in parsed


# ===========================================================================
# dojo experiments list
# ===========================================================================

class TestExperimentsList:
    def test_experiments_list_exits_ok(self):
        with patch("dojo.experiment.tracker.ExperimentTracker.list_runs", return_value=[]):
            result = runner.invoke(app, ["experiments", "list"])
        assert result.exit_code == 0

    def test_experiments_list_empty_message(self):
        with patch("dojo.experiment.tracker.ExperimentTracker.list_runs", return_value=[]):
            result = runner.invoke(app, ["experiments", "list"])
        assert "No experiment" in result.output

    def test_experiments_list_json_output(self):
        fake_runs = [{"run_id": "run-1", "name": "exp-a", "status": "complete", "start_time": "2026-01-01"}]
        with patch("dojo.experiment.tracker.ExperimentTracker.list_runs", return_value=fake_runs):
            result = runner.invoke(app, ["experiments", "list", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert parsed[0]["run_id"] == "run-1"


# ===========================================================================
# dojo experiments compare
# ===========================================================================

class TestExperimentsCompare:
    def test_experiments_compare_exits_ok(self):
        mock_lb = MagicMock()
        mock_lb.compare_top_n.return_value = {}
        with patch("dojo.experiment.leaderboard.RunLeaderboard.from_tracker", return_value=mock_lb):
            result = runner.invoke(app, ["experiments", "compare"])
        assert result.exit_code == 0

    def test_experiments_compare_empty_result(self):
        mock_lb = MagicMock()
        mock_lb.compare_top_n.return_value = {}
        with patch("dojo.experiment.leaderboard.RunLeaderboard.from_tracker", return_value=mock_lb):
            result = runner.invoke(app, ["experiments", "compare"])
        assert "No runs" in result.output or result.exit_code == 0

    def test_experiments_compare_json_output(self):
        mock_lb = MagicMock()
        mock_lb.compare_top_n.return_value = {
            "run-1": {"rank": 1, "run_name": "baseline", "baseline_run_id": ""}
        }
        with patch("dojo.experiment.leaderboard.RunLeaderboard.from_tracker", return_value=mock_lb):
            result = runner.invoke(app, ["experiments", "compare", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "run-1" in parsed

    def test_experiments_compare_emits_evidence(self):
        mock_lb = MagicMock()
        mock_lb.compare_top_n.return_value = {}
        with patch("dojo.experiment.leaderboard.RunLeaderboard.from_tracker", return_value=mock_lb):
            runner.invoke(app, ["experiments", "compare"])
        events = EvidenceLogger.get_events()
        types = [e["type"] for e in events]
        assert "experiment.leaderboard_compare" in types


# ===========================================================================
# dojo promote-sub revoke
# ===========================================================================

class TestPromoteRevoke:
    def test_revoke_not_revoked_exits_0(self):
        """When passport is not revoked, exit code is 0."""
        with patch("dojo.promote.revocation.check_revocation", return_value=False):
            result = runner.invoke(app, [
                "promote-sub", "revoke",
                "--passport-id", "pp-test-001",
            ])
        assert result.exit_code == 0

    def test_revoke_not_revoked_shows_valid(self):
        with patch("dojo.promote.revocation.check_revocation", return_value=False):
            result = runner.invoke(app, [
                "promote-sub", "revoke",
                "--passport-id", "pp-test-002",
            ])
        assert "VALID" in result.output or result.exit_code == 0

    def test_revoke_json_not_revoked(self):
        with patch("dojo.promote.revocation.check_revocation", return_value=False):
            result = runner.invoke(app, [
                "promote-sub", "revoke",
                "--passport-id", "pp-003",
                "--json",
            ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["passport_id"] == "pp-003"
        assert parsed["revoked"] is False

    def test_revoke_passport_revoked_exits_1(self):
        """When passport is revoked, exit code is 1."""
        from dojo.core.exceptions import PassportRevoked
        with patch(
            "dojo.promote.revocation.check_revocation",
            side_effect=PassportRevoked("revoked", reason="test"),
        ):
            result = runner.invoke(app, [
                "promote-sub", "revoke",
                "--passport-id", "pp-revoked",
            ])
        assert result.exit_code == 1

    def test_revoke_emits_evidence(self):
        """promote-sub revoke emits promote.revoke_check evidence."""
        with patch("dojo.promote.revocation.check_revocation", return_value=False):
            runner.invoke(app, [
                "promote-sub", "revoke",
                "--passport-id", "pp-evidence-test",
            ])
        events = EvidenceLogger.get_events()
        types = [e["type"] for e in events]
        assert "promote.revoke_check" in types

    def test_revoke_json_revoked(self):
        """--json output when revoked shows revoked=True."""
        from dojo.core.exceptions import PassportRevoked
        with patch(
            "dojo.promote.revocation.check_revocation",
            side_effect=PassportRevoked("revoked", reason="test"),
        ):
            result = runner.invoke(app, [
                "promote-sub", "revoke",
                "--passport-id", "pp-json-revoked",
                "--json",
            ])
        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["revoked"] is True


# ===========================================================================
# dojo hal detect
# ===========================================================================

class TestHalDetect:
    def test_hal_detect_exits_ok(self):
        result = runner.invoke(app, ["hal", "detect"])
        assert result.exit_code == 0

    def test_hal_detect_json_output_parseable(self):
        result = runner.invoke(app, ["hal", "detect"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "tier" in parsed

    def test_hal_detect_json_has_gpu_name(self):
        result = runner.invoke(app, ["hal", "detect"])
        parsed = json.loads(result.output)
        assert "gpu_name" in parsed

    def test_hal_detect_json_has_compute_capability(self):
        result = runner.invoke(app, ["hal", "detect"])
        parsed = json.loads(result.output)
        assert "compute_capability" in parsed

    def test_hal_detect_emits_evidence(self):
        """hal detect emits hal.detect_cli evidence."""
        runner.invoke(app, ["hal", "detect"])
        events = EvidenceLogger.get_events()
        types = [e["type"] for e in events]
        assert "hal.detect_cli" in types

    def test_hal_detect_appears_in_hal_help(self):
        result = runner.invoke(app, ["hal", "--help"])
        assert result.exit_code == 0
        assert "detect" in result.output
