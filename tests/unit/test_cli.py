"""Unit tests for dojo/cli/main.py."""

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from dojo.cli.main import app
from dojo.core.evidence import EvidenceLogger


@pytest.fixture(autouse=True)
def clear_logger():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


runner = CliRunner()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

class TestVersionCommand:
    def test_version_output(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "DanteDojo" in result.output

    def test_version_contains_semver(self):
        result = runner.invoke(app, ["version"])
        assert "0.1" in result.output


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

class TestDetectCommand:
    def test_detect_exits_ok(self):
        result = runner.invoke(app, ["detect"])
        assert result.exit_code == 0

    def test_detect_shows_tier(self):
        result = runner.invoke(app, ["detect"])
        assert "Tier" in result.output or "tier_" in result.output.lower()

    def test_detect_json_output(self):
        result = runner.invoke(app, ["detect", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "tier" in parsed
        assert "compute_capability" in parsed

    def test_detect_json_has_gpu_name(self):
        result = runner.invoke(app, ["detect", "--json"])
        parsed = json.loads(result.output)
        assert "gpu_name" in parsed

    def test_detect_json_has_fp8_flag(self):
        result = runner.invoke(app, ["detect", "--json"])
        parsed = json.loads(result.output)
        assert "has_native_fp8" in parsed


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

class TestEvidenceCommands:
    def test_evidence_chain_exits_ok(self):
        result = runner.invoke(app, ["evidence", "chain"])
        assert result.exit_code == 0

    def test_evidence_chain_shows_run_id(self):
        result = runner.invoke(app, ["evidence", "chain"])
        assert "Run ID" in result.output

    def test_evidence_chain_json(self):
        result = runner.invoke(app, ["evidence", "chain", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "run_id" in parsed
        assert "merkle_root" in parsed

    def test_evidence_export(self, tmp_path):
        out = tmp_path / "receipt.json"
        EvidenceLogger.get_instance().emit({"type": "test.event", "data": {}})
        result = runner.invoke(app, ["evidence", "export", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_evidence_export_shows_merkle(self, tmp_path):
        out = tmp_path / "r.json"
        EvidenceLogger.get_instance().emit({"type": "test.event"})
        result = runner.invoke(app, ["evidence", "export", str(out)])
        assert "Merkle root" in result.output


# ---------------------------------------------------------------------------
# stream status
# ---------------------------------------------------------------------------

class TestStreamCommands:
    def test_stream_status_exits_ok(self):
        result = runner.invoke(app, ["stream", "status"])
        assert result.exit_code == 0

    def test_stream_status_shows_torch(self):
        result = runner.invoke(app, ["stream", "status"])
        assert "torch" in result.output.lower()


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

class TestRegistryCommands:
    def test_registry_list_exits_ok(self):
        with patch("dojo.registry.store.LocalRegistryStore.list_artifacts", return_value=[]):
            result = runner.invoke(app, ["registry", "list"])
        assert result.exit_code == 0

    def test_registry_list_empty_message(self):
        with patch("dojo.registry.store.LocalRegistryStore.list_artifacts", return_value=[]):
            result = runner.invoke(app, ["registry", "list"])
        assert "No artifacts" in result.output


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

class TestExperimentCommands:
    def test_experiment_list_exits_ok(self):
        with patch("dojo.experiment.tracker.ExperimentTracker.list_runs", return_value=[]):
            result = runner.invoke(app, ["experiment", "list"])
        assert result.exit_code == 0

    def test_experiment_list_empty_message(self):
        with patch("dojo.experiment.tracker.ExperimentTracker.list_runs", return_value=[]):
            result = runner.invoke(app, ["experiment", "list"])
        assert "No experiment" in result.output


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------

class TestPromoteCommand:
    def test_promote_success(self):
        from datetime import datetime, timedelta, timezone

        from dojo.core.types import ConstitutionalPassport
        mock_passport = ConstitutionalPassport(
            passport_id="pp-001",
            model_artifact_id="dojo://test/v1",
            harvest_receipt_id="hr-001",
            harvest_check_receipt=b"h",
            dojo_check_receipt=b"d",
            forge_check_receipt=b"f",
            merkle_root="abc123",
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=90),
            revocation_url="http://127.0.0.1:8002/v1/passport/revoke",
        )
        with patch("dojo.promote.gate.promote", return_value=mock_passport):
            result = runner.invoke(app, [
                "promote", "dojo://test/v1",
                "--harvest-receipt", "hr-001",
                "--eval-merkle", "abc123",
            ])
        assert result.exit_code == 0
        assert "Passport issued" in result.output

    def test_promote_json(self):
        from datetime import datetime, timedelta, timezone

        from dojo.core.types import ConstitutionalPassport
        mock_passport = ConstitutionalPassport(
            passport_id="pp-002",
            model_artifact_id="dojo://test/v2",
            harvest_receipt_id="hr-002",
            harvest_check_receipt=b"h",
            dojo_check_receipt=b"d",
            forge_check_receipt=b"f",
            merkle_root="deadbeef",
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=90),
            revocation_url="http://127.0.0.1:8002/v1/passport/revoke",
        )
        with patch("dojo.promote.gate.promote", return_value=mock_passport):
            result = runner.invoke(app, [
                "promote", "dojo://test/v2", "--json",
                "--harvest-receipt", "hr-002",
                "--eval-merkle", "deadbeef",
            ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "passport_id" in parsed


# ---------------------------------------------------------------------------
# help / no-args behaviour
# ---------------------------------------------------------------------------

class TestHelpAndNoArgs:
    def test_top_level_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "detect" in result.output

    def test_pack_subcommand_help(self):
        result = runner.invoke(app, ["pack", "--help"])
        assert result.exit_code == 0
        assert "import" in result.output

    def test_registry_subcommand_help(self):
        result = runner.invoke(app, ["registry", "--help"])
        assert result.exit_code == 0
