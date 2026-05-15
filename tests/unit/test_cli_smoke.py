"""Comprehensive CLI smoke tests for DanteDojo — d49.

Covers every CLI command with Typer's CliRunner without requiring
DOJO_REAL_GPU=1 or any GPU hardware.  All heavy back-end calls are
patched at the boundary so the tests run in a plain CI environment.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from dojo.cli.main import app
from dojo.core.evidence import EvidenceLogger

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _fake_hal_profile():
    """Return a fully-populated HALProfile stub (no GPU required)."""
    from dojo.core.types import HALProfile, HardwareTier
    return HALProfile(
        tier=HardwareTier.TIER_CPU,
        gpu_name="Mock CPU",
        vram_bytes=0,
        compute_capability="0.0",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="0.0",
        profile_sha256="a" * 64,
    )


def _fake_finetune_result():
    """Return a minimal FinetuneResult stub."""
    from dojo.train.base import FinetuneResult
    return FinetuneResult(
        adapter_path=Path("/tmp/ckpt"),
        base_model="gpt2",
        training_receipt="a" * 64,
        final_loss=0.1,
        steps_completed=1,
        wall_time_seconds=0.01,
        backend_used="native",
    )


def _fake_eval_receipt(artifact_id: str = "dojo://test/v1"):
    """Return a minimal EvalReceipt stub."""
    from dojo.eval.harness import EvalReceipt
    return EvalReceipt(
        model_artifact_id=artifact_id,
        suite_type="dante",
        tasks=("task_a",),
        seed_scores={"task_a": [0.7] * 5},
        mean_scores={"task_a": 0.7},
        ci_lower={"task_a": 0.65},
        ci_upper={"task_a": 0.75},
        contamination_check_passed=True,
        merkle_hash="b" * 64,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )


def _fake_passport(artifact_id: str = "dojo://test/v1"):
    """Return a minimal ConstitutionalPassport stub."""
    from dojo.core.types import ConstitutionalPassport
    return ConstitutionalPassport(
        passport_id="pp-smoke-001",
        model_artifact_id=artifact_id,
        harvest_receipt_id="hr-smoke-001",
        harvest_check_receipt=b"h",
        dojo_check_receipt=b"d",
        forge_check_receipt=b"f",
        merkle_root="aabbccdd",
        issued_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(days=90),
        revocation_url="http://127.0.0.1:8002/v1/passport/revoke",
    )


# ---------------------------------------------------------------------------
# 1. Help and version (4 tests)
# ---------------------------------------------------------------------------

class TestHelpAndVersion:
    def test_cli_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_cli_version_exits_zero(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "DanteDojo" in result.output

    def test_train_help_exits_zero(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        # Either lowercase or uppercase positional name must appear
        assert "base" in result.output.lower()

    def test_eval_help_exits_zero(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "artifact" in result.output.lower()


# ---------------------------------------------------------------------------
# 2. Config validation — missing required args (4 tests)
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_train_missing_required_args_exits_nonzero(self):
        # train requires <base_model> positional arg
        result = runner.invoke(app, ["train"])
        assert result.exit_code != 0

    def test_eval_missing_required_args_exits_nonzero(self):
        # eval requires <artifact_id>
        result = runner.invoke(app, ["eval"])
        assert result.exit_code != 0

    def test_serve_missing_required_args_exits_nonzero(self):
        # serve requires <artifact_id>
        result = runner.invoke(app, ["serve"])
        assert result.exit_code != 0

    def test_promote_missing_required_args_exits_nonzero(self):
        # promote requires <artifact_id>
        result = runner.invoke(app, ["promote"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 3. Dry-run / CI-mode commands (6 tests)
# ---------------------------------------------------------------------------

class TestDryRunCIMode:
    def test_train_sft_ci_mode_exits_zero(self):
        """train with a stubbed finetune() — no GPU required."""
        with (
            patch("dojo.hal.detector.detect", return_value=_fake_hal_profile()),
            patch("dojo.train.finetune.finetune", return_value=_fake_finetune_result()),
        ):
            result = runner.invoke(app, ["train", "gpt2", "--epochs", "1"])
        assert result.exit_code == 0

    def test_eval_synthetic_ci_mode_exits_zero(self):
        """eval with mocked run_eval — no GPU needed."""
        with patch("dojo.eval.harness.run_eval", return_value=_fake_eval_receipt()):
            result = runner.invoke(app, ["eval", "dojo://test/v1"])
        assert result.exit_code == 0

    def test_registry_list_exits_zero(self):
        with patch("dojo.registry.store.LocalRegistryStore.list_artifacts", return_value=[]):
            result = runner.invoke(app, ["registry", "list"])
        assert result.exit_code == 0

    def test_registry_lookup_not_found_exits_nonzero(self):
        """Looking up a nonexistent artifact should surface an error."""
        from dojo.core.exceptions import RegistryError
        with patch(
            "dojo.registry.registry.lookup",
            side_effect=RegistryError("not found"),
        ):
            result = runner.invoke(app, ["registry", "lookup", "no-such-artifact"])
        # Either nonzero exit or error text in output
        assert result.exit_code != 0 or "not found" in result.output.lower()

    def test_pack_import_help_exits_zero(self):
        result = runner.invoke(app, ["pack", "import", "--help"])
        assert result.exit_code == 0
        # Command now takes manifest-path and data-path positional args
        assert "manifest" in result.output.lower() or "data" in result.output.lower()

    def test_promote_help_exits_zero(self):
        result = runner.invoke(app, ["promote", "--help"])
        assert result.exit_code == 0
        assert "artifact" in result.output.lower()


# ---------------------------------------------------------------------------
# 4. Evidence and output (5 tests)
# ---------------------------------------------------------------------------

class TestEvidenceAndOutput:
    def test_train_outputs_json_on_success_ci(self):
        """train --json with mocked backend returns parseable JSON."""
        with (
            patch("dojo.hal.detector.detect", return_value=_fake_hal_profile()),
            patch("dojo.train.finetune.finetune", return_value=_fake_finetune_result()),
        ):
            result = runner.invoke(app, ["train", "gpt2", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        # One of these keys must be present
        assert "training_receipt" in parsed or "adapter_path" in parsed

    def test_eval_outputs_result_json_ci(self):
        receipt = _fake_eval_receipt("dojo://test/v2")
        with patch("dojo.eval.harness.run_eval", return_value=receipt):
            result = runner.invoke(app, ["eval", "dojo://test/v2", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "artifact_id" in parsed or "mean_scores" in parsed

    def test_evidence_chain_json_output(self):
        EvidenceLogger.get_instance().emit({"type": "test.event", "data": {}})
        result = runner.invoke(app, ["evidence", "chain", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "run_id" in parsed
        assert "merkle_root" in parsed

    def test_cli_error_exit_code_nonzero(self):
        """Passing an unknown top-level command exits nonzero."""
        result = runner.invoke(app, ["totally-unknown-command-xyz"])
        assert result.exit_code != 0

    def test_cli_no_bare_exceptions_in_output(self):
        """Known-bad input (missing required arg) must not leak a raw Traceback."""
        result = runner.invoke(app, ["train"])
        assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# 5. Constitutional flags and sub-app help (6 tests)
# ---------------------------------------------------------------------------

class TestConstitutionalFlagsAndSubHelp:
    def test_serve_help_exits_zero(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "artifact" in result.output.lower()

    def test_registry_subapp_help_exits_zero(self):
        result = runner.invoke(app, ["registry", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output

    def test_evidence_subapp_help_exits_zero(self):
        result = runner.invoke(app, ["evidence", "--help"])
        assert result.exit_code == 0
        assert "chain" in result.output

    def test_stream_status_exits_zero(self):
        result = runner.invoke(app, ["stream", "status"])
        assert result.exit_code == 0
        assert "torch" in result.output.lower()

    def test_experiment_list_exits_zero(self):
        with patch("dojo.experiment.tracker.ExperimentTracker.list_runs", return_value=[]):
            result = runner.invoke(app, ["experiment", "list"])
        assert result.exit_code == 0

    def test_pack_subapp_help_exits_zero(self):
        result = runner.invoke(app, ["pack", "--help"])
        assert result.exit_code == 0
        assert "import" in result.output


# ---------------------------------------------------------------------------
# 6. Additional coverage (6 tests)
# ---------------------------------------------------------------------------

class TestAdditionalCoverage:
    def test_detect_exits_zero(self):
        result = runner.invoke(app, ["detect"])
        assert result.exit_code == 0

    def test_detect_json_parses(self):
        result = runner.invoke(app, ["detect", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "tier" in parsed

    def test_train_dry_run_json_is_dict(self):
        """train --json with mocked backend returns a dict with expected keys."""
        with (
            patch("dojo.hal.detector.detect", return_value=_fake_hal_profile()),
            patch("dojo.train.finetune.finetune", return_value=_fake_finetune_result()),
        ):
            result = runner.invoke(app, ["train", "bert-base", "--json", "--epochs", "1"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_version_contains_version_string(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert any(char.isdigit() for char in result.output)

    def test_promote_runs_with_mocked_gate(self):
        with patch("dojo.promote.gate.promote", return_value=_fake_passport()):
            result = runner.invoke(app, [
                "promote", "dojo://smoke/v1",
                "--harvest-receipt", "hr-smoke-001",
                "--eval-merkle", "aabbccdd",
            ])
        assert result.exit_code == 0
        assert "Passport" in result.output

    def test_evidence_export_creates_file(self, tmp_path):
        out = tmp_path / "smoke_receipt.json"
        EvidenceLogger.get_instance().emit({"type": "smoke.test", "data": {}})
        result = runner.invoke(app, ["evidence", "export", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "merkle_root" in data or "events" in data


# ---------------------------------------------------------------------------
# 7. Quantization CLI smoke tests (8 tests)
# ---------------------------------------------------------------------------

class TestQuantCommands:
    """Smoke tests for dojo quant gguf/gptq/awq CLI commands."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_model_dir(tmp_path: Path) -> Path:
        """Create a minimal fake model directory so Path(model_path).name works."""
        model_dir = tmp_path / "fake-model"
        model_dir.mkdir()
        # Write a minimal config so the directory is non-empty
        (model_dir / "config.json").write_text('{"model_type": "gpt2"}')
        return model_dir

    @staticmethod
    def _gguf_env():
        """Return env vars that put export_gguf into stub mode."""
        return {"DOJO_GGUF_STUB": "1", "DOJO_REAL_GPU": "0"}

    @staticmethod
    def _gpu_stub_env():
        """Return env vars that put gptq/awq into stub mode."""
        return {"DOJO_REAL_GPU": "0"}

    # ------------------------------------------------------------------
    # GGUF tests
    # ------------------------------------------------------------------

    def test_gguf_exits_zero_stub_mode(self, tmp_path):
        """dojo quant gguf exits 0 in stub mode (DOJO_GGUF_STUB=1)."""
        model_dir = self._make_model_dir(tmp_path)
        out_dir = tmp_path / "gguf_out"

        import sys
        # Stub llama_cpp so _find_convert_script's import probe doesn't error
        with patch.dict("os.environ", self._gguf_env()), \
             patch.dict(sys.modules, {"llama_cpp": None}):
            result = runner.invoke(app, [
                "quant", "gguf", str(model_dir),
                "--output-dir", str(out_dir),
            ])
        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"

    def test_gguf_json_output_contains_output_path(self, tmp_path):
        """dojo quant gguf --json output contains 'output_path' key."""
        model_dir = self._make_model_dir(tmp_path)
        out_dir = tmp_path / "gguf_out"

        import sys
        with patch.dict("os.environ", self._gguf_env()), \
             patch.dict(sys.modules, {"llama_cpp": None}):
            result = runner.invoke(app, [
                "quant", "gguf", str(model_dir),
                "--output-dir", str(out_dir),
                "--json",
            ])
        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        parsed = json.loads(result.output)
        assert "output_path" in parsed

    def test_gguf_quant_type_q5_k_m_in_filename(self, tmp_path):
        """dojo quant gguf --quant-type Q5_K_M uses Q5_K_M in the output filename."""
        model_dir = self._make_model_dir(tmp_path)
        out_dir = tmp_path / "gguf_out"

        import sys
        with patch.dict("os.environ", self._gguf_env()), \
             patch.dict(sys.modules, {"llama_cpp": None}):
            result = runner.invoke(app, [
                "quant", "gguf", str(model_dir),
                "--output-dir", str(out_dir),
                "--quant-type", "Q5_K_M",
                "--json",
            ])
        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        parsed = json.loads(result.output)
        assert "Q5_K_M" in parsed["output_path"]

    def test_gguf_passport_id_creates_sidecar(self, tmp_path):
        """dojo quant gguf --passport-id pp-test-001 creates a .meta.json sidecar."""
        model_dir = self._make_model_dir(tmp_path)
        out_dir = tmp_path / "gguf_out"

        import sys
        with patch.dict("os.environ", self._gguf_env()), \
             patch.dict(sys.modules, {"llama_cpp": None}):
            result = runner.invoke(app, [
                "quant", "gguf", str(model_dir),
                "--output-dir", str(out_dir),
                "--passport-id", "pp-test-001",
                "--json",
            ])
        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        parsed = json.loads(result.output)
        # sidecar field should be non-null when passport_id is set
        assert parsed.get("sidecar") is not None
        sidecar_path = Path(parsed["sidecar"])
        assert sidecar_path.exists(), f"Sidecar not found: {sidecar_path}"
        sidecar_data = json.loads(sidecar_path.read_text())
        assert sidecar_data.get("dojo.passport_id") == "pp-test-001"

    # ------------------------------------------------------------------
    # GPTQ tests
    # ------------------------------------------------------------------

    def test_gptq_exits_zero_stub_mode(self, tmp_path):
        """dojo quant gptq exits 0 in stub mode (DOJO_REAL_GPU=0)."""
        model_dir = self._make_model_dir(tmp_path)
        out_dir = tmp_path / "gptq_out"

        import sys
        import types
        fake_gptqmodel = types.ModuleType("gptqmodel")
        with patch.dict("os.environ", self._gpu_stub_env()), \
             patch.dict(sys.modules, {"gptqmodel": fake_gptqmodel}):
            result = runner.invoke(app, [
                "quant", "gptq", str(model_dir),
                "--output-dir", str(out_dir),
            ])
        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"

    def test_gptq_json_output_contains_output_path(self, tmp_path):
        """dojo quant gptq --json output contains 'output_path' key."""
        model_dir = self._make_model_dir(tmp_path)
        out_dir = tmp_path / "gptq_out"

        import sys
        import types
        fake_gptqmodel = types.ModuleType("gptqmodel")
        with patch.dict("os.environ", self._gpu_stub_env()), \
             patch.dict(sys.modules, {"gptqmodel": fake_gptqmodel}):
            result = runner.invoke(app, [
                "quant", "gptq", str(model_dir),
                "--output-dir", str(out_dir),
                "--json",
            ])
        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        parsed = json.loads(result.output)
        assert "output_path" in parsed

    # ------------------------------------------------------------------
    # AWQ tests
    # ------------------------------------------------------------------

    def test_awq_exits_zero_stub_mode(self, tmp_path):
        """dojo quant awq exits 0 in stub mode (DOJO_REAL_GPU=0)."""
        model_dir = self._make_model_dir(tmp_path)
        out_dir = tmp_path / "awq_out"

        import sys
        import types
        fake_awq = types.ModuleType("awq")
        with patch.dict("os.environ", self._gpu_stub_env()), \
             patch.dict(sys.modules, {"awq": fake_awq}):
            result = runner.invoke(app, [
                "quant", "awq", str(model_dir),
                "--output-dir", str(out_dir),
            ])
        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"

    def test_awq_json_output_contains_output_path(self, tmp_path):
        """dojo quant awq --json output contains 'output_path' key."""
        model_dir = self._make_model_dir(tmp_path)
        out_dir = tmp_path / "awq_out"

        import sys
        import types
        fake_awq = types.ModuleType("awq")
        with patch.dict("os.environ", self._gpu_stub_env()), \
             patch.dict(sys.modules, {"awq": fake_awq}):
            result = runner.invoke(app, [
                "quant", "awq", str(model_dir),
                "--output-dir", str(out_dir),
                "--json",
            ])
        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        parsed = json.loads(result.output)
        assert "output_path" in parsed
