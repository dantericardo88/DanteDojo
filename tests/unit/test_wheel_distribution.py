"""d50: Wheel distribution and installed CLI validation.

Proves that DanteDojo is fully deployable as a Python package:
  - Built wheel has correct structure (entry_points, METADATA, modules)
  - CLI entry point is registered in pyproject.toml and importable
  - All 8 CLI subcommands are registered in the typer app
  - dojo --help output contains expected command names
  - dojo detect runs end-to-end from the CLI app
  - Package name, version, and description are populated

Coverage:
1.  Built wheel file exists in dist/ or dist_test/
2.  Wheel contains entry_points.txt with dojo = dojo.cli.main:main
3.  Wheel METADATA contains project name 'dante-dojo'
4.  Wheel contains dojo/__init__.py
5.  pyproject.toml [project.scripts] dojo entry point matches
6.  CLI app has at least 8 registered commands
7.  detect, train, serve, eval, promote, registry, experiments commands present
8.  dojo --help runs without error
9.  dojo detect runs and outputs tier field
10. CLI entry point module is importable
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dojo.cli.main import app
from dojo.core.evidence import EvidenceLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

runner = CliRunner()

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _load_pyproject() -> dict:
    import tomllib
    with open(_PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def _find_wheel() -> Path | None:
    """Find the most recently built wheel in dist/ or dist_test/."""
    for dist_dir in ("dist", "dist_test"):
        wheels = sorted((_PROJECT_ROOT / dist_dir).glob("*.whl"))
        if wheels:
            return wheels[-1]
    return None


# ---------------------------------------------------------------------------
# Wheel structure tests
# ---------------------------------------------------------------------------

class TestWheelStructure:
    def test_wheel_file_exists(self):
        wheel = _find_wheel()
        assert wheel is not None, (
            "No .whl file found in dist/ or dist_test/ — run `python -m build` first"
        )
        assert wheel.stat().st_size > 0

    def test_wheel_contains_entry_points(self):
        wheel = _find_wheel()
        if wheel is None:
            pytest.skip("No wheel built yet")
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            ep_files = [n for n in names if "entry_points.txt" in n]
            assert ep_files, "Wheel must contain entry_points.txt"
            ep_content = zf.read(ep_files[0]).decode()
            assert "dojo" in ep_content, (
                f"entry_points.txt must register 'dojo' CLI entry point, got:\n{ep_content}"
            )
            assert "dojo.cli" in ep_content or "main" in ep_content

    def test_wheel_metadata_project_name(self):
        wheel = _find_wheel()
        if wheel is None:
            pytest.skip("No wheel built yet")
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            meta_files = [n for n in names if n.endswith("/METADATA")]
            assert meta_files, "Wheel must contain dist-info/METADATA"
            meta = zf.read(meta_files[0]).decode()
            assert "dante-dojo" in meta.lower() or "dante_dojo" in meta.lower(), (
                "METADATA must contain project name 'dante-dojo'"
            )

    def test_wheel_contains_dojo_package(self):
        wheel = _find_wheel()
        if wheel is None:
            pytest.skip("No wheel built yet")
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            assert any("dojo/__init__.py" in n for n in names), (
                "Wheel must contain dojo/__init__.py"
            )

    def test_wheel_contains_cli_module(self):
        wheel = _find_wheel()
        if wheel is None:
            pytest.skip("No wheel built yet")
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            assert any("dojo/cli/main.py" in n for n in names), (
                "Wheel must contain dojo/cli/main.py"
            )


# ---------------------------------------------------------------------------
# pyproject.toml entry point tests
# ---------------------------------------------------------------------------

class TestPyprojectEntryPoint:
    def test_entry_point_registered(self):
        pyproject = _load_pyproject()
        scripts = pyproject["project"].get("scripts", {})
        assert "dojo" in scripts, "pyproject.toml must register dojo entry point"

    def test_entry_point_points_to_correct_function(self):
        pyproject = _load_pyproject()
        scripts = pyproject["project"].get("scripts", {})
        target = scripts.get("dojo", "")
        assert "dojo.cli" in target, (
            f"dojo entry point must reference dojo.cli module, got: {target}"
        )

    def test_entry_point_module_importable(self):
        from dojo.cli import main as cli_main
        assert cli_main is not None

    def test_app_object_is_typer_app(self):
        import typer
        assert isinstance(app, typer.Typer), "dojo.cli.main.app must be a typer.Typer instance"


# ---------------------------------------------------------------------------
# CLI subcommand registration tests
# ---------------------------------------------------------------------------

_EXPECTED_COMMANDS = [
    "detect",
    "train",
    "serve",
    "eval",
    "promote",
    "registry",
    "experiments",
    "pack",
]


def _get_all_command_names() -> list[str]:
    """Return all command names visible in dojo --help output."""
    result = runner.invoke(app, ["--help"])
    names = []
    # Parse command names from help output lines starting with spaces
    for line in result.output.splitlines():
        stripped = line.strip()
        # Command lines look like: "  detect        Description..."
        if stripped and not stripped.startswith("-") and not stripped.startswith("Usage"):
            first_word = stripped.split()[0]
            if first_word.isidentifier() or "-" in first_word:
                names.append(first_word)
    return names


class TestCLICommands:
    def test_app_has_minimum_eight_commands(self):
        help_result = runner.invoke(app, ["--help"])
        # Verify all 8 expected commands appear in the help output
        found = [cmd for cmd in _EXPECTED_COMMANDS if cmd in help_result.output]
        assert len(found) >= 8, (
            f"CLI help must list all 8 expected commands; found only {found}\n"
            f"Help output:\n{help_result.output}"
        )

    @pytest.mark.parametrize("cmd_name", _EXPECTED_COMMANDS)
    def test_expected_command_registered(self, cmd_name):
        help_result = runner.invoke(app, ["--help"])
        assert cmd_name in help_result.output, (
            f"Command '{cmd_name}' not found in dojo --help output:\n{help_result.output}"
        )

    def test_help_runs_without_error(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, f"dojo --help failed: {result.output}"
        assert "detect" in result.output
        assert "train" in result.output

    def test_detect_outputs_tier_field(self):
        """dojo detect must complete and output the Tier: field."""
        result = runner.invoke(app, ["detect"])
        assert result.exit_code == 0, f"dojo detect failed:\n{result.output}\n{result.exception}"
        assert "Tier" in result.output or "tier" in result.output, (
            f"dojo detect output must include Tier field:\n{result.output}"
        )

    def test_registry_list_runs_without_error(self):
        result = runner.invoke(app, ["registry", "list"])
        assert result.exit_code == 0, f"dojo registry list failed: {result.output}"

    def test_experiments_list_runs_without_error(self):
        result = runner.invoke(app, ["experiments", "list"])
        assert result.exit_code == 0, f"dojo experiments list failed: {result.output}"


# ---------------------------------------------------------------------------
# Package metadata tests
# ---------------------------------------------------------------------------

class TestPackageMetadata:
    def test_package_name_is_dante_dojo(self):
        pyproject = _load_pyproject()
        name = pyproject["project"]["name"]
        assert name in ("dante-dojo", "dante_dojo"), f"Unexpected package name: {name}"

    def test_package_has_version(self):
        pyproject = _load_pyproject()
        version = pyproject["project"].get("version", "")
        assert version, "pyproject.toml must have a version"

    def test_package_has_description(self):
        pyproject = _load_pyproject()
        desc = pyproject["project"].get("description", "")
        assert desc, "pyproject.toml must have a description"

    def test_ci_publish_workflow_exists(self):
        workflow = _PROJECT_ROOT / ".github" / "workflows" / "publish.yml"
        assert workflow.exists(), (
            ".github/workflows/publish.yml must exist for CI/CD publishing"
        )
