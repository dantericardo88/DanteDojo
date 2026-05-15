"""Packaging readiness tests for DanteDojo — d50.

Verifies that pyproject.toml is complete for PyPI submission and that
all public submodules are importable from a clean install.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PYPROJECT = Path(__file__).parent.parent.parent / "pyproject.toml"


def _load_pyproject() -> dict:
    """Parse pyproject.toml using tomllib (stdlib in 3.11+)."""
    import tomllib  # available in Python 3.11+
    with open(_PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


# Cache the parsed pyproject so we don't re-parse on every test.
@pytest.fixture(scope="module")
def pyproject() -> dict:
    return _load_pyproject()


# ---------------------------------------------------------------------------
# 1. pyproject.toml field presence (7 tests)
# ---------------------------------------------------------------------------

class TestPyprojectFields:
    def test_pyproject_has_name(self, pyproject):
        assert pyproject["project"]["name"], "pyproject.toml missing [project] name"

    def test_pyproject_has_version(self, pyproject):
        assert pyproject["project"]["version"], "pyproject.toml missing [project] version"

    def test_pyproject_has_description(self, pyproject):
        desc = pyproject["project"].get("description", "")
        assert desc, "pyproject.toml missing [project] description"

    def test_pyproject_has_authors(self, pyproject):
        authors = pyproject["project"].get("authors", [])
        assert authors, "pyproject.toml missing [project] authors"
        assert any("name" in a or "email" in a for a in authors)

    def test_pyproject_has_license(self, pyproject):
        lic = pyproject["project"].get("license", "")
        assert lic, "pyproject.toml missing [project] license"

    def test_pyproject_has_requires_python(self, pyproject):
        rp = pyproject["project"].get("requires-python", "")
        assert rp, "pyproject.toml missing [project] requires-python"
        assert ">=" in rp or "~=" in rp

    def test_pyproject_has_entry_point(self, pyproject):
        scripts = pyproject["project"].get("scripts", {})
        assert "dojo" in scripts, "pyproject.toml missing [project.scripts] dojo entry point"
        assert "dojo.cli.main" in scripts["dojo"]


# ---------------------------------------------------------------------------
# 2. Build system (1 test)
# ---------------------------------------------------------------------------

class TestBuildSystem:
    def test_pyproject_has_build_system(self, pyproject):
        bs = pyproject.get("build-system", {})
        assert bs.get("build-backend"), "pyproject.toml missing [build-system] build-backend"
        assert bs.get("requires"), "pyproject.toml missing [build-system] requires"


# ---------------------------------------------------------------------------
# 3. Package importability (2 tests)
# ---------------------------------------------------------------------------

class TestPackageImportable:
    def test_package_importable(self):
        import dojo  # noqa: F401  — just verify import succeeds
        assert dojo is not None

    def test_package_version_accessible(self):
        import dojo
        # Either dojo.__version__ or importlib.metadata must work.
        has_attr = hasattr(dojo, "__version__")
        try:
            meta_version = importlib.metadata.version("dante-dojo")
            has_meta = bool(meta_version)
        except importlib.metadata.PackageNotFoundError:
            has_meta = False
        assert has_attr or has_meta, (
            "Neither dojo.__version__ nor importlib.metadata.version('dante-dojo') is available"
        )


# ---------------------------------------------------------------------------
# 4. Sub-module importability (multi-test)
# ---------------------------------------------------------------------------

_SUBMODULES = [
    "dojo.train",
    "dojo.eval",
    "dojo.route",
    "dojo.serve",
    "dojo.idle",
    "dojo.core",
    "dojo.cli",
    "dojo.hal",
    "dojo.pack",
    "dojo.promote",
    "dojo.registry",
    "dojo.experiment",
]


@pytest.mark.parametrize("module_name", _SUBMODULES)
def test_submodule_importable(module_name: str):
    """Each public sub-package must be importable without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None, f"Failed to import {module_name}"


# ---------------------------------------------------------------------------
# 5. Version consistency (1 test)
# ---------------------------------------------------------------------------

class TestVersionConsistency:
    def test_version_consistent_between_init_and_metadata(self):
        """dojo.__version__ and installed metadata version should agree."""
        import dojo
        pkg_version = getattr(dojo, "__version__", None)
        try:
            meta_version = importlib.metadata.version("dante-dojo")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("Package not installed via pip — metadata not available")

        # Normalize: "0.1.0-dev" vs "0.1.0.dev0" (setuptools normalises the tag)
        # Accept if they share the same numeric prefix.
        pkg_numeric = pkg_version.split("-")[0].split(".dev")[0] if pkg_version else ""
        meta_numeric = meta_version.split("-")[0].split(".dev")[0]
        assert pkg_numeric == meta_numeric, (
            f"dojo.__version__={pkg_version!r} does not match "
            f"metadata version={meta_version!r}"
        )
