"""Unit-test conftest — session-level pre-loading of PyO3 Rust extensions.

Many unit tests use ``patch.dict(sys.modules, {...})`` to inject mocks.
When that context exits, ``patch.dict`` restores sys.modules to its saved
pre-patch snapshot, deleting every module that was imported *during* the
patch but was absent *before* it.

If a Rust/PyO3 extension (torch, tokenizers, safetensors, …) happens to be
imported inside a patch context for the first time, it is deleted on exit.
The next call to ``import <that module>`` then re-runs the C extension
initializer → fatal "PyO3 modules compiled for CPython 3.8 or older may
only be initialized once per interpreter process" error.

Fixing every individual test file is fragile; instead this session fixture
pre-loads the affected extensions *before* any test (and before any
``patch.dict`` context), guaranteeing they are always present in the
pre-patch snapshot.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _preload_pyo3_extensions() -> None:
    """Pre-load Rust/PyO3 extension modules before any unit test runs."""
    import torch  # noqa: F401
    import tokenizers  # noqa: F401
    import safetensors  # noqa: F401
    import transformers  # noqa: F401
    import transformers.modeling_utils  # noqa: F401
