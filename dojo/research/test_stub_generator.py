"""dojo/research/test_stub_generator.py — Generate GPU test stubs from claims.

Given a CapabilityClaim, produces a pytest stub file ready for implementation.
The stub is annotated with the paper ID, technique, metric, and dimension IDs
so a human or another agent can fill in the validation logic.
"""

from __future__ import annotations

import re
from pathlib import Path

from dojo.research.claim_extractor import CapabilityClaim


def generate_stub(claim: CapabilityClaim, output_dir: Path) -> Path:
    """Write a pytest stub for *claim* to *output_dir*.

    Returns the path to the generated file.
    """
    safe_id = re.sub(r"[^\w]", "_", claim.paper_id)
    safe_tech = re.sub(r"[^\w]", "_", claim.technique)
    filename = f"test_arxiv_{safe_id}_{safe_tech}.py"
    path = output_dir / filename

    dims_str = ", ".join(claim.dimension_ids)
    stub = _STUB_TEMPLATE.format(
        paper_id=claim.paper_id,
        title=claim.title[:120],
        technique=claim.technique,
        metric=claim.metric,
        improvement_pct=claim.improvement_pct,
        hardware=claim.hardware,
        dims=dims_str,
        class_name=f"TestArXiv_{safe_id}_{safe_tech}",
        excerpt=claim.raw_excerpt[:300].replace("\n", " "),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(stub, encoding="utf-8")
    return path


_STUB_TEMPLATE = '''\
"""AUTO-GENERATED test stub from arXiv:{paper_id}

Title: {title}
Technique: {technique}
Metric: {metric} improved {improvement_pct:.1f}% vs baseline on {hardware}
Dimension(s): {dims}

Abstract excerpt:
  {excerpt}

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class {class_name}:
    """GPU-validated test for arXiv:{paper_id} — {technique}."""

    PAPER_ID = "{paper_id}"
    TECHNIQUE = "{technique}"
    METRIC = "{metric}"
    CLAIMED_IMPROVEMENT_PCT = {improvement_pct}
    HARDWARE = "{hardware}"
    DIMENSION_IDS = [{dims_quoted}]

    def test_claim_validation(self, tmp_path):
        """Validate: {technique} improves {metric} by {improvement_pct:.1f}% on {hardware}.

        TODO: implement — auto-generated stub awaiting GPU validation.
        """
        pytest.skip(
            f"{{self.TECHNIQUE}} claim validation not yet implemented — "
            f"see arXiv:{{self.PAPER_ID}}"
        )
'''


def _format_stub(claim: CapabilityClaim) -> str:
    """Helper used by generate_stub — exposed for testing."""
    dims_quoted = ", ".join(f'"{d}"' for d in claim.dimension_ids)
    safe_id = re.sub(r"[^\w]", "_", claim.paper_id)
    safe_tech = re.sub(r"[^\w]", "_", claim.technique)
    return _STUB_TEMPLATE.format(
        paper_id=claim.paper_id,
        title=claim.title[:120],
        technique=claim.technique,
        metric=claim.metric,
        improvement_pct=claim.improvement_pct,
        hardware=claim.hardware,
        dims=", ".join(claim.dimension_ids),
        dims_quoted=dims_quoted,
        class_name=f"TestArXiv_{safe_id}_{safe_tech}",
        excerpt=claim.raw_excerpt[:300].replace("\n", " "),
    )


# Override generate_stub to use _format_stub for consistent dims_quoted
def generate_stub(claim: CapabilityClaim, output_dir: Path) -> Path:  # noqa: F811
    """Write a pytest stub for *claim* to *output_dir*. Returns the path."""
    safe_id = re.sub(r"[^\w]", "_", claim.paper_id)
    safe_tech = re.sub(r"[^\w]", "_", claim.technique)
    filename = f"test_arxiv_{safe_id}_{safe_tech}.py"
    path = output_dir / filename

    stub = _format_stub(claim)
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(stub, encoding="utf-8")
    return path
