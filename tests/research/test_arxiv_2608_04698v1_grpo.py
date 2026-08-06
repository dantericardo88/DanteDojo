"""AUTO-GENERATED test stub from arXiv:2608.04698v1

Title: Teaching MLLMs to Say No: Generalized Referring Expression Comprehension via Refusal Calibrated GRPO
Technique: grpo
Metric: quality_score improved 10.0% vs baseline on unknown
Dimension(s): d18

Abstract excerpt:
  We tackle the challenging yet underexplored task of Generalized Referring Expression Comprehension (GREC), which requires a model to localize the object described by a textual expression when it exists (positive sample) and to refuse output when it does not (negative sample). Although Multimodal Lar

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class TestArXiv_2608_04698v1_grpo:
    """GPU-validated test for arXiv:2608.04698v1 — grpo."""

    PAPER_ID = "2608.04698v1"
    TECHNIQUE = "grpo"
    METRIC = "quality_score"
    CLAIMED_IMPROVEMENT_PCT = 10.0
    HARDWARE = "unknown"
    DIMENSION_IDS = ["d18"]

    def test_claim_validation(self, tmp_path):
        """Validate: grpo improves quality_score by 10.0% on unknown.

        TODO: implement — auto-generated stub awaiting GPU validation.
        """
        pytest.skip(
            f"{self.TECHNIQUE} claim validation not yet implemented — "
            f"see arXiv:{self.PAPER_ID}"
        )
