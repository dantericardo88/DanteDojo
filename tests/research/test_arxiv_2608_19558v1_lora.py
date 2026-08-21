"""AUTO-GENERATED test stub from arXiv:2608.19558v1

Title: Reliable Financial Named Entity Recognition under Domain Shift
Technique: lora
Metric: quality_score improved 10.0% vs baseline on unknown
Dimension(s): d17

Abstract excerpt:
  Financial AI systems often train information extractors on one textual register and deploy them across filings, news, and user-generated content, while standard F1 scores do not indicate which predictions remain safe to automate when the input distribution changes. We study confidence estimation and

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class TestArXiv_2608_19558v1_lora:
    """GPU-validated test for arXiv:2608.19558v1 — lora."""

    PAPER_ID = "2608.19558v1"
    TECHNIQUE = "lora"
    METRIC = "quality_score"
    CLAIMED_IMPROVEMENT_PCT = 10.0
    HARDWARE = "unknown"
    DIMENSION_IDS = ["d17"]

    def test_claim_validation(self, tmp_path):
        """Validate: lora improves quality_score by 10.0% on unknown.

        TODO: implement — auto-generated stub awaiting GPU validation.
        """
        pytest.skip(
            f"{self.TECHNIQUE} claim validation not yet implemented — "
            f"see arXiv:{self.PAPER_ID}"
        )
