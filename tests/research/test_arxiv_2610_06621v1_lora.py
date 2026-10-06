"""AUTO-GENERATED test stub from arXiv:2610.06621v1

Title: Frozen Factor or Spectral Band? Disentangling Two Choices in Low-Rank LoRA
Technique: lora
Metric: quality_score improved 10.0% vs baseline on unknown
Dimension(s): d17

Abstract excerpt:
  Spectral variants of low-rank adaptation (LoRA) choose both a subspace and which factor to freeze. We separate these choices by freezing the input factor A or output factor B on the top or bottom singular directions of pretrained weights, with learning rates selected separately. At rank 2, the same-

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class TestArXiv_2610_06621v1_lora:
    """GPU-validated test for arXiv:2610.06621v1 — lora."""

    PAPER_ID = "2610.06621v1"
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
