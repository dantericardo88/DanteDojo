"""AUTO-GENERATED test stub from arXiv:2605.30232v1

Title: How's it going? Reinforcement learning in language models recruits a functional welfare axis
Technique: lora
Metric: quality_score improved 10.0% vs baseline on unknown
Dimension(s): d17

Abstract excerpt:
  How does reinforcement learning shape a language model's internal representations? We present evidence that RL recruits a pre-existing representation of functional welfare: an estimate of how well or badly the system is doing, relative to its goals. We train several language models in a novel, seman

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class TestArXiv_2605_30232v1_lora:
    """GPU-validated test for arXiv:2605.30232v1 — lora."""

    PAPER_ID = "2605.30232v1"
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
