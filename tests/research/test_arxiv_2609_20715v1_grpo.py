"""AUTO-GENERATED test stub from arXiv:2609.20715v1

Title: Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL
Technique: grpo
Metric: quality_score improved 10.0% vs baseline on unknown
Dimension(s): d18

Abstract excerpt:
  Agent trajectories record what an agent does and what happens next. Yet standard supervised fine-tuning (SFT) applies loss only to agent-authored action tokens, using environment observations as context but not as prediction targets. We ask whether this convention provides the best initialization fo

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class TestArXiv_2609_20715v1_grpo:
    """GPU-validated test for arXiv:2609.20715v1 — grpo."""

    PAPER_ID = "2609.20715v1"
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
