"""AUTO-GENERATED test stub from arXiv:2607.21273v1

Title: The Dark Room in the Reward Channel: Dense Prediction Rewards Collapse GRPO-Trained LLM Agents -- and What Actually Work
Technique: grpo
Metric: quality_score improved 10.0% vs baseline on unknown
Dimension(s): d18

Abstract excerpt:
  Dense per-step supervision is an appealing remedy for sparse-reward, long-horizon LLM agents: reward the agent for predicting its next observation, and memory should follow. We show that under group-normalized RL (GRPO), this recipe does not merely fail -- it destroys the policy. Across Qwen3-1.7B/4

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class TestArXiv_2607_21273v1_grpo:
    """GPU-validated test for arXiv:2607.21273v1 — grpo."""

    PAPER_ID = "2607.21273v1"
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
