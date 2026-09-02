"""AUTO-GENERATED test stub from arXiv:2609.01455v1

Title: When Safety Routing Breaks: Understanding Alignment Fragility under Benign Fine-Tuning
Technique: lora
Metric: quality_score improved 10.0% vs baseline on unknown
Dimension(s): d17

Abstract excerpt:
  Benign fine-tuning severely weakens the safety alignment of large language models (LLMs), so we study why refusal behavior is so fragile. While prior work often attributes this failure to gradient conflict, we propose a fundamentally different Fisher-geometric explanation: safety Fisher is low-rank,

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class TestArXiv_2609_01455v1_lora:
    """GPU-validated test for arXiv:2609.01455v1 — lora."""

    PAPER_ID = "2609.01455v1"
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
