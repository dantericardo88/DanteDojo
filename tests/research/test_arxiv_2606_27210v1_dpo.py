"""AUTO-GENERATED test stub from arXiv:2606.27210v1

Title: Paved with True Intents: Intent-Aware Training Improves LLM Safety Classification Across Training Regimes
Technique: dpo
Metric: latency_ms improved 10.0% vs baseline on unknown
Dimension(s): d18

Abstract excerpt:
  We argue that safety classifiers should model user intent as an explicit signal between the prompt and the final label. To study this, we introduce AIMS, a human-annotated dataset of 1,724 difficult safety prompts, each paired with an intent description and harm label. We use AIMS to evaluate intent

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class TestArXiv_2606_27210v1_dpo:
    """GPU-validated test for arXiv:2606.27210v1 — dpo."""

    PAPER_ID = "2606.27210v1"
    TECHNIQUE = "dpo"
    METRIC = "latency_ms"
    CLAIMED_IMPROVEMENT_PCT = 10.0
    HARDWARE = "unknown"
    DIMENSION_IDS = ["d18"]

    def test_claim_validation(self, tmp_path):
        """Validate: dpo improves latency_ms by 10.0% on unknown.

        TODO: implement — auto-generated stub awaiting GPU validation.
        """
        pytest.skip(
            f"{self.TECHNIQUE} claim validation not yet implemented — "
            f"see arXiv:{self.PAPER_ID}"
        )
