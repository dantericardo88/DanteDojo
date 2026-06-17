"""AUTO-GENERATED test stub from arXiv:2606.17107v1

Title: Models Take Notes at Prefill: KV Cache Can Be Editable and Composable
Technique: rope
Metric: latency_ms improved 10.0% vs baseline on unknown
Dimension(s): d25

Abstract excerpt:
  Prefix caching reuses prefill only across an exactly shared prefix, so one changed field invalidates the entire downstream cache. Yet overwriting the field's own key/value vectors and reusing the rest leaves the model acting on the old value. The reason, established causally across four model famili

HUMAN ACTION REQUIRED: implement test_claim_validation below.
Replace the `pytest.skip` with real GPU validation code.
"""

from __future__ import annotations

import os
import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"


@pytest.mark.skipif(not _REAL_GPU, reason="requires DOJO_REAL_GPU=1")
class TestArXiv_2606_17107v1_rope:
    """GPU-validated test for arXiv:2606.17107v1 — rope."""

    PAPER_ID = "2606.17107v1"
    TECHNIQUE = "rope"
    METRIC = "latency_ms"
    CLAIMED_IMPROVEMENT_PCT = 10.0
    HARDWARE = "unknown"
    DIMENSION_IDS = ["d25"]

    def test_claim_validation(self, tmp_path):
        """Validate: rope improves latency_ms by 10.0% on unknown.

        TODO: implement — auto-generated stub awaiting GPU validation.
        """
        pytest.skip(
            f"{self.TECHNIQUE} claim validation not yet implemented — "
            f"see arXiv:{self.PAPER_ID}"
        )
