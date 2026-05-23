"""dojo/research/harvest_loop.py — arXiv harvest orchestrator.

Runs the full pipeline:
  1. Scan arXiv for papers matching queries
  2. Extract capability claims from each abstract
  3. Map claims to matrix dimensions
  4. Generate test stubs for novel claims
  5. Emit evidence for each step

CLI entry: dojo research scan --queries "quantization,flash attention"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dojo.core.evidence import EvidenceLogger
from dojo.research.arxiv_scanner import ArxivPaper, search
from dojo.research.claim_extractor import CapabilityClaim, extract_claims
from dojo.research.dimension_mapper import map_claims
from dojo.research.test_stub_generator import generate_stub


@dataclass
class HarvestResult:
    """Summary of one harvest loop run."""

    papers_scanned: int = 0
    claims_extracted: int = 0
    stubs_generated: int = 0
    stubs: list[Path] = field(default_factory=list)
    claims: list[CapabilityClaim] = field(default_factory=list)


def run_harvest(
    queries: list[str],
    output_dir: Path,
    max_results_per_query: int = 10,
    emit_evidence: bool = True,
) -> HarvestResult:
    """Run the full arXiv harvest loop for *queries*.

    Args:
        queries:                Query strings (e.g. ["quantization", "flash attention"]).
        output_dir:             Where to write generated test stubs.
        max_results_per_query:  Papers to fetch per query.
        emit_evidence:          Whether to emit EvidenceLogger events.

    Returns:
        HarvestResult with counts and generated stub paths.
    """
    logger = EvidenceLogger.get_instance() if emit_evidence else None
    result = HarvestResult()
    seen_ids: set[str] = set()

    for query in queries:
        try:
            papers = search(query, max_results=max_results_per_query)
        except Exception as e:
            if logger:
                logger.emit({
                    "type": "ARXIV_SCAN_ERROR",
                    "data": {"query": query, "error": str(e)},
                }, organ="research")
            continue

        for paper in papers:
            if paper.arxiv_id in seen_ids:
                continue
            seen_ids.add(paper.arxiv_id)
            result.papers_scanned += 1

            if logger:
                logger.emit({
                    "type": "ARXIV_PAPER_SCANNED",
                    "data": {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title[:120],
                        "published": paper.published,
                        "query": query,
                    },
                }, organ="research")

            claims = extract_claims(paper.arxiv_id, paper.title, paper.abstract)
            if not claims:
                continue

            claims = map_claims(claims)
            result.claims.extend(claims)
            result.claims_extracted += len(claims)

            for claim in claims:
                if logger:
                    logger.emit({
                        "type": "ARXIV_CLAIM_EXTRACTED",
                        "data": {
                            "arxiv_id": claim.paper_id,
                            "technique": claim.technique,
                            "metric": claim.metric,
                            "improvement_pct": claim.improvement_pct,
                            "hardware": claim.hardware,
                            "dimension_ids": claim.dimension_ids,
                        },
                    }, organ="research")

                stub_path = generate_stub(claim, output_dir)
                result.stubs.append(stub_path)
                result.stubs_generated += 1

                if logger:
                    logger.emit({
                        "type": "ARXIV_TEST_STUB_GENERATED",
                        "data": {
                            "arxiv_id": claim.paper_id,
                            "technique": claim.technique,
                            "stub_path": str(stub_path),
                            "dimension_ids": claim.dimension_ids,
                        },
                    }, organ="research")

            time.sleep(0.5)   # arXiv rate limit: 1 req/3s; 0.5s is safe between papers

    return result
