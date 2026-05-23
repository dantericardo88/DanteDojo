"""dojo/research — arXiv harvest loop, claim extraction, and combination testing."""

from dojo.research.arxiv_scanner import ArxivPaper, search as arxiv_search
from dojo.research.claim_extractor import CapabilityClaim, extract_claims
from dojo.research.combination_composer import CombinationComposer, CombinationPair, CombinationResult
from dojo.research.dimension_mapper import map_claims
from dojo.research.harvest_loop import HarvestResult, run_harvest
from dojo.research.interaction_model import InteractionEffect, classify_interaction
from dojo.research.test_stub_generator import generate_stub

__all__ = [
    "ArxivPaper",
    "arxiv_search",
    "CapabilityClaim",
    "extract_claims",
    "map_claims",
    "HarvestResult",
    "run_harvest",
    "generate_stub",
    "CombinationComposer",
    "CombinationPair",
    "CombinationResult",
    "InteractionEffect",
    "classify_interaction",
]
