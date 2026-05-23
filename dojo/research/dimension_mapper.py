"""dojo/research/dimension_mapper.py — Map capability claims to matrix dimensions.

Keyword rules first; each technique maps to one or more d01-d50 dimension IDs.
"""

from __future__ import annotations

from dojo.research.claim_extractor import CapabilityClaim


# Technique → dimension ID(s)
_TECHNIQUE_TO_DIMS: dict[str, list[str]] = {
    "fp8":                   ["d10"],
    "fp4":                   ["d10"],
    "lora":                  ["d17"],
    "grpo":                  ["d18"],
    "ppo":                   ["d18"],
    "dpo":                   ["d18"],
    "flash_attention":       ["d15"],
    "grouped_query_attention": ["d15"],
    "gptq":                  ["d22"],
    "awq":                   ["d23"],
    "gguf":                  ["d24"],
    "vllm":                  ["d31"],
    "speculative_decoding":  ["d41"],
    "moe":                   ["d41"],
    "kv_cache":              ["d25"],
    "rope":                  ["d25"],
    "fsdp":                  ["d14"],
    "int4_bnb":              ["d22"],
}

_FALLBACK_DIM = "d41"   # "general inference optimization"


def map_claims(claims: list[CapabilityClaim]) -> list[CapabilityClaim]:
    """Fill dimension_ids on each claim in *claims*.  Returns the modified list."""
    for claim in claims:
        dims = _TECHNIQUE_TO_DIMS.get(claim.technique, [_FALLBACK_DIM])
        claim.dimension_ids = dims
    return claims


def get_dimension_for_technique(technique: str) -> list[str]:
    """Return dimension IDs for a single technique name."""
    return list(_TECHNIQUE_TO_DIMS.get(technique, [_FALLBACK_DIM]))
