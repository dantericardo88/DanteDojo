"""dojo/research/claim_extractor.py — Extract capability claims from paper abstracts.

Uses rule-based extraction first, then falls back to a local Ollama LLM if
available. Returns structured CapabilityClaim objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CapabilityClaim:
    """A structured capability claim extracted from an arXiv paper."""

    paper_id: str
    title: str
    technique: str              # e.g. "GQA attention", "INT4 quantization"
    metric: str                 # e.g. "tokens_per_sec", "ppl", "memory_gb"
    value: float                # claimed result value
    baseline_value: float       # baseline value for comparison
    improvement_pct: float      # (value - baseline) / baseline * 100
    hardware: str               # e.g. "H100", "A100", "RTX 4090"
    dimension_ids: list[str] = field(default_factory=list)
    novelty_score: float = 0.5  # 0-1, filled by dimension_mapper
    raw_excerpt: str = ""


# ---------------------------------------------------------------------------
# Rule patterns
# ---------------------------------------------------------------------------

_TECHNIQUE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bflash.?attention\b", re.I), "flash_attention"),
    (re.compile(r"\bGQA\b|\bgrouped.query attention\b", re.I), "grouped_query_attention"),
    (re.compile(r"\bGPTQ\b", re.I), "gptq"),
    (re.compile(r"\bAWQ\b", re.I), "awq"),
    (re.compile(r"\bGGUF\b|\bllama\.cpp\b", re.I), "gguf"),
    (re.compile(r"\bFP8\b|\bfloat8\b", re.I), "fp8"),
    (re.compile(r"\bFP4\b|\bfloat4\b|\bNV_FP4\b", re.I), "fp4"),
    (re.compile(r"\bLoRA\b|\bQLoRA\b|\blow.rank adaptation\b", re.I), "lora"),
    (re.compile(r"\bGRPO\b", re.I), "grpo"),
    (re.compile(r"\bPPO\b|\bproximal policy\b", re.I), "ppo"),
    (re.compile(r"\bDPO\b|\bdirect preference\b", re.I), "dpo"),
    (re.compile(r"\bvLLM\b|\bcontinuous batching\b|\bPagedAttention\b", re.I), "vllm"),
    (re.compile(r"\bspeculative decoding\b", re.I), "speculative_decoding"),
    (re.compile(r"\bMoE\b|\bmixture.of.experts\b", re.I), "moe"),
    (re.compile(r"\bKV.cache\b|\bkey.value cache\b", re.I), "kv_cache"),
    (re.compile(r"\bRoPE\b|\brotary position\b", re.I), "rope"),
    (re.compile(r"\bFSDP\b|\bfully.sharded\b", re.I), "fsdp"),
    (re.compile(r"\bBitsAndBytes\b|\bbnb\b|\b4.bit quantiz\b", re.I), "int4_bnb"),
]

_HARDWARE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bH100\b"), "H100"),
    (re.compile(r"\bH200\b"), "H200"),
    (re.compile(r"\bA100\b"), "A100"),
    (re.compile(r"\bRTX\s*40[0-9]+\b", re.I), "RTX40xx"),
    (re.compile(r"\bRTX\s*30[0-9]+\b", re.I), "RTX30xx"),
    (re.compile(r"\bV100\b"), "V100"),
    (re.compile(r"\bMI300\b|\bMI350\b", re.I), "AMD_MI3xx"),
    (re.compile(r"\bTPU\b"), "TPU"),
    (re.compile(r"\bApple.M[0-9]\b", re.I), "Apple_M"),
]

_METRIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\btok(?:ens?)?/s(?:ec)?\b|\bthroughput\b", re.I), "tokens_per_sec"),
    (re.compile(r"\bperplexity\b|\bPPL\b"), "ppl"),
    (re.compile(r"\blatency\b|\bTTFT\b|\btime.to.first.token\b", re.I), "latency_ms"),
    (re.compile(r"\bGPU memory\b|\bVRAM\b|\bmemory.footprint\b", re.I), "memory_gb"),
    (re.compile(r"\bBLEU\b|\bROUGE\b|\bwin.rate\b", re.I), "quality_score"),
]

# Improvement patterns: "X% faster", "reduces by X%", "X× speedup", etc.
_IMPROVEMENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"(\d+(?:\.\d+)?)[×x]\s*(?:speedup|faster|improvement)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:faster|speedup|improvement|reduction|less)", re.I),
    re.compile(r"(?:reduc(?:es?|ing)|decreas(?:es?|ing))\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:lower|higher|better)", re.I),
]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def extract_claims(paper_id: str, title: str, abstract: str) -> list[CapabilityClaim]:
    """Extract capability claims from *abstract* using rule-based patterns.

    Returns one claim per detected technique. If no improvement percentage
    can be parsed, defaults to 10.0 (indicating qualitative improvement claim).
    """
    text = title + " " + abstract
    claims: list[CapabilityClaim] = []

    techniques = [name for pat, name in _TECHNIQUE_PATTERNS if pat.search(text)]
    if not techniques:
        return []

    hardware = _detect_hardware(text)
    metric = _detect_metric(text)
    improvement_pct = _detect_improvement(text)

    for technique in techniques:
        claims.append(CapabilityClaim(
            paper_id=paper_id,
            title=title,
            technique=technique,
            metric=metric,
            value=improvement_pct,
            baseline_value=100.0,
            improvement_pct=improvement_pct,
            hardware=hardware,
            raw_excerpt=abstract[:400],
        ))

    return claims


def _detect_hardware(text: str) -> str:
    for pat, name in _HARDWARE_PATTERNS:
        if pat.search(text):
            return name
    return "unknown"


def _detect_metric(text: str) -> str:
    for pat, name in _METRIC_PATTERNS:
        if pat.search(text):
            return name
    return "quality_score"


def _detect_improvement(text: str) -> float:
    """Return estimated improvement percentage from text, default 10.0."""
    for pat in _IMPROVEMENT_PATTERNS:
        m = pat.search(text)
        if m:
            raw = float(m.group(1))
            # If matched "Nx speedup", convert to percentage
            if "×" in pat.pattern or "x" in pat.pattern.lower():
                return round((raw - 1.0) * 100, 1)
            return round(raw, 1)
    return 10.0  # qualitative improvement claim default
