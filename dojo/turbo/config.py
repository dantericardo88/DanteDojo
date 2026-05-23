"""dojo/turbo/config.py — TurboConfig and ProfileResult dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from dojo.turbo.policy import PolicyType


@dataclass
class TurboConfig:
    """User-facing configuration for a turbocharge run."""

    model: str
    ppl_budget: float = 0.15
    optimize: str = "inference"      # "inference" | "training"
    output_path: Optional[str] = None
    bench_tokens: int = 512
    warmup_steps: int = 3
    bench_steps: int = 20
    device: str = "cuda:0"


@dataclass
class ProfileResult:
    """Benchmark result for a single (precision, quantization) configuration."""

    policy: PolicyType
    quant_scheme: str           # "none" | "gptq_4bit" | "awq_4bit"
    precision: str              # "fp8" | "bf16" | "fp16"
    tokens_per_sec: float
    ppl: float
    vram_gb: float
    success: bool = True
    error: str = ""

    @property
    def ppl_delta_pct(self) -> float:
        """Placeholder; optimizer fills this relative to the baseline."""
        return 0.0


@dataclass
class TurboResult:
    """Final output of a turbocharge session."""

    model: str
    winner: ProfileResult
    all_results: list[ProfileResult] = field(default_factory=list)
    gpu: str = ""
    sm: str = ""
