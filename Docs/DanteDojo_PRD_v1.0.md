# DanteDojo PRD v1.0

**Version:** 1.0.0
**Status:** Ready for Claude Code / Codex execution
**Date:** April 22, 2026
**Owner:** Ricky (solo founder, Real Empanada Corp + Dante ecosystem)
**Target completion:** 6 weeks (42 days) to v1.0.0-ship
**Donor foundation:** STEAK V3 (316K LOC Python, constitutional-architecture compliant)
**Repo:** `dante-dojo/`
**Language:** Python 3.11+
**Ecosystem peers:** DanteForge, DanteCode, DanteAgents (TypeScript); DanteHarvest (Python, parallel build)

---

## 0. Scope statement (what this PRD is and isn't)

This PRD specifies the **minimum lovable v1 of DanteDojo**: a sovereign-AI training, evaluation, specialization, and inference organ that runs on a solo founder's hardware (RTX 4060 laptop, RTX 3090 desktop, optional H100 cloud spot), consumes specialization packs from DanteHarvest, emits constitutional-passport-sealed fine-tuned models for DanteAgents, and validates them via the three-way promotion gate (Harvest provenance + Dojo evals + Forge policy).

**Two lanes:**
- **Lane 1 (Production):** LoRA fine-tuning of production OSS models, inference routing, local serving, quantization, idle-time training. Runs daily. Produces shippable specialist models.
- **Lane 2 (Foundry, v1.1 scope):** One proof-of-methodology experiment run during v1. Full Lane 2 orchestrator deferred to v1.1 after Lane 1 ships.

**NOT in scope for v1.0:**
- Full factorial ablation orchestrator (v1.1)
- Self-improvement RL loops beyond simple GRPO proof (v1.2)
- Diffusion/multimodal training (DanteStudio)
- Multi-node distributed training (v2)
- Managed-service integration (v1.1)
- EU AI Act compliance tooling (belongs in Forge)

**Architectural posture:** Dojo is Python. The ecosystem peers (Forge/Code/Agents) are TypeScript. Integration happens at artifact boundaries — signed JSON manifests, HTTP/JSON-RPC APIs, OCI-compatible model registry artifacts — not at the language level. This is how every serious ML platform works (HF TGI, Modal, Replicate all have polyglot control/execution splits).

---

## 1. Success criteria (how we know v1 shipped)

### 1.1 Functional acceptance

- [ ] Dojo installs cleanly on RTX 4060 laptop (Windows 11 + WSL2), RTX 3090 desktop (Ubuntu 22.04), and H100 cloud spot (Ubuntu 22.04) via `pip install dante-dojo` + one-command bootstrap.
- [ ] `dojo detect` autodetects hardware tier (4060 / 3090 / H100 / Apple / AMD / Intel) and emits a signed HAL profile.
- [ ] `dojo train --pack <harvest-pack-uri>` runs QLoRA fine-tune of Qwen2.5-7B on a Harvest pack on the 4060 laptop within 12 hours wall time.
- [ ] `dojo serve --model <model-uri>` starts a vLLM or llama.cpp server with OpenAI-compatible API; warm-up latency <60s; first token <500ms on 3090.
- [ ] `dojo route --task <task-name> --input <file>` dispatches inference to local model, API, or specialist based on routing policy and emits cost/latency receipt.
- [ ] `dojo eval --model <model-uri> --suite <suite-name>` runs lm-eval-harness + custom Dante suite and emits evidence-sealed eval receipt.
- [ ] `dojo promote --candidate <model-id>` walks the three-way gate (Harvest provenance → Dojo evals → Forge policy via HTTP) and emits a constitutional passport or blocks with reasons.
- [ ] `dojo idle-train` runs a fine-tune only during detected idle windows with thermal/VRAM/power guards.
- [ ] `dojo experiment run <experiment.yaml>` executes one Lane 2 proof experiment (μP-verified attention-variant sweep) with ≥5 seeds and emits bootstrapped CI report.

### 1.2 Constitutional acceptance

Every operation above must:
- Emit evidence events to the evidence chain (Merkle-rooted, SHA-256-hashed, thread-safe).
- Fail closed on missing preconditions (no silent degradation).
- Use an explicit finite-state machine (FSM) where non-trivial state exists.
- Be deterministically replayable from its evidence receipt.
- Pass the `one_door` doctrine (single public entry point per subsystem).

### 1.3 Integration acceptance

- [ ] Consumes DanteHarvest specialization packs (JSON + Parquet + signature) via `dojo pack import`.
- [ ] Emits model artifacts DanteAgents can load (OCI-manifest-compatible registry entries).
- [ ] Calls DanteForge policy gate via HTTP `/v1/policy/verify` for every promotion.
- [ ] MLflow 3.x tracing active for all training/eval runs; logs accessible via `dojo ui`.

### 1.4 Quality acceptance

- Every source file ≤ 500 LOC (KiloCode discipline). If a module exceeds 500 LOC, it must be split.
- No stubs, no `TODO`, no `NotImplementedError` in shipped code.
- `pytest` pass rate: 100% of written tests; coverage ≥ 70% of new code (not the transplanted STEAK code, which has its own test suite).
- `ruff` + `mypy --strict` clean on new code.
- Adversarial test suite passes: OOM cascade, FSM bypass attempts, evidence-chain tampering, promotion-gate bypass attempts.

### 1.5 Performance acceptance (on RTX 3090)

- Qwen2.5-7B QLoRA throughput: ≥ 600 tokens/sec training (Unsloth baseline).
- vLLM serving throughput: ≥ 80 tokens/sec single-stream, ≥ 400 tok/s batched (bs=16, seq=2048).
- llama.cpp GGUF Q4_K_M 7B: ≥ 40 tok/s single-stream.
- Idle-training resume latency: < 3 seconds from idle-detected to training-resumed.
- End-to-end promotion pipeline (pack → train → eval → gate → serve): < 18 hours on 3090 for a 20K-sample bookkeeping pack.

---

## 2. Product architecture

### 2.1 System context

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DANTE ECOSYSTEM (constitutional)                │
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │ DanteHarvest │────▶│  DanteDojo   │────▶│ DanteAgents  │         │
│  │  (packs)     │     │  (THIS PRD)  │     │ (runtime)    │         │
│  └──────────────┘     └──────┬───────┘     └──────────────┘         │
│                              │                                      │
│                              ▼ (policy gate via HTTP)               │
│                       ┌──────────────┐                              │
│                       │  DanteForge  │                              │
│                       │  (policy)    │                              │
│                       └──────────────┘                              │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  DanteCode   │  (used by Ricky + Codex to BUILD Dojo itself)     │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Dojo internal architecture

```
dante-dojo/
├── dojo/
│   ├── core/                    # Constitutional substrate (transplant from STEAK)
│   │   ├── evidence.py          # Evidence logger (transplant steak/core/evidence.py)
│   │   ├── one_door.py          # One-door doctrine (transplant steak/core/one_door.py)
│   │   ├── types.py             # Core types (DojoConfig, PrecisionType, TrainingMode)
│   │   ├── exceptions.py        # DojoError hierarchy
│   │   └── fsm.py               # FSM base class
│   │
│   ├── hal/                     # Hardware Abstraction Layer (transplant steak/hal/)
│   │   ├── detector.py          # Top-level autodetect
│   │   ├── nvidia.py            # NVIDIA detector (merged gpu_detector + architecture_classifier)
│   │   ├── apple.py             # Apple MLX profiler
│   │   ├── amd.py               # AMD ROCm profiler
│   │   ├── intel.py             # Intel Arc profiler
│   │   ├── vram.py              # VRAM auditor + monitor
│   │   └── profile.py           # HAL profile dataclass + signing
│   │
│   ├── turbo/                   # GPU Turbo Policy (transplant steak/gpu_turbo/)
│   │   ├── policy.py            # Arch→quantization policy FSM
│   │   └── arch_map.py          # ARCH_POLICY_MAP
│   │
│   ├── precision/               # Precision handling (transplant steak/precision/, fix Ampere FP8 reject)
│   │   ├── fp4.py
│   │   ├── fp8.py               # MUST reject on SM<8.9, route to BF16/W8A16-Marlin
│   │   ├── degradation.py
│   │   └── mixed.py
│   │
│   ├── pack/                    # Specialization pack ingest (DanteHarvest interface)
│   │   ├── schema.py            # Pack schema + signature validation
│   │   ├── importer.py          # Pack import FSM
│   │   ├── tokenize.py          # Tokenization + length bucketing
│   │   └── splits.py            # Train/val/held-out split determinism
│   │
│   ├── train/                   # Training engine (transplant + wrap Unsloth/axolotl)
│   │   ├── base.py              # BaseTrainer FSM
│   │   ├── finetune.py          # LoRA/QLoRA/Full-FT entry
│   │   ├── unsloth_backend.py   # Unsloth integration (4060 + 3090 single-GPU)
│   │   ├── axolotl_backend.py   # Axolotl integration (2x3090 multi-GPU)
│   │   ├── pretrain.py          # Lane 2 pretraining (transplant steak/training/pretrain.py)
│   │   ├── context.py           # Context length manager
│   │   └── checkpoint.py        # Checkpoint manager (transplant steak/checkpoints/)
│   │
│   ├── idle/                    # Idle training (transplant steak/s53_idle_training/)
│   │   ├── detector.py          # Idle window detection
│   │   ├── guardian.py          # Scheduler guardian (thermal/VRAM/power)
│   │   ├── resource.py          # Resource guard
│   │   └── accumulator.py       # Gradient accumulator
│   │
│   ├── stream/                  # Layer streaming (transplant steak/s51_layer_streaming/)
│   │   ├── loader.py            # Partial weight load for 4060
│   │   └── scheduler.py         # Layer swap scheduler
│   │
│   ├── quant/                   # Quantization (transplant steak/inference/quantization/)
│   │   ├── gguf.py              # GGUF export (llama.cpp target)
│   │   ├── gptq.py              # GPTQModel (NOT AutoGPTQ — archived)
│   │   ├── awq.py               # llm-compressor AWQ (NOT AutoAWQ — archived)
│   │   ├── nf4.py               # bitsandbytes NF4
│   │   └── validator.py         # Quality validation post-quant
│   │
│   ├── serve/                   # Inference serving (transplant steak/inference/serving/)
│   │   ├── launcher.py          # Unified serve launcher FSM
│   │   ├── vllm_backend.py      # vLLM launcher + lifecycle
│   │   ├── sglang_backend.py    # SGLang launcher + lifecycle
│   │   ├── llamacpp_backend.py  # llama.cpp server launcher
│   │   ├── mlx_backend.py       # mlx-lm server (Apple Silicon)
│   │   └── manifest.py          # Deployment manifest builder
│   │
│   ├── route/                   # Inference routing (NEW)
│   │   ├── router.py            # Task→endpoint policy FSM
│   │   ├── policy.py            # Routing policy (local/api/specialist/cost)
│   │   ├── litellm_adapter.py   # LiteLLM integration (pin compromised-version-safe)
│   │   └── cost_tracker.py      # API cost displacement tracking
│   │
│   ├── eval/                    # Evaluation harness (NEW — wraps lm-eval-harness + LiveBench)
│   │   ├── harness.py           # Unified eval runner
│   │   ├── lm_eval_adapter.py   # lm-evaluation-harness wrapper
│   │   ├── livebench_adapter.py # LiveBench rolling-monthly wrapper
│   │   ├── dante_suite.py       # Dante-specific evals (bookkeeping F1, etc.)
│   │   └── statistics.py        # Bootstrap CIs, seed variance
│   │
│   ├── registry/                # Model registry (transplant steak/foundry/model_registry.py)
│   │   ├── registry.py          # Model registry FSM
│   │   ├── artifact.py          # OCI-compatible artifact shape
│   │   ├── fingerprint.py       # Deterministic model fingerprinting
│   │   └── store.py             # Local filesystem backend
│   │
│   ├── experiment/              # Experiment tracking (transplant steak/experiment/)
│   │   ├── tracker.py           # MLflow 3.x GenAI tracing wrapper
│   │   ├── metrics.py           # Metric aggregation
│   │   ├── compare.py           # Run comparison
│   │   └── registry.py          # Run registry
│   │
│   ├── promote/                 # Three-way promotion gate (NEW)
│   │   ├── gate.py              # Top-level promotion FSM
│   │   ├── harvest_check.py     # Harvest provenance receipt verifier
│   │   ├── dojo_check.py        # Dojo eval threshold verifier
│   │   ├── forge_check.py       # Forge HTTP policy gate client
│   │   ├── passport.py          # Constitutional passport builder
│   │   └── sealer.py            # Merkle-seal passport (transplant evidence_sealer)
│   │
│   ├── cli/                     # CLI (transplant steak/cli/)
│   │   ├── main.py              # Entry point (Typer-based)
│   │   ├── detect.py            # `dojo detect`
│   │   ├── pack.py              # `dojo pack import/inspect`
│   │   ├── train.py             # `dojo train`
│   │   ├── idle_train.py        # `dojo idle-train`
│   │   ├── serve.py             # `dojo serve`
│   │   ├── route.py             # `dojo route`
│   │   ├── eval.py              # `dojo eval`
│   │   ├── promote.py           # `dojo promote`
│   │   ├── registry.py          # `dojo registry ls/show/rm`
│   │   ├── experiment.py        # `dojo experiment`
│   │   └── ui.py                # `dojo ui` (serves MLflow UI + custom dashboard)
│   │
│   └── ecosystem/               # Ecosystem integration (NEW)
│       ├── harvest_client.py    # HTTP client to DanteHarvest
│       ├── forge_client.py      # HTTP client to DanteForge
│       ├── agents_publisher.py  # Model-registry publish for DanteAgents
│       └── evidence_bridge.py   # Cross-organ evidence chain linking
│
├── tests/
│   ├── unit/                    # Per-module unit tests
│   ├── integration/             # Cross-module
│   ├── constitutional/          # One-door, evidence chain, fail-closed
│   ├── adversarial/             # OOM, FSM bypass, chain tampering
│   ├── e2e/                     # Full pack → promote → serve pipeline
│   └── fixtures/                # Test fixtures, mock HAL profiles, tiny packs
│
├── donors/
│   └── steak-v3/                # Full STEAK V3 repo, read-only reference
│
├── docs/
│   ├── architecture.md
│   ├── constitutional-guide.md
│   ├── hardware-tiers.md
│   ├── promotion-gate.md
│   └── api/                     # API reference
│
├── scripts/
│   ├── bootstrap.sh             # One-command install
│   ├── bootstrap.bat            # Windows equivalent
│   ├── benchmark.py             # Performance regression suite
│   └── replay.py                # Deterministic replay from receipt
│
├── examples/
│   ├── bookkeeping_finetune.py
│   ├── idle_train_overnight.py
│   └── lane2_attention_sweep.yaml
│
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── CONSTITUTIONAL.md            # The rules Dojo must obey
```

---

## 3. Component specifications

### 3.1 Constitutional substrate (`dojo/core/`)

#### 3.1.1 Evidence logger (`dojo/core/evidence.py`)

**Transplant source:** `steak/core/evidence.py` (1,958 lines, already constitutional).

**Responsibilities:**
- Thread-safe append-only event log.
- Merkle-root computation over events per session.
- SHA-256 hash chain: each event references previous event hash.
- `@contextmanager` operation spans (`start_operation` / `info` / `warn` / `error`).
- In-memory + filesystem persistence (JSONL + Merkle root file).
- Fail-closed: a broken evidence logger must block all operations, not silently drop events.

**Public API:**
```python
class EvidenceLogger:
    @classmethod
    def get_instance(cls) -> "EvidenceLogger": ...
    @contextmanager
    def start_operation(self, name: str, attrs: dict) -> Iterator["OperationSpan"]: ...
    def info(self, event: dict) -> None: ...
    def warn(self, event: dict) -> None: ...
    def error(self, event: dict) -> None: ...
    def compute_merkle_root(self) -> str: ...
    def export_receipt(self, path: Path) -> ReceiptManifest: ...
```

**Acceptance:**
- 100% backward-compatible with STEAK V3's test suite (`tests/unit/test_evidence_*`).
- Add Merkle-root cross-verification against `dojo/ecosystem/evidence_bridge.py`.

#### 3.1.2 One-door doctrine (`dojo/core/one_door.py`)

**Transplant source:** `steak/core/one_door.py` (1,609 lines).

**Responsibilities:**
- Decorator-based enforcement: every subsystem has exactly one public entry point.
- Runtime check: attempts to call internal functions from outside raise `OneDoorViolation`.
- Evidence emission on every one-door-protected call (method name, args hash, caller provenance).

**Public API:**
```python
@one_door("train.finetune")
def finetune(config: FinetuneConfig) -> FinetuneResult: ...
```

**Acceptance:**
- Adversarial test: attempting to import `dojo.train.unsloth_backend._internal_fn` from outside `dojo.train` raises `OneDoorViolation`.

#### 3.1.3 Core types (`dojo/core/types.py`)

New file, but structure derived from `steak/core/types.py` (1,179 lines).

**Required dataclasses:**
```python
@dataclass(frozen=True)
class DojoConfig:
    hardware_tier: HardwareTier
    training_mode: TrainingMode  # LORA, QLORA, FULL_FT, PRETRAIN
    precision: PrecisionType     # BF16, FP8, FP4, NF4, INT4
    backend: TrainingBackend     # UNSLOTH, AXOLOTL, NATIVE
    seed: int
    evidence_dir: Path

@dataclass(frozen=True)
class PackReference:
    uri: str                     # e.g. "harvest://empanada-bookkeeping/v3"
    sha256: str
    signature: bytes
    harvest_receipt_id: str

@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str             # e.g. "dojo://qwen25-7b-empanada-bookkeeping/v1"
    base_model: str
    lora_weights: Optional[Path]
    quantization: Optional[QuantizationSpec]
    training_receipt: str        # points into evidence chain
    passport: Optional[ConstitutionalPassport]
```

**Acceptance:** All dataclasses are `frozen=True` and pydantic-validated. Round-trip JSON serialization is deterministic (sorted keys, stable separators).

### 3.2 Hardware Abstraction Layer (`dojo/hal/`)

**Transplant source:** `steak/hal/` (8,593 lines, 15 files). Consolidate to 8 files under 500 LOC each.

**New file: `dojo/hal/detector.py`**

```python
@one_door("hal.detect")
def detect() -> HALProfile:
    """
    Autodetect current hardware and return a signed HALProfile.

    Detection order:
      1. NVIDIA CUDA (dojo.hal.nvidia)
      2. Apple MLX (dojo.hal.apple)
      3. AMD ROCm (dojo.hal.amd)
      4. Intel Arc (dojo.hal.intel)
      5. CPU-only fallback

    Emits evidence events:
      - hal.detect.start
      - hal.detect.{nvidia|apple|amd|intel|cpu}.found
      - hal.detect.profile.signed

    Fails closed if no viable backend found.
    """
```

**`HALProfile` fields:**
```python
@dataclass(frozen=True)
class HALProfile:
    tier: HardwareTier  # TIER_4060_LAPTOP | TIER_3090 | TIER_3090_DUAL | TIER_H100 | TIER_APPLE_M4 | TIER_AMD_MI300 | TIER_INTEL_B580 | TIER_CPU
    gpu_name: str
    vram_bytes: int
    compute_capability: str
    has_native_fp8: bool  # Critical: Ampere=False, Ada/Hopper/Blackwell=True
    has_native_fp4: bool  # Blackwell only
    has_nvlink: bool
    pcie_gen: int
    interconnect_bandwidth_gb_s: float
    memory_bandwidth_gb_s: float
    driver_version: str
    profile_sha256: str
    signed_at: datetime
```

**Ampere FP8 rejection (resolves Open Question 2 from audit):**
- `dojo/hal/nvidia.py::detect_fp8_capability()` MUST return `False` for compute capability < 8.9.
- `dojo/precision/fp8.py::FP8Handler.ensure_supported()` MUST raise `PrecisionUnsupported("FP8 requires SM >= 8.9; got SM {cc}")` on Ampere.
- Policy router in `dojo/turbo/policy.py` MUST route Ampere to BF16 (training) or W8A16-Marlin (serving), never to FP8.

**Acceptance:**
- On RTX 3090: profile.has_native_fp8 == False, attempting FP8 training raises PrecisionUnsupported.
- On RTX 4060: profile.has_native_fp8 == True, FP8 training path runs.
- Profile signing verifies against `dojo.hal.profile.verify_signature()`.

### 3.3 GPU Turbo Policy (`dojo/turbo/`)

**Transplant source:** `steak/gpu_turbo/` (1,602 lines, constitutional FSM verified).

**Key correction to STEAK's ARCH_POLICY_MAP** (in `dojo/turbo/arch_map.py`):

```python
ARCH_POLICY_MAP: dict[str, list[PolicyType]] = {
    "NVIDIA Blackwell": [PolicyType.FP4_QUANT, PolicyType.HET_COMM],  # SM 10.x, native NVFP4
    "NVIDIA Hopper":    [PolicyType.FP8_QUANT, PolicyType.HET_COMM],  # SM 9.0, native FP8
    "NVIDIA Ada":       [PolicyType.FP8_QUANT, PolicyType.HET_COMM],  # SM 8.9, native FP8 (4060/4090)
    "NVIDIA Ampere":    [PolicyType.BF16_DEFAULT, PolicyType.W8A16_MARLIN_SERVING],  # SM 8.6 (3090) — NO FP8
    "AMD MI350":        [PolicyType.FP4_QUANT, PolicyType.HET_COMM],
    "AMD MI300":        [PolicyType.FP8_QUANT, PolicyType.HET_COMM],
    "Apple M4 Max":     [PolicyType.INT4_QUANT],
    "Apple M5 Max":     [PolicyType.INT4_QUANT],
    "Intel Arc B580":   [PolicyType.INT4_QUANT],
}
```

**Acceptance:** Attempting FP8 training on an Ampere-reported profile fails closed with actionable error message suggesting BF16 or NF4 alternative.

### 3.4 Pack ingest (`dojo/pack/`)

**New subsystem** — no STEAK equivalent. Harvest delivers packs; Dojo consumes.

**Pack schema (`dojo/pack/schema.py`):**
```python
@dataclass(frozen=True)
class SpecializationPack:
    pack_id: str                  # e.g. "empanada-bookkeeping"
    version: str                  # semver
    schema_version: str           # Pack schema version
    sample_count: int
    format: PackFormat            # INSTRUCTION, CONVERSATION, COMPLETION, ALPACA, SHAREGPT
    splits: dict[str, int]        # {"train": 18000, "val": 1000, "held_out_gold": 1000}
    harvest_receipt: str          # Merkle root from Harvest
    license: LicenseSpec
    provenance: ProvenanceManifest
    signature: bytes              # Ed25519 signature by Harvest
    manifest_sha256: str
    data_parquet_sha256: str
```

**Ingest FSM states:** `IDLE → FETCHING → VERIFYING_SIGNATURE → VALIDATING_SCHEMA → DEDUPLICATING → TOKENIZING → BUCKETING → INDEXED`

**Contamination check:** `dojo/pack/importer.py` MUST reject a pack if any sample appears in Dojo's eval gold set (SHA-256 match on normalized text). Emits `pack.contamination.detected` evidence event.

**Acceptance:**
- Tampered signature → pack rejected, evidence emitted.
- Missing schema field → pack rejected with specific field name.
- Contaminated sample detected → pack rejected with conflicting eval set name.
- Valid pack → indexed locally, pack receipt emitted to evidence chain.

### 3.5 Training engine (`dojo/train/`)

**Transplant source:** `steak/training/` (9,354 lines) + wrap Unsloth/axolotl.

**`dojo/train/finetune.py` entry point:**
```python
@one_door("train.finetune")
def finetune(config: FinetuneConfig) -> FinetuneResult:
    """
    Fine-tune a base model on a specialization pack.

    Routes to backend based on HALProfile + config.backend:
      - Tier 4060 laptop → UNSLOTH backend (only option that fits 8GB)
      - Tier 3090 single → UNSLOTH (throughput leader)
      - Tier 3090 dual → AXOLOTL (FSDP2 config-driven)
      - Tier H100 → AXOLOTL or NATIVE
      - Tier Apple M4+ → MLX-LM backend

    Evidence emission:
      - train.finetune.started (config hash, pack id, hardware tier)
      - train.finetune.backend.selected
      - train.finetune.step (every N steps)
      - train.finetune.checkpoint.saved
      - train.finetune.completed (loss curve, final metrics)

    Fails closed on:
      - VRAM OOM (emit diagnostic + actionable config reduction)
      - Loss divergence (NaN/Inf detected)
      - Thermal throttle (>85°C sustained, idle-train only)
      - Pack signature invalid
    """
```

**Backend selection FSM:** `IDLE → PROFILING → ROUTING → PREPARING → TRAINING → CHECKPOINTING → EVALUATING → PROMOTING | FAILED`

**`dojo/train/unsloth_backend.py` (new):**
- Pins Unsloth `2026.3.x` (active channel).
- Memory budget enforcer: refuses configs that exceed HALProfile.vram_bytes × 0.85.
- 4060-laptop-safe defaults for Qwen2.5-7B: `r=16, bs=1, grad_accum=8, seq=2048, paged_adamw_8bit, use_gradient_checkpointing="unsloth"`.
- Emits evidence span per gradient accumulation boundary (not per step — too noisy).

**`dojo/train/axolotl_backend.py` (new):**
- Generates axolotl YAML from `FinetuneConfig` + HALProfile.
- Supports FSDP2 for 2×3090 and DeepSpeed ZeRO-3 for H100.
- Streams axolotl logs into evidence chain.

**Acceptance:**
- Qwen2.5-7B QLoRA on 20K-sample pack completes on 4060 laptop in ≤ 12 hours with no OOM.
- Qwen2.5-7B QLoRA on 3090 completes in ≤ 3 hours.
- Backend switch between Unsloth/axolotl produces compatible LoRA adapter checkpoints (deterministic hash on same seed + data).

### 3.6 Idle training (`dojo/idle/`)

**Transplant source:** `steak/s53_idle_training/` (1,728 lines, genuine differentiator).

**`dojo/idle/detector.py`:**
- Windows: polls `GetLastInputInfo` via `pywin32`.
- Linux: polls X11 `XScreenSaverQueryInfo` or Wayland `idle-inhibit-unstable-v1`.
- macOS: polls `CGEventSourceSecondsSinceLastEventType`.
- Idle threshold: 5 minutes (configurable).
- Emits `idle.detected` / `idle.interrupted` evidence events.

**`dojo/idle/guardian.py`:**
- Thermal guard: pauses training if GPU temp > 80°C sustained 30s.
- VRAM guard: pauses if foreground app requests VRAM (NVML query every 2s).
- Power guard: pauses on battery if laptop.
- Resume: automatic when idle conditions restored.

**Acceptance:**
- User moves mouse → training pauses within 1 second.
- GPU hits 82°C → training pauses, emits `idle.thermal.paused` event.
- User goes idle again + thermal recovers → training resumes, idempotent gradient accumulation continues from exact step.
- Checkpoint every 10 minutes during idle training; resume from last checkpoint on system reboot.

### 3.7 Layer streaming (`dojo/stream/`)

**Transplant source:** `steak/s51_layer_streaming/` (3,072 lines).

**Purpose:** Run models larger than VRAM on 4060 laptop by streaming layers from system RAM or disk.

**Key use case:** Qwen2.5-14B or 32B inference on 4060 at 3-5 tok/s (acceptable for offline batch workloads like overnight bookkeeping categorization of a 10K-txn backlog).

**Acceptance:** Llama-3-13B Q4_K_M runs on 4060 at ≥ 3 tok/s with layer streaming enabled.

### 3.8 Quantization (`dojo/quant/`)

**Transplant source:** `steak/inference/quantization/` (~2,700 lines). REPLACE `auto_gptq` (archived) with `gptq-model` and `autoawq` (archived) with `llm-compressor`.

**Supported export formats:**
- GGUF (llama.cpp target) — `dojo/quant/gguf.py`
- GPTQ via gptq-model — `dojo/quant/gptq.py`
- AWQ via llm-compressor — `dojo/quant/awq.py`
- NF4 via bitsandbytes (training-only) — `dojo/quant/nf4.py`

**Post-quantization validator (`dojo/quant/validator.py`):**
- Runs the base model + quantized model on a fixed 200-prompt held-out set.
- Rejects quantization if perplexity delta > 15% or task F1 delta > 5 points.
- Emits quality receipt into evidence chain.

**Acceptance:**
- Qwen2.5-7B Q4_K_M GGUF export passes validator with ≤ 5% perplexity regression.
- GPTQ-int4 Qwen2.5-7B export passes validator; loadable by vLLM.

### 3.9 Inference serving (`dojo/serve/`)

**Transplant source:** `steak/inference/serving/` (~2,560 lines of configurators).

**Critical change vs STEAK:** STEAK's configurators generate configs but may not launch servers. Dojo's `serve/` MUST actually launch and manage server lifecycle (Open Question 6 resolved by design).

**`dojo/serve/launcher.py` FSM states:** `IDLE → CONFIGURING → PROVISIONING → WARMING → READY → STOPPING → STOPPED | FAILED`

**Backends:**
- `vllm_backend.py`: pins vLLM 0.13+ (April 2026 active), launches via subprocess, health-checks `/v1/models`.
- `sglang_backend.py`: pins SGLang latest, launches via subprocess.
- `llamacpp_backend.py`: pins llama.cpp via `llama-cpp-python` or subprocess to `llama-server`.
- `mlx_backend.py`: pins `mlx-lm` for Apple Silicon.

**Backend selection:** policy routes by HALProfile:
- 4060 laptop → llama.cpp (vLLM too heavy, needs ≥16GB)
- 3090 → vLLM default, SGLang on opt-in
- 2×3090 → vLLM with TP=2
- H100 → vLLM or SGLang
- Apple → mlx-lm
- AMD → vLLM with ROCm (verified production-ready April 2026)

**Deployment manifest (`dojo/serve/manifest.py`):**
```python
@dataclass(frozen=True)
class DeploymentManifest:
    model_artifact_id: str
    backend: Backend
    endpoint_url: str            # e.g. "http://127.0.0.1:8080/v1"
    context_length: int
    quantization: Optional[str]
    tp_size: int
    dtype: str
    passport_id: str             # points to constitutional passport
    started_at: datetime
    pid: int
    manifest_sha256: str
```

**Acceptance:**
- `dojo serve --model qwen25-7b-empanada-bookkeeping/v1` starts server, emits manifest, DanteAgents can call it.
- Server crash → evidence event emitted, auto-restart optional.
- Graceful shutdown on SIGTERM (clean drain).

### 3.10 Inference routing (`dojo/route/`)

**New subsystem** — no direct STEAK equivalent.

**Purpose:** decide per-task whether to use a local specialist, a local generalist, or an external API.

**`dojo/route/policy.py`:**
```python
@dataclass(frozen=True)
class RoutingPolicy:
    task_name: str
    preference_order: list[RoutingOption]  # e.g. [LOCAL_SPECIALIST, LOCAL_GENERALIST, API_HAIKU, API_SONNET]
    max_latency_ms: int
    max_cost_usd: float
    privacy_tier: PrivacyTier  # PUBLIC, INTERNAL, CONFIDENTIAL, SECRETS
    fallback_on_unavailable: bool
```

**Privacy tier enforcement:** CONFIDENTIAL + SECRETS → never route to external API. This is load-bearing for Real Empanada bookkeeping.

**LiteLLM integration (`dojo/route/litellm_adapter.py`):**
- Pins `litellm` to a known-clean version (not v1.82.8 which was compromised per research).
- Uses LiteLLM for the "external API" option only; local serving goes direct to vLLM.

**Cost tracker (`dojo/route/cost_tracker.py`):**
- Every route decision emits a cost receipt: `{task, route_taken, tokens_in, tokens_out, cost_usd, latency_ms, local_displaced_estimate}`.
- Monthly rollup: `dojo route report --month 2026-04` shows API cost displaced.

**Acceptance:**
- Confidential task with API-only policy → route refuses, emits `route.privacy.blocked` event, proposes local fallback.
- Available local specialist that passes privacy tier → wins over API every time.
- Local specialist fails → falls back to API iff privacy tier permits.

### 3.11 Evaluation (`dojo/eval/`)

**New subsystem** — replaces STEAK's 82-line `evaluation_harness/` stub.

**`dojo/eval/harness.py`:**
```python
@one_door("eval.run")
def run_eval(model: ModelArtifact, suite: EvalSuite, seeds: int = 5) -> EvalReceipt:
    """
    Run an evaluation suite on a model, with multi-seed bootstrap CIs.

    Supported suites:
      - LmEvalHarness(tasks=["mmlu", "arc", ...])
      - LiveBench(category="reasoning" | "coding" | "math")
      - DanteSuite(suite_name="empanada_bookkeeping_f1")

    Statistics: bootstrap 95% CI over N seeds (N >= 5 for Lane 2 scientific validity).

    Fail-closed on:
      - Task contamination detected against pack
      - Judge LLM disagreement > threshold
      - Seed variance > 2σ (indicates unstable model)
    """
```

**`dojo/eval/dante_suite.py` — Dante-specific evals:**
- `empanada_bookkeeping_f1`: 1000 held-out 2026 Q1 txns, F1 on 25-category classification.
- `b2b_email_quality`: 200 drafts, judged by Sonnet 4.6 (frozen judge) on 5-dim rubric.
- `sop_extraction_faithfulness`: 20 SOPs, extracts exact-match on key steps.
- `code_review_actionability`: 500 reviews, judged on issue-finding precision/recall.
- `ocr_cleanup_exact_match`: 300 receipts, character-level accuracy.

**Frozen judge rule:** For any LLM-as-judge eval, the judge model MUST be a different model family than the candidate (no Qwen judging Qwen) AND the judge's weights MUST be pinned SHA-256 for 6 months minimum.

**Acceptance:**
- Eval receipt includes: task name, model artifact id, seed list, bootstrap CI, judge family/version, contamination check pass/fail, Merkle hash.
- 5-seed Qwen2.5-7B-base run on `empanada_bookkeeping_f1` produces CI that excludes 0 (baseline is deterministic).
- Judge-drift detector alerts if judge-vs-gold agreement drops below 80% over 30-day window.

### 3.12 Experiment tracking (`dojo/experiment/`)

**Transplant source:** `steak/experiment/` (2,970 lines).

**Wrap MLflow 3.x GenAI features** (tracing, AI Gateway, built-in judges).

**Required capabilities:**
- Every `dojo train` / `dojo eval` / `dojo experiment run` call creates an MLflow run.
- Parameters: config hash, pack id, hardware tier, git sha of Dojo, seed.
- Metrics: per-step loss, eval scores, wall-clock, VRAM peak, tokens/sec.
- Artifacts: evidence chain export, checkpoint manifest, deployment manifest.
- `dojo ui` launches MLflow UI on `http://127.0.0.1:5000` plus a custom dashboard overlay showing Dante-specific context (pack, passport, promotion status).

**Acceptance:**
- Two runs with identical config + seed produce identical evidence Merkle roots and MLflow metric histograms.
- Comparing runs via `dojo experiment compare --runs A,B,C` produces a side-by-side report.

### 3.13 Three-way promotion gate (`dojo/promote/`)

**New subsystem** — the core Dojo→Ecosystem contract.

**`dojo/promote/gate.py` FSM:**
```
INIT → HARVEST_CHECK → DOJO_CHECK → FORGE_CHECK → PASSPORT_BUILD → SEALED | BLOCKED
```

**All three gates must return `GREEN` for the candidate to receive a constitutional passport.**

**`dojo/promote/harvest_check.py`:**
- Verifies Harvest receipt referenced by the pack is still valid (not revoked).
- Checks pack license allows derived fine-tuned model redistribution (if required).
- Confirms pack provenance manifest signed by DanteHarvest's public key.
- Emits `promote.harvest.green` or `promote.harvest.blocked(reason)`.

**`dojo/promote/dojo_check.py`:**
- Runs the candidate model through the Dante eval suite for the pack's task.
- Compares against baseline (base model zero-shot + configured external API baseline like Haiku 4.5).
- Requires: candidate F1 ≥ baseline F1 + 5 points AND 95% CI lower bound ≥ baseline F1.
- Confirms: no regression on safety subset (TruthfulQA sample + harmbench sample).
- Emits `promote.dojo.green` or `promote.dojo.blocked(reason, metrics)`.

**`dojo/promote/forge_check.py`:**
```python
@one_door("promote.forge")
def forge_check(candidate: ModelArtifact) -> ForgePolicyReceipt:
    """
    HTTP POST to DanteForge /v1/policy/verify with full candidate manifest
    (model artifact, evidence chain, Dojo eval receipts, pack provenance).

    DanteForge returns signed policy receipt or policy-violation reasons.

    Timeout: 60s. Fail-closed on timeout (BLOCKED state, not GREEN).
    """
```

**Retry / circuit breaker:** 3 retries with exponential backoff, then circuit-open for 5 minutes. Never optimistically promote.

**`dojo/promote/passport.py` — Constitutional passport:**
```python
@dataclass(frozen=True)
class ConstitutionalPassport:
    passport_id: str
    model_artifact_id: str
    harvest_receipt_id: str
    harvest_check_receipt: bytes   # signed by Harvest key
    dojo_check_receipt: bytes      # signed by Dojo key
    forge_check_receipt: bytes     # signed by Forge key
    merkle_root: str               # combines all three
    issued_at: datetime
    expires_at: datetime           # default 90 days
    revocation_url: str            # Forge endpoint to check revocation
```

**`dojo/promote/sealer.py`:**
- Transplants `steak/certification/ultimate/evidence_sealer.py` (380 lines).
- Builds Merkle root over the three receipts + model artifact fingerprint.
- Signs with Dojo's long-lived signing key (from `~/.dojo/keys/`).

**Acceptance:**
- Attempt to promote a model on a revoked Harvest receipt → BLOCKED at harvest_check with actionable reason.
- Attempt to promote a model that doesn't beat baseline → BLOCKED at dojo_check with metric comparison.
- Forge unreachable → BLOCKED (not stalled, not optimistic-promoted).
- All three green → passport issued, DanteAgents can verify it offline using public keys.

### 3.14 CLI (`dojo/cli/`)

**Framework:** Typer (FastAPI-style type-driven CLI, rich help, shell completion).

**Top-level commands:**
```
dojo detect                              # Autodetect hardware, emit profile
dojo pack import <uri>                   # Import Harvest pack
dojo pack inspect <pack-id>              # Show pack metadata, samples
dojo train --pack <id> [--config <f>]    # Fine-tune
dojo idle-train --pack <id>              # Train only during idle windows
dojo eval --model <id> --suite <name>    # Run eval suite
dojo serve --model <id> [--backend <b>]  # Start inference server
dojo route --task <name> --input <f>     # Route single inference through policy
dojo registry ls                          # List models
dojo registry show <id>                  # Show model + passport
dojo promote <candidate-id>              # Run three-way gate
dojo experiment run <config.yaml>        # Lane 2 experiment
dojo experiment compare --runs A,B,C     # Compare runs
dojo replay <receipt-path>               # Deterministic replay
dojo ui                                  # Launch MLflow UI + custom dashboard
```

**Exit code contract:** 0 success, 1 operation failure, 2 usage error, 3 constitutional violation, 4 promotion blocked.

**Acceptance:** all commands ship with `--help`, examples, and integration tests.

### 3.15 Ecosystem integration (`dojo/ecosystem/`)

**New subsystem** — the cross-organ HTTP/artifact contracts.

**`dojo/ecosystem/harvest_client.py`:**
```python
class HarvestClient:
    def fetch_pack(self, pack_uri: str) -> SpecializationPack: ...
    def verify_receipt(self, receipt_id: str) -> ReceiptStatus: ...
    def check_revocation(self, receipt_id: str) -> bool: ...
```

Default Harvest endpoint: `http://127.0.0.1:8001` (local dev) or configured via `DANTE_HARVEST_URL`.

**`dojo/ecosystem/forge_client.py`:**
```python
class ForgeClient:
    def verify_policy(self, candidate_manifest: dict) -> ForgePolicyReceipt: ...
    def check_passport_revocation(self, passport_id: str) -> bool: ...
    def get_forge_public_key(self) -> bytes: ...
```

Default Forge endpoint: `http://127.0.0.1:8002`.

**`dojo/ecosystem/agents_publisher.py`:**
```python
class AgentsPublisher:
    def publish_model(self, artifact: ModelArtifact, passport: ConstitutionalPassport) -> None:
        """Register model + passport in local registry DanteAgents polls."""
```

Publishes via filesystem (`~/.dante/registry/models/`) or configured registry backend. Emits manifest Agents can discover via inotify / file watcher.

**Acceptance:**
- Mock Harvest + Mock Forge servers enable end-to-end test of promotion pipeline.
- DanteAgents mock consumer verifies passport using Dojo public key, loads model, runs inference.

---

## 4. Implementation phases (6 weeks, 42 days)

Each phase has a gate: all phase deliverables pass all tests before next phase starts. Fail closed.

### Phase 0 — Donor inspection + scaffolding (Days 1-3)

**Goal:** Resolve the 8 open questions from the donor audit and scaffold the repo.

**Day 1:**
- Clone STEAK V3 into `donors/steak-v3/` (read-only).
- Run STEAK's test suite; record pass/fail (Open Question 7).
- Inspect `training/finetune.py` — is it HF Transformers wrapper or Triton? (Open Question 1)
- Inspect `precision/fp8_handler.py` Ampere branch (Open Question 2).
- Inspect `self_improve/star/loop_initiator.py` end-to-end (Open Question 3).
- Inspect `s62_auto_improve/patch_validator.py` (Open Question 4).
- Inspect `s68_promotion_gate/policy.py` (Open Question 5).
- Inspect `vllm_configurator.py` — launcher or config-only? (Open Question 6).
- Read `pyproject.toml` (Open Question 8 — already resolved: has archived `auto-gptq` and `autoawq` deps, needs replacement).

**Day 2:**
- Scaffold `dante-dojo/` repo with full tree per Section 2.2.
- `pyproject.toml` with Python 3.11+ baseline, pydantic v2, pytest, ruff, mypy.
- `CONSTITUTIONAL.md` committed — the rules Dojo obeys.
- Initial `README.md`, `CHANGELOG.md`, `.gitignore`, `.github/workflows/ci.yml`.
- Pre-commit hooks: ruff, mypy --strict, pytest fast subset.

**Day 3:**
- Transplant Tier A (core substrate): `core/evidence.py`, `core/one_door.py`, `core/types.py`, `core/exceptions.py` from STEAK with passing tests.
- CI green on day 3.

**Phase 0 gate:** all 8 open questions have written answers committed to `docs/open-questions.md`; repo scaffolded; CI green; Tier A tests pass.

### Phase 1 — HAL + Turbo + Precision (Days 4-9)

**Goal:** Hardware autodetect + quantization policy + precision handling, all with Ampere FP8 rejection correctness.

**Day 4-5:** Transplant + consolidate `steak/hal/` → `dojo/hal/`. Max 8 files, each ≤ 500 LOC.
- `detector.py`, `nvidia.py`, `apple.py`, `amd.py`, `intel.py`, `vram.py`, `profile.py`, `types.py`.
- New tests: mock-HAL-profile fixtures for all 8 hardware tiers.

**Day 6:** Transplant `steak/gpu_turbo/` → `dojo/turbo/` with corrected Ampere routing (no FP8 on SM<8.9).

**Day 7:** Transplant `steak/precision/` → `dojo/precision/` with FP8 reject on Ampere.

**Day 8:** Adversarial tests: attempt FP8 on reported-Ampere profile → must raise `PrecisionUnsupported`. Attempt unknown architecture → fail closed.

**Day 9:** Integration test: `dojo detect` on actual 4060 laptop + actual 3090 desktop → HAL profiles signed, turbo policy routes correctly.

**Phase 1 gate:** `dojo detect` ships. Ampere FP8 refused correctly. Adversarial suite passes.

### Phase 2 — Pack ingest + Training + Idle training (Days 10-18)

**Goal:** End-to-end from Harvest pack to trained LoRA adapter, with idle training.

**Day 10-11:** Build `dojo/pack/` (new subsystem).
- Schema, importer FSM, tokenizer, splits, contamination check against eval gold.
- Tests: valid pack → indexed; tampered signature → rejected; contaminated pack → rejected with reason.

**Day 12-14:** Transplant + wrap `steak/training/` → `dojo/train/`.
- `base.py` (BaseTrainer FSM), `finetune.py`, `unsloth_backend.py`, `axolotl_backend.py`, `checkpoint.py`.
- Unsloth integration: pin version, wire into Evidence chain, 4060-safe defaults.
- Axolotl integration: config generator from DojoConfig.

**Day 15-16:** Transplant `steak/s53_idle_training/` → `dojo/idle/`.
- Idle detector (Windows/Linux/macOS backends), guardian, resource guard.
- Integration tests: mouse-move pauses training within 1s, thermal pauses, resume on idle.

**Day 17:** Transplant `steak/s51_layer_streaming/` → `dojo/stream/`.
- 4060-target: Llama-3-13B Q4 at ≥ 3 tok/s.

**Day 18:** End-to-end test: tiny pack (100 samples) → Unsloth QLoRA on tiny Qwen (0.5B) → adapter saved → reload → inference works. Full evidence chain replayable.

**Phase 2 gate:** `dojo train --pack ...` ships and runs end-to-end. Idle training verified on at least 2 OSes.

### Phase 3 — Quantization + Serving + Routing (Days 19-25)

**Goal:** Trained LoRA can be quantized, served, and routed to.

**Day 19-20:** Transplant `steak/inference/quantization/` → `dojo/quant/`. Replace `auto-gptq` with `gptq-model`, `autoawq` with `llm-compressor`.
- `gguf.py` (llama.cpp target), `gptq.py`, `awq.py`, `nf4.py`, `validator.py`.
- Post-quant validator: ≤ 15% perplexity regression required.

**Day 21-22:** Transplant `steak/inference/serving/` → `dojo/serve/`.
- Launcher FSM, vllm_backend, sglang_backend, llamacpp_backend, mlx_backend.
- **Critical:** launchers must actually launch processes, not just emit configs (resolves Open Question 6).
- Health checks, graceful shutdown, auto-restart options.

**Day 23-24:** Build `dojo/route/` (new subsystem).
- Routing policy FSM, LiteLLM adapter (pinned safe version), cost tracker.
- Privacy-tier enforcement: confidential → local only.

**Day 25:** End-to-end test: trained LoRA → GGUF export → llama.cpp serve on 4060 → `dojo route --task bookkeeping --input sample.csv` → routes to local, emits cost receipt.

**Phase 3 gate:** `dojo serve` + `dojo route` ship. API-displacement tracking works.

### Phase 4 — Evaluation + Experiment tracking + Registry (Days 26-32)

**Goal:** Rigorous multi-seed eval with bootstrap CIs, MLflow tracing, model registry.

**Day 26-28:** Build `dojo/eval/` (new subsystem).
- Harness wrapping lm-evaluation-harness + LiveBench + custom Dante suite.
- `dante_suite.py`: empanada_bookkeeping_f1, b2b_email_quality, sop_extraction_faithfulness, code_review_actionability, ocr_cleanup_exact_match.
- Bootstrap 95% CI over ≥ 5 seeds.
- Judge-drift detector.

**Day 29-30:** Transplant `steak/experiment/` → `dojo/experiment/`. Integrate MLflow 3.x GenAI tracing.

**Day 31:** Transplant `steak/foundry/{model_registry,certified_matrix,contracts}` → `dojo/registry/`. OCI-compatible artifact shape.

**Day 32:** End-to-end: train → quantize → eval (5 seeds, bootstrap CI) → register → inspect via `dojo registry show`.

**Phase 4 gate:** Full train+eval pipeline produces MLflow traces and registry entries with CIs. Judge-drift detection tested.

### Phase 5 — Three-way promotion gate + Ecosystem integration (Days 33-38)

**Goal:** Close the loop with Harvest + Forge + Agents.

**Day 33-34:** Build `dojo/promote/` (new subsystem).
- Gate FSM, harvest_check, dojo_check, forge_check, passport builder, sealer.
- Transplant `steak/certification/ultimate/evidence_sealer.py`.

**Day 35:** Build `dojo/ecosystem/` (new subsystem).
- Harvest client (HTTP), Forge client (HTTP), Agents publisher (filesystem registry).
- Mock Harvest + Mock Forge servers for testing.

**Day 36-37:** End-to-end three-way gate test.
- Real Empanada bookkeeping pack → fine-tune Qwen2.5-7B → eval passes Haiku-baseline → promote → Harvest green → Dojo green → Forge green → passport issued → DanteAgents mock loads and runs.
- Failure-mode tests: revoked Harvest receipt, failing eval, Forge unreachable.

**Day 38:** Adversarial tests: tampered passport, wrong key, expired receipt, contamination in eval set.

**Phase 5 gate:** Full constitutional passport lifecycle works. Mock Agents consumer verifies offline.

### Phase 6 — CLI polish + Lane 2 proof experiment + Docs + Ship (Days 39-42)

**Day 39:** CLI polish. Every command has `--help`, examples, shell completion. `dojo ui` launches.

**Day 40-41:** Lane 2 proof experiment.
- μP-verified attention-variant sweep at 100M-500M on 3090: {MHA, GQA, MLA, sliding-window} × 3 FLOP points × 5 seeds = 60 runs.
- Estimated ~48-72h wall time on 3090 (run in background during other days; day 40-41 is analysis + writeup).
- Transfer-validate winner at 1B on RunPod H100 spot (~$30).
- Emit pre-registered report with configs, seeds, bootstrap CI.

**Day 42:** Docs finalization, changelog, v1.0.0 tag, release.

**v1.0 SHIP GATE:**
- [ ] All 8 functional acceptance criteria (§1.1) pass.
- [ ] All constitutional acceptance (§1.2) passes — adversarial suite clean.
- [ ] All integration acceptance (§1.3) passes — mock Harvest + mock Forge + mock Agents all work.
- [ ] All quality acceptance (§1.4) passes — 100% test pass, ≥70% coverage, ruff+mypy clean, files ≤500 LOC.
- [ ] All performance acceptance (§1.5) passes on 3090.
- [ ] Lane 2 proof experiment report committed to `docs/experiments/phase6_attention_sweep.md`.
- [ ] README has working quickstart reproducible by a fresh clone.

---

## 5. Dependencies

### 5.1 Python dependencies (`pyproject.toml`)

```toml
[project]
name = "dante-dojo"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.5",
    "typer>=0.12",
    "rich>=13.7",
    "httpx>=0.27",
    "cryptography>=42.0",       # Ed25519 signing
    "pynvml>=11.5",
    "psutil>=5.9",
]

[project.optional-dependencies]
train = [
    "torch>=2.4.0",
    "transformers>=4.46.0",
    "datasets>=3.0.0",
    "tokenizers>=0.20.0",
    "peft>=0.13.0",
    "accelerate>=1.0.0",
    "trl>=0.12.0",
    "bitsandbytes>=0.44.0",
    "unsloth>=2026.3.0",        # single-GPU speed champion
    "axolotl>=0.16.0",          # multi-GPU config surface
    "flash-attn>=2.6.0",        # optional, Ada/Hopper only
    "triton>=3.0.0",
]
quant = [
    "gptq-model>=6.0.0",        # replaces archived auto-gptq
    "llm-compressor>=0.10.0",   # replaces archived autoawq, vLLM-aligned
    "safetensors>=0.4.0",
    "onnx>=1.17.0",
    "onnxruntime>=1.20.0",
]
serve = [
    "vllm>=0.13.0",
    "sglang>=0.5.0",            # optional
    "llama-cpp-python>=0.3.0",  # optional, for embedded llama.cpp
]
route = [
    "litellm>=1.55.0",          # pin != 1.82.8 (compromised)
]
eval = [
    "lm-eval>=0.4.5",
    "livebench @ git+https://github.com/LiveBench/LiveBench@main",
]
experiment = [
    "mlflow>=3.0.0",            # 3.x GenAI features
]
apple = [
    "mlx>=0.20.0",
    "mlx-lm>=0.21.0",
]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.14",
    "pytest-asyncio>=0.25",
    "ruff>=0.8",
    "mypy>=1.13",
    "pre-commit>=4.0",
]
```

### 5.2 System dependencies

- NVIDIA: CUDA 12.4+, cuDNN 9.x, nvidia-driver 550+.
- AMD: ROCm 6.2+ (MI300X supported April 2026).
- Apple: macOS 14+, Xcode CLT.
- llama.cpp: built from source OR `llama-cpp-python` wheel.

### 5.3 External service dependencies

- DanteHarvest HTTP endpoint (defaults to `http://127.0.0.1:8001`).
- DanteForge HTTP endpoint (defaults to `http://127.0.0.1:8002`).
- Optional: HuggingFace Hub (for base model downloads).
- Optional: API providers for `dojo route` external fallback (Anthropic, OpenAI, DeepSeek, etc.).

---

## 6. Non-functional requirements

### 6.1 Security

- All signing keys stored in `~/.dante/keys/` with 0600 permissions.
- Evidence chain storage encrypted at rest (`cryptography.fernet`).
- API keys for external providers read from environment (`DANTE_OPENAI_API_KEY`, etc.) — never logged.
- Pack imports verify Ed25519 signatures; tampered packs rejected with evidence.
- `dojo route --privacy confidential` NEVER allows external API routing.

### 6.2 Observability

- Every operation emits evidence events (Merkle-chained, SHA-256-hashed).
- MLflow 3.x tracing active for all training + eval runs.
- `dojo ui` serves MLflow + Dante dashboard on `http://127.0.0.1:5000`.
- Prometheus metrics endpoint at `:9090/metrics` (optional).
- Structured JSON logs via `structlog`.

### 6.3 Replayability

- Every training run is replayable from its evidence receipt.
- `dojo replay <receipt-path>` reproduces the run and emits a diff report vs original.
- Seeds, data splits, tokenization are all deterministic given DojoConfig.

### 6.4 Portability

- Works on: Windows 11 (WSL2 recommended), Ubuntu 22.04+, macOS 14+ (Apple Silicon only for full functionality).
- Containerized option: Dockerfile + docker-compose.yml for reproducible dev environment.

### 6.5 Documentation

Required docs at ship:
- `README.md` — quickstart, 5-minute first fine-tune walkthrough.
- `docs/architecture.md` — the Section 2 of this PRD, fleshed out.
- `docs/constitutional-guide.md` — how the constitutional substrate works.
- `docs/hardware-tiers.md` — capability matrix per tier.
- `docs/promotion-gate.md` — how three-way promotion works.
- `docs/open-questions.md` — the answers to the 8 audit questions.
- `docs/api/` — auto-generated API reference (mkdocs + mkdocstrings).

---

## 7. Risks and mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Open Questions 3-5 resolve unfavorably (STaR/patch_validator/promotion_policy are skeletons) | Medium | High | Phase 0 Day 1 inspection surfaces this; if real, cut Lane 2 scope and build minimal three-way gate from scratch |
| STEAK's pytest_errs.txt reveals systemic test decay in core substrate | Low | High | Phase 0 surfaces; if severe, rebuild `dojo/core/` from scratch following STEAK's patterns but writing fresh tests |
| 4060 laptop 8GB can't actually fit Qwen2.5-7B QLoRA under Unsloth | Low | Medium | Phase 2 Day 18 tests this; fallback is Qwen2.5-3B QLoRA + Qwen2.5-14B layer-streamed inference |
| NVLink bridge for 2×3090 unobtainable | Medium | Low | Plan as if unavailable; PCIe-only is fine for 70B Q4 serving at ~15-25 tok/s |
| vLLM / SGLang / Unsloth API churn | High | Medium | Version pins in pyproject; backend adapter abstraction isolates change |
| LiteLLM v1.82.8-style compromise recurs | Medium | High | Pin exact version, verify checksums, monitor CVE feed |
| Idle training causes data loss on sudden foreground resume | Low | Medium | Checkpoint every 10min; gradient_accumulator is idempotent |
| DanteHarvest or DanteForge not ready for Phase 5 integration | High | Medium | Mock servers in `tests/fixtures/` enable parallel dev; real integration is additive |
| Lane 2 proof experiment takes > 4 weeks | Medium | Low | Cut to 2 compute points instead of 3 if running over; v1 ships without full experiment if necessary |
| Overall timeline slip past 6 weeks | Medium | Medium | Hard cut points at each phase gate; v1 ships with Phase 5 minimum, Phase 6 can slip to v1.1 |

---

## 8. Post-v1 roadmap preview (NOT in scope)

- **v1.1 (weeks 7-10):** Full Lane 2 factorial orchestrator. IsoFLOP sweeps, seed budgeting, pre-registration registry, cross-scale transfer validation, multi-variable factorial DOE. Integrate `steak/self_improve/` for real self-rewarding DPO on verifiable-reward domains.
- **v1.2 (weeks 11-14):** Self-improvement loops. rStar-Math-style MCTS + PPM for verifiable-reward domains. Integrate `s58_agent_loop/` Darwin-Gödel scaffolding on frozen local 7-13B.
- **v1.3 (weeks 15-18):** Managed-fine-tuning escape hatch. Together AI + Modal Labs + Fireworks connectors. Transparent routing between local and managed based on time-to-fine-tune budget.
- **v2 (month 5+):** Multi-node distributed training. DanteStudio handoff for diffusion/multimodal. DanteBody handoff for robotics policies.

---

## 9. Execution instructions for Claude Code / Codex

1. **Read this PRD end-to-end before writing any code.** Every component spec is load-bearing.
2. **Work phase by phase, gate by gate.** Do not start Phase 2 until Phase 1 gate is green.
3. **KiloCode discipline:** every file ≤ 500 LOC, no stubs, no TODO markers, no NotImplementedError in shipped code. Tests + docstrings per the STEAK pattern.
4. **Constitutional checklist per module:**
   - [ ] One public entry point (`@one_door` decorator)
   - [ ] Evidence emission on all operations
   - [ ] Fail-closed error handling with specific error types
   - [ ] Deterministic behavior given seed
   - [ ] Explicit FSM where state is non-trivial
   - [ ] Tests: unit + constitutional + adversarial
5. **Transplants:** when transplanting from STEAK V3, preserve the evidence event names and state machine names. Rewrite only what's needed for Dojo's boundary (no Foundry monolith, no EU-AI-Act compliance, no diffusion).
6. **Before every commit:** `ruff check`, `mypy --strict`, `pytest tests/unit/<touched-module>/`. Pre-commit hook enforces.
7. **Before every phase gate:** full `pytest`, adversarial suite, integration tests. Phase-gate tests listed in each phase above.
8. **Evidence artifacts:** at every phase gate, commit the evidence chain export (`phase_N_evidence.jsonl`) under `docs/evidence/`. This is the Dogfood: Dojo uses Dojo's own evidence substrate to prove its own construction.
9. **Human-in-loop points:** Ricky reviews and signs off at each phase gate. Phase-gate review is ≤ 30 minutes.
10. **When you encounter an open question:** consult `docs/open-questions.md`. If it's not answered there, STOP, flag it in a commit message, and defer to Ricky.

---

## 10. Appendix — mapping this PRD to research dossier findings

| Dossier finding | PRD component addressing it |
|---|---|
| STEAK ships ~76K transplantable lines | Phases 1-5 transplant schedules |
| Ampere has no native FP8 | `dojo/hal/nvidia.py` + `dojo/precision/fp8.py` reject FP8 on SM<8.9 |
| AutoGPTQ archived April 2025 | `dojo/quant/gptq.py` uses `gptq-model` |
| AutoAWQ archived May 2025 | `dojo/quant/awq.py` uses `llm-compressor` |
| LiteLLM v1.82.8 compromised | `pyproject.toml` pins safe version |
| TGI archived March 2026 | Not included in supported backends |
| torchtune stagnating | Not included in supported backends |
| Unsloth + axolotl + TRL are the live triad | Phase 2 + Phase 5 |
| MLflow 3.x pivoted to GenAI tracing | Phase 4 integration |
| 3090 + 4060 can reach induction-head / grokking thresholds but not emergence | Lane 2 scope scoped accordingly |
| Verifiable rewards work at 1.5-7B (rStar-Math); judge LLMs need 30-70B | Lane 2 self-improvement deferred to v1.2 on verifiable-reward domains only |
| DanteHarvest parallel build | `dojo/pack/` + `dojo/ecosystem/harvest_client.py` with mock server for parallel dev |
| 4060 laptop is minimum target | Every backend selection checks HALProfile and picks 4060-safe defaults |
| Confidence 76/100 on research dossier | 8 open questions resolved in Phase 0 Day 1 raise this to ship confidence |

---

**End of PRD v1.0. Hand to Claude Code + Codex. Go build.**

---

## 11. Context Economy Addendum

This addendum is required by DanteForge Article XIV: Context Economy. Dojo is Python-native, but it must still emit context-economy telemetry compatible with the shared Dante ledger.

| Surface | Expected context footprint | Expected compression ratio | Sacred content types |
|---|---:|---:|---|
| Training run summaries | <= 10 KB per orchestration turn | 40%+ on routine logs; raw passthrough on failures | OOM errors, CUDA errors, data-loader failures, checkpoint failures, eval regressions |
| Eval and promotion receipts | <= 8 KB per prompt unless full audit is requested | 40%+ on successful eval receipts | promotion gate failures, score regressions, policy denials, tamper evidence |
| Hardware/HAL reports | <= 4 KB per prompt | 50%+ on repeated hardware probes | unsupported precision warnings, thermal/VRAM guard failures |
| Harvest pack import summaries | <= 6 KB per prompt | 40%+ on manifest/provenance summaries | rights violations, missing signatures, provenance gaps |

Implementation requirements:

- Dojo must add a Python context-economy adapter that writes the same JSONL fields as PRD-26.
- Evidence writes under `docs/evidence/` and future Dojo evidence dirs must use per-artifact compression rules.
- Sacred content must be preserved verbatim or passed through raw when uncertain.
- Phase gates must report observed token footprint, compression ratio, and sacred bypass count.
- Dojo is production-ready for Context Economy only at 7.0+ on the shared five-submetric rubric.
