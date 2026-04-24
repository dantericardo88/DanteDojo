# DanteDojo Architecture

## Overview

DanteDojo is a sovereign AI training, evaluation, specialization, and inference organ designed to run on solo-founder hardware (RTX 4060 laptop, RTX 3090 desktop, optional H100 cloud). It enforces a constitutional architecture: every operation is observable, every entry point is one-door, and every promotion requires three-way agreement.

## Constitutional Substrate (`dojo/core/`)

Four invariants hold unconditionally across all modules:

### 1. One-Door Doctrine
Every subsystem exposes exactly one `@one_door(name)` decorated entry point. All external callers must go through it. The decorator:
- Emits `ONE_DOOR_ENTRY` evidence on every call
- Raises `OneDoorViolation` if `DOJO_BYPASS` env var is set
- Sets `__one_door__` and `__organ__` attributes on the wrapper

### 2. Merkle-Chained Evidence
`EvidenceLogger` (singleton) chains every emitted event into a SHA-256 Merkle tree. Each `EvidenceBundle` contains:
- `seq`: monotonically increasing sequence number
- `hash`: SHA-256 of the normalized evidence payload
- `prevHash`: hash of the preceding bundle
- `at`: ISO 8601 timestamp

The `EvidenceChain` returned by `logger.chain()` provides `verify_chain()` which checks both hashes and the Merkle root.

### 3. FSM-Based State Machines
All multi-step processes use `BaseFSM[S]`. Subclasses declare:
- `initial_state: S`
- `_TRANSITIONS: dict[S, frozenset[S]]`

Calling `transition(target)` on an illegal edge raises `FSMTransitionError`.

### 4. Fail-Closed Error Handling
All public functions either succeed completely or raise a typed `DojoError` subclass. There are no silent failures. `EvidenceLogger.emit()` raises `TypeError` on non-dict input.

## Module Map

```
dojo/
├── core/           Constitutional substrate
│   ├── evidence.py     MerkleTree, EvidenceBundle, EvidenceLogger, EvidenceType
│   ├── exceptions.py   DojoError hierarchy
│   ├── fsm.py          BaseFSM, fsm_guard
│   ├── one_door.py     @one_door, ok(), fail()
│   └── types.py        HALProfile, DojoConfig, ConstitutionalPassport, ...
│
├── hal/            Hardware abstraction layer
│   ├── detector.py     @one_door detect() — NVIDIA→Apple→AMD→Intel→CPU
│   ├── nvidia.py       pynvml probe + profile builder
│   ├── apple.py        platform.machine probe
│   ├── amd.py          rocm-smi probe
│   ├── intel.py        lspci/wmic probe
│   └── profile.py      sign_profile() using SHA-256
│
├── precision/      Precision tier enforcement
│   └── fp8.py          FP8Handler — raises PrecisionUnsupported if CC < 8.9
│
├── turbo/          GPU turbo policy dispatch
│   └── policy.py       ARCH_POLICY_MAP (Ampere → no FP8), get_policies_for_profile()
│
├── pack/           Specialization pack ingestion
│   ├── importer.py     @one_door import_pack() — 8-state ImportState FSM
│   ├── schema.py       validate_schema() — checks 12 required fields
│   ├── splits.py       train/eval/test split logic
│   └── tokenize.py     Tokenizer wrapper
│
├── train/          Fine-tuning pipeline
│   ├── finetune.py     @one_door finetune() — routes Unsloth/Axolotl/MLX/Native
│   ├── base.py         TrainingState FSM, FinetuneConfig, FinetuneResult
│   ├── unsloth_backend.py
│   ├── axolotl_backend.py
│   └── checkpoint.py
│
├── idle/           Idle-window accumulation
│   ├── accumulator.py  GradientAccumulator — crash-recoverable JSON state
│   ├── detector.py     CPU/GPU utilization probe
│   ├── guardian.py     ThermalGuard — throttles on GPU temp
│   └── resource.py     ResourceMonitor
│
├── quant/          Post-training quantization
│   ├── validator.py    QuantValidator — 15% PPL delta / 5pt F1 delta gates
│   ├── gptq.py         GPTQ via gptq-model
│   ├── awq.py          AWQ via llm-compressor
│   ├── gguf.py         GGUF via llama.cpp
│   └── nf4.py          NF4 via bitsandbytes
│
├── serve/          Inference server lifecycle
│   ├── launcher.py     @one_door launch_server() — auto-selects backend by tier
│   ├── vllm_backend.py
│   ├── llamacpp_backend.py
│   └── manifest.py     DeploymentManifest builder
│
├── route/          Intelligent routing
│   ├── router.py       @one_door dispatch() — privacy-tier enforcement
│   ├── policy.py       RoutingPolicy loader
│   └── cost_tracker.py CostReceipt emitter
│
├── eval/           Evaluation harness
│   ├── harness.py      @one_door run_eval() — multi-seed, bootstrap CI
│   ├── statistics.py   bootstrap_ci(), seeds_are_stable()
│   ├── dante_suite.py  DanteEvalTask registry
│   ├── lm_eval_adapter.py  lm-eval stub with synthetic fallback
│   └── judge_drift.py  JudgeDriftDetector — 80% agreement threshold
│
├── experiment/     Experiment tracking
│   ├── tracker.py      ExperimentTracker — MLflow 3.x / in-memory fallback
│   └── metrics.py      aggregate_metrics(), compare_runs()
│
├── registry/       Model artifact registry
│   ├── registry.py     @one_door register(), lookup()
│   ├── store.py        LocalRegistryStore — JSONL-backed at ~/.dante/registry/
│   └── fingerprint.py  SHA-256 fingerprinting
│
├── promote/        Three-way promotion gate
│   ├── gate.py         @one_door promote() — HARVEST→DOJO→FORGE→SEAL FSM
│   ├── harvest_check.py  Revocation + license check
│   ├── dojo_check.py   F1 delta ≥ 5pt gate
│   ├── forge_check.py  HTTP POST to DanteForge (3-retry, fail-closed)
│   ├── passport.py     build_passport() — 90-day ConstitutionalPassport
│   └── sealer.py       Merkle seal over all three receipts
│
├── ecosystem/      Cross-organ contracts
│   ├── harvest_client.py   httpx client for DanteHarvest
│   ├── forge_client.py     httpx client for DanteForge
│   ├── agents_publisher.py Writes manifests to ~/.dante/registry/models/
│   └── evidence_bridge.py  cross_link_evidence() between organ chains
│
├── stream/         Layer streaming (transplanted from STEAKV3 s51)
│   ├── weight_streamer.py   Single-layer VRAM residency, accelerate device_map
│   ├── kv_cache_streamer.py Sovereign cache size limit (1M tokens ceiling)
│   ├── prefetch_governor.py Lookahead window (max 16 layers)
│   ├── stream_balancer.py   Compute/memory overlap (max 8 CUDA streams)
│   ├── meta_device_manager.py  meta-device lazy load, full-model alloc forbidden
│   ├── block_quantizer.py   On-the-fly block quant with precision floor
│   └── evidence_sovereign.py SHA-256 operation seal/halt
│
└── cli/            Typer-based CLI
    └── main.py     dojo detect|pack|train|serve|eval|promote|registry|...
```

## Data Flow: Training a Specialist

```
User: dojo train qwen2.5-7b --pack harvest://empanada-bookkeeping/v3

1. dojo detect            → HALProfile (tier=4060_laptop, CC=8.9, FP8=True)
2. pack import            → ImportState FSM (8 stages) → indexed pack
3. finetune               → backend=UNSLOTH (4060 tier)
                          → FinetuneResult(training_receipt=<merkle>)
4. quant validate         → QuantValidator (PPL delta < 15%)
5. eval run               → EvalReceipt(mean_scores, bootstrap CI, merkle_hash)
6. promote                → HARVEST_CHECK → DOJO_CHECK → FORGE_CHECK
                          → ConstitutionalPassport(expires_at=+90d)
7. registry register      → artifact saved to ~/.dante/registry/
8. serve launch           → LLAMACPP backend (4060 tier) → ready on :8000
```

## Hardware Routing

| Tier          | Training  | Inference |
|---------------|-----------|-----------|
| 4060 Laptop   | Unsloth   | llama.cpp |
| 3090          | Unsloth   | vLLM      |
| 3090 Dual     | Axolotl   | vLLM TP=2 |
| H100          | Axolotl   | vLLM      |
| Apple M4      | MLX       | MLX-LM    |
| CPU           | (blocked) | llama.cpp |

## Privacy Tier Enforcement

`PrivacyTier.CONFIDENTIAL` and `PrivacyTier.SECRETS` tasks **never** route to any external API. `router.dispatch()` raises `PrivacyTierViolation` and emits `ROUTE_PRIVACY_BLOCKED` evidence before any network call is attempted.
