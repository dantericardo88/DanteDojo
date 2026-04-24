# DANTEDOJO Research Dossier — Decision-Grade Findings

## Section 1 — STEAK V3 donor audit (grounded, file-level)

STEAK V3 is a **687,669-line codebase**: 316,344 lines of source across 92 subsystems in `steak/`, 119,539 lines of tests across 356 test files, 251,786 lines of docs including 191 PlanningDocs. **Test-to-source ratio ~1:2.6.** The first-pass dossier treated STEAK as a module list and concluded most Dojo subsystems would need to be built from scratch. Direct code traversal inverts that conclusion: **STEAK is much more mature than inventory suggests**, and several subsystems previously labeled "must build" are substantially shipped with constitutional evidence emission.

### What's actually in the box

**Inference stack (~11,182 lines).** `steak/inference/serving/` contains `vllm_configurator.py` (614), `sglang_configurator.py` (612), `tensorrt_configurator.py` (612), and `deployment_manifest_builder.py` (722). Quantization exporters ship for GGUF (556), EXL2 (532), GPTQ (524), AWQ (520). Edge packagers exist for llama.cpp (468), MLC-LLM (448), and ExecuTorch (442). These are **"constitutional configurators"** — they produce configs and emit evidence spans, not actual servers.

**Training stack (~9,354 lines).** `steak/training/finetune.py` is 1,807 lines with LORA/QLORA/FULL_FT modes and fail-closed dataset validation. `pretrain.py` (1,024), `base.py` (1,384), `context_length.py` (886), `rope_scaling.py` (952).

**Hardware abstraction (~8,593 lines).** NVIDIA `gpu_detector.py` (394), `apple_mlx_profiler.py` (758), `amd_rocm_profiler.py` (662), `fp8_handler_v3.py` (536), `profile_assembler.py` (986), `vram_auditor.py` (831), `interconnect_prober.py` (776), `multimodal_compute_profiler.py` (896).

**Self-improvement family (~18,000 lines).** S29 STaR (`loop_initiator.py` 1,246; retrain_dispatcher 480; correctness_filter 404; convergence_monitor 416; reasoning_generator 296) with constitutional `LoopState` enum. S30 Self-Rewarding (`dpo_trainer` 860; `constitutional_judge` 490 with JudgementReason enum for HELPFULNESS/HARMLESSNESS/HONESTY/ACCURACY/CLARITY/COMPLETENESS/CONSTITUTIONAL_ALIGNMENT; `preference_generator` 498; `reward_stability_monitor` 498; `guardrailer` 438; `query_translator` 486; `citation_assembler` 398). S31 Breakthrough Harvesting (`arxiv_monitor` 685; `paper_analyzer` 1,040; `integration_recommender` 664; `safety_reviewer` 664; `metric_computer` 422). S32 Gradient Surgery (`cagrad_resolver` 514; `pcgrad_resolver` 424; `pareto_tracker` 460; `conflict_detector` 446; `ttl_manager` 404). S58 Agent Loop — Darwin–Gödel-style scaffolding evolution (`loop_controller` 328; `self_rewrite_proposer` 426; `convergence_checker` 260). S59 Self-Tooling (`tool_synthesizer` 330; `tool_validator` 398; `tool_registry` 314). S62 Auto-Improve — code-level self-modification (`patch_proposer` 326; `patch_validator` 408; `improvement_cycle_manager` 262).

**Core substrate.** `steak/core/evidence.py` (1,958 lines, thread-safe, Merkle-root support, `@contextmanager` spans), `steak/core/one_door.py` (1,609), `steak/core/types.py` (1,179), `steak/checkpoints/manager.py` (921).

**Promotion machinery.** `s68_promotion_gate/` (682 lines with `PromotionPolicy` and `PromotionGate` classes), `certification/ultimate/` (2,000+ lines with `replay_runner` 450, `evidence_sealer` 380, `cert_checker` 330, `green_emitter` 338, `metric_selector` 426), `validation/gate_all.py` (3,224).

**GPU turbo (~1,372 lines policy + 1,086 lines tests).** `steak/gpu_turbo/policy.py` includes a `TurboState` FSM enum and `ARCH_POLICY_MAP` covering NVIDIA Blackwell→FP4_QUANT+HET_COMM, Ada→FP8+HET_COMM, Ampere→FP8, AMD MI350→FP4+HET_COMM, MI300→FP8+HET_COMM, Apple M4 Max→INT4.

**Idle training (1,728 lines across 9 files).** `idle_detector` 324, `scheduler_guardian` 276, `resource_guard` 328, `gradient_accumulator` 252, `compatibility_bridge` 274, plus three evidence modules. **Genuine differentiator** — almost nothing in OSS does this well.

**Foundry monolith.** `foundry/service.py` is 7,226 lines, `workflow.py` 3,936, `ui.py` 3,746 — keep as donor reference only. But `model_registry.py` (222, already seeded with `_CANONICAL_4060_TRAINING_RECIPES`), `certified_matrix.py` (207), and `contracts.py` (302) are transplant gems. `video_assembler.py` (512) and `brand_conditioner.py` (470) belong in DanteStudio.

**Layer streaming (3,072 lines).** `s51_layer_streaming/` is critical for 4060-laptop inference of models larger than VRAM.

**EU AI Act compliance (2,906 lines).** `compliance/eu_ai_act/` has `provider_info_collector` (764), `training_summary_builder` (728), `annex_xi_generator` (724), `copyright_policy_enforcer` (690). **Belongs in DanteForge, not Dojo.**

### Monolith-disease risk

92 subsystems is a lot. The S51–S85 tower is ~40% speculative research tracks from PlanningDocs that may not be fully implemented. **`pytest_errs.txt` is committed at repo root**, suggesting known failing tests. `s65_research_harvester/` overlaps with DanteHarvest — treat as pattern donor only.

### Transplant ledger

| Tier | Scope | Lines | Contents |
|---|---|---|---|
| **A** | Constitutional substrate | ~8,000 | `core/evidence`, `core/one_door`, `core/types`, `checkpoints/manager`, `evidence/` |
| **B** | Lane 1 core | ~42,000 | `gpu_turbo`, `hal`, `precision`, `training/{finetune,base,context_length,rope_scaling,config_receiver}`, `inference/{serving,quantization,edge}`, `s51_layer_streaming`, `s53_idle_training`, `foundry/{model_registry,certified_matrix,contracts}`, `experiment/`, `cli/{bench,router,main}`, `s68_promotion_gate`, `certification/ultimate`, `validation/gate_all` |
| **C** | Lane 2 | ~26,000 | `training/pretrain`, `self_improve/`, `s58_agent_loop`, `s59_self_tooling`, `s62_auto_improve`, `s56_test_time_scaling`, `s64_synthetic_data`, `s66_scientist_lab`, `s70_coscale_rl`, `composition/`, `compression/` |
| **D** | Donor reference only | ~62,000 | Foundry monolith, speculative S51–S85 tracks, overlap with other organs |
| **E** | Retire | ~1,900 | Dead code, superseded drafts |

**Total: ~76,000 lines transplant, ~62,000 reference, ~1,900 retire, ~176,000 out of scope for v1.**

### Eight questions only direct code inspection can resolve

1. Does `training/finetune.py` wrap HF Transformers directly or have Triton kernels? (Determines whether Unsloth should replace it.)
2. Does `precision/fp8_handler.py` on Ampere emulate (slower than BF16) or reject at runtime?
3. Does STaR `loop_initiator.py` run end-to-end or is it skeleton-only?
4. Is `patch_validator.py` a real code-modification safety gate or a stub?
5. Does `s68_promotion_gate/policy.py` connect to actual metrics or is it threshold-only?
6. Does `vllm_configurator.py` launch vLLM or only generate configs?
7. What's the current test pass rate (given `pytest_errs.txt` exists)?
8. What's in `pyproject.toml`?

**Net revision vs prior dossier:** Dojo is not a 3-month greenfield build. It's a **6-week integration pass** on an existing codebase where the hard parts (evidence substrate, promotion gate, HAL, idle training, layer streaming, self-improvement scaffolding) already exist as pluggable modules.

---

## Section 2 — Hardware reality check

### RTX 4060 laptop (8GB, Ada SM 8.9, ~272 GB/s, native FP8)

Unsloth's published minimum-VRAM table (unsloth.ai/docs) is the defensible reference. Assume 7.0–7.3 GB usable after OS/driver overhead.

| Scenario | What fits | Memory math |
|---|---|---|
| **QLoRA + Unsloth (paged AdamW 8-bit)** | **Qwen2.5-7B comfortably** (~5 GB); **9B comfortable**; 14B marginal | 7B = 5 GB NF4 base + 0.3 GB LoRA (r=16) + 0.5 GB paged optimizer + 1.0–1.5 GB activations at bs=2/seq=2048 |
| **Full FT (BF16 AdamW)** | **~0.5–1B** | 12 bytes/param ceiling = ~650M before activations |
| **Pretrain from scratch** | **100M–500M comfortably; 1B only with bnb 8-bit optim + grad checkpoint + bs=1** | Lane-2 micro-experiment range |
| **Inference 4-bit** | **~13B Q4_K_M (~7.5 GB)** with seq≤4k | 7B Q4 ≈ 4.1 GB; Llama-3-8B BF16 KV at 8k ≈ 1 GB |
| **Layer streaming** | **30–34B Q4 viable** at 3–5 tok/s with `--n-gpu-layers` partial | |

**Qwen2.5-7B QLoRA on 4060: YES but tight.** `per_device_train_batch_size=1–2`, `gradient_accumulation_steps=4–8`, `max_seq_length=2048` (up to 4096 with `use_gradient_checkpointing="unsloth"`), `paged_adamw_8bit`. Community reports 3080 10GB runs 7B QLoRA; 4060 8GB works with lower batch.

**OpenMythos feasibility:** OpenMythos is real (github.com/kyegomez/OpenMythos, released April 19, 2026) — an independent PyTorch reconstruction of a speculated "Claude Mythos" architecture (Recurrent-Depth Transformer + MoE + MLA/GQA). `mythos_1b` (dim=1024, 8 heads, 4 experts, 4 loop iters) **cannot be pretrained from scratch on 4060** in BF16 (~12 GB for params+grads+Adam alone) — only smoke tests with 8-bit optim + checkpointing + bs=1 are tractable, not a Chinchilla-scale 30B-token run. `mythos_3b` **requires cloud** (≥1× A100 40GB or 2× 3090). No public pretrained checkpoint ships — repo is code-only. **Flag:** Kye Gomez has a history of speculative reconstructions; the cited "Parcae paper" is arXiv:2604.12946 (SandyResearch/UCSD, April 14, 2026) — preprint only, not peer-reviewed. Treat OpenMythos as a Lane-2 reference architecture, not a reproducible baseline.

### RTX 3090 (24GB, Ampere SM 8.6, no native FP8, 936 GB/s)

**FP8 on Ampere = hard reject.** NVIDIA Transformer Engine throws `AssertionError: Device compute capability 8.9 or higher required for FP8 execution` on SM 8.6. vLLM/Triton fails with `type fp8e4nv not supported in this architecture`. Qwen FP8 w8a8 is explicitly Hopper/Ada-only per Qwen org. **Workaround:** vLLM Marlin kernel supports weight-only FP8 (W8A16), dequantizing to FP16 at load — same or slightly slower than native BF16. torchao float8 has the same rejection. **Conclusion: don't request FP8 on 3090; `hal/fp8_handler_v3.py` must route to BF16 or W8A16-Marlin.**

Scenarios: QLoRA up to **32B** (tight at 22 GB); LoRA 16-bit up to **9B**; full FT BF16 up to **~1.5–2B**; Q4 inference up to **34B** (bs=1).

### 2× RTX 3090 (48GB aggregate)

**Unlocked:** 13B full FT with ZeRO-2/3 sharding; 27B QLoRA with headroom; **70B Q4_K_M inference** (~40 GB + KV) via vLLM TP=2. Llama-3-405B is still infeasible (Q4 ≈ 230 GB).

**NVLink bridge sourcing April 2026:** Discontinued by NVIDIA; aftermarket only. Bridges appear on eBay at ~$230–$235 (item 336219247033 confirmed). Supply is sporadic. Most 3090s are 3+ slot, so a 3-slot or 4-slot bridge is required; Gigabyte 3090 TURBO blower (2-slot compatible) is discontinued. NVLink provides ~112 GB/s bidirectional vs PCIe 4.0 x16 = 32 GB/s — helps prefill/TTFT by 10–20%, minimal decode gain. **Plan as if NVLink is unavailable.** PCIe-only TP=2 for 70B Q4 via vLLM reaches ~15–25 tok/s single-stream.

### H100 80GB cloud tier (April 2026)

| Provider | On-demand | Spot / Community |
|---|---|---|
| RunPod | $2.69 SXM / $1.99 PCIe | **$1.25 Community spot** |
| Lambda | $3.29–$4.29 | — |
| Vast.ai | $1.49–$1.87 marketplace | variable |
| AWS p5 | $4.24 (post-June-2025 44% cut) | Capacity Blocks |

**Lane-2 verified-finding cost:** 3 IsoFLOP points × 5 seeds × 24 h = **360 GPU-hours**. At RunPod spot $1.25 = **$450/finding**; on-demand $2.69 = **$968/finding**; Vast = **~$536**; AWS = **~$1,526**. Add 10–20% for spot preemption retries and egress (~$0.05/GB). **Decision rule: Lane-2 H100 budget = $1,000–$2,000/month allows 2–4 verified findings.**

### Edge

**Apple Silicon / MLX (April 2026):** M5 chip neural accelerators give **3.5–4× TTFT speedup** vs M4 on LLMs (Apple ML Research Nov 2025). MLX 0.31.0 (Feb 2026) added a CUDA backend. `mlx-lm` supports LoRA/DoRA/QLoRA/full FT. 4,255+ pre-converted models on mlx-community. **Training maturity: LoRA/QLoRA production-ready; full pretraining still behind CUDA** — no FlashAttention-3, no multi-Mac distributed. M4 Max 128GB → ~96GB usable → Llama-70B at 10–15 tok/s. M5 Max bandwidth ~546 GB/s is 2–3× slower than RTX 4090 at equal param count if model fits NVIDIA VRAM. **Unsloth MLX support: "in the works," not released April 2026.**

**AMD MI300X / ROCm 7.0 (April 2026):** **Production-ready for inference; training gaps remain.** vLLM is first-class on ROCm since Jan 2026 (93% CI passing, up from 37% in Nov 2025). Pre-built `vllm/vllm-openai-rocm` Docker image. FP8/FP4 kernels (AITER) native on MI300X/MI350X. **bitsandbytes:** ROCm fork (`ROCm/bitsandbytes`, branch `rocm6.2_internal_testing`) must be compiled from source — no official wheels. gptq_marlin unsupported on ROCm. **Training gaps:** GRPO + vLLM + DeepSpeed on MI300X has open hang issues (huggingface/open-r1 #482). Axolotl/Unsloth are CUDA-first; Unsloth AMD is "coming soon." Meta runs 100% of Llama-3.1-405B production inference on MI300X — inference legit, training not yet.

**Intel Arc / IPEX-LLM:** Arc B580 (12GB, $249) runs 8B at 15–25 tok/s via IPEX-LLM — ~70–75% of 4060 at 50% price. Viable for budget inference, not production training. Arc A770/B580 can run DeepSeek-R1 671B and Qwen3-MoE-235B via FlashMoE (May 2025).

### Honest API-vs-local cost for Dante workloads (April 2026)

Pricing per 1M tokens (verified on claude.com/pricing, openai.com/api/pricing, api-docs.deepseek.com, ai.google.dev): **Opus 4.7 $5/$25**, **Sonnet 4.6 $3/$15**, **Haiku 4.5 $1/$5**, **GPT-5 $1.25/$10**, **GPT-5 mini $0.25/$2**, **GPT-5 nano $0.05/$0.40**, **DeepSeek V3.2 $0.28/$0.42**, **DeepSeek-reasoner $0.55/$2.19**, **Gemini 3 Pro preview $2/$12**, **Gemini 2.5 Flash $0.30/$2.50**, **Grok 4 Fast $0.20/$0.50**.

Monthly API cost for bundled Dante workloads (12.9M input + 2.66M output):

| Workload | Volume | Haiku 4.5 | Sonnet 4.6 |
|---|---|---|---|
| Bookkeeping categorization | 10K × 500/100 | $10.00 | $30.00 |
| B2B email drafting | 200 × 2000/500 | $0.90 | $2.70 |
| SOP extraction | 20 × 50K/3K | $1.30 | $3.90 |
| Code review | 500 × 5K/1K | $5.00 | $15.00 |
| OCR cleanup | 2K × 2K/500 | $9.00 | $27.00 |
| **Total** | | **$26.20** | **$78.60** |

Bundled cost on **GPT-5 nano = $1.71**; **Grok 4 Fast = $3.91**; **DeepSeek V3.2 = $4.73**; **Gemini 2.5 Flash = $10.52**. **Opus 4.7 = $131** (plus ~0–35% tokenizer overhead per claude.com docs).

**Local breakeven:** 4060 laptop + 3090, 10h/day × 250W × $0.12/kWh = $9/month electricity. Amortized GPU over 3 years: 4060 laptop (already owned) = $0; 3090 used ~$900 → $25/month. **Total local cost ≈ $35/month for effectively unlimited throughput** (bounded by GPU tok/s, not API budget).

**The honest verdict: API wins on cost for Haiku-tier workloads at Dante's volume.** Local only beats API when: (a) latency matters (on-device), (b) data sovereignty is non-negotiable (tax, health, contracts), (c) volume exceeds ~10× Dante's current, or (d) fine-tuning produces a small specialist that API generalists can't match. **The strategic case for Dojo is NOT "save money on API calls." It's "own the specialist models that API generalists can't replicate."**

---

## Section 3 — External OSS / research landscape (April 2026)

### Inference servers

| Project | License | Last commit | Maturity | Harvest | Avoid |
|---|---|---|---|---|---|
| **vLLM** | Apache-2.0 | Apr 2026 daily | **5** | Industry-default; PagedAttention, FP8/NVFP4, 200+ archs, P/D disagg | Churning internal APIs; heavy Blackwell bias |
| **SGLang** | Apache-2.0 | Apr 2026 daily | **5** | De-facto RL rollout backend (verl, slime); RadixAttention prefix cache; TP/EP/PP, native GB200 | Apple/MLX backend still Q1 2026 roadmap; velocity = churn |
| **llama.cpp** (now `ggml-org/llama.cpp`) | MIT | Apr 2026 daily | **5** | GGUF format, broad backends (CUDA/Metal/Vulkan/SYCL/ROCm), Merkle-friendly rolling builds | No stable releases; ggml ABI drift — pin commits |
| **TGI** | Apache-2.0 | **Archived Mar 21 2026** | **2** | Rust router patterns worth mining | HF redirects to vLLM/SGLang — do not adopt |
| **exllamav2** | MIT | Dec 2025; **superseded by EXL3** (Feb 2026) | **3** | Best-in-class EXL2 mixed-bit | Winding down; EXL3 semi-dormant since Feb 2026 |
| **TabbyAPI** | **AGPL-3.0** | Apr 1 2026 | **3** | Clean FastAPI OpenAI-compat wrapper | AGPL is viral — **avoid for commercial product** |
| **Ollama** | MIT (CLI) / proprietary (desktop) | Apr 2026 daily (v0.20.7) | **5** | Model registry UX, MLX backend, launch agent pattern | Desktop GUI unlicensed (Issue #11634); don't depend on as backend at scale |
| **mlc-llm** | Apache-2.0 | Apr 14 2026 | **4** | Unique WebGPU/iOS/Android compilation via TVM | Fragile TVM pins, steep build |
| **MLX-LM** | MIT | Apr 20 2026 (v0.31.2) | **4** | Cleanest Apple-Silicon stack; LoRA/QLoRA/quant/server | Mac-only; "not recommended for production" |

**Anchor choice:** vLLM for server-grade, llama.cpp for portable/edge, MLX-LM for Mac. Skip TGI, EXL2, TabbyAPI.

### Fine-tuning

| Project | License | Status | Maturity | Native algorithms |
|---|---|---|---|---|
| **axolotl** (axolotl-ai-cloud) | Apache-2.0 | v0.16.1 Apr 2 2026, active | **5** | QLoRA, DoRA, rsLoRA, LoRA+, GRPO, DPO, FSDP2 |
| **Unsloth** | Apache-2.0 | 2026.3.x very active | **5** | QLoRA, rsLoRA, GRPO, DPO — **no DoRA, no LoRA+** |
| **torchtune** | BSD-3 | Last release v0.6.1 Apr 2025 — **stagnating** | **2** | QLoRA, DoRA, DPO, GRPO — but deprecating |
| **TRL** | Apache-2.0 | **v1.0 Apr 1 2026** | **5** | DPO, GRPO, KTO, ORPO, SimPO (via CPO), PPO, RLOO, Online-DPO |
| **LitGPT** | Apache-2.0 | v0.5.12 Dec 2025 | **3** | QLoRA only — thin post-training |
| **Llama-Factory** | Apache-2.0 | v0.9.4 Dec 2025, active | **5** | All PEFT variants + GUI + GRPO via EasyR1 |
| **HF PEFT** | Apache-2.0 | Active | **5** | Adapter library only — canonical |

**Anchor choice:** Unsloth for single-GPU Triton speed (2× faster, 70% less VRAM), axolotl for config-driven reproducibility, TRL for RL trainers. torchtune is stagnating — don't anchor there.

### Quantization

| Project | License | Status | Notes |
|---|---|---|---|
| **bitsandbytes** | MIT | v0.49.2 Feb 16 2026, active | QLoRA standard — training, not serving |
| **GPTQModel** (ModelCloud) | Apache-2.0 | v6.0.3 Apr 3 2026 — **replaces archived AutoGPTQ** | HF Transformers' official GPTQ backend |
| **AutoAWQ** | MIT | **Archived May 11 2025** | Use only to load existing checkpoints; replaced by llm-compressor |
| **torchao** | BSD-3 | v0.17 Mar 25 2026, active | PyTorch-native INT4/INT8/FP8/MXFP4/NVFP4 + QAT |
| **NVIDIA TransformerEngine** | Apache-2.0 | v2.13 Mar 31 2026 | FP8/MXFP8 prod; **NVFP4 shipping on Blackwell** (SM100 only) |
| **llm-compressor** (vllm-project) | Apache-2.0 | v0.10 Apr 20 2026, very active | **Canonical path to vLLM-ready checkpoints** — W8A8, W4A16, AWQ, GPTQ, NVFP4, MXFP4 |
| **exllamav2 EXL2** | MIT | **Archived** (EXL3 active) | |

**Critical correction vs prior dossier:** AutoGPTQ (archived April 2025) is dead. Use GPTQModel. AutoAWQ (archived May 2025) is dead. Use llm-compressor for AWQ production.

### RL / RLAIF / alignment

| Project | License | Status | Algorithms |
|---|---|---|---|
| **TRL** | Apache-2.0 | v1.0 Apr 2026 | GRPO, DPO, KTO, ORPO, SimPO, PPO, RLOO, Online-DPO |
| **veRL** (volcengine) | Apache-2.0 | Daily through Apr 2026 | HybridFlow 3D engine, FSDP2/Megatron/vLLM/SGLang rollouts, GRPO/PPO/DAPO |
| **OpenRLHF** | Apache-2.0 | Feb 6 2026; v0.10 adds VLM RLHF Apr 2026 | Ray+vLLM hybrid-engine PPO/GRPO/REINFORCE++/RLOO |
| **NeMo-Aligner** | Apache-2.0 | **Frozen** May 2025 — NVIDIA redirects to NeMo-RL | Legacy only |
| **Unsloth GRPO** | Apache-2.0 | Very active | **80% VRAM reduction**, QLoRA+vLLM, 5 GB for 1.5B RL |

**MiniMax OpenClaw:** OpenClaw is a third-party agent runtime (docs.openclaw.ai), NOT a MiniMax project. MiniMax has not released any public RL training harness for M2.5/M2.7 — only the "OpenRoom" demo UI. **The RL harness is proprietary; do not treat as reproducible.**

**Sakana Darwin-Gödel Machine:** Code is public at `github.com/jennyzzt/dgm` (~2k stars, 400 forks). Contains full DGM loop, SWE-bench/Polyglot eval harness, Docker scripts. **Flag: no explicit top-level LICENSE file** — verify before incorporating. Repo warns it executes untrusted model-generated code.

### Evaluation

| Project | License | Status | Harvest |
|---|---|---|---|
| **lm-evaluation-harness** (EleutherAI) | MIT | Apr 2026 very active, ~9k★ | De-facto standard; 60+ benchmarks; vLLM/HF/API backends |
| **LiveBench** | Apache-2.0 | Monthly question drops | Contamination-free, objective, fresh monthly |
| **arena-hard-auto** (lmarena) | Apache-2.0 | v2.0-Preview Apr 2025 — slowed | GPT-4.1+Gemini-2.5 ensemble judge; highest LMArena correlation |
| **HELM** (stanford-crfm) | Apache-2.0 | v0.5.x active | Holistic — includes bias, toxicity, efficiency, MedHELM |
| **opencompass** | Apache-2.0 | v0.4.x active | Chinese-led 100+ datasets, NeedleBench long-context |

### Tracking and routing

| Project | License | Status |
|---|---|---|
| **MLflow** | Apache-2.0 | v3.11.x active — **pivoted to GenAI** (tracing, judges, AI Gateway, OTel-native) |
| **Aim** | Apache-2.0 | v3.28.x late 2025, slower |
| **ClearML** | Apache-2.0 | Active, full MLOps |
| **LiteLLM** (BerriAI) | MIT | v1.83.x daily, 44k★ — **⚠️ v1.82.8 PyPI was compromised, pin carefully** |
| **Portkey gateway** | MIT | 11k★, <1ms edge-deployable, 1600+ models |

### Research-scale architectures

| Project | License | Status | Harvest |
|---|---|---|---|
| **nanoGPT** | MIT | Author flagged **"deprecated" Nov 2025**; superseded by nanochat | Pedagogical baseline only |
| **OLMo** | Apache-2.0 | Migrated to OLMo-core (active Apr 18 2026) | Full pipeline with open data |
| **OLMoE** | Apache-2.0 | ~early 2025 low activity | 1B-active/7B-total configs |
| **Megatron-LM** | Custom BSD-3-ish | Apr 2026 daily | Production 3D/4D/5D parallelism, FP8/FP4, ~47% MFU on H100 |
| **nanotron** (HF) | Apache-2.0 | v0.5 Apr 2026 | Minimal 3D-parallel + EP-MoE + µTransfer; used for SmolLM3 |
| **picotron** (HF) | Apache-2.0 | 2025 tutorial-paced | Educational 4D-parallel |
| **Marin** (absorbed levanter Nov 2025) | Apache-2.0 | Feb 28 2026 | Reproducible open-lab executor; trained Marin-8B/32B |
| **lingua** (Meta) | BSD-3 | Early 2025 lightly maintained | Lean FSDP+torch.compile research stack |
| **TinyLlama** | Apache-2.0 | **Archived Jul 30 2025** | Models still usable; repo frozen |

### Distillation and MoE

| Project | License | Status |
|---|---|---|
| **DistillKit** (arcee-ai) | Apache-2.0 | Active; powers Arcee models; logit/hidden-state losses |
| **Megablocks** (databricks) | Apache-2.0 | v0.10.0 May 2025 **quiet since**; torch 2.6/triton 3.2 compat issues — **flag stale** |
| **ScatterMoE** (shawntan) | Apache-2.0 | 2025 steady; Triton ParallelLinear; **preferred Megablocks replacement** |
| **DeepEP** (deepseek-ai) | MIT | v1.2.1 Sep 2025, Apr 16 2026 active | Production NVLink+RDMA MoE all-to-all |

---

## Section 4 — Integration hypothesis deep dive

**Founder's thesis:** small-scale integration experiments predict large-scale behavior, ML research is over-specialized, frontier wins come from combining architecture + training + data + RL variants (DeepSeek as proof), and a disciplined multi-variable ablation foundry is an unfilled methodology niche. The thesis has **real support and real risk**, and the literature lets us be precise about each.

### What transfers from small to large

**μP/μTransfer (Yang & Hu 2022, arXiv 2203.03466)** is the single strongest pro-transfer result. Under Maximal Update Parametrization, optimal learning rate, init scale, and parameter multipliers remain stable across width, depth, batch size, and sequence length. Their headline: HPs tuned on a 40M proxy beat published GPT-3 6.7B numbers using **~7% of pretraining cost**. **What does NOT transfer:** regularization HPs (dropout, weight decay), which depend on data:compute ratio.

**Chinchilla (Hoffmann 2022, arXiv 2203.15556)** trained >400 models (70M–16B, 5B–500B tokens) and established ~**20 tokens per parameter** as compute-optimal — contradicting Kaplan 2020 (arXiv 2001.08361) which proposed N∝C^0.73, D∝C^0.27 (scale model ~5.5× per 10× compute, data only ~1.8×). **Porian et al. 2024 (arXiv 2406.19146)** resolved the discrepancy by identifying three specific Kaplan methodology errors: (1) non-embedding param counts inflating small-model compute cost, (2) overly long warmup hurting small models, (3) scale-dependent optimizer HPs (especially AdamW β₂ at small batch) not retuned per scale. **Decision-grade lesson: small-scale findings are only trustworthy if you retune at each scale point and use full param counts.** This is exactly what a rigorous ablation foundry must enforce.

**Schaeffer 2023 "Mirage" (arXiv 2304.15004)** showed emergent abilities on **92% of BIG-Bench "emergent" tasks** dissolve when you switch from exact-match/MCQ metrics to continuous metrics (token edit distance, Brier score). Genuinely emergent remainder is small but real. **Lesson: use continuous metrics as primary signal for Lane 2; discrete benchmarks last.**

**Pythia/PolyPythias (Biderman 2023, arXiv 2304.01373 + extension):** training dynamics across seeds are highly consistent at 14M–410M, with only rare outliers. **Single-seed small-scale is defensible — but bootstrap CIs across ≥5 seeds is the standard.**

### What fails to transfer (failure modes that INVERT at scale)

1. **Emergent capability thresholds.** CoT usefulness emerges ~60B. MMLU > random ~7B. True emergence (non-metric-artifact) exists above Lane 2 hardware ceiling.
2. **MoE-at-small-scale paradox.** DeepSeekMoE (arXiv 2401.06066) and OLMoE (arXiv 2409.02060) both report MoE benefits scale non-monotonically. DeepSeekMoE validates at 2B, 16B, 145B — with larger scales showing 28.5–40% compute reduction vs dense; small-scale ablations underpredict this. OLMoE's §4 ablations show shared experts HELP at 1B-active but HURT in other setups — so the correct architectural choice flips.
3. **Optimizer ranking instability.** AdamW vs Lion vs Shampoo rankings flip with scale and batch size.
4. **Regularization-vs-capacity inversion.** Dropout HP transfer fails per μP.
5. **Attention-variant perplexity-parity with downstream divergence.** Linear-attention / Mamba / RWKV can match perplexity at small scale but diverge on recall-heavy downstream tasks at larger scale.

### Even frontier labs ablate one-at-a-time and compose at full scale

**DeepSeek-V3 (arXiv 2412.19437), 671B total / 37B active, 14.8T tokens, 2.788M H800-hours.** Their ablation methodology (§4) compares auxiliary-loss-free vs loss-based balancing on two baselines (V2-Lite-like ~16B, V2-like ~200B) — **integrated/cumulative, not isolated single-variable sweeps**. They ablate batch-wise vs sequence-wise balance, MTP, but not the full factorial. **This is evidence for the founder's gap thesis:** the frontier does NOT do disciplined multi-variable factorial ablation, so an OSS foundry that does would be a genuine methodology niche.

### OSS tooling gap

Components that exist as pieces but no integrated orchestrator: nanotron has μTransfer. Marin has a reproducible executor graph. lm-eval-harness handles eval. MLflow 3.x has tracing. **No OSS tool ties IsoFLOP sweep orchestration + seed-variance budgeting + pre-registration + cross-scale transfer validation + multi-variable factorial DOE + ablation provenance together.** This is the defensible Lane-2 wedge.

### Minimum scientific validity bar

Non-negotiable for Dojo Lane 2: **(1)** ≥5 seeds with bootstrap 95% CIs; **(2)** IsoFLOP sweeps at ≥3 compute points (e.g., 10^17, 10^18, 10^19 FLOPs); **(3)** μP parameterization so HPs transfer; **(4)** pre-registered hypotheses with locked metrics before experiment start; **(5)** mandatory transfer validation at next scale tier before claim promotion; **(6)** continuous metrics as primary, discrete last; **(7)** open release of configs + seeds + logs.

### What 4060+3090 can and cannot reach

**Reachable on-prem:** induction heads (~10M), grokking (~1M synthetic), small-MoE win-loss, attention variant perplexity comparison at 100M–500M, optimizer ranking at 100M–1B, μP HP transfer sanity check, rStar-Math-style verifiable-reward RL at 1.5B–7B.

**Unreachable on-prem:** CoT emergence (~60B), MMLU>random (~7B+), true discrete-metric emergence, MoE gains vs dense at realistic scale. **For these, H100 spot at ~$450/finding is the only path.**

### Three concrete tractable Lane-2 experiments

1. **μP-verified attention-variant comparison** (100M–1B scale, 4060+3090, ~$0 cloud, 2 weeks). Compare MHA / GQA / MLA / sliding-window at matched FLOPs with μP, 5 seeds, continuous eval. Transfer-validate at 3B on one 24h H100 spot run (~$30). Deliverable: open dataset + pre-registered report.

2. **MoE-at-small-scale honesty paper** (300M–2B active, 4060+3090 + ~$200 H100 spot, 3 weeks). Reproduce DeepSeekMoE's fine-grained experts + shared-expert isolation at small scale; test whether small-scale MoE vs dense rankings match the 16B and 145B scales. If they don't — **report the inversion and its magnitude**. This is the kind of methodology paper labs avoid writing.

3. **GRPO verifiable-reward self-improvement at 1.5–7B** (2× 3090 + ~$100 H100 spot for validation, 2 weeks). Reproduce rStar-Math on Qwen2.5-Math-1.5B (Unsloth GRPO, 5 GB VRAM per paper). Target: 40%→70% on MATH at 1.5B, then transfer-validate at 7B. Provides Lane 1 donor recipe + Lane 2 finding.

---

## Section 5 — Self-improvement loop architecture

### What the literature actually says

**rStar-Math (arXiv 2501.04519, Guan et al., Microsoft, Jan 2025)** — MCTS + Process Preference Model (PPM) + 4 rounds of self-evolution on 747K problems. Qwen2.5-Math-7B on MATH: **58.8% → 90.0%**. Phi3-mini-3.8B: **41.4% → 86.4%** (surpassing o1-preview by +0.9%). Solves 53.3% of AIME 2024. **Critical insight: this works at 1.5B–7B because the reward is verifiable (final answer correct/incorrect) — no judge LLM needed.**

**DeepSeek-R1 (arXiv 2501.12948, Jan 2025)** — pure RL with GRPO on DeepSeek-V3-Base (no SFT) produced R1-Zero with emergent reasoning. AIME 2024 pass@1: **15.6% → 71.0%** (86.7% with majority vote, matching o1-0912). GRPO mechanics come from **DeepSeekMath (arXiv 2402.03300)**: critic-free PPO variant using group-mean reward as baseline, saving value-function memory. Known instabilities: reward hacking on long-completion policies, KL collapse under low β.

**DPO (arXiv 2305.18290)** reparameterizes RLHF into binary cross-entropy — no reward model, no RL sampling. **But Rafailov et al. 2406.02900 (NeurIPS 2024)** shows DPO/IPO/SLiC exhibit the same reward-hacking curves as PPO-RLHF: **GPT-4 win-rate rises then collapses within 1 epoch**; best performance often at only ~25% of data across 3 objectives × 7 β values × 3 sizes (1B/2.8B/6.9B).

**Gao 2023 (arXiv 2210.10760)** — reward overoptimization scaling laws. Gold score vs √KL is quadratic for best-of-n, log-linear for RL. Coefficients scale smoothly with proxy RM size. **RL consumes much more KL than BoN** for equal optimization. **Lesson: use best-of-n for safer early iterations; add RL only after reward saturation is understood.**

**Zheng 2306.05685** — LLM judges have **position bias, verbosity bias, self-enhancement bias, limited reasoning**. GPT-4 achieves >80% agreement with human preferences on MT-bench (matching human-human). **At smaller judge sizes, agreement drops sharply.**

**Constitutional AI (Bai 2022, arXiv 2212.08073)** — two-phase (SL self-critique/revision + RLAIF) from a written constitution, no human harmfulness labels. **Works competitively with human-feedback models ONLY with CoT evaluation at >52B scale.** Below that, self-critique fails.

**Self-Rewarding LMs (Yuan 2401.10020)** — Llama-2-70B iterative DPO with LLM-as-judge for 3 rounds beats Claude 2, Gemini Pro, GPT-4-0613 on AlpacaEval 2.0. **Works at 70B. Not demonstrated at <30B.**

**Sakana Darwin-Gödel Machine (Zhang et al., arXiv 2505.22954, May 2025, code at github.com/jennyzzt/dgm)** — iteratively modifies its own coding-agent code, keeps archive of validated variants. **SWE-bench 20.0% → 50.0%; Polyglot 14.2% → 30.7%.** Reproducible at smaller scale per code release, but executes untrusted model-generated code — sandboxing mandatory. No top-level LICENSE file visible; license status unclear.

### Size-threshold rule

**Verifiable-reward self-improvement works at 1.5B–7B** (rStar-Math proof point). **Open-ended self-judging needs ≥30–70B** (Constitutional AI, Self-Rewarding LMs proof points). Dojo-on-3090 = 7B ceiling → **must route all self-improvement through verifiable rewards (unit tests, math answers, JSON schema conformance, SQL exec results, grounded citation check) — not through a judge LLM.**

### Re-evaluation given STEAK already ships S29/S30/S32/S58/S59/S62

Prior dossier assumed these were greenfield. Grounded audit shows:
- **Constitutional Judge** (490 lines with 7-dimension JudgementReason enum) — present
- **DPO Trainer** (860 lines) — present
- **STaR loop_initiator** (1,246 lines) with LoopState enum — present
- **Gradient Surgery** (CAGrad, PCGrad) — present
- **Scaffolding evolution** (S58 Agent Loop) — present
- **Patch Validator** (408 lines) — present

**Revised estimate: self-improvement loop is ~70% donor-ready.** Remaining work is **not** building these modules; it is wiring them to a verifiable-reward pipeline and a three-way promotion gate (per below), plus resolving Open Questions 3–5 (do they actually run end-to-end or are they skeletons).

### Three-way promotion gate contract

Candidate model promotes from Lane-1 quarantine to DanteAgents prod only when ALL THREE emit green evidence:

1. **Harvest provenance receipt** — dataset hash, license check, copyright policy scan, synthetic/human provenance, deduplication stats.
2. **Dojo eval receipt** — task-specific eval pass (e.g., bookkeeping F1 ≥ 0.95 on held-out 2026-Q1 Real Empanada txns), ≥5-seed CI, continuous metric ≥ policy threshold, safety eval (TruthfulQA, harmbench subset).
3. **Forge policy receipt** — constitutional check (honesty/harmlessness/helpfulness Merkle), EU AI Act training summary valid, human-in-loop attestation.

All three sealed into a single Merkle root → becomes the "constitutional passport" the Agents runtime requires before loading a checkpoint.

### Anti-patterns to budget for

**KL collapse** (Gao 2210.10760 — RL consumes KL faster than BoN), **length exploitation** (policies learn to game verbosity-biased judges — cite Singhal "A Long Way to Go: Investigating Length Correlations in RLHF" and Dubois AlpacaEval length-controlled), **mode collapse** (DPO 2406.02900 — policies narrow to mean response within 1 epoch), **reward hacking** (verifiable-reward domains mitigate), **judge drift** (Zheng 2306.05685 — mandates multi-judge + periodic calibration against human spot-check).

---

## Section 6 — Product boundary pressure test

### Dojo ↔ Harvest

**Contract:** Harvest delivers a signed dataset pack (Arrow/Parquet + Merkle manifest + license/provenance JSON). Dojo consumes and owns everything from tokenization onward (pack construction, length bucketing, sample-packing, curriculum ordering, augmentation).

**Erosion signal:** if Dojo starts calling Harvest for corpus shaping mid-training, the line has failed. **Rule: Harvest never sees a tokenizer; Dojo never sees a raw scrape URL.**

### Dojo ↔ Agents

**Contract:** Dojo emits versioned model artifacts (weights + LoRA + quant + deploy manifest + constitutional passport) into a model registry. Agents pulls by manifest ID. Serving runtime (vLLM/SGLang/llama.cpp launcher) lives in Dojo because it's training-adjacent and needs the same HAL; Agents only consumes the OpenAI-compat endpoint URL + passport.

**Erosion signal:** if Agents starts doing LoRA merges or runtime quantization, the line has failed. **Rule: Agents never touches weights; Dojo never handles live user requests.**

### Dojo ↔ Forge

**Contract:** Forge defines WHICH eval policies must pass (constitutional rules, EU AI Act Annex XI checklist, copyright scan results). Dojo MECHANIZES the checks — runs lm-eval-harness, HELM subsets, custom Dante evals, the constitutional judge. Dojo emits evidence; Forge interprets and signs.

**Erosion signal:** if Dojo starts writing policy rules, the line has failed. **Rule: Forge is policy; Dojo is mechanism.**

### Lane 1 ↔ Lane 2

Shared: eval harness, HAL, evidence substrate, tracking (MLflow 3.x GenAI stack), promotion gate framework.
Separated: Lane 1 = production LoRA recipes; Lane 2 = factorial experiment orchestrator (IsoFLOP sweeps, seed-variance budgeting, pre-registration registry, cross-scale transfer validation).

---

## Section 7 — Minimum lovable v1

### In for v1 (6-week integration pass, not 3 months)

Based on ~76,000 transplant lines already existing, the work is integration + wiring + end-to-end test, not construction.

**Included:** HAL autodetect (3090/4060/edge), GPU turbo policy, training/finetune (Unsloth for 4060, native for 3090), vLLM serving launcher, llama.cpp edge launcher, layer streaming for 4060, idle training, evaluation harness (lm-eval + bookkeeping-specific), promotion gate with three-way contract, constitutional passport emission, MLflow 3.x GenAI tracing, first Lane-1 fine-tune run.

**Deferred to v2:** Full Lane-2 factorial orchestrator (ship IsoFLOP sweep stub only), self-improvement loop end-to-end (ship S29/S30 wiring test only), Darwin-Gödel scaffolding, multi-node training, NVFP4 (Blackwell not owned).

### End-to-end v1 workflow

```
Harvest pack (signed) → Dojo tokenize → Dojo QLoRA (Unsloth, 4060)
→ Dojo eval (bookkeeping F1 on held-out) → Dojo serve (vLLM or llama.cpp)
→ Promotion gate (3-way Merkle) → Model registry → Agents pulls by manifest ID
```

### First real fine-tune

**Qwen2.5-7B QLoRA on Real Empanada bookkeeping pack, Unsloth, 4060 laptop.**
- Config: bs=1, grad_accum=8, seq=2048, paged_adamw_8bit, r=16, 3 epochs on ~20K labeled txns
- Unsloth VRAM table: 5 GB minimum for 7B QLoRA; with 2048 seq + activations ≈ 7–7.5 GB — **tight but verified feasible**
- Expected wall time: 6–12 hours (Unsloth 2× speedup on Ada)
- Eval: F1 ≥ 0.95 on 2026-Q1 held-out txns vs Haiku 4.5 zero-shot baseline ($10/month API equivalent)
- Success metric: beats Haiku-4.5 zero-shot F1 by ≥5 points, passes constitutional check

### First Lane-2 experiment proving scientific foundry is sound

**μP-verified attention-variant comparison at 100M–500M** on 3090 (2 weeks):
- 4 variants: MHA, GQA, MLA, sliding-window
- 3 compute points (IsoFLOP at ~10^17, 10^18, 10^19)
- 5 seeds each → 60 runs × ~4h on 3090 = ~240 GPU-hours
- μP-parameterized HPs
- Pre-registered metric: continuous bpd on held-out C4 subset + downstream HellaSwag+PIQA+SciQ
- Transfer-validate winner at 3B on 24h H100 spot (~$30)
- Deliverable: open configs, logs, seeds + a short report that names one finding and its CI

**If this Lane-2 run ships a clean result with pre-registered methodology, the foundry is real.** If it drags past 4 weeks or produces an un-defensible finding, Lane 2 is suspect — go to Section 8 kill criteria.

### Timeline revision

Prior dossier estimated 3 months for v1 based on a from-scratch module list. Grounded audit shows ~76K lines of transplant material. **Revised estimate: 6-week integration pass** assuming the eight open questions resolve favorably. If Open Questions 3–6 reveal skeleton-only implementations, add 3–4 weeks. **Worst-case realistic: 10 weeks.**

---

## Section 8 — Kill criteria and counter-argument

### Case for Dojo first, Harvest second

Dojo produces immediate operational value (Real Empanada bookkeeping fine-tune saves ~$30/month and solves the data-sovereignty concern for tax records). Harvest only delivers value once it has something to feed — and Dojo's first pack can be hand-curated (20K txns from QuickBooks export). Dojo also validates the HAL/GPU-turbo/evidence substrate that every other organ depends on.

### Case for managed fine-tuning (steelman)

Together AI: LoRA FT ~$0.48/M tokens ≤16B, ~$4/M Llama-70B LoRA, $14/M full-FT 70B. Any HF Hub model. **Weights downloadable**, explicit no-lock-in. SOC2/HIPAA on enterprise tier.

Fireworks AI: LoRA-only from ~$0.50/M ≤16B, ~$6/M 70B. SOC2 Type II + HIPAA + GDPR. Same-price inference on tuned model with up to 100 LoRAs free on one deployment.

Modal Labs: per-second billing — A100-80G ~$4.56/hr, H100 ~$4.56/hr, B200 $6.25/hr. BYO code (Axolotl/torchtune/TRL). Weights stay in your Volume.

RunPod: H100 SXM $2.69/hr on-demand, $1.25/hr Community spot. Raw GPU rental, BYO stack.

**OpenPipe** post-CoreWeave acquisition (Sep 2025): RL/agent-focused, pricing opaque. **Fly.io GPU: DEPRECATED — shutting off July 31, 2026. Do not adopt.**

**Managed-FT decision rule:** If a Dante fine-tune is a one-shot (20K txns, 6-hour run), managed beats on-prem for a solo founder on wall-clock. Together at ~$5–15 for 10–30M-token run is trivial. **But:** (1) proprietary Real Empanada data under enterprise tier only + BAA; (2) no claim of data sovereignty if shared tier; (3) can't run idle training at 3am on owned hardware; (4) no evidence-substrate integration; (5) no Lane-2 custom scaffolding. Managed is a **fast-iteration escape hatch for Lane 1**, not a substitute for Dojo.

### Case Lane 2 is a rabbit hole

- Lane 2 findings that don't transfer waste $450–$1500 each at H100 spot.
- Solo founder writing factorial ablation papers competes directly with AllenAI, Meta FAIR, HF science — each with 20× the compute and 10× the people.
- OSS methodology papers don't convert to ARR.
- Every hour on Lane 2 is an hour off DanteCode/Agents/Forge shipping revenue.
- The history of "disciplined ablation foundry" as a business model is thin — Eleuther.ai runs on donations; EpochAI runs on grants.

### Signals to kill Lane 2

Kill Lane 2 if ANY of:
1. First experiment runs past 4 weeks without a clean pre-registered finding.
2. Second experiment finds zero-sign-flip between small-scale and transfer-validation scale — i.e., small scale was perfectly predictive → the "integration gap" thesis is wrong and Lane 2 has no edge.
3. H100 spot spend exceeds $2K/month for two consecutive months without a shipped report.
4. The methodology paper niche gets filled by a better-resourced lab (Marin, nanotron, or similar publishing factorial ablation tooling).
5. Real Empanada or Dante revenue demands all of Ricky's attention for >4 consecutive weeks.

### Decision rule

**Ship Dojo Lane 1 first (6 weeks). Run ONE Lane 2 experiment (μP-verified attention comparison) as part of v1 to prove the foundry works and produce a shippable report. Then evaluate Lane 2 against the five kill criteria before funding a second experiment.**

---

## Section 9 — Final verdict

**Recommendation: Build Dojo as a 6-week integration pass over STEAK V3 donor code, with Lane 1 production-critical and one Lane 2 experiment run as proof-of-foundry within v1.** Use Unsloth for 4060 single-GPU, vLLM+llama.cpp for serving, Together AI as managed escape hatch, MLflow 3.x + lm-eval + HELM subsets for eval. Route all self-improvement through verifiable rewards (7B ceiling on local hardware cannot sustain judge-LLM RL). Kill Lane 2 against the five signals above.

**Confidence: 76/100** (up from prior 62).

The grounded STEAK audit is the dominant upward factor: ~76K lines of transplant material with constitutional evidence substrate, promotion gate, HAL, idle training, layer streaming, and self-improvement scaffolding already shipped. A prior estimate treating Dojo as greenfield produced 62; treating it as integration-over-donor produces 76.

**Three findings driving the score most:**

1. **STEAK's self-improvement family (~18K lines across S29/S30/S32/S58/S59/S62) is ~70% donor-ready** — Constitutional Judge with 7-dimension enum, 860-line DPO trainer, STaR loop with LoopState FSM, gradient surgery, patch validator, scaffolding evolution. This collapses what was a 2–3-month greenfield into a wire-up + verifiable-reward adapter.

2. **The GPU turbo ARCH_POLICY_MAP already codifies Blackwell→FP4, Ada→FP8, Ampere→FP8 routing and Apple M4 Max→INT4** — the exact Ampere-FP8-rejection failure mode confirmed in external research (TransformerEngine asserts CC≥8.9) already has a policy branch. HAL maturity is higher than any OSS alternative.

3. **4060 + 3090 is genuinely sufficient for Lane 1 (Qwen2.5-7B QLoRA verified by Unsloth's published table) and for three specific Lane-2 experiments** (μP attention sweep, small-MoE honesty paper, GRPO verifiable-reward at 1.5B). H100 transfer-validation at ~$450/finding is affordable for a solo founder at 2–4 findings/month.

**Three things that would lower the score if different:**

1. If Open Questions 3–6 resolve unfavorably — STaR is skeleton-only, `vllm_configurator.py` only writes configs without a launcher, `patch_validator` is a stub, promotion gate has no real metric connections — subtract 10–15 points; the 6-week estimate moves to 10–12 weeks.

2. If `pytest_errs.txt` reveals systemic test decay across the core substrate (evidence, one_door, types), subtract 8–10 points — the substrate is the non-negotiable foundation, and any rot there blocks everything downstream.

3. If NVLink for 3090 becomes genuinely unobtainable AND a second 3090 cannot be acquired, subtract 5 points — Lane 2 becomes marginal on a single 3090 for 1B+ experiments, and H100 spot cost rises to ~$800/finding for equivalent seed coverage.

**Most important next action:** resolve the **eight open questions via 2–3 days of direct STEAK code inspection** before any v1 work starts. Specifically: run the test suite and report pass/fail count (Q7), `grep -r "subprocess\|Popen" steak/inference/serving/` to check if configurators actually launch servers (Q6), read `self_improve/star/loop_initiator.py` end-to-end (Q3), read `s62_auto_improve/patch_validator.py` end-to-end (Q4), read `s68_promotion_gate/policy.py` end-to-end (Q5), read `precision/fp8_handler.py` Ampere branch (Q2), read `training/finetune.py` for Triton kernels or HF wrap (Q1), and read `pyproject.toml` for dependency alignment (Q8). The answers determine whether the revised 6-week estimate holds or reverts toward 10–12 weeks — and they cost 2–3 days to resolve vs the 4+ weeks a wrong estimate would cost mid-build.