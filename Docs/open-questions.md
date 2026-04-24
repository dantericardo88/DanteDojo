# Open Questions — STEAK V3 Donor Audit

Resolved during Phase 0 inspection of the STEAK V3 donor codebase.
All answers inform architectural decisions in DanteDojo v1.x.

---

## Q1: Does `training/finetune.py` wrap HF Transformers or have Triton kernels?

**ANSWER:** HF Transformers wrapper. Imports `AutoModelForCausalLM` and
`BitsAndBytesConfig` from `transformers`. No Triton kernels present.

**Decision:** Unsloth replaces it as the primary training backend for single-GPU
targets (RTX 4060 Ti / RTX 3090). Axolotl remains the multi-GPU path.

---

## Q2: Does `precision/fp8_handler.py` reject FP8 on Ampere or emulate it?

**ANSWER:** Partial support only. RTX 3090 is listed as
`fp8_compute=False, fp8_storage=True`. STEAK does NOT hard-reject FP8 on Ampere;
it allows FP8 storage. This is a known issue: NVIDIA Transformer Engine asserts
SM >= 8.9 for FP8 compute. Running FP8 storage-only on Ampere is slower than BF16.

**Decision:** Dojo MUST add an explicit compute-capability gate. If CC < 8.9,
raise `PrecisionUnsupported` and route to BF16 or W8A16-Marlin. No silent
performance degradation.

---

## Q3: Does `STaR/loop_initiator.py` run end-to-end or is it a skeleton?

**ANSWER:** Fully functional — confirmed via direct inspection. The design separates
`initiate()` (returns `STaRConfig`) from `execute_loop(config)` (drives the real loop).
`execute_loop` implements a proper state machine:
`INITIALIZING → GENERATING_RATIONALES → FILTERING → RETRAINING → EVALUATING → CONVERGED`.
Calls `_generate_rationales()`, `_filter_rationales()`, `_dispatch_retraining()`,
`_evaluate_model()` with convergence tracking and metrics history. Not a skeleton.

**Decision:** STaR is ~70% donor-ready. v1.1 adds verifiable-reward adapter wiring
(rStar-Math-style GRPO on verifiable domains). v1.0 ships STaR config+FSM, defers
full RL training integration.

---

## Q4: Is `patch_validator.py` a real safety gate or a stub?

**ANSWER:** Fully functional — confirmed via direct inspection (203 lines).
`validate()` runs four real checks using Python introspection:
1. **Syntax check** via `py_compile` — rejects malformed Python.
2. **Safety check** via `ast.walk()` — flags dangerous patterns (eval, exec, os.system, etc.).
3. **Regression check** — flags non-trivially-small patches.
4. **Constitutional check** — detects unbounded loops in patch AST.
Returns a `PatchValidationResult` with `is_valid`, `risk_score`, `checks_passed`,
`checks_failed`. Not a stub.

**Decision:** Transplant `patch_validator.py` as-is into `dojo/promote/` for v1.1
self-improvement gate. v1.0 uses it as the patch safety check in the promotion sealer.

---

## Q5: Does `s68_promotion_gate/policy.py` connect to actual metrics?

**ANSWER:** Threshold-only. Parameters: `quality_threshold` (0.7),
`efficiency_threshold` (0.5), `max_cost_tokens` (500K). The `evaluate` method
accepts a `candidate` object and a `certified_matrix_support` dict and checks
provided metrics against thresholds. It does NOT connect to lm-eval or live eval
metrics.

**Decision:** Dojo's `dojo/promote/gate.py` will wire real lm-eval metrics into
the threshold check so the gate evaluates live benchmark results, not caller-supplied
numbers.

---

## Q6: Does `vllm_configurator.py` launch vLLM or only generate configs?

**ANSWER:** Config-only. Creates `VLLMConfig` and `VLLMDeployment` dataclass
objects. No `subprocess`, `Popen`, or `os.exec` calls present.

**Decision:** `dojo/serve/vllm_backend.py` MUST add real subprocess-based process
lifecycle management: start, health-check, graceful stop, and restart.

---

## Q7: What is the current STEAK test pass rate?

**ANSWER:** 8 collection errors out of 1,692 collected items. All failures are
`ModuleNotFoundError: No module named 'torch'` — environment errors on a Windows
machine without CUDA/PyTorch installed, not code quality issues. Core substrate
tests (evidence, one_door, types) are expected to pass on a machine with the
required dependencies installed.

**Decision:** DanteDojo CI runs on ubuntu-latest with `.[dev]` only (no torch),
so torch-gated tests must be guarded with `pytest.importorskip("torch")` or
placed in a separate `tests/gpu/` suite excluded from the base CI job.

---

## Q8: What is in `pyproject.toml` — any problematic dependencies?

**ANSWER:** Uses `pydantic>=2.0` and the standard HF stack. No `auto_gptq` or
`autoawq` in the inspected version.

**Decision:** Dojo's `pyproject.toml` uses `gptq-model` (replaces the archived
AutoGPTQ) and `llm-compressor` (replaces the archived AutoAWQ). `litellm` is
pinned with `!=1.82.8` to exclude the known-compromised release.
