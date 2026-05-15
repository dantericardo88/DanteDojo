# DanteDojo — Agent Context Brief
> Every sub-agent dispatched by goal-loop-matrix MUST read this before writing any code.

**Last updated:** 2026-05-14 | **Harsh regrade applied** | **Target: all dims 9.0+**

---

## 1. What You Are Building

DanteDojo is a **sovereign AI training, eval, and inference organ**. The core philosophy:
- Every operation emits a cryptographic evidence bundle (SHA-256 Merkle chain)
- All state transitions go through typed FSMs — no side effects outside `@one_door`
- Privacy is enforced at the routing layer — CONFIDENTIAL data never touches an external API
- The system is fail-closed: no silent Nones, no bare excepts, no stubs that pretend to work

You are not refactoring for style. You are implementing real, working functionality that meets harsh scoring criteria.

---

## 2. Harsh Scoring Rules (Applied to Your Work)

An adversarial scorer will re-score your work after you submit. Know these rules:

| Condition | Max Score |
|-----------|-----------|
| Feature fully working, tested on real hardware/infrastructure | 9-10 |
| Feature logic correct, tested with real external calls (mocked at boundary only) | 8 |
| Feature wired, tested with real framework (e.g. real GPU training, not mock) | 7 |
| Feature logic correct but tested on mock only (no real hardware/API) | 6 |
| Feature wired but DOJO_REAL_GPU-gated and never actually runs | 5 max |
| Feature stubs + constants only | 1-2 |
| Feature absent | 0 |

**The adversarial scorer will downgrade you if:**
- You claim "tested" but only mocked the entire system
- You gate real behavior behind DOJO_REAL_GPU=1 without proving it works
- Your implementation is wired but has no integration test with real data
- You say "provenance chain updated" but evidence events are not emitted
- Streaming governors (d26/d27/d28) claim 9 without a real inference loop

**What earns a 9:**
- Real data in, real result out, evidence chain has events, error paths tested

---

## 3. Hardware Context

**Available:** RTX 4060 Ti 16GB — Ada Lovelace, SM 8.9
- ✅ BF16/FP16 training
- ✅ FP8 training (Ada Lovelace supports FP8 via TransformerEngine or torchao)
- ✅ QLoRA (4-bit base + 16-bit adapters, 7B model fits in 16GB)
- ✅ SFT on 1B-7B models
- ✅ Triton custom kernels (SM 8.9 supported)
- ✅ GGUF inference via llama.cpp
- ❌ FP4 training — needs SM10.0 (Blackwell) — do NOT attempt, hard cap at 5

**DOJO_REAL_GPU=1** must be set when testing real GPU paths. In CI, mock at the torch level. But for dimension scores above 6, you need a real GPU test result.

---

## 4. Constitutional Requirements (Do Not Bypass)

Every piece of code you write MUST follow these rules or it will be rejected by merge-court:

1. **@one_door decorator** on every public entry point — no naked functions that accept untrusted input
2. **Evidence emission** — every significant operation emits evidence via `EvidenceLog.add(...)` with a typed event (e.g. `TRAIN_START`, `SERVE_REQUEST`, `REVOCATION_CHECKED`)
3. **Typed exceptions** — no `raise Exception(...)`, no `raise ValueError(...)` — use the DojoError subclass hierarchy
4. **FSM lifecycle** — multi-step operations (training, serving) must use BaseFSM; no free-floating `state = "running"` strings
5. **No silent failures** — if something fails, raise a typed exception or log evidence with `status=FAILED`
6. **Privacy before routing** — any routing function must check PrivacyTier first
7. **Merkle seal** — complete operations seal their evidence with `evidence_log.seal()` returning a Merkle root that is stored in the result

---

## 5. Wave Dependencies (Do Not Work Out of Order)

```
Wave 1 (first):  d16, d17, d24, d29    — training foundation + GGUF + server
Wave 2:          d22, d23, d25, d31    — needs d16 for GPU quant tests
Wave 3:          d11, d13, d20, d09, d10  — needs Wave 1 done
Wave 4:          d30, d26, d27, d28    — needs d24 (GGUF) + d29 (server)
Wave 5:          d32, d35, d36, d08    — routing real calls
Wave 6:          d42, d43, d44, d45    — experiment tracking (independent)
Wave 7:          d39, d41, d21, d46, d47, d48  — near-9s (independent)
Wave 8:          d49, d50             — CLI smoke + PyPI
Wave 9 (last):   d15                  — Triton kernels (hardest, schedule last)
BLOCKED:         d12                  — Blackwell hardware required, cap at 5
```

If you are dispatched to work on a dimension whose prerequisite isn't done, check `.danteforge/matrix/scores.json`. If the prereq is < 7, build the prereq first and note it in your output.

---

## 6. What "Done" Looks Like Per Dimension (Quick Reference)

| Dim | Done = |
|-----|--------|
| d08 | HTTP GET to revocation_url at model load; PassportRevoked raised on `{revoked:true}` |
| d09 | dojo detect returns SM=8.9 from real torch.cuda.get_device_capability() |
| d10 | FP8 training run completes on RTX 4060 Ti; loss logged |
| d11 | BF16 training run completes; mixed precision scaler active |
| d13 | QuantQualityRejected fires in real GPTQ run with bad quant |
| d15 | 2 Triton kernels; ≥20% speedup measured on 4060 Ti |
| d16 | TRL SFTTrainer runs on real GPU; loss < initial; checkpoint saved |
| d17 | QLoRA adapter trains on real GPU; VRAM ≤ 14GB; adapter loads back |
| d20 | Idle daemon triggers real GPU training; checkpoint saved |
| d21 | 10 injected gold items detected in ≥1M example corpus |
| d22 | GPTQ 4-bit quantized on real GPU; PPL measured |
| d23 | AWQ 4-bit quantized on real GPU; F1 measured |
| d24 | GGUF file generated; dojo_passport_id in metadata; loadable by llama.cpp |
| d25 | 5 adversarial bypass vectors all blocked |
| d26 | VRAM streaming during real inference; VRAM never exceeds 1-layer budget |
| d27 | KV eviction fires during real 512-token generation |
| d28 | SovereignHalt fires when lookahead > 16 attempted in real inference |
| d29 | /v1/chat/completions returns real tokens; openai client works |
| d30 | GGUF loaded locally; tokens generated; no network calls |
| d31 | vLLM serves 3 concurrent requests; throughput logged |
| d32 | dispatch() routes to ≥3 real providers; routing decision in evidence |
| d35 | Budget enforcement fires on over-limit request; cost in evidence |
| d36 | Real dead provider triggers fallback or RoutingProviderUnavailable |
| d39 | 3 distinct judge models compared; gate fires at <80% pairwise agreement |
| d41 | All 8 canonical benchmarks covered synthetically with correct few-shot |
| d42 | Runs appear in real MLflow server after training |
| d43 | Params + metrics + artifacts all queryable from MLflow API |
| d44 | dojo experiment compare returns ranked table, best checkpoint shown |
| d45 | W&B callback logs params + metrics to W&B (mocked server OK) |
| d46 | Full CRUD on dojo:// registry; versions selectable |
| d47 | Real HuggingFace pack imported; provenance in evidence chain |
| d48 | Apache/Llama/CC-BY-NC license paths all tested |
| d49 | All 12 dojo <cmd> --help pass CI without DOJO_REAL_GPU |
| d50 | pip install dantedojo works; mkdocs/sphinx builds; CI badge on README |

---

## 7. Evidence Events to Emit

When implementing a new feature, emit evidence at these points:

```python
# Start of operation
evidence_log.add(EventType.YOUR_EVENT_START, {
    "dimension": "d16",
    "detail": "starting SFT training run"
})

# Completion with Merkle seal
receipt = evidence_log.seal()  # returns Merkle root
# Store receipt in result object
```

New event types should be added to `dojo/core/types.py` EventType enum.

---

## 8. Test Requirements

A dimension is not done until:
1. Happy path test passes (feature works correctly)
2. Failure path test passes (typed exception raised, evidence emitted)
3. Evidence chain has the right events (check with `verify_chain()`)
4. If GPU-dependent: at least one test runs with `DOJO_REAL_GPU=1`

Do not mark a dimension complete if only the happy path exists.
