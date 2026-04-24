# DanteDojo Constitutional Guide

## What Is the Constitutional Architecture?

DanteDojo's constitutional architecture means that every AI operation is:
1. **Observable** — every call emits a SHA-256-linked evidence bundle
2. **One-door** — every subsystem has exactly one public entry point
3. **FSM-governed** — multi-step processes use explicit finite state machines
4. **Fail-closed** — failures raise typed exceptions, never return None or silently continue

These are not optional patterns. They are enforced by the substrate.

## The One-Door Doctrine

Every module's public API is a single `@one_door` decorated function:

```python
from dojo.core.one_door import one_door

@one_door("pack.import", organ="pack")
def import_pack(uri: str, raw_data: list, signature: bytes) -> PackResult:
    ...
```

The decorator:
- Emits `ONE_DOOR_ENTRY` evidence before calling the wrapped function
- Checks `DOJO_BYPASS` env var and raises `OneDoorViolation` if set
- Sets `__one_door__` attribute so callers can introspect the entry point name

**Why one door?** Multi-entry modules become untestable and auditable only at arbitrary depth. One door means one place to check logs, one place to gate access, one place to add cross-cutting concerns.

## Evidence Chain

### Emit an evidence bundle

```python
from dojo.core.evidence import EvidenceLogger, emit_evidence

# Module-level shorthand
bundle = emit_evidence({"type": "custom.event", "data": {"key": "value"}})

# Via singleton logger
logger = EvidenceLogger.get_instance()
bundle = logger.emit({"type": "train.finetune.started", "data": {"model": "qwen2.5-7b"}})
```

### Operation context manager

```python
with logger.start_operation("load_model", organ="stream") as op:
    result = load_weights(model_id)
# Emits operation.start and operation.end bundles automatically
# Emits operation.error on exception
```

### Verify the chain

```python
chain = logger.chain()
assert chain.verify_chain()  # checks hashes and Merkle root
```

### Export a receipt

```python
receipt = logger.export_receipt(Path("/tmp/dojo/receipts/run-001.json"))
# receipt["merkle_root"] is the chain fingerprint
```

## FSM Design Pattern

All multi-step operations use `BaseFSM[S]`:

```python
from enum import Enum
from dojo.core.fsm import BaseFSM

class ImportState(Enum):
    IDLE = "idle"
    FETCHING = "fetching"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"

class ImportFSM(BaseFSM[ImportState]):
    initial_state = ImportState.IDLE
    _TRANSITIONS = {
        ImportState.IDLE: frozenset({ImportState.FETCHING}),
        ImportState.FETCHING: frozenset({ImportState.VERIFYING, ImportState.FAILED}),
        ImportState.VERIFYING: frozenset({ImportState.DONE, ImportState.FAILED}),
    }

fsm = ImportFSM()
fsm.transition(ImportState.FETCHING)   # OK
fsm.transition(ImportState.DONE)       # raises FSMTransitionError — skipped VERIFYING
```

## Exception Hierarchy

All Dojo exceptions inherit from `DojoError`:

```
DojoError
├── OneDoorViolation          DOJO_BYPASS env var set
├── FSMTransitionError        Illegal FSM transition
├── PrivacyTierViolation      CONFIDENTIAL/SECRETS → external API
├── ConstitutionalViolation   Generic constitutional breach
├── PrecisionUnsupported      FP8 on SM < 8.9
├── PackSchemaInvalid         Missing required pack fields
├── PackContaminationDetected Eval-gold hash found in training data
├── QuantQualityRejected      PPL delta > 15% or F1 delta > 5pt
├── EvalContaminationError    Pack hash in eval set
├── EvalError                 Stability gate failure
├── JudgeDriftError           Judge agreement < 80%
├── PromotionBlocked(gate, reasons)   Three-way gate failure
├── HarvestClientError        DanteHarvest unreachable
└── ForgeClientError          DanteForge unreachable
```

`PromotionBlocked` carries `gate` (which check failed) and `reasons` (list of human-readable explanations).

`DojoError` accepts an optional `reason` kwarg for actionable detail:

```python
raise PrecisionUnsupported(
    "FP8 not supported on this hardware",
    reason="Upgrade to Ada (SM 8.9+) or use W8A16-Marlin instead",
)
```

## DOJO_BYPASS Guard

The `DOJO_BYPASS` environment variable is a constitutional tripwire. If set to any non-empty value, every `@one_door` decorated function raises `OneDoorViolation` before executing. This exists so CI can detect accidental test code that skips constitutional checks.

**Never set DOJO_BYPASS in production.** It is only meaningful in adversarial test scenarios.

## Privacy Tier Enforcement

```python
from dojo.core.types import PrivacyTier, RoutingOption, RoutingPolicy

policy = RoutingPolicy(
    task_name="bookkeeping",
    preference_order=(RoutingOption.LOCAL_SPECIALIST,),
    max_latency_ms=5000,
    max_cost_usd=0.0,
    privacy_tier=PrivacyTier.CONFIDENTIAL,
    fallback_on_unavailable=False,
)
assert policy.allows_external_api() is False  # always False for CONFIDENTIAL/SECRETS
```

`router.dispatch()` checks `privacy_tier` before evaluating any routing option. If a CONFIDENTIAL task reaches an external API option, it raises `PrivacyTierViolation` and emits `ROUTE_PRIVACY_BLOCKED` evidence — the network is never touched.

## Envelope Helpers

Short-circuit responses can use the `ok()` / `fail()` envelope helpers:

```python
from dojo.core.one_door import ok, fail

return ok({"model_id": artifact_id, "status": "ready"})
# → {"ok": True, "data": {"model_id": ..., "status": "ready"}}

return fail("vram-insufficient", "Not enough VRAM for this model", details={"required_gb": 14})
# → {"ok": False, "error": {"code": "vram-insufficient", "message": "...", "details": {...}}}
```
