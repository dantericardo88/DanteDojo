# DanteDojo Three-Way Promotion Gate

## Overview

A model cannot be loaded by DanteAgents until it holds a `ConstitutionalPassport`. Obtaining a passport requires all three of:

1. **Harvest check** — provenance, revocation, and license validation by DanteHarvest
2. **Dojo check** — evaluation quality gate (F1 delta ≥ 5pt over baseline)
3. **Forge check** — policy approval from DanteForge

All three must return GREEN. Any single failure → BLOCKED → `PromotionBlocked` raised.

## FSM

```
INIT
 └─► HARVEST_CHECK
      ├─► DOJO_CHECK
      │    ├─► FORGE_CHECK
      │    │    ├─► PASSPORT_BUILD
      │    │    │    └─► SEALED ✓
      │    │    └─► BLOCKED ✗
      │    └─► BLOCKED ✗
      └─► BLOCKED ✗
```

## Running the Gate

```python
from dojo.promote.gate import promote
from dojo.core.types import ModelArtifact

passport = promote(
    candidate=artifact,
    harvest_receipt_id="harvest-receipt-abc123",
    eval_receipt_merkle="<merkle from run_eval>",
    baseline_f1=0.70,
    candidate_f1=0.78,   # must be >= baseline + 0.05
)
# → ConstitutionalPassport(passport_id=..., expires_at=now+90d, merkle_root=...)
```

### CLI

```bash
dojo promote dojo://empanada-bookkeeping/v1 \
    --harvest-receipt harvest-receipt-abc123 \
    --eval-merkle <merkle> \
    --baseline-f1 0.70 \
    --candidate-f1 0.78
```

## Gate 1: Harvest Check (`dojo/promote/harvest_check.py`)

Validates:
1. Receipt is **not revoked** (mock: any receipt ID starting with `revoked_` → BLOCKED)
2. Pack license **allows derived fine-tunes** (`allows_derived_ft=True`)
3. Pack signature is **valid** (mock: `b"test_signature"` always passes)

Emits `PROMOTE_HARVEST_GREEN` or `PROMOTE_HARVEST_BLOCKED`.

## Gate 2: Dojo Check (`dojo/promote/dojo_check.py`)

Validates:
- `candidate_f1 >= baseline_f1 + 0.05` (5 percentage-point improvement)
- Optional: `ci_lower >= baseline_f1` (95% CI lower bound above baseline)

Emits `PROMOTE_DOJO_GREEN` or `PROMOTE_DOJO_BLOCKED`.

## Gate 3: Forge Check (`dojo/promote/forge_check.py`)

Posts to DanteForge's policy endpoint with 3-retry exponential backoff and a 60-second timeout. **Fail-closed**: any timeout, connection error, or 4xx/5xx → BLOCKED.

When `forge_client=None` (no Forge instance running), the mock always returns GREEN.

Emits `PROMOTE_FORGE_GREEN` or `PROMOTE_FORGE_BLOCKED`.

## The Passport (`dojo/core/types.py`)

```python
@dataclass(frozen=True)
class ConstitutionalPassport:
    passport_id: str
    model_artifact_id: str
    harvest_receipt_id: str
    harvest_check_receipt: bytes   # signed by Harvest key
    dojo_check_receipt: bytes      # signed by Dojo key
    forge_check_receipt: bytes     # signed by Forge key
    merkle_root: str               # Merkle over all three receipts + artifact ID + issued_at
    issued_at: datetime
    expires_at: datetime           # issued_at + 90 days
    revocation_url: str            # DanteForge revocation endpoint
```

`is_expired()` checks `datetime.now(UTC) > expires_at`.

`to_dict()` is JSON-serializable (bytes fields are hex-encoded).

## Seal Verification (`dojo/promote/sealer.py`)

The `merkle_root` is a Merkle tree over:
- SHA-256(harvest_receipt_id)
- SHA-256(dojo_check_receipt)
- SHA-256(forge_check_receipt)
- SHA-256(artifact_id)
- SHA-256(issued_at ISO string)

`verify_seal(passport)` recomputes this root and compares it against `passport.merkle_root`. Any tampering with the passport fields will fail verification.

## Evidence Emitted

A successful promotion run emits 7 evidence bundles:
1. `one_door.entry` (promote.gate)
2. `promote.harvest.green`
3. `promote.dojo.green`
4. `promote.forge.green`
5. `operation.start` (passport build)
6. `operation.end`
7. `promote.passport.issued`

A blocked run emits fewer bundles; the last will be the appropriate `*.blocked` type.

## Revocation

DanteForge hosts a revocation endpoint at `passport.revocation_url`. DanteAgents calls this endpoint at model load time. If revoked, the passport is invalidated regardless of the `expires_at` date.
