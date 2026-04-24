# DanteDojo Constitutional Rules

These rules are non-negotiable. Every subsystem, every commit, every model load
must comply. No exceptions, no silent waivers.

---

## 1. One-Door Doctrine

Each subsystem exposes exactly one public entry point. All external callers must
use that door. Internal helpers are private. No subsystem may import from another
subsystem's internal modules — only from its public door. Violations break the
module boundary and are rejected at code review.

## 2. Evidence Emission

Every operation that mutates state or makes an external call MUST emit a signed
evidence receipt. Receipts are SHA-256 hashed and Merkle-chained to the previous
receipt in the session. The chain is append-only and must be verifiable offline.
No operation completes without a committed receipt.

## 3. Fail-Closed on Missing Preconditions

If a required precondition is absent (missing config key, absent GPU, absent
model file, uninitialized registry entry), the subsystem raises immediately with
a typed exception. It does NOT degrade silently, fall back to a default, or
proceed in a degraded mode unless the caller explicitly opts in with a documented
override flag. Silence equals safety failure.

## 4. Explicit FSM for Non-Trivial State

Any process with more than two distinct lifecycle phases MUST be modeled as an
explicit finite state machine with a named enum for states and named enum for
valid transitions. Undeclared state transitions are programming errors and are
raised as such. No spaghetti lifecycle management.

## 5. Deterministic Replayability

Given an evidence receipt chain, any operation must be exactly reproducible. This
means: random seeds are captured in receipts, external inputs are logged before
use, and all branching conditions are recorded. The `dojo replay <receipt-id>`
command must reconstruct the operation's outputs without side effects.

## 6. KiloCode Discipline

No source file exceeds 500 lines of code (LOC), excluding blank lines and
comments. Files approaching the limit must be split along natural responsibility
boundaries. This is enforced by CI and is a hard block on merge.

## 7. No Stubs in Shipped Code

`TODO`, `FIXME`, `NotImplementedError`, `raise NotImplementedError`, and `pass`
used as a body in non-abstract methods are forbidden in any code that reaches the
`main` or `dev` branch. Placeholder implementations must be replaced before
promotion. Abstract base classes may use `...` as a body only when decorated with
`@abstractmethod`.

## 8. Privacy-Tier Enforcement

Data classified as CONFIDENTIAL (PII, customer data, internal weights, API keys)
must never be routed to any external API, logging sink, or third-party service.
The router layer checks privacy tier before dispatch. Routing CONFIDENTIAL data
externally is an unrecoverable constitutional violation — the operation is aborted
and flagged in the audit log.

## 9. Three-Way Promotion Gate

A model checkpoint or code change may only be promoted to the next stage when all
three gates are simultaneously GREEN:

- **Quality Gate**: lm-eval benchmark score meets or exceeds the declared
  threshold for the target tier.
- **Efficiency Gate**: tokens/second and memory footprint meet or beat the
  baseline for the hardware target.
- **Safety Gate**: constitutional patch validator passes with zero violations.

Any single RED gate blocks promotion. There is no override path that bypasses all
three gates.

## 10. Constitutional Passport

Before any Agent loads, fine-tunes, quantizes, or serves a model, it must obtain
a Constitutional Passport from `dojo.core.passport`. The passport certifies that
the model has passed the three-way promotion gate, the privacy-tier check, and the
hardware precondition check for the target device. An Agent that proceeds without
a valid passport raises `ConstitutionalViolation` and halts. Passports are
time-bounded and tied to the evidence chain of the issuing promotion run.

---

*Version: 0.1.0-dev — Phase 0 Constitutional Substrate*
*Last updated: 2026-04-22*
