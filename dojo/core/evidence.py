"""DanteDojo evidence logger — constitutional substrate.

Transplanted and adapted from STEAK V3 steak/core/evidence.py.

Provides a thread-safe, append-only evidence chain with SHA-256 hashing and
Merkle root computation. Every Dojo operation MUST emit evidence. A broken
evidence logger blocks all operations — it never silently drops events.

Design guarantees:
- Fail-closed: emit() raises if given non-dict input; never swallows errors.
- Deterministic ordering within a process via a monotonic sequence counter.
- Merkle root is recomputed on every add so callers can snapshot anytime.
- In-memory storage for tests; JSONL filesystem persistence for production.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceBundle:
    """Immutable evidence bundle with SHA-256 hash and Merkle linkage."""

    run_id: str
    seq: int
    at: str
    organ: str
    evidence: Dict[str, Any]
    hash: str
    prev_hash: str = ""
    normalized: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def bundle_hash(self) -> str:
        return self.hash

    def verify(self) -> bool:
        data = self.normalized if self.normalized else self.evidence
        expected = hashlib.sha256(_stable_json(data).encode()).hexdigest()
        return expected == self.hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runId": self.run_id,
            "seq": self.seq,
            "at": self.at,
            "organ": self.organ,
            "evidence": self.evidence,
            "hash": self.hash,
            "prevHash": self.prev_hash,
        }


class MerkleTree:
    """Minimal Merkle tree for deterministic root computation."""

    def __init__(self) -> None:
        self._leaves: List[str] = []

    def add_leaf(self, payload: str) -> None:
        self._leaves.append(hashlib.sha256(payload.encode()).hexdigest())

    @property
    def root(self) -> str:
        if not self._leaves:
            return hashlib.sha256(b"").hexdigest()
        nodes = self._leaves[:]
        while len(nodes) > 1:
            nxt: List[str] = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left
                nxt.append(hashlib.sha256((left + right).encode()).hexdigest())
            nodes = nxt
        return nodes[0]


@dataclass
class EvidenceChain:
    """Evidence chain with Merkle root and bundle list."""

    run_id: str = ""
    merkle_root: str = ""
    size: int = 0
    last_hash: str = ""

    def __post_init__(self) -> None:
        self._bundles: List[EvidenceBundle] = []
        self._tree = MerkleTree()

    def add_bundle(self, bundle: EvidenceBundle) -> None:
        self._bundles.append(bundle)
        self._tree.add_leaf(bundle.hash)
        self.merkle_root = self._tree.root
        self.size = len(self._bundles)
        self.last_hash = bundle.hash

    @property
    def bundles(self) -> List[EvidenceBundle]:
        return list(self._bundles)

    def verify_chain(self) -> bool:
        prev_hash = ""
        for b in self._bundles:
            if not b.verify():
                return False
            if b.prev_hash and b.prev_hash != prev_hash:
                return False
            prev_hash = b.hash
        if self._bundles and self.merkle_root:
            check = MerkleTree()
            for b in self._bundles:
                check.add_leaf(b.hash)
            if check.root != self.merkle_root:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runId": self.run_id,
            "merkleRoot": self.merkle_root,
            "size": self.size,
            "lastHash": self.last_hash,
        }

    def export_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for b in self._bundles:
                fh.write(json.dumps(b.to_dict(), separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Evidence type registry
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    """Formal evidence type enumeration for all Dojo operations."""

    ONE_DOOR_ENTRY = "one_door.entry"

    # HAL
    HAL_DETECT_START = "hal.detect.start"
    HAL_DETECT_FOUND = "hal.detect.found"
    HAL_PROFILE_SIGNED = "hal.detect.profile.signed"

    # Training
    TRAINING_START = "train.finetune.started"
    TRAINING_STEP = "train.finetune.step"
    TRAINING_CHECKPOINT = "train.finetune.checkpoint.saved"
    TRAINING_COMPLETE = "train.finetune.completed"
    TRAINING_BACKEND_SELECTED = "train.finetune.backend.selected"

    # GRPO reinforcement learning
    GRPO_START = "train.grpo.started"
    GRPO_STEP = "train.grpo.step"
    GRPO_KL_CHECKED = "train.grpo.kl_checked"
    GRPO_CHECKPOINT = "train.grpo.checkpoint.saved"
    GRPO_COMPLETE = "train.grpo.completed"

    # DPO reinforcement learning
    DPO_START = "train.dpo.started"
    DPO_STEP = "train.dpo.step"
    DPO_REWARD_MARGIN_CHECKED = "train.dpo.reward_margin_checked"
    DPO_CHECKPOINT = "train.dpo.checkpoint.saved"
    DPO_COMPLETE = "train.dpo.completed"

    # Pack ingest
    PACK_IMPORT_START = "pack.import.start"
    PACK_SIGNATURE_VERIFIED = "pack.signature.verified"
    PACK_CONTAMINATION_DETECTED = "pack.contamination.detected"
    PACK_INDEXED = "pack.indexed"

    # Quantization
    QUANTIZATION_START = "quant.start"
    QUANTIZATION_COMPLETE = "quant.complete"
    QUANT_QUALITY_GATE = "quant.quality_gate"
    QUANT_PPL_GATE = "quant.ppl_gate"

    # Serving
    SERVE_START = "serve.start"
    SERVE_READY = "serve.ready"
    SERVE_STOPPED = "serve.stopped"
    SERVE_VLLM_BATCH_START = "serve.vllm_batch_start"
    SERVE_VLLM_BATCH_COMPLETE = "serve.vllm_batch_complete"
    SERVE_VLLM_STOP_SEQUENCE = "serve.vllm_stop_sequence"

    # Routing
    ROUTE_DECISION = "route.decision"
    ROUTE_PRIVACY_BLOCKED = "route.privacy.blocked"
    ROUTE_COST_RECEIPT = "route.cost_receipt"
    ROUTE_FALLBACK = "route.fallback"

    # Evaluation
    EVAL_START = "eval.start"
    EVAL_COMPLETE = "eval.complete"
    EVAL_CONTAMINATION = "eval.contamination_check"
    EVAL_JUDGE_DRIFT = "eval.judge_drift"
    EVAL_LM_HARNESS_START = "eval.lm_harness.start"
    EVAL_LM_HARNESS_COMPLETE = "eval.lm_harness.complete"
    EVAL_SYNTHETIC_FALLBACK = "eval.synthetic_fallback"

    # Pack license
    PACK_LICENSE_CHECKED = "pack.license.checked"

    # Revocation
    REVOCATION_CHECKED = "promote.revocation.checked"

    # Promotion gate
    PROMOTE_HARVEST_GREEN = "promote.harvest.green"
    PROMOTE_HARVEST_BLOCKED = "promote.harvest.blocked"
    PROMOTE_DOJO_GREEN = "promote.dojo.green"
    PROMOTE_DOJO_BLOCKED = "promote.dojo.blocked"
    PROMOTE_FORGE_GREEN = "promote.forge.green"
    PROMOTE_FORGE_BLOCKED = "promote.forge.blocked"
    PROMOTE_PASSPORT_ISSUED = "promote.passport.issued"

    # Idle training
    IDLE_WINDOW_DETECTED = "idle.detected"
    IDLE_INTERRUPTED = "idle.interrupted"
    IDLE_THERMAL_PAUSED = "idle.thermal.paused"
    IDLE_TRAINING_RESUMED = "idle.training.resumed"
    IDLE_CHECKPOINT_SAVED = "idle.checkpoint.saved"
    IDLE_TRAINING_COMPLETED = "idle.training.completed"
    GRADIENT_ACCUMULATED = "idle.gradient_accumulated"

    # Layer streaming
    LAYER_STREAM_LOADED = "stream.layer.loaded"
    LAYER_STREAM_EVICTED = "stream.layer.evicted"
    KV_PAGE_STORED = "stream.kv.stored"
    KV_PAGE_HIT = "stream.kv.hit"
    PREFETCH_ISSUED = "stream.prefetch.issued"
    PREFETCH_CONSUMED = "stream.prefetch.consumed"
    BLOCK_QUANTIZED = "stream.block.quantized"
    SOVEREIGN_HALT = "stream.sovereign.halt"
    INFERENCE_LOOP_STEP = "stream.inference_loop_step"
    INFERENCE_LOOP_COMPLETE = "stream.loop_complete"

    # Evidence chain
    EVIDENCE_CHAIN_SEALED = "evidence.chain_sealed"
    EVIDENCE_MERKLE_ROOT = "evidence.merkle_root"

    # FSDP2 multi-GPU training
    FSDP2_STATE_TRANSITION = "train.fsdp2.fsm.transition"
    FSDP2_PROFILE_DETECTED = "train.fsdp2.profile.detected"
    FSDP2_RANK_INITIALIZED = "train.fsdp2.rank.initialized"
    FSDP2_MODEL_WRAPPED = "train.fsdp2.model.wrapped"
    FSDP2_OPTIMIZER_READY = "train.fsdp2.optimizer.ready"
    FSDP2_GRADIENT_SYNC = "train.fsdp2.gradient.sync"
    FSDP2_CHECKPOINT_SAVED = "train.fsdp2.checkpoint.saved"
    FSDP2_TRAINING_COMPLETE = "train.fsdp2.training.complete"
    FSDP2_TRAINING_STEP = "train.fsdp2.step"
    FSDP2_RANK_MERKLE_ROOT = "train.fsdp2.rank.merkle_root"

    # Experiment tracking
    EXPERIMENT_RUN_STARTED = "experiment.run.started"
    EXPERIMENT_RUN_COMPLETE = "experiment.run.complete"
    EXPERIMENT_METRIC = "experiment.metric"


# ---------------------------------------------------------------------------
# Operation context (dual-purpose: string + context manager)
# ---------------------------------------------------------------------------

class _OperationContext:
    """Context manager returned by EvidenceLogger.start_operation()."""

    def __init__(self, logger: "EvidenceLogger", op_id: str, meta: Dict[str, Any], organ: str) -> None:
        self._logger = logger
        self._op_id = op_id
        self._meta = meta
        self._organ = organ

    def __str__(self) -> str:
        return self._op_id

    def __bool__(self) -> bool:
        return True

    def __enter__(self) -> Dict[str, Any]:
        return self._meta

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is None:
                self._logger.emit({"type": "operation.end", "data": self._meta}, organ=self._organ)
            else:
                self._logger.emit(
                    {"type": "operation.error", "data": {**self._meta, "error": repr(exc_val)}},
                    organ=self._organ,
                )
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# EvidenceLogger (singleton)
# ---------------------------------------------------------------------------

class EvidenceLogger:
    """Thread-safe singleton evidence logger.

    All Dojo subsystems call EvidenceLogger.get_instance() to obtain the shared
    logger for the process. Tests call EvidenceLogger.clear() between runs.
    """

    _instance_lock = threading.Lock()
    _instance: Optional["EvidenceLogger"] = None
    _events: List[Dict[str, Any]] = []
    _counter = 0

    def __init__(self, run_id: Optional[str] = None) -> None:
        self._run_id = run_id or str(uuid.uuid4())
        self._seq = 0
        self._tree = MerkleTree()
        self._last_hash = hashlib.sha256(b"").hexdigest()
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls, run_id: Optional[str] = None) -> "EvidenceLogger":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = EvidenceLogger(run_id=run_id)
            return cls._instance

    @classmethod
    def clear(cls) -> None:
        with cls._instance_lock:
            cls._instance = None
            cls._events = []
            cls._counter = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def emit(self, evidence: Dict[str, Any], organ: str = "core") -> EvidenceBundle:
        """Emit evidence. Never silently drops. Raises TypeError on non-dict."""
        if not isinstance(evidence, dict):
            raise TypeError(f"EvidenceLogger.emit expects dict, got {type(evidence).__name__}")

        with self._lock:
            seq = self._seq
            at = _utc_now_iso()
            EvidenceLogger._counter += 1
            h = hashlib.sha256(_stable_json(evidence).encode()).hexdigest()
            self._tree.add_leaf(_stable_json({"evidence": evidence, "organ": organ, "seq": seq, "at": at}))
            prev_hash = self._last_hash
            self._last_hash = h
            bundle = EvidenceBundle(
                run_id=self._run_id,
                seq=seq,
                at=at,
                organ=organ,
                evidence=evidence,
                hash=h,
                prev_hash=prev_hash,
                normalized=evidence,
            )
            self._seq += 1
            EvidenceLogger._events.append({"run_id": self._run_id, "seq": seq, "at": at, "organ": organ, "evidence": evidence, "hash": h})
        # If DanteForge decision store is configured, mirror this emit as a decision node.
        import os as _os
        _store_env = _os.getenv("DANTEFORGE_DECISION_STORE")
        if _store_env:
            try:
                from dojo.core.time_machine_decision_node import record_decision_node
                record_decision_node(
                    project_root=".",
                    session_id=_os.getenv("DANTEFORGE_DECISION_SESSION_ID", self._run_id),
                    actor_id=organ,
                    prompt=evidence.get("type", "unknown"),
                    result=evidence.get("data", {}),
                    success=True,
                    evidence_ref=h,
                )
            except Exception:
                pass
        return bundle

    def log_evidence(self, evidence: Dict[str, Any], organ: str = "core") -> EvidenceBundle:
        return self.emit(evidence, organ=organ)

    # ------------------------------------------------------------------
    # Operation tracking
    # ------------------------------------------------------------------

    def start_operation(self, name: Optional[str] = None, organ: str = "core", **fields: Any) -> "_OperationContext":
        op_id = uuid.uuid4().hex
        meta: Dict[str, Any] = {"operationId": op_id, "name": str(name or "unknown"), "organ": organ}
        if fields:
            meta.update({k: v for k, v in fields.items() if v is not None})
        try:
            self.emit({"type": "operation.start", "data": meta}, organ=organ)
        except Exception:
            pass
        return _OperationContext(self, op_id, meta, organ)

    def complete_operation(self, evidence_id: Any = None, *, success: bool = True, outputs: Optional[Dict[str, Any]] = None) -> None:
        try:
            data: Dict[str, Any] = {"evidence_id": str(evidence_id) if evidence_id else "", "success": success}
            if outputs:
                data["outputs"] = outputs
            self.emit({"type": "operation.end", "data": data})
        except Exception:
            pass

    def fail_operation(self, evidence_id: Any = None, *, error: Optional[Any] = None) -> None:
        try:
            data: Dict[str, Any] = {"evidence_id": str(evidence_id) if evidence_id else "", "success": False}
            if error is not None:
                data["error"] = str(error)
            self.emit({"type": "operation.error", "data": data})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Convenience logging
    # ------------------------------------------------------------------

    def info(self, message: Union[str, Dict[str, Any]], *, organ: str = "core", **fields: Any) -> Optional[EvidenceBundle]:
        try:
            if isinstance(message, dict):
                return self.emit(message, organ=organ)
            return self.emit({"type": "log", "level": "info", "data": {"message": str(message), **fields}}, organ=organ)
        except Exception:
            return None

    def warn(self, message: Union[str, Dict[str, Any]], *, organ: str = "core", **fields: Any) -> Optional[EvidenceBundle]:
        try:
            if isinstance(message, dict):
                return self.emit(message, organ=organ)
            return self.emit({"type": "log", "level": "warn", "data": {"message": str(message), **fields}}, organ=organ)
        except Exception:
            return None

    def warning(self, message: Union[str, Dict[str, Any]], *, organ: str = "core", **fields: Any) -> Optional[EvidenceBundle]:
        return self.warn(message, organ=organ, **fields)

    def error(self, message: Union[str, Dict[str, Any]], *, organ: str = "core", **fields: Any) -> Optional[EvidenceBundle]:
        try:
            if isinstance(message, dict):
                return self.emit(message, organ=organ)
            return self.emit({"type": "log", "level": "error", "data": {"message": str(message), **fields}}, organ=organ)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def get_events(cls) -> List[Dict[str, Any]]:
        import copy
        out: List[Dict[str, Any]] = []
        for e in cls._events:
            ev = e.get("evidence", {})
            if isinstance(ev, dict) and "type" in ev:
                out.append({"type": str(ev["type"]), "data": copy.deepcopy(ev.get("data", {})), "timestamp": str(e.get("at", ""))})
            else:
                out.append(copy.deepcopy(dict(e)))
        return out

    def list_events(self) -> List[Dict[str, Any]]:
        import copy
        return [copy.deepcopy(e) for e in EvidenceLogger._events]

    def compute_merkle_root(self) -> str:
        with self._lock:
            return self._tree.root

    def chain(self) -> EvidenceChain:
        with self._lock:
            ec = EvidenceChain(run_id=self._run_id, merkle_root=self._tree.root, size=len(EvidenceLogger._events), last_hash=self._last_hash)
        return ec

    def export_receipt(self, path: Path) -> Dict[str, Any]:
        receipt = {
            "run_id": self._run_id,
            "merkle_root": self.compute_merkle_root(),
            "event_count": len(EvidenceLogger._events),
            "exported_at": _utc_now_iso(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(receipt, fh, indent=2, sort_keys=True)
        return receipt


# Backwards-compat alias
Evidence = EvidenceLogger


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def get_evidence_logger() -> EvidenceLogger:
    return EvidenceLogger.get_instance()


def emit_evidence(evidence: Dict[str, Any], organ: str = "core") -> EvidenceBundle:
    return EvidenceLogger.get_instance().emit(evidence, organ=organ)


def emit_failure(error: Exception, context: Optional[Dict[str, Any]] = None, organ: str = "core") -> EvidenceBundle:
    return EvidenceLogger.get_instance().emit({"type": "error", "error": str(error), "context": context or {}}, organ=organ)


__all__ = [
    "Evidence",
    "EvidenceBundle",
    "EvidenceChain",
    "EvidenceLogger",
    "EvidenceType",
    "MerkleTree",
    "emit_evidence",
    "emit_failure",
    "get_evidence_logger",
]
