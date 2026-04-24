"""Unit tests for dojo/core/evidence.py."""

import hashlib
import pytest

from dojo.core.evidence import (
    EvidenceBundle,
    EvidenceLogger,
    EvidenceType,
    MerkleTree,
    emit_evidence,
    emit_failure,
    get_evidence_logger,
)


@pytest.fixture(autouse=True)
def clear_logger():
    """Reset singleton between every test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# MerkleTree
# ---------------------------------------------------------------------------

class TestMerkleTree:
    def test_empty_root_is_deterministic(self):
        t = MerkleTree()
        assert t.root == hashlib.sha256(b"").hexdigest()

    def test_single_leaf(self):
        t = MerkleTree()
        t.add_leaf("hello")
        # Single leaf: root == sha256(payload) with no further combination
        expected = hashlib.sha256("hello".encode()).hexdigest()
        assert t.root == expected

    def test_two_leaves_deterministic(self):
        t1 = MerkleTree()
        t1.add_leaf("a")
        t1.add_leaf("b")
        t2 = MerkleTree()
        t2.add_leaf("a")
        t2.add_leaf("b")
        assert t1.root == t2.root

    def test_order_matters(self):
        t1 = MerkleTree()
        t1.add_leaf("a")
        t1.add_leaf("b")
        t2 = MerkleTree()
        t2.add_leaf("b")
        t2.add_leaf("a")
        assert t1.root != t2.root


# ---------------------------------------------------------------------------
# EvidenceBundle
# ---------------------------------------------------------------------------

class TestEvidenceBundle:
    def _make(self, evidence: dict) -> EvidenceBundle:
        import json
        normalized = evidence
        h = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        return EvidenceBundle(run_id="r1", seq=0, at="2026-04-22T00:00:00Z", organ="test", evidence=evidence, hash=h, normalized=normalized)

    def test_verify_valid(self):
        b = self._make({"type": "test.event", "data": {"x": 1}})
        assert b.verify() is True

    def test_verify_tampered(self):
        b = self._make({"type": "test.event", "data": {"x": 1}})
        # Forge a bundle with wrong hash
        tampered = EvidenceBundle(run_id="r1", seq=0, at="2026-04-22T00:00:00Z", organ="test", evidence={"type": "tampered"}, hash=b.hash, normalized={"type": "tampered"})
        assert tampered.verify() is False

    def test_to_dict_keys(self):
        b = self._make({"type": "t"})
        d = b.to_dict()
        assert all(k in d for k in ("runId", "seq", "at", "organ", "evidence", "hash", "prevHash"))


# ---------------------------------------------------------------------------
# EvidenceLogger
# ---------------------------------------------------------------------------

class TestEvidenceLogger:
    def test_singleton(self):
        a = EvidenceLogger.get_instance()
        b = EvidenceLogger.get_instance()
        assert a is b

    def test_emit_returns_bundle(self):
        logger = EvidenceLogger.get_instance()
        bundle = logger.emit({"type": "test.event", "data": {}})
        assert isinstance(bundle, EvidenceBundle)
        assert bundle.verify()

    def test_emit_rejects_non_dict(self):
        logger = EvidenceLogger.get_instance()
        with pytest.raises(TypeError):
            logger.emit("not a dict")  # type: ignore[arg-type]

    def test_sequential_ordering(self):
        logger = EvidenceLogger.get_instance()
        b1 = logger.emit({"type": "e1"})
        b2 = logger.emit({"type": "e2"})
        assert b2.seq == b1.seq + 1

    def test_merkle_root_changes_on_emit(self):
        logger = EvidenceLogger.get_instance()
        r1 = logger.compute_merkle_root()
        logger.emit({"type": "any"})
        r2 = logger.compute_merkle_root()
        assert r1 != r2

    def test_info_warn_error_do_not_raise(self):
        logger = EvidenceLogger.get_instance()
        logger.info("message")
        logger.warn("message")
        logger.error("message")
        events = EvidenceLogger.get_events()
        assert len(events) == 3

    def test_info_accepts_dict(self):
        logger = EvidenceLogger.get_instance()
        logger.info({"type": "custom.event", "data": {"k": "v"}})
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "custom.event" for e in events)

    def test_start_operation_context_manager(self):
        logger = EvidenceLogger.get_instance()
        with logger.start_operation("test_op", organ="test"):
            pass
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert "operation.start" in types
        assert "operation.end" in types

    def test_start_operation_on_exception_emits_error(self):
        logger = EvidenceLogger.get_instance()
        try:
            with logger.start_operation("failing_op", organ="test"):
                raise ValueError("boom")
        except ValueError:
            pass
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert "operation.error" in types

    def test_clear_resets_singleton_and_events(self):
        logger = EvidenceLogger.get_instance()
        logger.emit({"type": "x"})
        assert len(EvidenceLogger.get_events()) > 0
        EvidenceLogger.clear()
        assert len(EvidenceLogger.get_events()) == 0
        new_logger = EvidenceLogger.get_instance()
        assert new_logger is not logger

    def test_export_receipt_creates_file(self, tmp_path):
        logger = EvidenceLogger.get_instance()
        logger.emit({"type": "x"})
        path = tmp_path / "receipt.json"
        receipt = logger.export_receipt(path)
        assert path.exists()
        assert "merkle_root" in receipt
        assert "run_id" in receipt

    def test_thread_safety(self):
        import threading
        logger = EvidenceLogger.get_instance()
        errors: list[Exception] = []

        def emit_many() -> None:
            try:
                for _ in range(50):
                    logger.emit({"type": "concurrent"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emit_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(EvidenceLogger.get_events()) == 200


# ---------------------------------------------------------------------------
# EvidenceChain
# ---------------------------------------------------------------------------

class TestEvidenceChain:
    def test_chain_via_logger(self):
        logger = EvidenceLogger.get_instance()
        logger.emit({"type": "e1"})
        logger.emit({"type": "e2"})
        c = logger.chain()
        assert c.size == 2
        assert c.merkle_root != ""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_evidence_logger_returns_singleton(self):
        a = get_evidence_logger()
        b = get_evidence_logger()
        assert a is b

    def test_emit_evidence_shorthand(self):
        b = emit_evidence({"type": "shorthand"})
        assert b.verify()

    def test_emit_failure(self):
        b = emit_failure(ValueError("test error"), context={"op": "test"})
        assert b.verify()


# ---------------------------------------------------------------------------
# EvidenceType registry
# ---------------------------------------------------------------------------

class TestEvidenceType:
    def test_all_values_are_strings(self):
        for et in EvidenceType:
            assert isinstance(et.value, str)

    def test_no_duplicate_values(self):
        values = [et.value for et in EvidenceType]
        assert len(values) == len(set(values)), "Duplicate EvidenceType values"

    def test_key_types_present(self):
        required = [
            EvidenceType.HAL_DETECT_START,
            EvidenceType.TRAINING_START,
            EvidenceType.PACK_IMPORT_START,
            EvidenceType.PROMOTE_PASSPORT_ISSUED,
            EvidenceType.ROUTE_PRIVACY_BLOCKED,
        ]
        for t in required:
            assert t in EvidenceType
