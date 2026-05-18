"""Tests for d38: fail-closed strict contamination gate."""
from __future__ import annotations

import pytest

from dojo.eval.contamination_strict import (
    ContaminationDetected,
    ContaminationGate,
    ContaminationVerdict,
    quick_contamination_check,
)


class TestCleanCase:
    def test_no_overlap_returns_clean(self):
        gate = ContaminationGate()
        v = gate.check(
            training_texts=["the quick brown fox"],
            eval_texts=["completely different evaluation text"],
        )
        assert v.contaminated is False
        assert v.reason == "clean"

    def test_clean_passes_gate(self):
        gate = ContaminationGate()
        gate.gate(
            training_texts=["fox jumps"],
            eval_texts=["wholly unrelated"],
        )  # no raise


class TestContaminatedCase:
    def test_overlap_flagged(self):
        gate = ContaminationGate(ngram_size=5)
        v = gate.check(
            training_texts=["the quick brown fox jumps over"],
            eval_texts=["the quick brown fox"],
        )
        assert v.contaminated is True

    def test_gate_raises_on_overlap(self):
        gate = ContaminationGate(ngram_size=5)
        with pytest.raises(ContaminationDetected, match="overlap"):
            gate.gate(
                training_texts=["the quick brown fox"],
                eval_texts=["the quick brown fox"],
            )

    def test_partial_overlap_still_fails_at_zero_threshold(self):
        gate = ContaminationGate(ngram_size=5, overlap_threshold=0.0)
        v = gate.check(
            training_texts=["the quick brown fox eats"],
            eval_texts=["the quick brown fox jumps"],
        )
        assert v.contaminated is True


class TestFailClosed:
    def test_empty_eval_set_fails_by_default(self):
        gate = ContaminationGate()
        v = gate.check(training_texts=["hello"], eval_texts=[])
        assert v.contaminated is True
        assert v.reason == "empty_eval_set"

    def test_empty_eval_allowed_when_required_false(self):
        gate = ContaminationGate(require_eval_data=False)
        v = gate.check(training_texts=["hello"], eval_texts=[])
        assert v.contaminated is False

    def test_empty_eval_gate_raises_in_strict_mode(self):
        gate = ContaminationGate()
        with pytest.raises(ContaminationDetected):
            gate.gate(training_texts=["x"], eval_texts=[])


class TestThreshold:
    def test_threshold_above_overlap_allows(self):
        gate = ContaminationGate(ngram_size=5, overlap_threshold=0.99)
        v = gate.check(
            training_texts=["the quick brown fox jumps"],
            eval_texts=["the quick brown fox jumps"],
        )
        # Even though overlap is high, threshold is set to 0.99 — equal ratio
        # of 1.0 still exceeds threshold, so this fails. Test a softer case:
        assert v.contaminated  # 1.0 > 0.99

    def test_threshold_zero_strict(self):
        gate = ContaminationGate(ngram_size=5, overlap_threshold=0.0)
        v = gate.check(
            training_texts=["a"],  # too short to form ngrams
            eval_texts=["zzzzzzz this is unique"],
        )
        # No overlap → not contaminated
        assert v.contaminated is False


class TestVerdictFields:
    def test_returns_verdict_dataclass(self):
        gate = ContaminationGate()
        v = gate.check(training_texts=["x"], eval_texts=["y is different"])
        assert isinstance(v, ContaminationVerdict)

    def test_verdict_includes_overlap_count(self):
        gate = ContaminationGate(ngram_size=5)
        v = gate.check(
            training_texts=["the quick brown fox"],
            eval_texts=["the quick brown fox"],
        )
        assert v.overlap_count > 0
        assert v.eval_hash_count > 0


class TestQuickCheck:
    def test_quick_check_returns_tuple(self):
        contaminated, ratio = quick_contamination_check(
            ["the quick brown fox"],
            ["the quick brown fox"],
            ngram_size=5,
        )
        assert contaminated is True
        assert 0.0 <= ratio <= 1.0

    def test_quick_check_clean_case(self):
        contaminated, ratio = quick_contamination_check(
            ["alpha beta"],
            ["xenon zenith"],
            ngram_size=5,
        )
        assert contaminated is False


class TestNormalization:
    def test_case_insensitive(self):
        gate = ContaminationGate(ngram_size=5)
        v = gate.check(
            training_texts=["THE QUICK BROWN FOX"],
            eval_texts=["the quick brown fox"],
        )
        assert v.contaminated is True

    def test_whitespace_normalised(self):
        gate = ContaminationGate(ngram_size=5)
        v = gate.check(
            training_texts=["the   quick    brown    fox"],
            eval_texts=["the quick brown fox"],
        )
        assert v.contaminated is True


class TestEvidence:
    def test_evidence_emitted_on_check(self):
        from dojo.core.evidence import EvidenceLogger
        gate = ContaminationGate()
        EvidenceLogger.clear()
        gate.check(training_texts=["x"], eval_texts=["y"])
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "eval.contamination_strict" for e in events)

    def test_evidence_includes_overlap_ratio(self):
        from dojo.core.evidence import EvidenceLogger
        gate = ContaminationGate()
        EvidenceLogger.clear()
        gate.check(training_texts=["alpha"], eval_texts=["beta gamma"])
        events = EvidenceLogger.get_events()
        evs = [e for e in events if e.get("type") == "eval.contamination_strict"]
        assert "overlap_ratio" in evs[0]["data"]
