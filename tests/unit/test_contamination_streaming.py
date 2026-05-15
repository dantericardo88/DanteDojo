"""Unit tests for dojo/eval/contamination.py — d21 streaming contamination detection."""

from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import EvalContaminationError
from dojo.eval.contamination import (
    ContaminationReport,
    hash_corpus_sample,
    stream_check_contamination,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _make_gold(texts):
    """Return a frozenset of hashes for all supplied texts (sample_rate=1.0)."""
    return hash_corpus_sample(texts, sample_rate=1.0, seed=0)


# ===========================================================================
# ContaminationReport tests (5)
# ===========================================================================

def test_clean_corpus_report_is_clean():
    gold = _make_gold(["alpha", "beta"])
    report = stream_check_contamination(iter(["gamma", "delta"]), gold, seed=0)
    assert report.clean is True
    assert report.contaminated_count == 0


def test_contaminated_corpus_raises_error():
    gold = _make_gold(["secret eval text"])
    with pytest.raises(EvalContaminationError):
        stream_check_contamination(iter(["secret eval text", "other"]), gold, seed=0)


def test_sample_rate_reduces_checked_count():
    texts = [f"record_{i}" for i in range(200)]
    gold: frozenset = frozenset()  # nothing contaminates — just count
    report = stream_check_contamination(iter(texts), gold, sample_rate=0.1, seed=42)
    # With sample_rate=0.1 over 200 records we expect ~20, definitely < 200.
    assert report.checked_count < 200
    assert report.corpus_size_hint == 200


def test_contamination_report_has_checked_count():
    gold: frozenset = frozenset()
    texts = ["a", "b", "c", "d", "e"]
    report = stream_check_contamination(iter(texts), gold, sample_rate=1.0, seed=0)
    assert report.checked_count == 5
    assert report.corpus_size_hint == 5


def test_contamination_evidence_emitted():
    gold: frozenset = frozenset()
    stream_check_contamination(iter(["x", "y"]), gold, sample_rate=1.0, seed=0)
    events = EvidenceLogger.get_events()
    types = [e["type"] for e in events]
    assert EvidenceType.EVAL_CONTAMINATION.value in types


# ===========================================================================
# hash_corpus_sample tests (5)
# ===========================================================================

def test_hash_corpus_sample_returns_frozenset():
    result = hash_corpus_sample(["hello", "world"], sample_rate=1.0)
    assert isinstance(result, frozenset)


def test_hash_corpus_sample_deterministic():
    texts = ["foo", "bar", "baz"]
    a = hash_corpus_sample(texts, sample_rate=0.5, seed=7)
    b = hash_corpus_sample(texts, sample_rate=0.5, seed=7)
    assert a == b


def test_hash_corpus_sample_sample_rate_1_hashes_all():
    texts = ["one", "two", "three"]
    result = hash_corpus_sample(texts, sample_rate=1.0, seed=0)
    assert len(result) == 3


def test_hash_corpus_sample_sample_rate_0_returns_empty():
    texts = ["one", "two", "three"]
    result = hash_corpus_sample(texts, sample_rate=0.0, seed=0)
    assert result == frozenset()


def test_hash_corpus_sample_normalizes_whitespace():
    # "hello   world" and "hello world" should hash to the same value.
    a = hash_corpus_sample(["hello   world"], sample_rate=1.0)
    b = hash_corpus_sample(["hello world"], sample_rate=1.0)
    assert a == b


# ===========================================================================
# stream_check tests (5)
# ===========================================================================

def test_stream_check_clean_data_no_error():
    gold = _make_gold(["eval gold"])
    report = stream_check_contamination(
        iter(["train sample 1", "train sample 2"]),
        gold,
        sample_rate=1.0,
        seed=0,
    )
    assert report.clean is True


def test_stream_check_raises_on_contaminated_item():
    poison = "this text is in the eval set"
    gold = _make_gold([poison])
    with pytest.raises(EvalContaminationError) as exc_info:
        stream_check_contamination(iter([poison, "harmless"]), gold, sample_rate=1.0, seed=0)
    assert "Contamination detected" in str(exc_info.value)


def test_stream_check_chunk_size_respected():
    """Processing 50 records with chunk_size=10 must still find all contaminations."""
    poison = "poisoned record"
    gold = _make_gold([poison])
    records = ["clean"] * 25 + [poison] + ["clean"] * 24
    with pytest.raises(EvalContaminationError):
        stream_check_contamination(iter(records), gold, sample_rate=1.0, chunk_size=10, seed=0)


def test_stream_check_sample_rate_half():
    """With sample_rate=0.5 and a deterministic seed, checked_count ≈ half."""
    gold: frozenset = frozenset()
    texts = [f"item_{i}" for i in range(100)]
    report = stream_check_contamination(iter(texts), gold, sample_rate=0.5, seed=0)
    # Expect roughly 50; allow generous range to avoid flakiness.
    assert 20 <= report.checked_count <= 80
    assert report.corpus_size_hint == 100


def test_stream_check_empty_iterator_no_error():
    gold = _make_gold(["anything"])
    report = stream_check_contamination(iter([]), gold, sample_rate=1.0, seed=0)
    assert report.clean is True
    assert report.checked_count == 0
    assert report.corpus_size_hint == 0
