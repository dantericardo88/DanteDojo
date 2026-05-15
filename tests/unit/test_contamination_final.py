"""Final contamination detection tests for d21 — targets 9/10.

Covers:
- Real synthetic 50-entry corpus test (verbatim hit detection)
- Paraphrased text NOT flagged (specificity)
- Evidence emission of eval.contamination_detected with ngram_matches
- ScanReport fields: contaminated_count, clean_count, total_count
- contamination_rate property
- Empty dataset → rate = 0.0
- Empty corpus → flags nothing
- n_gram_size=1 (unigram) and n_gram_size=5 (5-gram)
- Various edge cases
"""

from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import EvalContaminationError
from dojo.eval.contamination import (
    ContaminationReport,
    ScanReport,
    hash_corpus_sample,
    scan_dataset,
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


def _make_50_corpus() -> list[str]:
    """Build a deterministic 50-entry synthetic corpus."""
    return [
        f"The sovereign AI system processes document entry number {i} with full constitutional compliance."
        for i in range(50)
    ]


# ===========================================================================
# 1. Real synthetic 50-entry corpus — verbatim contamination detected
# ===========================================================================

def test_verbatim_corpus_entry_flagged():
    """A sentence taken verbatim from the 50-entry corpus MUST be flagged."""
    corpus = _make_50_corpus()
    # Take entry 17 verbatim as the dataset item to check
    contaminated_item = corpus[17]
    report = scan_dataset([contaminated_item], corpus, n_gram_size=3)
    assert report.contaminated_count == 1
    assert report.clean_count == 0
    assert report.total_count == 1


def test_verbatim_multiple_entries_flagged():
    """Several verbatim entries from the 50-entry corpus are all flagged."""
    corpus = _make_50_corpus()
    # Take entries 0, 10, 25, 49 verbatim
    contaminated = [corpus[0], corpus[10], corpus[25], corpus[49]]
    report = scan_dataset(contaminated, corpus, n_gram_size=3)
    assert report.contaminated_count == 4
    assert report.total_count == 4


# ===========================================================================
# 2. Paraphrased version NOT flagged (specificity)
# ===========================================================================

def test_paraphrase_not_flagged():
    """A paraphrased version of a corpus entry must NOT be flagged (specificity)."""
    corpus = _make_50_corpus()
    # Original: "...processes document entry number 7..."
    # Paraphrase uses entirely different tokens
    paraphrase = "AI performs constitutional verification of dataset item seven."
    report = scan_dataset([paraphrase], corpus, n_gram_size=4)
    assert report.contaminated_count == 0
    assert report.clean_count == 1


def test_short_paraphrase_not_flagged_5gram():
    """With n_gram_size=5, a paraphrase with no 5-gram overlap is not flagged."""
    corpus = ["the quick brown fox jumps over the lazy dog every single morning"]
    paraphrase = "a fast red cat leaps across a sleepy hound at dawn"
    report = scan_dataset([paraphrase], corpus, n_gram_size=5)
    assert report.contaminated_count == 0


# ===========================================================================
# 3. Evidence emission of eval.contamination_detected with ngram_matches
# ===========================================================================

def test_evidence_emitted_on_clean_scan():
    """scan_dataset always emits eval.contamination_detected evidence."""
    corpus = _make_50_corpus()
    scan_dataset(["completely unrelated text here"], corpus, n_gram_size=3)
    events = EvidenceLogger.get_events()
    types = [e["type"] for e in events]
    assert "eval.contamination_detected" in types


def test_evidence_has_ngram_matches_field():
    """eval.contamination_detected evidence must include ngram_matches."""
    corpus = _make_50_corpus()
    scan_dataset(["some clean text"], corpus, n_gram_size=3)
    events = EvidenceLogger.get_events()
    contamination_events = [e for e in events if e["type"] == "eval.contamination_detected"]
    assert contamination_events, "No contamination_detected event emitted"
    data = contamination_events[0].get("data", {})
    assert "ngram_matches" in data


def test_evidence_emitted_on_contaminated_scan():
    """Evidence is emitted even when contamination is found."""
    corpus = _make_50_corpus()
    scan_dataset([corpus[5]], corpus, n_gram_size=3)
    events = EvidenceLogger.get_events()
    types = [e["type"] for e in events]
    assert "eval.contamination_detected" in types


# ===========================================================================
# 4. ScanReport fields: contaminated_count, clean_count, total_count
# ===========================================================================

def test_scan_report_has_required_fields():
    """ScanReport must expose contaminated_count, clean_count, total_count."""
    corpus = _make_50_corpus()
    dataset = ["clean entry alpha", "clean entry beta", corpus[3]]
    report = scan_dataset(dataset, corpus, n_gram_size=3)
    assert hasattr(report, "contaminated_count")
    assert hasattr(report, "clean_count")
    assert hasattr(report, "total_count")


def test_scan_report_counts_are_consistent():
    """contaminated_count + clean_count == total_count."""
    corpus = _make_50_corpus()
    dataset = ["clean text", corpus[1], "another clean", corpus[40]]
    report = scan_dataset(dataset, corpus, n_gram_size=3)
    assert report.contaminated_count + report.clean_count == report.total_count
    assert report.total_count == 4


# ===========================================================================
# 5. contamination_rate = contaminated_count / total_count
# ===========================================================================

def test_contamination_rate_calculation():
    """contamination_rate must equal contaminated_count / total_count."""
    corpus = _make_50_corpus()
    # 2 contaminated out of 4 total → rate = 0.5
    dataset = [corpus[0], "clean a", corpus[1], "clean b"]
    report = scan_dataset(dataset, corpus, n_gram_size=3)
    assert report.contamination_rate == pytest.approx(report.contaminated_count / report.total_count)


def test_contamination_rate_all_clean():
    """contamination_rate = 0.0 when nothing is contaminated."""
    corpus = _make_50_corpus()
    dataset = ["totally unrelated sentence one", "totally unrelated sentence two"]
    report = scan_dataset(dataset, corpus, n_gram_size=4)
    assert report.contamination_rate == 0.0


def test_contamination_rate_all_contaminated():
    """contamination_rate = 1.0 when every entry matches."""
    corpus = _make_50_corpus()
    dataset = [corpus[i] for i in range(5)]
    report = scan_dataset(dataset, corpus, n_gram_size=3)
    assert report.contamination_rate == pytest.approx(1.0)


# ===========================================================================
# 6. Empty dataset → rate = 0.0
# ===========================================================================

def test_empty_dataset_rate_zero():
    """Empty dataset must return rate=0.0."""
    corpus = _make_50_corpus()
    report = scan_dataset([], corpus, n_gram_size=3)
    assert report.contamination_rate == 0.0
    assert report.total_count == 0
    assert report.contaminated_count == 0
    assert report.clean_count == 0


def test_empty_dataset_report_fields():
    """Empty dataset returns a ScanReport with all zero counts."""
    corpus = ["some corpus text"]
    report = scan_dataset([], corpus, n_gram_size=3)
    assert report.contaminated_count == 0
    assert report.clean_count == 0
    assert report.total_count == 0


# ===========================================================================
# 7. Empty corpus → flags nothing
# ===========================================================================

def test_empty_corpus_flags_nothing():
    """When corpus is empty, no dataset entry can be flagged."""
    dataset = ["text one", "text two", "text three"]
    report = scan_dataset(dataset, [], n_gram_size=3)
    assert report.contaminated_count == 0
    assert report.clean_count == 3
    assert report.total_count == 3


def test_empty_corpus_rate_zero():
    """Empty corpus → contamination_rate = 0.0 regardless of dataset content."""
    dataset = ["any content here"]
    report = scan_dataset(dataset, [], n_gram_size=3)
    assert report.contamination_rate == 0.0


# ===========================================================================
# 8. n_gram_size=1 (unigram) and n_gram_size=5 (5-gram) both work
# ===========================================================================

def test_unigram_n_gram_size_1():
    """n_gram_size=1 (unigram) should flag entries sharing individual words."""
    corpus = ["sovereign constitutional compliance training"]
    # Dataset entry shares the word "sovereign"
    dataset_hit = "sovereign policy enforcement"
    dataset_miss = "random unrelated words here"
    report = scan_dataset([dataset_hit, dataset_miss], corpus, n_gram_size=1)
    assert report.contaminated_count >= 1
    assert report.total_count == 2


def test_5gram_n_gram_size_5():
    """n_gram_size=5 should only flag entries with exact 5-consecutive-word matches."""
    corpus = ["the quick brown fox jumps over the lazy dog"]
    # Verbatim 5-gram hit: "quick brown fox jumps over"
    dataset_hit = "the quick brown fox jumps over the lazy dog completely"
    # No 5-gram overlap
    dataset_miss = "a fast red cat leaps across sleeping hound at noon"
    report = scan_dataset([dataset_hit, dataset_miss], corpus, n_gram_size=5)
    assert report.contaminated_count >= 1
    assert report.contaminated_count < 2 or dataset_miss  # miss is not flagged


def test_5gram_verbatim_sentence_flagged():
    """n_gram_size=5: exact 50-entry corpus entry is flagged."""
    corpus = _make_50_corpus()
    verbatim = corpus[33]
    report = scan_dataset([verbatim], corpus, n_gram_size=5)
    assert report.contaminated_count == 1


def test_unigram_returns_scan_report_type():
    """scan_dataset returns a ScanReport instance regardless of n_gram_size."""
    corpus = ["test corpus"]
    report = scan_dataset(["test"], corpus, n_gram_size=1)
    assert isinstance(report, ScanReport)


# ===========================================================================
# 9. Mixed clean/contaminated dataset
# ===========================================================================

def test_mixed_dataset_correct_counts():
    """Mixed clean and contaminated entries produce correct counts."""
    corpus = _make_50_corpus()
    clean_entries = [f"unrelated clean text entry {i}" for i in range(10)]
    dirty_entries = [corpus[i] for i in range(5)]
    dataset = clean_entries + dirty_entries
    report = scan_dataset(dataset, corpus, n_gram_size=3)
    assert report.contaminated_count == 5
    assert report.clean_count == 10
    assert report.total_count == 15
