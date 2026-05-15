"""Streaming contamination detection for DanteDojo — d21.

Provides:
- hash_corpus_sample(): hash a sample of texts for use as contamination_hashes.
- stream_check_contamination(): streaming/chunked contamination check.
- ContaminationReport: dataclass summarising a contamination check run.

Design goals:
- Memory-constant: corpora are never fully loaded; processed in chunks.
- Sampling: sample_rate fraction controls how many records are actually hashed.
- Fail-closed: raises EvalContaminationError on the first contaminated record.
- Evidence: emits EvidenceType.EVAL_CONTAMINATION on every run (clean or not).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import FrozenSet, Generator, Iterable, Iterator, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import EvalContaminationError


# ---------------------------------------------------------------------------
# Text normalisation (mirrors dojo/pack/tokenize.py to keep hashes consistent)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Strip whitespace, collapse runs, lowercase — canonical contamination form."""
    import re
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip().lower()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ContaminationReport
# ---------------------------------------------------------------------------

@dataclass
class ContaminationReport:
    """Summary of a contamination check run.

    Attributes:
        checked_count: Number of records actually hashed (after sampling).
        contaminated_count: Number of records whose hash matched gold_hashes.
        sample_rate: Fraction of the iterator that was sampled.
        corpus_size_hint: Total records seen (sampled + skipped), if known.
        contaminated_samples: First up to 5 contaminated hashes found.
        clean: True iff contaminated_count == 0.
    """

    checked_count: int
    contaminated_count: int
    sample_rate: float
    corpus_size_hint: int = 0
    contaminated_samples: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.contaminated_count == 0


# ---------------------------------------------------------------------------
# hash_corpus_sample
# ---------------------------------------------------------------------------

def hash_corpus_sample(
    texts: List[str],
    sample_rate: float = 0.1,
    *,
    seed: Optional[int] = None,
) -> FrozenSet[str]:
    """Hash a sampled fraction of *texts* for use as contamination_hashes.

    Uses SHA-256 on the normalized (strip whitespace, lowercase) form of each
    text, mirroring compute_sample_hash() in dojo/pack/tokenize.py.

    Args:
        texts: List of raw text strings to sample from.
        sample_rate: Fraction of texts to hash (0.0 → empty, 1.0 → all).
            Clamped to [0.0, 1.0].
        seed: Optional RNG seed for deterministic sampling.

    Returns:
        frozenset of SHA-256 hex digests.
    """
    sample_rate = max(0.0, min(1.0, sample_rate))
    if not texts or sample_rate == 0.0:
        return frozenset()

    rng = random.Random(seed)
    hashes: List[str] = []
    for text in texts:
        if rng.random() < sample_rate:
            hashes.append(_sha256_text(text))

    return frozenset(hashes)


# ---------------------------------------------------------------------------
# stream_check_contamination
# ---------------------------------------------------------------------------

def stream_check_contamination(
    data_iter: Iterable[str],
    gold_hashes: FrozenSet[str],
    *,
    sample_rate: float = 1.0,
    chunk_size: int = 1000,
    seed: Optional[int] = None,
) -> ContaminationReport:
    """Streaming contamination check over an iterable of text records.

    Processes *data_iter* in chunks of *chunk_size* records. Within each chunk
    a random fraction *sample_rate* of records are hashed and compared against
    *gold_hashes*. Never materialises the full corpus in memory.

    Raises EvalContaminationError on the first detected hit (fail-closed).
    Always emits an EvidenceType.EVAL_CONTAMINATION evidence event summarising
    the run, whether clean or contaminated.

    Args:
        data_iter: Iterable of text records (strings) to check.
        gold_hashes: frozenset of SHA-256 hex digests known to be in the eval
            gold set. Produced by hash_corpus_sample() or compute_sample_hash().
        sample_rate: Fraction of records to actually hash. 1.0 = check all.
        chunk_size: Number of records to buffer per processing batch.
        seed: Optional RNG seed for reproducible sampling.

    Returns:
        ContaminationReport summarising the run (only returned when clean).

    Raises:
        EvalContaminationError: If any sampled record matches gold_hashes.
    """
    sample_rate = max(0.0, min(1.0, sample_rate))
    logger = EvidenceLogger.get_instance()
    rng = random.Random(seed)

    checked_count = 0
    contaminated_count = 0
    contaminated_samples: List[str] = []
    corpus_size_hint = 0

    def _iter_chunks(it: Iterable[str], n: int) -> Generator[List[str], None, None]:
        chunk: List[str] = []
        for item in it:
            chunk.append(item)
            if len(chunk) >= n:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    for chunk in _iter_chunks(data_iter, chunk_size):
        corpus_size_hint += len(chunk)
        for text in chunk:
            if rng.random() >= sample_rate:
                continue  # skipped by sampling

            checked_count += 1
            h = _sha256_text(text)
            if h in gold_hashes:
                contaminated_count += 1
                if len(contaminated_samples) < 5:
                    contaminated_samples.append(h)

                # Fail-closed: emit evidence then raise immediately.
                report = ContaminationReport(
                    checked_count=checked_count,
                    contaminated_count=contaminated_count,
                    sample_rate=sample_rate,
                    corpus_size_hint=corpus_size_hint,
                    contaminated_samples=contaminated_samples,
                )
                _emit_contamination_evidence(logger, report)
                raise EvalContaminationError(
                    f"Contamination detected: record hash {h[:16]}… appears in eval gold set",
                    reason=f"hash={h} found in gold_hashes (checked {checked_count} records)",
                )

    report = ContaminationReport(
        checked_count=checked_count,
        contaminated_count=contaminated_count,
        sample_rate=sample_rate,
        corpus_size_hint=corpus_size_hint,
        contaminated_samples=contaminated_samples,
    )
    _emit_contamination_evidence(logger, report)
    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _emit_contamination_evidence(logger: EvidenceLogger, report: ContaminationReport) -> None:
    logger.emit(
        {
            "type": EvidenceType.EVAL_CONTAMINATION.value,
            "data": {
                "checked_count": report.checked_count,
                "contaminated_count": report.contaminated_count,
                "sample_rate": report.sample_rate,
                "corpus_size_hint": report.corpus_size_hint,
                "clean": report.clean,
                "contaminated_samples": report.contaminated_samples[:5],
            },
        },
        organ="eval",
    )


# ---------------------------------------------------------------------------
# N-gram based contamination detection  (scan_dataset / ScanReport)
# ---------------------------------------------------------------------------

@dataclass
class ScanReport:
    """Report from scan_dataset() — n-gram based contamination scan.

    Attributes:
        contaminated_count: Number of dataset entries flagged as contaminated.
        clean_count: Number of dataset entries that passed clean.
        total_count: Total entries in the dataset (contaminated + clean).
        contamination_rate: contaminated_count / total_count (0.0 if total=0).
        ngram_size: N-gram window size used for the scan.
        details: Per-entry detail dicts with keys "text", "ngram_matches", "contaminated".
    """

    contaminated_count: int
    clean_count: int
    total_count: int
    ngram_size: int = 3
    details: List[dict] = field(default_factory=list)

    @property
    def contamination_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.contaminated_count / self.total_count


def _build_ngram_hashes(text: str, n: int) -> FrozenSet[str]:
    """Return the set of SHA-256 digests of every n-gram in *text*."""
    normalized = _normalize_text(text)
    tokens = normalized.split()
    if len(tokens) < n:
        # Fall back to whole-text hash for very short texts
        return frozenset([_sha256_text(normalized)])
    hashes: List[str] = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i : i + n])
        hashes.append(hashlib.sha256(ngram.encode("utf-8")).hexdigest())
    return frozenset(hashes)


def _build_corpus_ngram_index(corpus: List[str], n: int) -> FrozenSet[str]:
    """Build a single frozenset of all n-gram hashes from every text in *corpus*."""
    all_hashes: List[str] = []
    for text in corpus:
        all_hashes.extend(_build_ngram_hashes(text, n))
    return frozenset(all_hashes)


def scan_dataset(
    dataset: List[str],
    corpus: List[str],
    *,
    n_gram_size: int = 3,
    match_threshold: int = 1,
    fail_closed: bool = False,
) -> ScanReport:
    """Scan *dataset* entries for n-gram contamination against *corpus*.

    For each entry in *dataset*, computes its n-gram hashes and checks how many
    appear in the corpus n-gram index.  An entry is flagged *contaminated* when
    the number of matching n-grams is >= *match_threshold*.

    Always emits ``eval.contamination_detected`` evidence via EvidenceLogger
    regardless of whether any contamination was found.

    Args:
        dataset: List of text strings to scan (the evaluation set to protect).
        corpus: List of text strings to build the contamination fingerprint from
                (typically the training corpus).  Empty corpus → flags nothing.
        n_gram_size: Window size for n-gram tokenisation (1 = unigram, …).
        match_threshold: Minimum number of matching n-grams to flag an entry.
        fail_closed: If True, raise EvalContaminationError on first hit.

    Returns:
        ScanReport with contaminated_count, clean_count, total_count, etc.

    Raises:
        EvalContaminationError: Only when fail_closed=True and contamination found.
    """
    logger = EvidenceLogger.get_instance()

    # Empty corpus → no fingerprints, nothing can match
    if not corpus:
        report = ScanReport(
            contaminated_count=0,
            clean_count=len(dataset),
            total_count=len(dataset),
            ngram_size=n_gram_size,
            details=[],
        )
        logger.emit(
            {
                "type": "eval.contamination_detected",
                "data": {
                    "contaminated_count": 0,
                    "clean_count": len(dataset),
                    "total_count": len(dataset),
                    "ngram_matches": 0,
                    "n_gram_size": n_gram_size,
                    "corpus_size": 0,
                },
            },
            organ="eval",
        )
        return report

    corpus_index = _build_corpus_ngram_index(corpus, n_gram_size)

    contaminated_count = 0
    clean_count = 0
    details: List[dict] = []

    for text in dataset:
        entry_ngrams = _build_ngram_hashes(text, n_gram_size)
        matches = len(entry_ngrams & corpus_index)
        is_contaminated = matches >= match_threshold
        if is_contaminated:
            contaminated_count += 1
        else:
            clean_count += 1
        details.append({
            "text": text[:120],
            "ngram_matches": matches,
            "contaminated": is_contaminated,
        })

        if fail_closed and is_contaminated:
            # Emit partial evidence then raise
            logger.emit(
                {
                    "type": "eval.contamination_detected",
                    "data": {
                        "contaminated_count": contaminated_count,
                        "clean_count": clean_count,
                        "total_count": contaminated_count + clean_count,
                        "ngram_matches": matches,
                        "n_gram_size": n_gram_size,
                        "corpus_size": len(corpus),
                        "fail_closed": True,
                    },
                },
                organ="eval",
            )
            raise EvalContaminationError(
                f"Contamination detected via {n_gram_size}-gram scan ({matches} matches)",
                reason=f"text={text[:60]!r} matched {matches} n-grams in corpus",
            )

    total_count = len(dataset)
    report = ScanReport(
        contaminated_count=contaminated_count,
        clean_count=clean_count,
        total_count=total_count,
        ngram_size=n_gram_size,
        details=details,
    )

    logger.emit(
        {
            "type": "eval.contamination_detected",
            "data": {
                "contaminated_count": contaminated_count,
                "clean_count": clean_count,
                "total_count": total_count,
                "ngram_matches": contaminated_count,  # total entries flagged
                "n_gram_size": n_gram_size,
                "corpus_size": len(corpus),
            },
        },
        organ="eval",
    )

    return report


__all__ = [
    "ContaminationReport",
    "ScanReport",
    "hash_corpus_sample",
    "scan_dataset",
    "stream_check_contamination",
]
