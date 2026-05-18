"""Tests for d04: evidence emission coverage audit."""
from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.evidence_coverage import (
    EvidenceCoverage,
    EvidenceCoverageError,
    assert_operations_emitted,
)


def _emit(event_type: str, **data) -> None:
    EvidenceLogger.get_instance().emit(
        {"type": event_type, "data": dict(data)},
        organ="test",
    )


class TestContextManager:
    def test_records_event_emitted_inside_block(self):
        with EvidenceCoverage() as cov:
            _emit("custom.event")
        assert cov.contains("custom.event")

    def test_excludes_events_emitted_outside_block(self):
        _emit("outside.event")
        with EvidenceCoverage() as cov:
            _emit("inside.event")
        assert not cov.contains("outside.event")
        assert cov.contains("inside.event")

    def test_total_emitted_counts_block_only(self):
        with EvidenceCoverage() as cov:
            _emit("a")
            _emit("b")
            _emit("a")
        assert cov.total_emitted() == 3

    def test_count_of_specific_type(self):
        with EvidenceCoverage() as cov:
            _emit("repeat")
            _emit("repeat")
            _emit("other")
        assert cov.count_of("repeat") == 2

    def test_event_types_is_distinct_set(self):
        with EvidenceCoverage() as cov:
            _emit("a")
            _emit("b")
            _emit("a")
        assert cov.event_types() == {"a", "b"}


class TestMissingAndReport:
    def test_missing_reports_unfired_types(self):
        with EvidenceCoverage() as cov:
            _emit("a")
        assert "b" in cov.missing(["a", "b", "c"])
        assert "c" in cov.missing(["a", "b", "c"])

    def test_missing_empty_when_all_fired(self):
        with EvidenceCoverage() as cov:
            _emit("a")
            _emit("b")
        assert cov.missing(["a", "b"]) == []

    def test_report_has_total_and_distinct(self):
        with EvidenceCoverage() as cov:
            _emit("a")
            _emit("b")
            _emit("a")
        r = cov.report()
        assert r["total_emitted"] == 3
        assert r["distinct_types"] == 2

    def test_report_counts_per_type(self):
        with EvidenceCoverage() as cov:
            _emit("a")
            _emit("a")
            _emit("b")
        r = cov.report()
        assert r["counts"]["a"] == 2
        assert r["counts"]["b"] == 1


class TestAssertEmitted:
    def test_passes_when_all_expected_present(self):
        with EvidenceCoverage() as cov:
            _emit("a")
            _emit("b")
        cov.assert_emitted(["a", "b"])

    def test_raises_when_missing(self):
        with EvidenceCoverage() as cov:
            _emit("a")
        with pytest.raises(EvidenceCoverageError, match="missing"):
            cov.assert_emitted(["a", "b"])


class TestAssertOperationsEmitted:
    def test_runs_op_and_validates(self):
        def op():
            _emit("op.start")
            _emit("op.complete")
            return "value"

        result = assert_operations_emitted(op, ["op.start", "op.complete"])
        assert result == "value"

    def test_raises_when_op_skips_event(self):
        def bad_op():
            _emit("op.start")
            # forgot op.complete

        with pytest.raises(EvidenceCoverageError):
            assert_operations_emitted(bad_op, ["op.start", "op.complete"])

    def test_passes_kwargs_through(self):
        def op(x, y=0):
            _emit("op.complete", x=x, y=y)
            return x + y

        result = assert_operations_emitted(op, ["op.complete"], 5, y=10)
        assert result == 15


class TestEmptyBlock:
    def test_no_emissions_inside_block(self):
        with EvidenceCoverage() as cov:
            pass
        assert cov.total_emitted() == 0
        assert cov.event_types() == set()
