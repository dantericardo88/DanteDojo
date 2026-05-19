"""Unit tests for d44 SQLite persistence in RunLeaderboard.

Covers:
- add_run() writes to SQLite when db_path is given
- from_db() loads all entries with correct metrics
- best() after from_db() returns correct top entry
- multiple runs → correct ranking after db reload
- save_to_db() then from_db() round-trip preserves all data
- EvidenceLogger emits for add_run() (evidence chain intact)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.experiment.leaderboard import LeaderboardEntry, RunLeaderboard


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset the evidence logger before and after every test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ===========================================================================
# Test 1: add_run() with db_path writes to SQLite
# ===========================================================================

def test_add_run_writes_to_sqlite(tmp_path: Path):
    db = tmp_path / "lb.db"
    lb = RunLeaderboard(primary_metric="f1", higher_is_better=True, db_path=db)
    lb.add_run("r1", "run-1", "exp-a", {"f1": 0.85}, {"lr": 1e-4})

    assert db.exists()
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute("SELECT run_id, run_name, experiment_name FROM runs").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "r1"
    assert rows[0][1] == "run-1"
    assert rows[0][2] == "exp-a"


# ===========================================================================
# Test 2: from_db() loads all entries with correct metrics
# ===========================================================================

def test_from_db_loads_entries_with_correct_metrics(tmp_path: Path):
    db = tmp_path / "lb.db"
    lb = RunLeaderboard(primary_metric="loss", higher_is_better=False, db_path=db)
    lb.add_run("r1", "run-1", "exp", {"loss": 0.5, "f1": 0.70})
    lb.add_run("r2", "run-2", "exp", {"loss": 0.2, "f1": 0.88})

    lb2 = RunLeaderboard.from_db(db, primary_metric="loss", higher_is_better=False)
    assert len(lb2._entries) == 2

    by_id = {e.run_id: e for e in lb2._entries}
    assert by_id["r1"].metrics["loss"] == pytest.approx(0.5)
    assert by_id["r1"].metrics["f1"] == pytest.approx(0.70)
    assert by_id["r2"].metrics["loss"] == pytest.approx(0.2)
    assert by_id["r2"].metrics["f1"] == pytest.approx(0.88)


# ===========================================================================
# Test 3: best() after from_db() returns correct top entry
# ===========================================================================

def test_best_after_from_db_returns_correct_entry(tmp_path: Path):
    db = tmp_path / "lb.db"
    lb = RunLeaderboard(primary_metric="f1", higher_is_better=True, db_path=db)
    lb.add_run("r1", "run-1", "exp", {"f1": 0.70})
    lb.add_run("r2", "run-2", "exp", {"f1": 0.92})
    lb.add_run("r3", "run-3", "exp", {"f1": 0.80})

    lb2 = RunLeaderboard.from_db(db, primary_metric="f1", higher_is_better=True)
    best = lb2.best()
    assert best is not None
    assert best.run_id == "r2"
    assert best.metrics["f1"] == pytest.approx(0.92)


# ===========================================================================
# Test 4: multiple runs → correct ranking after db reload
# ===========================================================================

def test_multiple_runs_correct_ranking_after_db_reload(tmp_path: Path):
    db = tmp_path / "lb.db"
    lb = RunLeaderboard(primary_metric="loss", higher_is_better=False, db_path=db)
    lb.add_run("r1", "run-1", "exp", {"loss": 0.5})
    lb.add_run("r2", "run-2", "exp", {"loss": 0.1})
    lb.add_run("r3", "run-3", "exp", {"loss": 0.3})
    lb.add_run("r4", "run-4", "exp", {"loss": 0.4})

    lb2 = RunLeaderboard.from_db(db, primary_metric="loss", higher_is_better=False)
    ranked = lb2.rank()
    assert len(ranked) == 4
    # Lower loss → better rank
    assert ranked[0].run_id == "r2"  # loss=0.1
    assert ranked[1].run_id == "r3"  # loss=0.3
    assert ranked[2].run_id == "r4"  # loss=0.4
    assert ranked[3].run_id == "r1"  # loss=0.5
    assert [e.rank for e in ranked] == [1, 2, 3, 4]


# ===========================================================================
# Test 5: save_to_db() then from_db() round-trip preserves all data
# ===========================================================================

def test_save_to_db_then_from_db_round_trip(tmp_path: Path):
    # Create an in-memory leaderboard (no db_path) and populate it
    lb = RunLeaderboard(primary_metric="accuracy", higher_is_better=True)
    lb.add_run("a", "run-a", "exp-x", {"accuracy": 0.60, "loss": 0.8}, {"lr": 1e-3})
    lb.add_run("b", "run-b", "exp-x", {"accuracy": 0.95, "loss": 0.2}, {"lr": 5e-4})
    lb.add_run("c", "run-c", "exp-x", {"accuracy": 0.75, "loss": 0.5}, {"lr": 2e-4})

    # Export to DB
    db = tmp_path / "export.db"
    lb.save_to_db(db)
    assert db.exists()

    # Reload from DB
    lb2 = RunLeaderboard.from_db(db, primary_metric="accuracy", higher_is_better=True)
    assert len(lb2._entries) == 3

    by_id = {e.run_id: e for e in lb2._entries}
    assert by_id["b"].metrics["accuracy"] == pytest.approx(0.95)
    assert by_id["a"].params["lr"] == pytest.approx(1e-3)
    assert by_id["c"].run_name == "run-c"
    assert by_id["c"].experiment_name == "exp-x"

    # Ranking should survive round-trip
    ranked = lb2.rank()
    assert ranked[0].run_id == "b"
    assert ranked[1].run_id == "c"
    assert ranked[2].run_id == "a"


# ===========================================================================
# Test 6: EvidenceLogger emits for add_run() — evidence chain intact
# ===========================================================================

def test_add_run_with_db_emits_evidence(tmp_path: Path):
    db = tmp_path / "lb.db"
    lb = RunLeaderboard(primary_metric="f1", higher_is_better=True, db_path=db)
    lb.add_run("r1", "run-1", "exp", {"f1": 0.88})

    events = EvidenceLogger.get_events()
    run_complete_events = [
        e for e in events
        if e.get("type") == EvidenceType.EXPERIMENT_RUN_COMPLETE.value
    ]
    assert len(run_complete_events) >= 1
    data = run_complete_events[0]["data"]
    assert data["run_id"] == "r1"
    assert data["action"] == "leaderboard.add_run"
