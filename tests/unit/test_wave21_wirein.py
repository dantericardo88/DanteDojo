"""Wave 21 wire-in integration tests.

Wires 4 more orphans into production. Each test exercises the
production-facing entry point, not the orphan in isolation.

Maps to:
  d34: SpecialistRegistry imported by dojo/route/router.py;
       configure_specialist_registry + _is_viable consults it for LOCAL_SPECIALIST.
  d36: HealthMonitor lifecycle helpers (start/stop) added to router public API.
  d44: RunLeaderboard.pareto + .weighted_rank delegate to leaderboard_advanced.
  d47: serve_with_cascade in dojo/serve/launcher.py uses BackendCascade.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# d34 SpecialistRegistry wire-in
# ---------------------------------------------------------------------------

class TestSpecialistRegistryWiredIn:
    def test_router_source_imports_specialist_registry(self):
        from dojo.route import router
        src = inspect.getsource(router)
        assert "from dojo.route.specialist_registry import" in src

    def test_configure_specialist_registry_exists(self):
        from dojo.route.router import configure_specialist_registry
        assert callable(configure_specialist_registry)

    def test_get_active_specialist_registry_starts_none(self):
        from dojo.route.router import (
            configure_specialist_registry,
            get_active_specialist_registry,
        )
        configure_specialist_registry(None)
        assert get_active_specialist_registry() is None

    def test_configure_then_get_returns_registry(self):
        from dojo.route.router import (
            configure_specialist_registry,
            get_active_specialist_registry,
        )
        from dojo.route.specialist_registry import SpecialistRegistry
        reg = SpecialistRegistry()
        reg.register("code-7b", ["code"])
        try:
            configure_specialist_registry(reg)
            assert get_active_specialist_registry() is reg
        finally:
            configure_specialist_registry(None)


# ---------------------------------------------------------------------------
# d36 HealthMonitor wire-in
# ---------------------------------------------------------------------------

class TestHealthMonitorWiredIn:
    def test_start_health_monitor_exists(self):
        from dojo.route.router import start_health_monitor
        assert callable(start_health_monitor)

    def test_get_active_starts_none(self):
        from dojo.route.router import get_active_health_monitor, stop_health_monitor
        stop_health_monitor()
        assert get_active_health_monitor() is None

    def test_start_then_get_returns_instance(self):
        from dojo.route.router import (
            start_health_monitor,
            stop_health_monitor,
            get_active_health_monitor,
        )
        try:
            mon = start_health_monitor(interval_seconds=0.05)
            assert get_active_health_monitor() is mon
        finally:
            stop_health_monitor()

    def test_stop_clears_active(self):
        from dojo.route.router import (
            start_health_monitor,
            stop_health_monitor,
            get_active_health_monitor,
        )
        start_health_monitor(interval_seconds=0.05)
        stop_health_monitor()
        assert get_active_health_monitor() is None

    def test_start_is_idempotent(self):
        from dojo.route.router import (
            start_health_monitor,
            stop_health_monitor,
            get_active_health_monitor,
        )
        try:
            mon1 = start_health_monitor(interval_seconds=0.05)
            mon2 = start_health_monitor(interval_seconds=0.05)
            assert mon1 is mon2
        finally:
            stop_health_monitor()


# ---------------------------------------------------------------------------
# d44 leaderboard_advanced wire-in
# ---------------------------------------------------------------------------

class TestLeaderboardAdvancedWiredIn:
    def test_leaderboard_pareto_method_exists(self):
        from dojo.experiment.leaderboard import RunLeaderboard
        assert hasattr(RunLeaderboard, "pareto")
        assert hasattr(RunLeaderboard, "weighted_rank")

    def test_leaderboard_source_imports_advanced(self):
        from dojo.experiment import leaderboard
        src = inspect.getsource(leaderboard)
        assert "leaderboard_advanced" in src

    def test_pareto_returns_subset_of_runs(self):
        from dojo.experiment.leaderboard import RunLeaderboard
        from dojo.experiment.leaderboard_advanced import MetricSpec

        lb = RunLeaderboard(primary_metric="accuracy", higher_is_better=True)
        lb.add_run(run_id="a", run_name="A", experiment_name="exp",
                   metrics={"accuracy": 0.9, "cost": 5.0}, params={})
        lb.add_run(run_id="b", run_name="B", experiment_name="exp",
                   metrics={"accuracy": 0.7, "cost": 9.0}, params={})

        frontier = lb.pareto([
            MetricSpec("accuracy", higher_is_better=True),
            MetricSpec("cost", higher_is_better=False),
        ])
        ids = {r["run_id"] for r in frontier}
        # 'b' is dominated by 'a' (higher acc, lower cost)
        assert "a" in ids
        assert "b" not in ids

    def test_weighted_rank_returns_ranked_list(self):
        from dojo.experiment.leaderboard import RunLeaderboard
        from dojo.experiment.leaderboard_advanced import MetricSpec

        lb = RunLeaderboard(primary_metric="accuracy", higher_is_better=True)
        lb.add_run(run_id="a", run_name="A", experiment_name="exp",
                   metrics={"accuracy": 0.9}, params={})
        lb.add_run(run_id="b", run_name="B", experiment_name="exp",
                   metrics={"accuracy": 0.7}, params={})
        ranked = lb.weighted_rank([MetricSpec("accuracy")], limit=10)
        assert ranked[0]["run_id"] == "a"


# ---------------------------------------------------------------------------
# d47 BackendCascade wire-in
# ---------------------------------------------------------------------------

class TestBackendCascadeWiredIn:
    def test_serve_with_cascade_exists(self):
        from dojo.serve.launcher import serve_with_cascade
        assert callable(serve_with_cascade)

    def test_launcher_source_imports_backend_cascade(self):
        from dojo.serve import launcher
        src = inspect.getsource(launcher)
        assert "BackendCascade" in src

    def test_serve_with_cascade_uses_first_successful_backend(self):
        from dojo.serve.launcher import serve_with_cascade

        def primary(prompt, **kwargs):
            return {"backend": "primary", "text": prompt.upper()}

        result = serve_with_cascade([("primary", primary)], "hello")
        assert result["backend"] == "primary"
        assert result["text"] == "HELLO"

    def test_serve_with_cascade_falls_back_on_failure(self):
        from dojo.serve.launcher import serve_with_cascade

        def bad(prompt, **kwargs):
            raise RuntimeError("primary down")

        def good(prompt, **kwargs):
            return {"backend": "fallback", "text": prompt}

        result = serve_with_cascade([("bad", bad), ("good", good)], "hello")
        assert result["backend"] == "fallback"

    def test_serve_with_cascade_raises_when_all_fail(self):
        from dojo.serve.launcher import serve_with_cascade, BackendCascadeExhausted

        def bad(prompt, **kwargs):
            raise RuntimeError("boom")

        with pytest.raises(BackendCascadeExhausted):
            serve_with_cascade([("bad", bad)], "hello")


# ---------------------------------------------------------------------------
# Production-import audit
# ---------------------------------------------------------------------------

class TestWave21NotOrphaned:
    DOJO_ROOT = Path(__file__).resolve().parents[2] / "dojo"

    def _count(self, module_path: str) -> int:
        n = 0
        for p in self.DOJO_ROOT.rglob("*.py"):
            if "tests" in p.parts:
                continue
            if module_path in p.read_text(encoding="utf-8", errors="ignore"):
                n += 1
        return n

    def test_specialist_registry_imported_in_production(self):
        assert self._count("dojo.route.specialist_registry") >= 1

    def test_leaderboard_advanced_imported_in_production(self):
        assert self._count("dojo.experiment.leaderboard_advanced") >= 1

    def test_backend_cascade_imported_in_production(self):
        assert self._count("dojo.serve.backend_cascade") >= 1

    def test_health_monitor_instantiated_in_production(self):
        """HealthMonitor class lives in router.py — the wire-in check is
        that something in production CALLS HealthMonitor(...) (not just the
        class def, which is `class HealthMonitor:` without parens)."""
        router_src = (self.DOJO_ROOT / "route" / "router.py").read_text(encoding="utf-8")
        assert router_src.count("HealthMonitor(") >= 1, (
            "router.py must instantiate HealthMonitor in start_health_monitor()"
        )
