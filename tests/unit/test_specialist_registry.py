"""Tests for d34: capability-aware specialist registry.

Verifies SpecialistRegistry registration, capability lookup, filter_for_task
audit trail, and evidence emission for blocked specialists.
"""
from __future__ import annotations

from dojo.route.specialist_registry import (
    SpecialistCapability,
    SpecialistRegistry,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_returns_capability(self):
        r = SpecialistRegistry()
        cap = r.register("code-7b", ["code", "qa"])
        assert isinstance(cap, SpecialistCapability)
        assert cap.specialist_id == "code-7b"

    def test_register_stored_in_registry(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        assert r.get("code-7b") is not None

    def test_re_registration_overwrites(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        r.register("code-7b", ["code", "qa"])
        cap = r.get("code-7b")
        assert "qa" in cap.supported_task_types

    def test_unregister_removes(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        r.unregister("code-7b")
        assert r.get("code-7b") is None

    def test_alive_default_true(self):
        r = SpecialistRegistry()
        cap = r.register("code-7b", ["code"])
        assert cap.alive is True

    def test_register_dead_specialist(self):
        r = SpecialistRegistry()
        cap = r.register("code-7b", ["code"], alive=False)
        assert cap.alive is False


# ---------------------------------------------------------------------------
# Liveness toggles
# ---------------------------------------------------------------------------

class TestLiveness:
    def test_mark_dead_updates_state(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        r.mark_dead("code-7b")
        assert r.get("code-7b").alive is False

    def test_mark_alive_updates_state(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"], alive=False)
        r.mark_alive("code-7b")
        assert r.get("code-7b").alive is True

    def test_mark_dead_preserves_capabilities(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code", "qa"])
        r.mark_dead("code-7b")
        cap = r.get("code-7b")
        assert "code" in cap.supported_task_types
        assert "qa" in cap.supported_task_types


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_supports_method(self):
        cap = SpecialistCapability("code-7b", ("code",))
        assert cap.supports("code")
        assert not cap.supports("creative")

    def test_find_capable_returns_matching(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        r.register("creative-7b", ["creative"])
        results = r.find_capable("code")
        assert len(results) == 1
        assert results[0].specialist_id == "code-7b"

    def test_find_capable_excludes_dead(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"], alive=False)
        assert r.find_capable("code") == []

    def test_can_handle_returns_true_when_capable(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        assert r.can_handle("code") is True

    def test_can_handle_returns_false_when_no_capable(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        assert r.can_handle("creative") is False

    def test_all_alive_filters_dead(self):
        r = SpecialistRegistry()
        r.register("alive", ["code"], alive=True)
        r.register("dead", ["code"], alive=False)
        ids = [c.specialist_id for c in r.all_alive()]
        assert "alive" in ids
        assert "dead" not in ids


# ---------------------------------------------------------------------------
# filter_for_task + audit
# ---------------------------------------------------------------------------

class TestFilterForTask:
    def test_capable_specialists_returned(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        r.register("creative-7b", ["creative"])
        capable, _ = r.filter_for_task("code")
        assert len(capable) == 1

    def test_rejection_reasons_include_capability_mismatch(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        r.register("creative-7b", ["creative"])
        _, rejections = r.filter_for_task("code")
        joined = " ".join(rejections)
        assert "creative-7b" in joined
        assert "cannot handle" in joined

    def test_rejection_reasons_include_dead(self):
        r = SpecialistRegistry()
        r.register("code-7b", ["code"], alive=False)
        _, rejections = r.filter_for_task("code")
        assert any("dead" in r for r in rejections)

    def test_evidence_emitted_for_capability_block(self):
        from dojo.core.evidence import EvidenceLogger
        r = SpecialistRegistry()
        r.register("code-7b", ["code"])
        r.register("creative-7b", ["creative"])
        EvidenceLogger.clear()
        r.filter_for_task("code")
        events = EvidenceLogger.get_events()
        blocks = [e for e in events if e.get("type") == "route.specialist_capability_blocked"]
        assert len(blocks) >= 1
        assert blocks[0]["data"]["specialist_id"] == "creative-7b"

    def test_no_evidence_for_dead_only_block(self):
        from dojo.core.evidence import EvidenceLogger
        r = SpecialistRegistry()
        r.register("code-7b", ["code"], alive=False)
        EvidenceLogger.clear()
        r.filter_for_task("code")
        events = EvidenceLogger.get_events()
        blocks = [e for e in events if e.get("type") == "route.specialist_capability_blocked"]
        # Dead specialists are filtered before the capability check —
        # they get a rejection reason but not the capability_blocked event.
        assert len(blocks) == 0
