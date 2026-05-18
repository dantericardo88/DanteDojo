"""Tests for d48: recipe diff + migration helpers."""
from __future__ import annotations

from dojo.train.recipe_diff import (
    FieldChange,
    Migration,
    RecipeDiff,
    diff_recipes,
    migrate_recipe,
    recipe_compatibility,
    rename_field,
    set_default,
)


class TestDiff:
    def test_no_diff_for_identical(self):
        d = diff_recipes({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert not d.any_change

    def test_detects_added_field(self):
        d = diff_recipes({"a": 1}, {"a": 1, "b": 2})
        assert d.added == ["b"]
        assert d.removed == []
        assert d.changed == []

    def test_detects_removed_field(self):
        d = diff_recipes({"a": 1, "b": 2}, {"a": 1})
        assert d.removed == ["b"]

    def test_detects_changed_field(self):
        d = diff_recipes({"a": 1}, {"a": 2})
        assert len(d.changed) == 1
        assert d.changed[0].old == 1
        assert d.changed[0].new == 2

    def test_dotted_path_for_nested(self):
        d = diff_recipes({"lr": {"warmup": 100}}, {"lr": {"warmup": 200}})
        assert d.changed[0].path == "lr.warmup"

    def test_any_change_flag(self):
        d = diff_recipes({"a": 1}, {"a": 2})
        assert d.any_change is True

    def test_returns_recipe_diff_dataclass(self):
        d = diff_recipes({}, {})
        assert isinstance(d, RecipeDiff)


class TestMigrations:
    def test_rename_field_works(self):
        m = rename_field("lr", "learning_rate")
        out = m.apply({"lr": 0.001})
        assert out == {"learning_rate": 0.001}

    def test_rename_skips_when_new_already_present(self):
        m = rename_field("lr", "learning_rate")
        out = m.apply({"lr": 0.001, "learning_rate": 0.002})
        assert "lr" in out
        assert out["learning_rate"] == 0.002

    def test_rename_no_op_when_old_missing(self):
        m = rename_field("lr", "learning_rate")
        out = m.apply({"other": 1})
        assert out == {"other": 1}

    def test_set_default_adds_missing_key(self):
        m = set_default("seed", 42)
        out = m.apply({})
        assert out["seed"] == 42

    def test_set_default_preserves_existing(self):
        m = set_default("seed", 42)
        out = m.apply({"seed": 99})
        assert out["seed"] == 99


class TestMigrateRecipe:
    def test_applies_migrations_in_order(self):
        ms = [
            rename_field("lr", "learning_rate"),
            set_default("seed", 42),
        ]
        result = migrate_recipe({"lr": 0.001}, ms)
        assert result == {"learning_rate": 0.001, "seed": 42}

    def test_no_migrations_returns_copy(self):
        original = {"a": 1}
        result = migrate_recipe(original, [])
        assert result == original
        # ensure it's a copy, not the same dict
        result["a"] = 999
        assert original["a"] == 1

    def test_empty_recipe_with_defaults(self):
        ms = [set_default("epochs", 3), set_default("batch_size", 8)]
        result = migrate_recipe({}, ms)
        assert result == {"epochs": 3, "batch_size": 8}


class TestCompatibility:
    def test_compatible_when_required_match(self):
        old = {"base_model": "m", "training_mode": "sft", "num_epochs": 3}
        new = {"base_model": "m", "training_mode": "sft", "num_epochs": 3, "extra": "ok"}
        info = recipe_compatibility(old, new)
        assert info["compatible"] is True

    def test_incompatible_when_base_model_differs(self):
        old = {"base_model": "m1", "training_mode": "sft", "num_epochs": 3}
        new = {"base_model": "m2", "training_mode": "sft", "num_epochs": 3}
        info = recipe_compatibility(old, new)
        assert info["compatible"] is False
        assert "base_model" in info["incompatible_fields"]

    def test_incompatible_when_training_mode_differs(self):
        old = {"base_model": "m", "training_mode": "sft", "num_epochs": 3}
        new = {"base_model": "m", "training_mode": "qlora", "num_epochs": 3}
        info = recipe_compatibility(old, new)
        assert info["compatible"] is False
        assert "training_mode" in info["incompatible_fields"]

    def test_returns_diff_objects(self):
        old = {"base_model": "m1", "training_mode": "sft", "num_epochs": 3}
        new = {"base_model": "m2", "training_mode": "sft", "num_epochs": 3}
        info = recipe_compatibility(old, new)
        assert len(info["diffs"]) == 1
        assert isinstance(info["diffs"][0], FieldChange)


class TestEvidence:
    def test_diff_emits_evidence(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        diff_recipes({"a": 1}, {"a": 2})
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "train.recipe_diff" for e in events)

    def test_migration_emits_evidence(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        migrate_recipe({"a": 1}, [set_default("b", 2)])
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "train.recipe_migrated" for e in events)
