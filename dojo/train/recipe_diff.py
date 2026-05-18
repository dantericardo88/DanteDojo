"""Training-recipe diff + migration helpers (d48).

The inspector noted d48 with "sealed recipes" — recipe export was solid but
there was no way to diff two recipes to see what changed between training
runs.  This module adds:

  - ``diff_recipes(old, new)`` → structured diff (added/removed/changed)
  - ``migrate_recipe(recipe, migrations)`` → apply a list of rename/default rules
  - ``recipe_compatibility(old, new)`` → simple "safe to swap" check
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError


class RecipeDiffError(DojoError):
    """Raised when a recipe diff or migration cannot proceed."""


@dataclass
class FieldChange:
    """One field-level change between two recipes."""

    path: str
    old: Any
    new: Any


@dataclass
class RecipeDiff:
    """Structured diff between two recipe dicts."""

    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    changed: List[FieldChange] = field(default_factory=list)

    @property
    def any_change(self) -> bool:
        return bool(self.added or self.removed or self.changed)


def _walk(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested dicts to dotted-key paths."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_walk(path, v))
        else:
            out[path] = v
    return out


def diff_recipes(old: Dict[str, Any], new: Dict[str, Any]) -> RecipeDiff:
    """Compute a flat, dotted-path diff between two recipe dicts.

    Returns a RecipeDiff with:
      - ``added``:   keys present only in ``new``
      - ``removed``: keys present only in ``old``
      - ``changed``: keys present in both but with different values
    """
    flat_old = _walk("", old)
    flat_new = _walk("", new)
    old_keys = set(flat_old)
    new_keys = set(flat_new)

    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    changed: List[FieldChange] = []
    for k in sorted(old_keys & new_keys):
        if flat_old[k] != flat_new[k]:
            changed.append(FieldChange(path=k, old=flat_old[k], new=flat_new[k]))

    diff = RecipeDiff(added=added, removed=removed, changed=changed)
    EvidenceLogger.get_instance().emit(
        {
            "type": "train.recipe_diff",
            "data": {
                "added_count": len(added),
                "removed_count": len(removed),
                "changed_count": len(changed),
            },
        },
        organ="train",
    )
    return diff


@dataclass
class Migration:
    """A single recipe migration step."""

    name: str
    apply: Callable[[Dict[str, Any]], Dict[str, Any]]


def rename_field(old_key: str, new_key: str) -> Migration:
    """Create a Migration that renames a top-level key."""
    def _apply(recipe: Dict[str, Any]) -> Dict[str, Any]:
        if old_key in recipe and new_key not in recipe:
            recipe = dict(recipe)
            recipe[new_key] = recipe.pop(old_key)
        return recipe
    return Migration(name=f"rename:{old_key}->{new_key}", apply=_apply)


def set_default(key: str, value: Any) -> Migration:
    """Create a Migration that sets a default value if a key is missing."""
    def _apply(recipe: Dict[str, Any]) -> Dict[str, Any]:
        if key not in recipe:
            recipe = dict(recipe)
            recipe[key] = value
        return recipe
    return Migration(name=f"default:{key}={value!r}", apply=_apply)


def migrate_recipe(
    recipe: Dict[str, Any],
    migrations: List[Migration],
) -> Dict[str, Any]:
    """Apply a sequence of migrations to a recipe.

    Migrations are applied in order; each receives the result of the previous.
    """
    current = dict(recipe)
    applied: List[str] = []
    for m in migrations:
        before = current
        current = m.apply(current)
        if current is not before:
            applied.append(m.name)

    EvidenceLogger.get_instance().emit(
        {
            "type": "train.recipe_migrated",
            "data": {
                "applied_migrations": applied,
                "total_migrations": len(migrations),
            },
        },
        organ="train",
    )
    return current


_REQUIRED_FOR_COMPAT: List[str] = ["base_model", "training_mode", "num_epochs"]


def recipe_compatibility(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict describing whether ``new`` can drop-in replace ``old``.

    Compatible iff every key in ``_REQUIRED_FOR_COMPAT`` has the same value
    in both recipes.
    """
    incompatible: List[FieldChange] = []
    for key in _REQUIRED_FOR_COMPAT:
        ov = old.get(key)
        nv = new.get(key)
        if ov != nv:
            incompatible.append(FieldChange(path=key, old=ov, new=nv))

    return {
        "compatible": not incompatible,
        "incompatible_fields": [c.path for c in incompatible],
        "diffs": incompatible,
    }


__all__ = [
    "FieldChange",
    "RecipeDiff",
    "RecipeDiffError",
    "Migration",
    "diff_recipes",
    "rename_field",
    "set_default",
    "migrate_recipe",
    "recipe_compatibility",
]
