"""Axolotl config schema validation (d46).

Validates a generated axolotl config dict against the subset of fields that
axolotl requires before launching training.  Uses no external dependencies
(no jsonschema, no pydantic) — pure Python validation tailored to the
axolotl YAML shape DanteDojo emits.

Raises ``AxolotlConfigValidationError`` (a DojoError subclass) with field-level
detail so callers can correct the config before disk write.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from dojo.core.exceptions import DojoError


class AxolotlConfigValidationError(DojoError):
    """Raised when an axolotl config dict fails schema validation."""


# Required top-level fields. Maps field name → (expected python type, description).
_REQUIRED_FIELDS: Dict[str, Tuple[type, str]] = {
    "base_model": (str, "HuggingFace model id or local path"),
    "datasets": (list, "List of dataset specifications"),
    "output_dir": (str, "Path where checkpoints and adapters are written"),
    "num_epochs": ((int, float), "Total training epochs"),
    "micro_batch_size": ((int, float), "Per-device batch size"),
    "learning_rate": ((int, float), "Initial learning rate"),
}

# Required keys for each dataset entry in datasets[]
_DATASET_REQUIRED: Dict[str, type] = {
    "path": str,
    "type": str,
}

# Fields that must be non-negative numbers if present.
_NONNEG_FIELDS: List[str] = [
    "num_epochs",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "warmup_ratio",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "sequence_len",
    "max_steps",
]

# Fields that must be strictly positive numbers if present.
_POSITIVE_FIELDS: List[str] = [
    "learning_rate",
]


def _check_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        return isinstance(value, expected)
    return isinstance(value, expected)


def _format_type(expected: Any) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


def validate_axolotl_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ``cfg`` against the axolotl schema. Returns ``cfg`` on success.

    Raises:
        AxolotlConfigValidationError: with all field-level issues concatenated
        when any required field is missing or has the wrong type / range.
    """
    if not isinstance(cfg, dict):
        raise AxolotlConfigValidationError(
            f"axolotl config must be a dict, got {type(cfg).__name__}",
            reason="type_error",
        )

    errors: List[str] = []

    # Required top-level fields
    for field, (expected_type, _desc) in _REQUIRED_FIELDS.items():
        if field not in cfg:
            errors.append(f"missing required field: {field}")
            continue
        if not _check_type(cfg[field], expected_type):
            errors.append(
                f"field '{field}' must be {_format_type(expected_type)}, "
                f"got {type(cfg[field]).__name__}"
            )

    # Datasets must be a non-empty list of dicts with required keys
    datasets = cfg.get("datasets")
    if isinstance(datasets, list):
        if len(datasets) == 0:
            errors.append("'datasets' must be a non-empty list")
        for i, entry in enumerate(datasets):
            if not isinstance(entry, dict):
                errors.append(f"datasets[{i}] must be a dict")
                continue
            for k, t in _DATASET_REQUIRED.items():
                if k not in entry:
                    errors.append(f"datasets[{i}] missing required key: {k}")
                elif not isinstance(entry[k], t):
                    errors.append(
                        f"datasets[{i}].{k} must be {t.__name__}, got {type(entry[k]).__name__}"
                    )

    # Numeric range checks (only when field present and is a number)
    for field in _NONNEG_FIELDS:
        if field in cfg and isinstance(cfg[field], (int, float)):
            if cfg[field] < 0:
                errors.append(f"field '{field}' must be >= 0, got {cfg[field]}")

    for field in _POSITIVE_FIELDS:
        if field in cfg and isinstance(cfg[field], (int, float)):
            if cfg[field] <= 0:
                errors.append(f"field '{field}' must be > 0, got {cfg[field]}")

    # Mutually exclusive: adapter must be 'lora' or 'qlora' when lora_r > 0
    if cfg.get("lora_r", 0) and cfg.get("adapter") not in ("lora", "qlora"):
        errors.append(
            f"lora_r={cfg['lora_r']} requires adapter in ('lora', 'qlora'), "
            f"got adapter={cfg.get('adapter')!r}"
        )

    if errors:
        joined = "; ".join(errors)
        raise AxolotlConfigValidationError(
            f"axolotl config validation failed: {joined}",
            reason=f"{len(errors)} error(s)",
        )

    return cfg


def required_field_names() -> List[str]:
    """Return the list of required top-level field names (for help/docs)."""
    return list(_REQUIRED_FIELDS.keys())


__all__ = [
    "AxolotlConfigValidationError",
    "validate_axolotl_config",
    "required_field_names",
]
