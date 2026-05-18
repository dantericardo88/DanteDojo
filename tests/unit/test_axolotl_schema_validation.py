"""Tests for d46: axolotl config schema validation.

Verifies validate_axolotl_config() raises on missing fields, wrong types,
out-of-range values, and bad LoRA configuration — and returns the config
unchanged when valid.
"""
from __future__ import annotations

import pytest

from dojo.train.axolotl_schema import (
    AxolotlConfigValidationError,
    required_field_names,
    validate_axolotl_config,
)


def _valid_config() -> dict:
    return {
        "base_model": "qwen/qwen2-1.5b",
        "datasets": [{"path": "/data/train.json", "type": "alpaca"}],
        "output_dir": "/out",
        "num_epochs": 3,
        "micro_batch_size": 2,
        "learning_rate": 2e-4,
    }


# ---------------------------------------------------------------------------
# Valid input passes through
# ---------------------------------------------------------------------------

class TestValid:
    def test_returns_config_unchanged(self):
        cfg = _valid_config()
        assert validate_axolotl_config(cfg) is cfg

    def test_extra_fields_are_allowed(self):
        cfg = _valid_config()
        cfg["custom_field"] = "anything"
        validate_axolotl_config(cfg)


# ---------------------------------------------------------------------------
# Required field checks
# ---------------------------------------------------------------------------

class TestRequiredFields:
    def test_required_field_names_returns_list(self):
        names = required_field_names()
        assert isinstance(names, list)
        assert "base_model" in names
        assert "datasets" in names

    def test_missing_base_model_rejected(self):
        cfg = _valid_config()
        del cfg["base_model"]
        with pytest.raises(AxolotlConfigValidationError, match="base_model"):
            validate_axolotl_config(cfg)

    def test_missing_datasets_rejected(self):
        cfg = _valid_config()
        del cfg["datasets"]
        with pytest.raises(AxolotlConfigValidationError, match="datasets"):
            validate_axolotl_config(cfg)

    def test_missing_output_dir_rejected(self):
        cfg = _valid_config()
        del cfg["output_dir"]
        with pytest.raises(AxolotlConfigValidationError, match="output_dir"):
            validate_axolotl_config(cfg)

    def test_missing_num_epochs_rejected(self):
        cfg = _valid_config()
        del cfg["num_epochs"]
        with pytest.raises(AxolotlConfigValidationError, match="num_epochs"):
            validate_axolotl_config(cfg)


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------

class TestTypes:
    def test_non_dict_input_rejected(self):
        with pytest.raises(AxolotlConfigValidationError, match="must be a dict"):
            validate_axolotl_config([1, 2, 3])

    def test_base_model_must_be_string(self):
        cfg = _valid_config()
        cfg["base_model"] = 123
        with pytest.raises(AxolotlConfigValidationError, match="base_model"):
            validate_axolotl_config(cfg)

    def test_datasets_must_be_list(self):
        cfg = _valid_config()
        cfg["datasets"] = "not_a_list"
        with pytest.raises(AxolotlConfigValidationError, match="datasets"):
            validate_axolotl_config(cfg)

    def test_empty_datasets_rejected(self):
        cfg = _valid_config()
        cfg["datasets"] = []
        with pytest.raises(AxolotlConfigValidationError, match="non-empty"):
            validate_axolotl_config(cfg)

    def test_dataset_entry_must_have_path(self):
        cfg = _valid_config()
        cfg["datasets"] = [{"type": "alpaca"}]
        with pytest.raises(AxolotlConfigValidationError, match="path"):
            validate_axolotl_config(cfg)

    def test_dataset_entry_must_have_type(self):
        cfg = _valid_config()
        cfg["datasets"] = [{"path": "/data"}]
        with pytest.raises(AxolotlConfigValidationError, match="type"):
            validate_axolotl_config(cfg)


# ---------------------------------------------------------------------------
# Numeric range checks
# ---------------------------------------------------------------------------

class TestNumericRanges:
    def test_negative_num_epochs_rejected(self):
        cfg = _valid_config()
        cfg["num_epochs"] = -1
        with pytest.raises(AxolotlConfigValidationError, match=">= 0"):
            validate_axolotl_config(cfg)

    def test_zero_learning_rate_rejected(self):
        cfg = _valid_config()
        cfg["learning_rate"] = 0
        with pytest.raises(AxolotlConfigValidationError, match="> 0"):
            validate_axolotl_config(cfg)

    def test_negative_learning_rate_rejected(self):
        cfg = _valid_config()
        cfg["learning_rate"] = -1e-4
        with pytest.raises(AxolotlConfigValidationError, match="> 0"):
            validate_axolotl_config(cfg)

    def test_negative_lora_r_rejected(self):
        cfg = _valid_config()
        cfg["lora_r"] = -8
        with pytest.raises(AxolotlConfigValidationError, match="lora_r"):
            validate_axolotl_config(cfg)


# ---------------------------------------------------------------------------
# LoRA / adapter coupling
# ---------------------------------------------------------------------------

class TestLoraCoupling:
    def test_lora_r_requires_adapter_lora_or_qlora(self):
        cfg = _valid_config()
        cfg["lora_r"] = 8
        cfg["adapter"] = "full"
        with pytest.raises(AxolotlConfigValidationError, match="lora_r"):
            validate_axolotl_config(cfg)

    def test_lora_r_with_qlora_adapter_passes(self):
        cfg = _valid_config()
        cfg["lora_r"] = 8
        cfg["adapter"] = "qlora"
        validate_axolotl_config(cfg)

    def test_zero_lora_r_does_not_require_adapter(self):
        cfg = _valid_config()
        cfg["lora_r"] = 0
        validate_axolotl_config(cfg)


# ---------------------------------------------------------------------------
# Multi-error reporting
# ---------------------------------------------------------------------------

class TestMultiErrors:
    def test_multiple_errors_concatenated(self):
        cfg = {"datasets": [{}]}  # missing many fields + bad dataset
        with pytest.raises(AxolotlConfigValidationError) as exc_info:
            validate_axolotl_config(cfg)
        # Should mention several missing required fields
        msg = str(exc_info.value)
        assert "base_model" in msg
        assert "output_dir" in msg
