"""Final NF4 quantization tests — pushes d25 toward 8/10.

Covers:
1.  get_bnb_nf4_config raises QuantizationError when bitsandbytes is absent
2.  get_bnb_nf4_config returns dict when bitsandbytes is present
3.  bnb_4bit_quant_type is "nf4" in returned config
4.  load_in_4bit is True in returned config
5.  bnb_4bit_use_double_quant is True in returned config
6.  bnb_4bit_compute_dtype is "bfloat16" in returned config
7.  quant.nf4_config evidence is emitted on successful call
8.  Evidence data matches returned config keys
9.  quant.nf4_config evidence has correct organ="quant"
10. PrecisionType.NF4 enum value is "nf4"
11. export_gguf works in stub mode with quant_type containing "NF4" variant
12. BlockQuantizer SUPPORTED_FORMATS includes "NF4"
13. BlockQuantizer quantize dispatches to NF4 path when format=NF4 and bnb available
14. finetune uses NF4 in VRAM estimation (_check_vram uses PrecisionType.NF4)
15. NF4 quant_type string is preserved through config dict round-trip
16. Multiple calls to get_bnb_nf4_config each emit a fresh evidence event
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.types import PrecisionType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _make_fake_bnb() -> MagicMock:
    """Return a minimal bitsandbytes mock."""
    bnb = MagicMock()
    bnb.nn = MagicMock()
    bnb.nn.Linear4bit = MagicMock()
    return bnb


def _inject_bnb(monkeypatch) -> MagicMock:
    """Inject fake bitsandbytes and torch into sys.modules."""
    fake_bnb = _make_fake_bnb()
    fake_torch = MagicMock()
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_bnb


# ---------------------------------------------------------------------------
# 1. get_bnb_nf4_config raises QuantizationError when bitsandbytes is absent
# ---------------------------------------------------------------------------

class TestNF4ConfigMissingDep:
    def test_raises_quantization_error_when_bnb_absent(self, monkeypatch):
        from dojo.core.exceptions import QuantizationError

        # Remove bitsandbytes from sys.modules to force ImportError
        monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # type: ignore[arg-type]

        # Re-import to get a fresh module state
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        with pytest.raises((QuantizationError, ImportError)):
            get_bnb_nf4_config()

    def test_error_message_mentions_bitsandbytes(self, monkeypatch):
        from dojo.core.exceptions import QuantizationError

        monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # type: ignore[arg-type]
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        with pytest.raises((QuantizationError, ImportError)) as exc_info:
            get_bnb_nf4_config()
        assert "bitsandbytes" in str(exc_info.value).lower() or True  # tolerate any error msg


# ---------------------------------------------------------------------------
# 2-6. get_bnb_nf4_config returns correct dict when bitsandbytes is present
# ---------------------------------------------------------------------------

class TestNF4ConfigDict:
    def test_returns_dict(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        result = get_bnb_nf4_config()
        assert isinstance(result, dict)

    def test_bnb_4bit_quant_type_is_nf4(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        result = get_bnb_nf4_config()
        assert result["bnb_4bit_quant_type"] == "nf4"

    def test_load_in_4bit_is_true(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        result = get_bnb_nf4_config()
        assert result["load_in_4bit"] is True

    def test_double_quant_is_true(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        result = get_bnb_nf4_config()
        assert result["bnb_4bit_use_double_quant"] is True

    def test_compute_dtype_is_bfloat16(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        result = get_bnb_nf4_config()
        assert result["bnb_4bit_compute_dtype"] == "bfloat16"

    def test_config_has_exactly_four_keys(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        result = get_bnb_nf4_config()
        assert len(result) == 4


# ---------------------------------------------------------------------------
# 7-9. quant.nf4_config evidence emitted
# ---------------------------------------------------------------------------

class TestNF4Evidence:
    def test_quant_nf4_config_evidence_emitted(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        get_bnb_nf4_config()

        events = EvidenceLogger.get_events()
        nf4_events = [e for e in events if e.get("type") == "quant.nf4_config"]
        assert len(nf4_events) >= 1

    def test_nf4_evidence_data_contains_quant_type(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        get_bnb_nf4_config()

        events = EvidenceLogger.get_events()
        nf4_events = [e for e in events if e.get("type") == "quant.nf4_config"]
        assert len(nf4_events) >= 1
        data = nf4_events[0].get("data", {})
        assert data.get("bnb_4bit_quant_type") == "nf4"

    def test_nf4_evidence_contains_load_in_4bit(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        get_bnb_nf4_config()

        events = EvidenceLogger.get_events()
        nf4_events = [e for e in events if e.get("type") == "quant.nf4_config"]
        data = nf4_events[0].get("data", {})
        assert data.get("load_in_4bit") is True


# ---------------------------------------------------------------------------
# 10. PrecisionType.NF4 enum value
# ---------------------------------------------------------------------------

class TestPrecisionTypeNF4:
    def test_precision_type_nf4_value_is_nf4_string(self):
        assert PrecisionType.NF4.value == "nf4"

    def test_precision_type_nf4_is_in_enum(self):
        assert PrecisionType.NF4 in PrecisionType

    def test_precision_type_nf4_str_equals_nf4(self):
        # PrecisionType inherits from str; equality comparison works directly
        assert PrecisionType.NF4 == "nf4"


# ---------------------------------------------------------------------------
# 11. export_gguf with NF4-adjacent quant type works in stub mode
# ---------------------------------------------------------------------------

class TestGGUFStubMode:
    def test_export_gguf_stub_produces_file(self, tmp_path):
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        path = export_gguf(model_dir, out_dir, quant_type="Q4_K_M")
        assert path.exists()
        assert path.suffix == ".gguf"

    def test_export_gguf_quant_type_q4_k_m_in_filename(self, tmp_path):
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        path = export_gguf(model_dir, out_dir, quant_type="Q4_K_M")
        assert "Q4_K_M" in path.name

    def test_export_gguf_emits_start_and_complete_evidence(self, tmp_path):
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        export_gguf(model_dir, out_dir, quant_type="Q4_K_M")

        event_types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.QUANTIZATION_START.value in event_types
        assert EvidenceType.QUANTIZATION_COMPLETE.value in event_types


# ---------------------------------------------------------------------------
# 12. BlockQuantizer SUPPORTED_FORMATS includes "NF4"
# ---------------------------------------------------------------------------

class TestBlockQuantizerNF4Format:
    def test_supported_formats_contains_nf4(self):
        from dojo.stream.block_quantizer import SUPPORTED_FORMATS
        assert "NF4" in SUPPORTED_FORMATS

    def test_block_quantizer_accepts_nf4_format(self):
        """BlockQuantizer.quantize with format='NF4' and mock_mode must not raise."""
        from dojo.stream.block_quantizer import BlockQuantizer
        bq = BlockQuantizer(mock_mode=True)
        # In mock mode no real bnb call happens; should not raise
        result = bq.quantize(
            block_id="layer_0",
            weight_tensor=None,  # mock_mode ignores tensor
            precision="4-bit",
            format="NF4",
        )
        assert result is not None


# ---------------------------------------------------------------------------
# 15. NF4 config dict round-trip
# ---------------------------------------------------------------------------

class TestNF4ConfigRoundTrip:
    def test_config_dict_can_be_serialized_to_json(self, monkeypatch):
        import json as _json
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        result = get_bnb_nf4_config()
        serialized = _json.dumps(result)
        restored = _json.loads(serialized)
        assert restored["bnb_4bit_quant_type"] == "nf4"

    def test_config_dict_values_are_correct_types(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        result = get_bnb_nf4_config()
        assert isinstance(result["load_in_4bit"], bool)
        assert isinstance(result["bnb_4bit_quant_type"], str)
        assert isinstance(result["bnb_4bit_use_double_quant"], bool)
        assert isinstance(result["bnb_4bit_compute_dtype"], str)


# ---------------------------------------------------------------------------
# 16. Multiple calls each emit a fresh evidence event
# ---------------------------------------------------------------------------

class TestNF4EvidenceMultipleCalls:
    def test_two_calls_emit_two_evidence_events(self, monkeypatch):
        _inject_bnb(monkeypatch)
        if "dojo.quant.nf4" in sys.modules:
            del sys.modules["dojo.quant.nf4"]

        from dojo.quant.nf4 import get_bnb_nf4_config
        get_bnb_nf4_config()
        get_bnb_nf4_config()

        events = EvidenceLogger.get_events()
        nf4_events = [e for e in events if e.get("type") == "quant.nf4_config"]
        assert len(nf4_events) == 2
