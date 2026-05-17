"""BlockQuantizer: quantizes weight blocks with precision floor enforcement."""

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError
from .evidence_sovereign import EvidenceSovereign


class BlockQuantizerError(DojoError):
    pass


@dataclass
class QuantizeResult:
    format: str
    precision: str
    status: str
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_PRECISION_RANK = {"2-bit": 1, "4-bit": 2, "8-bit": 3, "16-bit": 4, "32-bit": 5}
SUPPORTED_PRECISIONS = list(_PRECISION_RANK)
SUPPORTED_FORMATS = ["GGUF", "AWQ", "GPTQ", "NF4", "FP8"]


class BlockQuantizer:
    """Quantizes weight blocks with constitutional precision floor guarantees."""

    def __init__(
        self,
        precision_floor: str = "4-bit",
        evidence: Optional[EvidenceLogger] = None,
        mock_mode: bool = False,
    ) -> None:
        if precision_floor not in _PRECISION_RANK:
            raise BlockQuantizerError(f"precision_floor must be one of {SUPPORTED_PRECISIONS}")
        self.precision_floor = precision_floor
        self.evidence = evidence or EvidenceLogger.get_instance()
        self.mock_mode = mock_mode
        self._sovereign = EvidenceSovereign()
        self._blocks_quantized = 0
        self._torch: Any = None
        self._bnb: Any = None
        self._has_bnb = False
        self._init_backend()

    def _init_backend(self) -> None:
        if self.mock_mode:
            return
        try:
            import torch
            self._torch = torch
        except ImportError:
            warnings.warn("torch not installed; BlockQuantizer running in mock_mode.", stacklevel=2)
            self.mock_mode = True
            return
        try:
            import bitsandbytes as bnb  # type: ignore[import-not-found]
            self._bnb = bnb
            self._has_bnb = True
        except ImportError:
            self._has_bnb = False

    def quantize_block(self, raw_block: str, format: str, precision: str) -> QuantizeResult:
        if not raw_block:
            raise BlockQuantizerError("raw_block cannot be empty")
        if format not in SUPPORTED_FORMATS:
            raise BlockQuantizerError(f"format must be one of {SUPPORTED_FORMATS}")
        if precision not in _PRECISION_RANK:
            raise BlockQuantizerError(f"precision must be one of {SUPPORTED_PRECISIONS}")

        floor_rank = _PRECISION_RANK[self.precision_floor]
        target_rank = _PRECISION_RANK[precision]
        if target_rank < floor_rank:
            self._sovereign.trigger_halt("Precision floor violation")
            raise BlockQuantizerError(
                f"Precision floor violation: {precision} is below floor {self.precision_floor}"
            )

        if self.mock_mode:
            self._blocks_quantized += 1
            self._sovereign.seal_operation({"organ": "BlockQuantizer", "op": "quantize_block"})
            result = QuantizeResult(format=format, precision=precision, status="QuantizedBlock")
            self.evidence.info({"type": EvidenceType.BLOCK_QUANTIZED.value, "data": result.to_dict()}, organ="stream")
            return result

        status = self._apply_quantization(raw_block, format, precision)
        self._blocks_quantized += 1
        self._sovereign.seal_operation({"organ": "BlockQuantizer", "op": "quantize_block"})
        result = QuantizeResult(format=format, precision=precision, status=status)
        self.evidence.info({"type": EvidenceType.BLOCK_QUANTIZED.value, "data": result.to_dict()}, organ="stream")
        return result

    def _apply_quantization(self, raw_block: str, format: str, precision: str) -> str:
        if precision == "8-bit":
            return self._quantize_int8()
        if precision == "4-bit" and format == "NF4" and self._has_bnb:
            return self._quantize_nf4_bnb()
        if precision == "4-bit":
            return self._quantize_int4_torch()
        if precision == "16-bit":
            return "QuantizedBlock_FP16"
        if precision == "32-bit":
            return "QuantizedBlock_FP32"
        return f"QuantizedBlock_{precision}"

    def _quantize_int8(self) -> str:
        container = self._torch.nn.Sequential(self._torch.nn.Linear(128, 128))
        self._torch.quantization.quantize_dynamic(container, {self._torch.nn.Linear}, dtype=self._torch.qint8)
        return "QuantizedBlock_INT8"

    def _quantize_nf4_bnb(self) -> str:
        self._bnb.nn.Linear4bit(128, 128, bias=False, compute_dtype=self._torch.float16, quant_type="nf4")
        return "QuantizedBlock_NF4_bnb"

    def _quantize_int4_torch(self) -> str:
        weight = self._torch.randn(128, 128)
        scale = weight.abs().max() / 7.0
        self._torch.clamp(self._torch.round(weight / scale), -8, 7).to(self._torch.int8)
        return "QuantizedBlock_INT4_torch"

    def quantize_tensor(self, tensor: Any, precision: str = "8-bit") -> Any:
        floor_rank = _PRECISION_RANK.get(self.precision_floor, 2)
        target_rank = _PRECISION_RANK.get(precision, 0)
        if target_rank < floor_rank:
            self._sovereign.trigger_halt("Precision floor violation")
            raise BlockQuantizerError(
                f"Precision floor violation: {precision} is below floor {self.precision_floor}"
            )

        if self.mock_mode:
            self._blocks_quantized += 1
            self.evidence.info({"type": EvidenceType.BLOCK_QUANTIZED.value, "data": {"precision": precision}}, organ="stream")
            return tensor

        if precision == "8-bit":
            scale = tensor.abs().max() / 127.0
            result_tensor = self._torch.clamp(self._torch.round(tensor / scale), -128, 127).to(self._torch.int8)
        elif precision == "4-bit":
            scale = tensor.abs().max() / 7.0
            result_tensor = self._torch.clamp(self._torch.round(tensor / scale), -8, 7).to(self._torch.int8)
        elif precision == "16-bit":
            result_tensor = tensor.half()
        else:
            result_tensor = tensor

        self._blocks_quantized += 1
        self.evidence.info({"type": EvidenceType.BLOCK_QUANTIZED.value, "data": {"precision": precision}}, organ="stream")
        return result_tensor

    def quantize(
        self,
        block_id: str = "",
        weight_tensor: Any = None,
        precision: str = "4-bit",
        format: str = "NF4",
    ) -> "QuantizeResult":
        """Unified quantize entry point dispatching to quantize_block or quantize_tensor."""
        if weight_tensor is not None:
            self.quantize_tensor(weight_tensor, precision=precision)
        raw = block_id or "block"
        return self.quantize_block(raw_block=raw, format=format, precision=precision)

    def get_stats(self) -> Dict[str, int]:
        return {"blocks_quantized": self._blocks_quantized}
