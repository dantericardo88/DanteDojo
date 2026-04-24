"""MetaDeviceManager: routes tensor ops without full-model allocation."""

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError
from .evidence_sovereign import EvidenceSovereign


class MetaDeviceManagerError(DojoError):
    pass


@dataclass
class TensorOpResult:
    op: str
    status: str
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetaDeviceManager:
    """Routes tensor ops with constitutional single-layer residency guarantees.

    Full-model allocation is unconditionally forbidden — raises immediately.
    """

    SUPPORTED_OPS = frozenset({"forward", "backward", "attention", "norm", "mlp"})

    def __init__(
        self,
        evidence: Optional[EvidenceLogger] = None,
        mock_mode: bool = False,
    ) -> None:
        self.evidence = evidence or EvidenceLogger.get_instance()
        self.mock_mode = mock_mode
        self.virtual_ready = True
        self._sovereign = EvidenceSovereign()
        self._ops_routed = 0
        self._torch: Any = None
        self._init_empty_weights: Any = None
        self._lazy_model: Any = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.mock_mode:
            return
        try:
            import torch
            self._torch = torch
        except ImportError:
            warnings.warn("torch not installed; MetaDeviceManager running in mock_mode.", stacklevel=2)
            self.mock_mode = True
            return
        try:
            from accelerate import init_empty_weights  # type: ignore[import-not-found]
            self._init_empty_weights = init_empty_weights
        except ImportError:
            warnings.warn("accelerate not installed; MetaDeviceManager running in mock_mode.", stacklevel=2)
            self.mock_mode = True

    def load_model_lazy(self, model_id: str) -> Dict[str, Any]:
        if self.mock_mode:
            result: Dict[str, Any] = {
                "model_id": model_id, "device": "meta",
                "parameters": 0, "status": "MockLazyLoaded", "mock": True,
            }
            self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result}, organ="stream")
            return result

        try:
            from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-not-found]
            config = AutoConfig.from_pretrained(model_id)
            with self._init_empty_weights():
                self._lazy_model = AutoModelForCausalLM.from_config(config)
            param_count = sum(p.numel() for p in self._lazy_model.parameters())
            result = {"model_id": model_id, "device": "meta", "parameters": param_count, "status": "LazyLoaded"}
            self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result}, organ="stream")
            return result
        except MetaDeviceManagerError:
            raise
        except Exception as e:
            raise MetaDeviceManagerError(f"load_model_lazy failed: {e}") from e

    def allocate_full_model(self) -> None:
        self._sovereign.trigger_halt("Full-model tensor allocation attempted")
        raise MetaDeviceManagerError(
            "Full-model tensor allocation attempted: violates single-layer residency"
        )

    def forward_tensor_op(self, op: str, tensors: Any) -> TensorOpResult:
        if not op:
            raise MetaDeviceManagerError("op cannot be empty")

        if self.mock_mode:
            self._ops_routed += 1
            self._sovereign.seal_operation({"organ": "MetaDeviceManager", "op": "forward_tensor_op"})
            result = TensorOpResult(op=op, status="RoutedOp")
            self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result.to_dict()}, organ="stream")
            return result

        try:
            device_str = "cpu"
            if self._torch is not None and self._torch.cuda.is_available():
                device_str = "cuda:0"
            if self._torch is not None and isinstance(tensors, self._torch.Tensor):
                if tensors.device.type != "meta":
                    tensors = tensors.to(device_str)
            self._ops_routed += 1
            self._sovereign.seal_operation({"organ": "MetaDeviceManager", "op": "forward_tensor_op"})
            result = TensorOpResult(op=op, status="RoutedOp", device=device_str)
            self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result.to_dict()}, organ="stream")
            return result
        except MetaDeviceManagerError:
            raise
        except Exception as e:
            raise MetaDeviceManagerError(f"forward_tensor_op failed: {e}") from e

    def get_stats(self) -> Dict[str, int]:
        return {"ops_routed": self._ops_routed}
