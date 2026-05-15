"""WeightStreamer: streams model layers to/from VRAM with constitutional guarantees."""

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError
from .evidence_sovereign import EvidenceSovereign


class WeightStreamerError(DojoError):
    pass


@dataclass
class StreamResult:
    layer_id: str
    source_tier: str
    status: str
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WeightStreamer:
    """Streams model layers to/from VRAM. Constitutional: single layer in VRAM at a time."""

    SUPPORTED_TIERS = frozenset({"vram", "ram", "disk", "network", "ssd"})
    MAX_ACTIVE_LAYERS = 1

    def __init__(
        self,
        evidence: Optional[EvidenceLogger] = None,
        mock_mode: bool = False,
        max_gpu_memory: str = "7GiB",
        max_cpu_memory: str = "24GiB",
    ) -> None:
        self.evidence = evidence or EvidenceLogger.get_instance()
        self.mock_mode = mock_mode
        self.active_layer: Optional[str] = None
        self._sovereign = EvidenceSovereign()
        self._layers_streamed = 0
        self.max_gpu_memory = max_gpu_memory
        self.max_cpu_memory = max_cpu_memory
        self._device_map: Dict[str, str] = {}
        self._model: Any = None
        self._torch: Any = None
        self._accelerate_dispatch: Any = None
        self._accelerate_device_map: Any = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.mock_mode:
            return
        try:
            import torch
            self._torch = torch
        except ImportError:
            warnings.warn("torch not installed; WeightStreamer running in mock_mode.", stacklevel=2)
            self.mock_mode = True
            return
        try:
            from accelerate import dispatch_model, infer_auto_device_map
            self._accelerate_dispatch = dispatch_model
            self._accelerate_device_map = infer_auto_device_map
        except ImportError:
            warnings.warn("accelerate not installed; WeightStreamer running in mock_mode.", stacklevel=2)
            self.mock_mode = True

    def load_model(self, model_id: str, max_gpu_memory: Optional[str] = None) -> Dict[str, Any]:
        gpu_mem = max_gpu_memory or self.max_gpu_memory
        if self.mock_mode:
            result: Dict[str, Any] = {
                "model_id": model_id,
                "device_map": {"layer_0": "cuda:0", "layer_1": "cpu"},
                "max_memory": {"0": gpu_mem, "cpu": self.max_cpu_memory},
                "status": "MockModelLoaded",
                "mock": True,
            }
            self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result}, organ="stream")
            return result

        try:
            from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-not-found]
            has_cuda = bool(self._torch.cuda.is_available() and self._torch.cuda.device_count() > 0)
            max_memory: Dict[Any, str] = {"cpu": self.max_cpu_memory}
            if has_cuda:
                max_memory[0] = gpu_mem
            config = AutoConfig.from_pretrained(model_id)
            device_map = self._accelerate_device_map(
                AutoModelForCausalLM.from_config(config), max_memory=max_memory
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=device_map, max_memory=max_memory,
                torch_dtype=self._torch.float16,
            )
            self._device_map = dict(device_map)
            result = {"model_id": model_id, "device_map": self._device_map, "status": "ModelLoaded"}
            self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result}, organ="stream")
            return result
        except Exception as e:
            raise WeightStreamerError(f"load_model failed: {e}") from e

    def stream_layer(self, layer_id: str, source_tier: str = "disk") -> StreamResult:
        if not layer_id:
            raise WeightStreamerError("layer_id cannot be empty")
        if source_tier not in self.SUPPORTED_TIERS:
            raise WeightStreamerError(f"source_tier must be one of {sorted(self.SUPPORTED_TIERS)}")
        if self.active_layer is not None:
            self._sovereign.trigger_halt("VRAM residency breach")
            raise WeightStreamerError("VRAM residency breach: attempted to load >1 layer into VRAM")

        if self.mock_mode:
            self._layers_streamed += 1
            self.active_layer = layer_id
            self._sovereign.seal_operation({"organ": "WeightStreamer", "op": "stream_layer"})
            result = StreamResult(layer_id=layer_id, source_tier=source_tier, status="LayerReady")
            self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result.to_dict()}, organ="stream")
            return result

        try:
            target = self._torch.device("cuda:0") if self._torch.cuda.is_available() else self._torch.device("cpu")
            if self._model is not None:
                for name, module in self._model.named_modules():
                    if name == layer_id or name.endswith(layer_id):
                        module.to(target)
                        self._device_map[layer_id] = str(target)
                        break
            self._layers_streamed += 1
            self.active_layer = layer_id
            self._sovereign.seal_operation({"organ": "WeightStreamer", "op": "stream_layer"})
            result = StreamResult(layer_id=layer_id, source_tier=source_tier, status="LayerReady", device=str(target))
            self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result.to_dict()}, organ="stream")
            return result
        except WeightStreamerError:
            raise
        except Exception as e:
            raise WeightStreamerError(f"stream_layer failed: {e}") from e

    def evict_current(self) -> Optional[str]:
        if self.active_layer is None:
            return None
        evicted = self.active_layer
        if not self.mock_mode and self._model is not None and self._torch is not None:
            cpu = self._torch.device("cpu")
            for name, module in self._model.named_modules():
                if name == evicted or name.endswith(evicted):
                    module.to(cpu)
                    self._device_map[evicted] = "cpu"
                    break
        self.active_layer = None
        self._sovereign.seal_operation({"organ": "WeightStreamer", "op": "evict_current"})
        self.evidence.info({"type": EvidenceType.LAYER_STREAM_EVICTED.value, "data": {"layer_id": evicted}}, organ="stream")
        return evicted

    def vram_budget_check(self) -> Dict[str, Any]:
        """Return VRAM budget status and verify 1-layer maximum."""
        active = self.active_layer
        result = {
            "active_layer": active,
            "layers_streamed": self._layers_streamed,
            "budget_ok": active is None,  # True iff no layer currently in VRAM
            "max_active_layers": self.MAX_ACTIVE_LAYERS,
        }
        self.evidence.info(
            {"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result},
            organ="stream",
        )
        return result

    def get_device_map(self) -> Dict[str, str]:
        return dict(self._device_map)

    def get_stats(self) -> Dict[str, int]:
        return {"layers_streamed": self._layers_streamed}
