"""Constitutional VRAM-bounded inference loop using WeightStreamer."""
from __future__ import annotations

from typing import Any, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError
from dojo.stream.weight_streamer import WeightStreamer


class InferenceLoopError(DojoError):
    pass


class LayerStreamingInferenceLoop:
    """Runs inference layer-by-layer with constitutional VRAM budget.

    Ensures exactly 1 layer is in VRAM at any time:
    1. Load layer i into VRAM
    2. Run forward pass for that layer
    3. Evict layer i from VRAM
    4. Repeat for all layers
    """

    def __init__(
        self,
        layer_ids: List[str],
        streamer: Optional[WeightStreamer] = None,
        mock_mode: bool = True,
    ) -> None:
        self._layer_ids = layer_ids
        self._streamer = streamer or WeightStreamer(mock_mode=mock_mode)
        self._evidence = EvidenceLogger.get_instance()
        self._mock_mode = mock_mode

    def run(self, prompt_tokens: List[int]) -> List[int]:
        """Run a complete inference pass, streaming one layer at a time.

        Args:
            prompt_tokens: input token IDs

        Returns:
            output_tokens: generated token IDs

        Raises:
            WeightStreamerError: if VRAM budget is exceeded (>1 layer loaded)
            InferenceLoopError: if layer forward pass fails
        """
        output_tokens: List[int] = []

        self._evidence.emit(
            {
                "type": EvidenceType.SERVE_START.value,
                "data": {
                    "mode": "layer_streaming",
                    "num_layers": len(self._layer_ids),
                    "prompt_tokens": len(prompt_tokens),
                    "vram_budget_layers": 1,  # constitutional maximum
                },
            },
            organ="stream",
        )

        for layer_id in self._layer_ids:
            # Load this layer (enforces MAX_ACTIVE_LAYERS=1)
            stream_result = self._streamer.stream_layer(layer_id, source_tier="ram")

            # Forward pass (stub: identity pass; real: layer computation)
            token = self._forward_pass(layer_id, prompt_tokens)
            output_tokens.append(token)

            # Evict before loading next layer
            evicted = self._streamer.evict_current()

            # Assert VRAM budget not exceeded
            assert self._streamer.active_layer is None, (
                f"VRAM breach: layer {evicted} still active after eviction"
            )

        self._evidence.emit(
            {
                "type": EvidenceType.SERVE_READY.value,
                "data": {
                    "mode": "layer_streaming",
                    "layers_processed": len(self._layer_ids),
                    "output_tokens": len(output_tokens),
                    "vram_peak_layers": 1,  # never exceeded
                },
            },
            organ="stream",
        )

        return output_tokens

    def _forward_pass(self, layer_id: str, tokens: List[int]) -> int:
        """Simulate a layer forward pass. Returns one output token."""
        if self._mock_mode:
            return hash(layer_id) % 50256  # deterministic mock token
        # Real: run actual transformer layer forward
        raise InferenceLoopError(
            "Real forward pass not implemented; use mock_mode=True in CI",
            reason="real_mode_not_implemented",
        )
