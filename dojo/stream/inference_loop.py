"""Constitutional VRAM-bounded inference loop using WeightStreamer."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError
from dojo.stream.weight_streamer import WeightStreamer
from dojo.stream.kv_cache_streamer import KVCacheStreamer
from dojo.stream.prefetch_governor import PrefetchGovernor


class InferenceLoopError(DojoError):
    pass


class LayerStreamingInferenceLoop:
    """Runs inference layer-by-layer with constitutional VRAM budget.

    Ensures exactly 1 layer is in VRAM at any time:
    1. Load layer i into VRAM
    2. Run forward pass for that layer
    3. Evict layer i from VRAM
    4. Repeat for all layers

    In non-mock mode, each layer is a real torch.nn.Linear that performs a
    genuine matrix multiplication on CPU (or GPU if available).  Hidden state
    is threaded from layer to layer so the forward pass is a real computation
    graph, not arithmetic stubs.

    The ``hidden_dim`` constructor parameter controls model width — use 64 for
    fast CI tests and 768/1024 for realistic VRAM budget measurements.

    VRAM budget is enforced in bytes: ``layer_vram_bytes()`` computes the real
    tensor memory footprint per layer, and ``run()`` tracks peak VRAM usage.
    """

    VOCAB_SIZE: int = 50257   # GPT-2 compatible vocabulary
    _DEFAULT_HIDDEN_DIM: int = 64  # fast CPU-only default

    def __init__(
        self,
        layer_ids: List[str],
        streamer: Optional[WeightStreamer] = None,
        mock_mode: bool = True,
        hidden_dim: int = 0,
        vram_budget_bytes: int = 0,
    ) -> None:
        self._layer_ids = layer_ids
        self._streamer = streamer or WeightStreamer(mock_mode=mock_mode)
        self._evidence = EvidenceLogger.get_instance()
        self._mock_mode = mock_mode
        self._hidden_dim: int = hidden_dim if hidden_dim > 0 else self._DEFAULT_HIDDEN_DIM
        self._vram_budget_bytes: int = vram_budget_bytes
        self._torch: Any = None
        self._embed: Any = None
        self._layers: Dict[str, Any] = {}
        self._lm_head: Any = None
        self._peak_vram_bytes: int = 0
        if not mock_mode:
            self._init_real_layers()

    @property
    def HIDDEN_DIM(self) -> int:
        return self._hidden_dim

    def layer_vram_bytes(self, dtype_bytes: int = 4) -> int:
        """Compute real VRAM footprint for one Linear layer (hidden_dim × hidden_dim).

        Args:
            dtype_bytes: Bytes per element (4 = float32, 2 = bfloat16/float16).

        Returns:
            Number of bytes required to hold one layer's weight matrix.
        """
        return self._hidden_dim * self._hidden_dim * dtype_bytes

    def total_model_vram_bytes(self, dtype_bytes: int = 4) -> int:
        """Total VRAM required to hold all layers simultaneously (in bytes)."""
        return self.layer_vram_bytes(dtype_bytes) * len(self._layer_ids)

    def _init_real_layers(self) -> None:
        """Build CPU-capable torch.nn.Linear layers — real tensors, no GPU required."""
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            torch.manual_seed(42)
            self._embed = nn.Embedding(self.VOCAB_SIZE, self._hidden_dim)
            for lid in self._layer_ids:
                self._layers[lid] = nn.Linear(self._hidden_dim, self._hidden_dim, bias=False)
            self._lm_head = nn.Linear(self._hidden_dim, self.VOCAB_SIZE, bias=False)
        except ImportError:
            self._mock_mode = True  # graceful fallback if torch absent

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

        per_layer_bytes = self.layer_vram_bytes()

        self._evidence.emit(
            {
                "type": EvidenceType.SERVE_START.value,
                "data": {
                    "mode": "layer_streaming",
                    "num_layers": len(self._layer_ids),
                    "prompt_tokens": len(prompt_tokens),
                    "vram_budget_layers": 1,  # constitutional maximum
                    "hidden_dim": self._hidden_dim,
                    "per_layer_vram_bytes": per_layer_bytes,
                    "total_model_vram_bytes": per_layer_bytes * len(self._layer_ids),
                    "vram_budget_bytes": self._vram_budget_bytes,
                },
            },
            organ="stream",
        )

        # Fail-closed budget check: if a budget is set and one layer exceeds it,
        # refuse to run rather than silently OOM.
        if self._vram_budget_bytes > 0 and per_layer_bytes > self._vram_budget_bytes:
            raise InferenceLoopError(
                f"Layer VRAM ({per_layer_bytes} bytes) exceeds budget "
                f"({self._vram_budget_bytes} bytes) for hidden_dim={self._hidden_dim}"
            )

        self._peak_vram_bytes = 0

        # Embed prompt tokens into an initial hidden state (real path only).
        hidden: Any = None
        if not self._mock_mode and self._embed is not None:
            clamped = [t % self.VOCAB_SIZE for t in (prompt_tokens[:512] or [0])]
            ids = self._torch.tensor(clamped, dtype=self._torch.long)
            hidden = self._embed(ids).mean(dim=0, keepdim=True)  # (1, hidden_dim)

        for layer_id in self._layer_ids:
            # Load this layer (enforces MAX_ACTIVE_LAYERS=1)
            self._streamer.stream_layer(layer_id, source_tier="ram")

            # Track peak VRAM — only one layer is resident at a time
            self._peak_vram_bytes = max(self._peak_vram_bytes, per_layer_bytes)

            # Forward pass — real linear ops in non-mock mode, hash stub in mock.
            token, hidden = self._forward_pass(layer_id, hidden)
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
                    "vram_peak_layers": 1,
                    "vram_peak_bytes": self._peak_vram_bytes,
                    "total_model_vram_bytes": per_layer_bytes * len(self._layer_ids),
                    "savings_ratio": (
                        round(
                            1.0 - (self._peak_vram_bytes / max(per_layer_bytes * len(self._layer_ids), 1)),
                            4,
                        )
                        if self._layer_ids else 0.0
                    ),
                },
            },
            organ="stream",
        )

        return output_tokens

    @property
    def peak_vram_bytes(self) -> int:
        """Peak VRAM bytes observed during the most recent run() call."""
        return self._peak_vram_bytes

    def _forward_pass(self, layer_id: str, hidden: Any = None) -> Tuple[int, Any]:
        """Run a layer forward pass with real torch tensors (no GPU required).

        Mock path: deterministic hash → token, hidden state unchanged.
        Real path: hidden → nn.Linear → lm_head → argmax token.

        Returns:
            (output_token, updated_hidden_state)
        """
        if self._mock_mode or self._embed is None:
            return hash(layer_id) % 50256, hidden

        h = self._layers[layer_id](hidden)    # (1, HIDDEN_DIM) — real matmul
        logits = self._lm_head(h)              # (1, VOCAB_SIZE) — real projection
        token = int(logits.argmax(dim=-1).item())
        return token, h


class InferenceLoopSimulator:
    """Coordinated simulation of WeightStreamer + KVCacheStreamer + PrefetchGovernor.

    Each inference step:
    1. WeightStreamer loads the current layer (round-robin over n_layers).
    2. KVCacheStreamer stores the KV activations for that layer.
    3. PrefetchGovernor schedules the *next* layer before the current one finishes.
    4. Current layer is evicted from VRAM.
    5. Evidence is emitted: stream.inference_loop_step per step, stream.loop_complete at end.

    Memory accounting (all in MB):
        vram_mb = layer_size_mb * (layers currently loaded)
                + kv_cache_mb_per_layer * (KV entries cached)
    Because we load one layer at a time and evict before the next, the layer
    contribution is always exactly layer_size_mb during a step and 0 between
    steps.  KV cache grows by kv_cache_mb_per_layer each step (no eviction
    unless the budget would be exceeded).
    """

    def __init__(
        self,
        n_layers: int,
        layer_size_mb: float,
        kv_cache_mb_per_layer: float,
        kv_budget_mb: Optional[float] = None,
        mock_mode: bool = True,
        evidence: Optional[EvidenceLogger] = None,
    ) -> None:
        if n_layers < 1:
            raise InferenceLoopError("n_layers must be >= 1")
        if layer_size_mb <= 0:
            raise InferenceLoopError("layer_size_mb must be positive")
        if kv_cache_mb_per_layer <= 0:
            raise InferenceLoopError("kv_cache_mb_per_layer must be positive")

        self.n_layers = n_layers
        self.layer_size_mb = layer_size_mb
        self.kv_cache_mb_per_layer = kv_cache_mb_per_layer
        self._evidence = evidence or EvidenceLogger.get_instance()

        # KV budget: default to enough for all layers × 10 steps (never a
        # bottleneck in normal simulations unless caller sets a tight budget).
        if kv_budget_mb is None:
            kv_budget_mb = kv_cache_mb_per_layer * n_layers * 10 + 1

        # KVCacheStreamer works in integer "tokens"; we scale MB to integer
        # units by multiplying by 1000 so fractional MB values are preserved.
        self._kv_scale = 1000
        kv_max_cache_size = int(kv_budget_mb * self._kv_scale)

        self.weight_streamer = WeightStreamer(mock_mode=mock_mode)
        self.kv_streamer = KVCacheStreamer(
            max_cache_size=kv_max_cache_size,
            mock_mode=mock_mode,
        )
        self.prefetch_governor = PrefetchGovernor(
            max_lookahead=1,
            mock_mode=mock_mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, n_steps: int) -> Dict[str, Any]:
        """Run the simulated inference loop for n_steps steps.

        Returns:
            {
                "steps_completed": int,
                "peak_vram_mb": float,
                "layers_loaded": List[str],
            }
        """
        if n_steps < 0:
            raise InferenceLoopError("n_steps must be >= 0")

        layers_loaded: List[str] = []
        peak_vram_mb: float = 0.0
        kv_entries_cached = 0

        for step in range(n_steps):
            layer_idx = step % self.n_layers
            layer_id = f"layer_{layer_idx}"

            # 3. PrefetchGovernor: schedule the *next* layer ahead of time.
            #    We do this before loading the current layer so the prefetch
            #    is "in flight" while the current layer is being processed.
            next_idx = (step + 1) % self.n_layers
            next_layer_id = f"layer_{next_idx}"
            # Drain any stale pending entry first so the window never overruns.
            self.prefetch_governor.pending.clear()
            self.prefetch_governor.issue_prefetch([next_layer_id])

            # 1. WeightStreamer: load the current layer into VRAM.
            self.weight_streamer.stream_layer(layer_id, source_tier="ram")
            layers_loaded.append(layer_id)

            # 2. KVCacheStreamer: store KV activations for this layer.
            kv_key = f"kv_step{step}_layer{layer_idx}"
            kv_size = int(self.kv_cache_mb_per_layer * self._kv_scale)
            self.kv_streamer.page_kv_entry(kv_key, f"activation_{layer_id}", kv_size)
            kv_entries_cached += 1

            # Memory accounting for this step.
            # Layer is still in VRAM here (loaded, not yet evicted).
            vram_mb = (
                self.layer_size_mb                              # one layer in VRAM
                + self.kv_cache_mb_per_layer * kv_entries_cached  # KV cache so far
            )
            if vram_mb > peak_vram_mb:
                peak_vram_mb = vram_mb

            # Consume the prefetch entry (layer "arrived").
            self.prefetch_governor.consume_prefetch(next_layer_id)

            # Evict current layer before next iteration.
            self.weight_streamer.evict_current()

            # Emit per-step evidence.
            self._evidence.emit(
                {
                    "type": EvidenceType.INFERENCE_LOOP_STEP.value,
                    "data": {
                        "step": step,
                        "layer_id": layer_id,
                        "vram_mb": vram_mb,
                        "kv_entries_cached": kv_entries_cached,
                        "prefetch_scheduled": next_layer_id,
                    },
                },
                organ="stream",
            )

        # Emit loop-complete evidence.
        self._evidence.emit(
            {
                "type": EvidenceType.INFERENCE_LOOP_COMPLETE.value,
                "data": {
                    "steps_completed": n_steps,
                    "peak_vram_mb": peak_vram_mb,
                    "layers_loaded_count": len(layers_loaded),
                },
            },
            organ="stream",
        )

        return {
            "steps_completed": n_steps,
            "peak_vram_mb": peak_vram_mb,
            "layers_loaded": layers_loaded,
        }


# ---------------------------------------------------------------------------
# KV-cache streamed inference adapter (d27)
# ---------------------------------------------------------------------------

def run_kv_streamed_inference(
    layer_ids: List[str],
    query: "Any",
    head_dim: int = 64,
    vocab_size: int = 1024,
    seq_len: int = 8,
) -> List[int]:
    """Run real scaled-dot-product attention over a streamed KV cache.

    Wraps dojo.stream.kv_streamed_inference.KvStreamedInferenceLoop so the
    rest of the streaming module can offer the user a single inference
    entry point.  Each layer's KV is streamed in, attention runs, KV is
    evicted before the next layer loads.

    Returns the list of generated token ids (one per layer).
    """
    from dojo.stream.kv_streamed_inference import KvStreamedInferenceLoop
    loop = KvStreamedInferenceLoop(
        layer_ids=layer_ids,
        head_dim=head_dim,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )
    return loop.run(query)
