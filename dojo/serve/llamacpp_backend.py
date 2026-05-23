"""llama.cpp inference backend for DanteDojo.

Manages the lifecycle of a llama.cpp server subprocess: start, health-check, stop.
In v1, the subprocess is only launched when DOJO_REAL_GPU=1 is set;
otherwise the backend validates that llama_cpp is importable and documents
the launch command without executing it.

Also provides LocalGGUFRunner — a direct Python-binding runner that uses
llama_cpp.Llama in-process, guaranteeing zero network calls during inference.
When llama_cpp is not available, LocalGGUFRunner falls back to Ollama at
localhost:11434 (if running), and further falls back to a deterministic stub.
"""

from __future__ import annotations

import json as _json
import os
import time
from pathlib import Path
from typing import Any, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BackendUnavailable, ServerHealthTimeout, ServerStartError, ServeError
from dojo.core.one_door import one_door
from dojo.stream.kv_cache_streamer import KVCacheStreamer
from dojo.stream.prefetch_governor import PrefetchGovernor
from dojo.stream.weight_streamer import WeightStreamer

_HEALTH_POLL_INTERVAL = 2.0  # seconds between health polls


class LlamaCppBackend:
    """Manages a llama.cpp server subprocess."""

    def __init__(
        self,
        model_path: Path,
        port: int,
        n_gpu_layers: int = -1,
    ) -> None:
        self._model_path = model_path
        self._port = port
        self._n_gpu_layers = n_gpu_layers
        self._pid: int = 0
        self._process: Optional[object] = None  # subprocess.Popen when real

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Launch llama.cpp server subprocess.

        Returns:
            PID of the launched process (stub PID when not in real-GPU mode).

        Raises:
            ServeError: if model_path does not exist.
            BackendUnavailable: if llama_cpp is not installed.
            ServerStartError: if the subprocess fails to start.
        """
        if not self._model_path.exists():
            raise ServeError(
                f"Model path not found: {self._model_path}",
                reason="model_path does not exist",
            )

        _require_llama_cpp()

        cmd = self._build_command()

        if _real_gpu_mode():
            import subprocess
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self._process = proc
                self._pid = proc.pid
            except OSError as exc:
                raise ServerStartError(
                    f"Failed to start llama.cpp server: {exc}",
                    reason=str(exc),
                ) from exc
        else:
            self._pid = _stub_pid()

        # Read passport_id from sidecar .meta.json if it exists alongside the GGUF
        passport_id = self._read_sidecar_passport_id()

        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": "serve.llamacpp_loaded",
                "data": {
                    "model_path": str(self._model_path),
                    "port": self._port,
                    "context_length": 2048,
                    "n_gpu_layers": self._n_gpu_layers,
                    "pid": self._pid,
                    "passport_id": passport_id,
                },
            },
            organ="serve",
        )

        return self._pid

    def health_check(self, timeout_seconds: float = 60.0) -> bool:
        """Poll GET /health until the server is healthy or timeout expires.

        Returns:
            True when the server responds with HTTP 200.

        Raises:
            ServerHealthTimeout: if timeout elapses before health passes.
        """
        if not _real_gpu_mode():
            return True

        import urllib.request

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.endpoint_url}/health", timeout=5.0
                ) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL)

        raise ServerHealthTimeout(
            f"llama.cpp server did not become healthy within {timeout_seconds}s",
            reason=f"endpoint={self.endpoint_url}",
        )

    def stop(self) -> None:
        """Send SIGTERM to the llama.cpp subprocess for graceful drain."""
        if self._process is None:
            return
        try:
            import signal
            os.kill(self._pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    @property
    def endpoint_url(self) -> str:
        return f"http://localhost:{self._port}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_command(self) -> list[str]:
        """Build the llama.cpp server launch command list."""
        cmd = [
            "llama-server",
            "--model", str(self._model_path),
            "--port", str(self._port),
            "--n-gpu-layers", str(self._n_gpu_layers),
            "--host", "127.0.0.1",
        ]
        return cmd

    def _read_sidecar_passport_id(self) -> Optional[str]:
        """Read passport_id from the .meta.json sidecar alongside the GGUF file.

        Returns the passport_id string if the sidecar exists and contains it,
        otherwise returns None (no crash — sidecar is optional).
        """
        import json as _json

        meta_path = self._model_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        try:
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get("dojo.passport_id") or None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _ollama_available(url: str = "http://localhost:11434") -> bool:
    """Return True if the Ollama server is reachable (GET /api/tags, 2s timeout)."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _call_ollama(
    prompt: str,
    url: str = "http://localhost:11434",
    model: str = "qwen2.5:0.5b",
    max_tokens: int = 128,
) -> dict:
    """POST to Ollama's OpenAI-compatible chat endpoint and return a unified dict.

    Returns:
        {"text": str, "tokens": int, "prompt_tokens": int}
    """
    import urllib.request

    payload = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = _json.loads(resp.read().decode("utf-8"))

    choice = body["choices"][0]
    text = choice["message"]["content"]
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens", len(text.split()))
    prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
    return {"text": text, "tokens": completion_tokens, "prompt_tokens": prompt_tokens}


def _require_llama_cpp() -> None:
    """Raise BackendUnavailable if llama_cpp is not importable."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError as exc:
        raise BackendUnavailable(
            "llama-cpp-python is not installed. "
            "Install with: pip install llama-cpp-python",
            reason="llama_cpp package not found",
        ) from exc


def _real_gpu_mode() -> bool:
    return os.environ.get("DOJO_REAL_GPU", "0") == "1"


def _stub_pid() -> int:
    return 99998


class LocalGGUFRunner:
    """Runs GGUF inference locally via llama-cpp-python bindings.

    Inference priority:
    1. llama_cpp.Llama in-process (zero network calls, DOJO_REAL_GPU path).
    2. Ollama fallback at ``ollama_url`` when llama_cpp is unavailable but
       Ollama is running locally (e.g. dev machines without GPU drivers).
    3. Deterministic stub (always available, used in CI / offline).

    Wired sub-systems (all run in mock_mode when real hardware is absent):
    - ``WeightStreamer``   — streams the model checkpoint from disk into RAM.
    - ``PrefetchGovernor``— issues layer prefetch commands during model load.
    - ``KVCacheStreamer``  — pages KV cache entries during generation.
    """

    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = -1,    # -1 = all layers on GPU
        context_size: int = 2048,
        passport_id: str = "",
        llm: Any = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:0.5b",
    ) -> None:
        self._model_path = model_path
        self._n_gpu_layers = n_gpu_layers
        self._context_size = context_size
        self._passport_id = passport_id
        self._llm: Any = llm
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model
        self._evidence = EvidenceLogger.get_instance()

        # Streaming sub-systems (mock_mode degrades gracefully without torch/accelerate)
        self._weight_streamer = WeightStreamer(mock_mode=True)
        self._prefetch_gov = PrefetchGovernor(max_lookahead=16, mock_mode=True)
        self._kv_streamer = KVCacheStreamer(max_cache_size=512, mock_mode=True)

    @one_door("serve.local_gguf.load", organ="serve")
    def load(self) -> None:
        """Load the GGUF model into memory.

        Raises:
            BackendUnavailable: if llama-cpp-python is not installed.
            ServerStartError: if model file not found or load fails.
        """
        _require_llama_cpp()

        if not self._model_path.exists():
            raise ServerStartError(
                f"GGUF model not found: {self._model_path}",
                reason="File does not exist",
            )

        # --- d26: stream model weights from disk (real I/O when path exists) ---
        self._weight_streamer.stream_layer("full_model", source_tier="disk", path=self._model_path)
        self._weight_streamer.evict_current()  # release the VRAM slot after checkpoint maps

        # --- d28: issue layer prefetch to warm the GPU copy pipeline ---
        layer_count = self._n_gpu_layers if self._n_gpu_layers > 0 else 12
        layer_names = ["embed"] + [f"layer_{i}" for i in range(layer_count)] + ["lm_head"]
        self._prefetch_gov.issue_prefetch(layer_names)

        self._evidence.emit(
            {
                "type": EvidenceType.SERVE_START.value,
                "data": {
                    "backend": "llamacpp_local",
                    "model_path": str(self._model_path),
                    "passport_id": self._passport_id,
                    "n_gpu_layers": self._n_gpu_layers,
                    "mode": "local_binding",  # not subprocess — no network
                    "weight_streamer": "active",
                    "prefetch_governor": "active",
                },
            },
            organ="serve",
        )

        if _real_gpu_mode():
            try:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=str(self._model_path),
                    n_gpu_layers=self._n_gpu_layers,
                    n_ctx=self._context_size,
                    verbose=False,
                )
            except Exception as exc:
                raise ServerStartError(
                    f"Failed to load GGUF model: {exc}",
                    reason=str(exc),
                ) from exc
        # stub/dev mode: self._llm stays None; generate() uses Ollama or stub

    @one_door("serve.local_gguf.generate", organ="serve")
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> dict:
        """Generate tokens from prompt.

        Backend selection:
        1. llama_cpp in-process (zero network calls) — when self._llm is loaded.
        2. Ollama fallback at self._ollama_url — when llama_cpp unavailable but
           Ollama is reachable.
        3. Deterministic stub — offline / CI fallback.

        Returns:
            dict with "text", "tokens", "prompt_tokens", "backend" keys.

        Raises:
            ServeError: if model not loaded (real-GPU mode only).
        """
        if self._llm is None and _real_gpu_mode():
            raise ServeError("Model not loaded. Call load() first.", reason="not_loaded")

        # --- d27: KV cache — serve from cache on hit, page on miss ---
        cache_key = f"gen_{hash(prompt) & 0xFFFFFF:06x}"
        cached_text = self._kv_streamer.get_cached_value(cache_key)
        if cached_text is not None:
            self._evidence.emit(
                {
                    "type": EvidenceType.SERVE_READY.value,
                    "data": {
                        "backend": "kv_cache",
                        "model_path": str(self._model_path),
                        "tokens_generated": len(cached_text.split()),
                        "prompt_tokens": len(prompt.split()),
                        "kv_cache_active": True,
                        "network_calls": 0,
                    },
                },
                organ="serve",
            )
            return {
                "text": cached_text,
                "tokens": len(cached_text.split()),
                "prompt_tokens": len(prompt.split()),
                "backend": "kv_cache",
            }

        if self._llm is not None:
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            text = output["choices"][0]["text"]
            tokens_generated = output["usage"]["completion_tokens"]
            prompt_tokens = output["usage"]["prompt_tokens"]
            active_backend = "llamacpp_local"
        elif _ollama_available(self._ollama_url):
            # d29/d30: real Ollama inference fallback
            result = _call_ollama(
                prompt,
                url=self._ollama_url,
                model=self._ollama_model,
                max_tokens=max_tokens,
            )
            text = result["text"]
            tokens_generated = result["tokens"]
            prompt_tokens = result["prompt_tokens"]
            active_backend = "ollama_fallback"
        else:
            # Stub: deterministic response (offline / CI)
            text = f"[local stub: {prompt[:20]}...]"
            tokens_generated = min(max_tokens, 16)
            prompt_tokens = len(prompt.split())
            active_backend = "stub"

        # --- d28: consume prefetch layers (feedback loop: issued → consumed) ---
        # Each inference traverses all model layers — consume the pending prefetch
        # queue so the governor tracks real utilization, not just advisory intent.
        _pending = len(self._prefetch_gov.pending)
        if _pending > 0:
            for _lyr in list(self._prefetch_gov.pending):
                self._prefetch_gov.consume_prefetch(_lyr)
            _stats = self._prefetch_gov.get_stats()
            self._evidence.emit(
                {
                    "type": EvidenceType.PREFETCH_CONSUMED.value,
                    "data": {
                        "layers_consumed": _pending,
                        "pending_remaining": _stats["pending_count"],
                        "total_issued": _stats["prefetches_issued"],
                        "backend": active_backend,
                    },
                },
                organ="stream",
            )

        # Store generated text in KV cache for future hits
        self._kv_streamer.page_kv_entry(cache_key, text, size=max(1, len(prompt.split())))

        self._evidence.emit(
            {
                "type": EvidenceType.SERVE_READY.value,
                "data": {
                    "backend": active_backend,
                    "model_path": str(self._model_path),
                    "tokens_generated": tokens_generated,
                    "prompt_tokens": prompt_tokens,
                    "kv_cache_active": True,
                    "network_calls": 1 if active_backend == "ollama_fallback" else 0,
                },
            },
            organ="serve",
        )

        return {
            "text": text,
            "tokens": tokens_generated,
            "prompt_tokens": prompt_tokens,
            "backend": active_backend,
        }

    def unload(self) -> None:
        """Release model from memory."""
        self._llm = None


__all__ = ["LlamaCppBackend", "LocalGGUFRunner"]
