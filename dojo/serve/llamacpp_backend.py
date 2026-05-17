"""llama.cpp inference backend for DanteDojo.

Manages the lifecycle of a llama.cpp server subprocess: start, health-check, stop.
In v1, the subprocess is only launched when DOJO_REAL_GPU=1 is set;
otherwise the backend validates that llama_cpp is importable and documents
the launch command without executing it.

Also provides LocalGGUFRunner — a direct Python-binding runner that uses
llama_cpp.Llama in-process, guaranteeing zero network calls during inference.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BackendUnavailable, ServerHealthTimeout, ServerStartError, ServeError
from dojo.core.one_door import one_door

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

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

    Constitutional guarantee: no network calls during inference.
    All generation happens in-process via the llama_cpp Python API,
    NOT via subprocess or any HTTP server.
    """

    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = -1,    # -1 = all layers on GPU
        context_size: int = 2048,
        passport_id: str = "",
    ) -> None:
        self._model_path = model_path
        self._n_gpu_layers = n_gpu_layers
        self._context_size = context_size
        self._passport_id = passport_id
        self._llm: Any = None
        self._evidence = EvidenceLogger.get_instance()

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

        self._evidence.emit(
            {
                "type": EvidenceType.SERVE_START.value,
                "data": {
                    "backend": "llamacpp_local",
                    "model_path": str(self._model_path),
                    "passport_id": self._passport_id,
                    "n_gpu_layers": self._n_gpu_layers,
                    "mode": "local_binding",  # not subprocess — no network
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
        # stub mode: self._llm stays None

    @one_door("serve.local_gguf.generate", organ="serve")
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> dict:
        """Generate tokens from prompt. No network calls made.

        Returns:
            dict with "text", "tokens", "prompt_tokens" keys.

        Raises:
            ServeError: if model not loaded (real-GPU mode only).
        """
        if self._llm is None and _real_gpu_mode():
            raise ServeError("Model not loaded. Call load() first.", reason="not_loaded")

        if _real_gpu_mode():
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            text = output["choices"][0]["text"]
            tokens_generated = output["usage"]["completion_tokens"]
            prompt_tokens = output["usage"]["prompt_tokens"]
        else:
            # Stub: deterministic response
            text = f"[local stub: {prompt[:20]}...]"
            tokens_generated = min(max_tokens, 16)
            prompt_tokens = len(prompt.split())

        self._evidence.emit(
            {
                "type": EvidenceType.SERVE_READY.value,
                "data": {
                    "backend": "llamacpp_local",
                    "model_path": str(self._model_path),
                    "tokens_generated": tokens_generated,
                    "prompt_tokens": prompt_tokens,
                    "network_calls": 0,  # constitutional guarantee
                },
            },
            organ="serve",
        )

        return {
            "text": text,
            "tokens": tokens_generated,
            "prompt_tokens": prompt_tokens,
        }

    def unload(self) -> None:
        """Release model from memory."""
        self._llm = None


__all__ = ["LlamaCppBackend", "LocalGGUFRunner"]
