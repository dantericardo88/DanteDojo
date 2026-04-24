"""llama.cpp inference backend for DanteDojo.

Manages the lifecycle of a llama.cpp server subprocess: start, health-check, stop.
In v1, the subprocess is only launched when DOJO_REAL_GPU=1 is set;
otherwise the backend validates that llama_cpp is importable and documents
the launch command without executing it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from dojo.core.exceptions import BackendUnavailable, ServerHealthTimeout, ServerStartError

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
            BackendUnavailable: if llama_cpp is not installed.
            ServerStartError: if the subprocess fails to start.
        """
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


__all__ = ["LlamaCppBackend"]
