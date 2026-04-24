"""vLLM inference backend for DanteDojo.

Manages the lifecycle of a vLLM subprocess: start, health-check, stop.
In v1, the subprocess is only launched when DOJO_REAL_GPU=1 is set;
otherwise the backend validates that vllm is importable and documents
the launch command without executing it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from dojo.core.exceptions import BackendUnavailable, ServerHealthTimeout, ServerStartError
from dojo.core.types import HALProfile

_HEALTH_POLL_INTERVAL = 2.0  # seconds between health polls


class VLLMBackend:
    """Manages a vLLM server subprocess."""

    def __init__(
        self,
        model_path: Path,
        profile: HALProfile,
        port: int,
        context_length: int,
    ) -> None:
        self._model_path = model_path
        self._profile = profile
        self._port = port
        self._context_length = context_length
        self._pid: int = 0
        self._process: Optional[object] = None  # subprocess.Popen when real

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Launch vLLM subprocess.

        Returns:
            PID of the launched process (stub PID when not in real-GPU mode).

        Raises:
            BackendUnavailable: if vllm is not installed.
            ServerStartError: if the subprocess fails to start.
        """
        _require_vllm()

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
                    f"Failed to start vLLM: {exc}",
                    reason=str(exc),
                ) from exc
        else:
            # Stub mode: document command, return synthetic PID
            self._pid = 0
            self._pid = _stub_pid()

        return self._pid

    def health_check(self, timeout_seconds: float = 60.0) -> bool:
        """Poll GET /v1/models until the server is healthy or timeout expires.

        Returns:
            True when the server responds with HTTP 200.

        Raises:
            ServerHealthTimeout: if timeout elapses before health passes.
        """
        if not _real_gpu_mode():
            # In stub mode, immediately report healthy
            return True

        import urllib.request
        import urllib.error

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.endpoint_url}/v1/models", timeout=5.0
                ) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL)

        raise ServerHealthTimeout(
            f"vLLM did not become healthy within {timeout_seconds}s",
            reason=f"endpoint={self.endpoint_url}",
        )

    def stop(self) -> None:
        """Send SIGTERM to the vLLM subprocess for graceful drain."""
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
        """Build the vLLM server launch command list."""
        tp = _tp_size_for_profile(self._profile)
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self._model_path),
            "--port", str(self._port),
            "--max-model-len", str(self._context_length),
            "--tensor-parallel-size", str(tp),
            "--trust-remote-code",
        ]
        return cmd


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _require_vllm() -> None:
    """Raise BackendUnavailable if vllm is not importable."""
    try:
        import vllm  # noqa: F401
    except ImportError as exc:
        raise BackendUnavailable(
            "vllm is not installed. Install with: pip install vllm",
            reason="vllm package not found",
        ) from exc


def _real_gpu_mode() -> bool:
    return os.environ.get("DOJO_REAL_GPU", "0") == "1"


def _stub_pid() -> int:
    """Return a deterministic non-zero stub PID for tests."""
    return 99999


def _tp_size_for_profile(profile: HALProfile) -> int:
    """Derive tensor-parallel size from HAL profile."""
    from dojo.core.types import HardwareTier
    if profile.tier == HardwareTier.TIER_3090_DUAL:
        return 2
    return 1


__all__ = ["VLLMBackend"]
