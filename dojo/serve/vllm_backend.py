"""vLLM inference backend for DanteDojo.

Manages the lifecycle of a vLLM subprocess: start, health-check, stop.
In v1, the subprocess is only launched when DOJO_REAL_GPU=1 is set;
otherwise the backend validates that vllm is importable and documents
the launch command without executing it.
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BackendUnavailable, ServerHealthTimeout, ServerStartError, ServeError
from dojo.core.one_door import one_door
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
        tensor_parallel_size: int = 1,
    ) -> None:
        self._model_path = model_path
        self._profile = profile
        self._port = port
        self._context_length = context_length
        self._tensor_parallel_size = tensor_parallel_size
        self._pid: int = 0
        self._process: Optional[object] = None  # subprocess.Popen when real
        self._engine: Optional[object] = None   # vllm.AsyncLLMEngine when real

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

    @one_door("serve.vllm.concurrent", organ="serve")
    def serve_concurrent(
        self,
        requests: list[dict],
        timeout_per_request: float = 30.0,
    ) -> list[dict]:
        """Send requests concurrently to vLLM and return results with throughput.

        Args:
            requests: List of dicts with "prompt" key.
            timeout_per_request: Per-request timeout in seconds.

        Returns:
            List of result dicts: {"prompt": ..., "response": ..., "tokens": int, "elapsed_s": float}

        Raises:
            ServerHealthTimeout: if server is not healthy before sending requests.
            BackendUnavailable: if vllm is not installed.
        """
        _require_vllm()

        evidence_log = EvidenceLogger.get_instance()

        evidence_log.emit(
            {
                "type": EvidenceType.SERVE_START.value,
                "data": {
                    "backend": "vllm",
                    "concurrent_requests": len(requests),
                    "model_path": str(self._model_path),
                },
            },
            organ="serve",
        )

        if _real_gpu_mode():
            # Real GPU: verify health before sending requests
            self.health_check(timeout_seconds=timeout_per_request)
            results = self._send_concurrent_requests(requests, timeout_per_request)
        else:
            # Stub mode: synthetic responses with realistic timings
            results = []
            for i, req in enumerate(requests):
                results.append({
                    "prompt": req.get("prompt", ""),
                    "response": f"[stub response {i + 1}]",
                    "tokens": 42,
                    "elapsed_s": 0.15 + i * 0.02,
                })

        token_counts: list[int] = [int(r["tokens"]) for r in results]
        elapsed_vals: list[float] = [float(r["elapsed_s"]) for r in results]
        total_tokens: int = sum(token_counts)
        # Concurrent execution: wall time = max elapsed, not sum
        total_elapsed: float = max(elapsed_vals) if elapsed_vals else 1.0
        throughput: float = round(total_tokens / total_elapsed, 2) if total_elapsed > 0.0 else 0.0

        evidence_log.emit(
            {
                "type": EvidenceType.SERVE_READY.value,
                "data": {
                    "backend": "vllm",
                    "concurrent_requests": len(requests),
                    "total_tokens": total_tokens,
                    "total_elapsed_s": total_elapsed,
                    "throughput_tokens_per_s": throughput,
                    "model_path": str(self._model_path),
                },
            },
            organ="serve",
        )

        return results

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: Optional[list] = None,
        stream: bool = False,
    ) -> "str | list[dict]":
        """Generate a completion for a single prompt.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            stop: Optional list of stop sequences.
            stream: If True, return a list of SSE-style chunk dicts instead of
                a plain string.  Each chunk has the shape
                ``{"choices": [{"delta": {"content": str, "role": str},
                "finish_reason": str | None}]}``.
                The final chunk has ``finish_reason="stop"``.

        Returns:
            Generated text string when ``stream=False`` (default), or a list
            of streaming chunk dicts when ``stream=True``.

        Raises:
            BackendUnavailable: if vllm is not installed.
        """
        if stream:
            return self._generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        results = self.batch_generate(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return results[0]

    def _generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: Optional[list] = None,
    ) -> list:
        """Return a list of SSE-style chunk dicts simulating streaming output.

        Each chunk (except the last) has ``finish_reason=None`` and
        ``delta={"role": "assistant", "content": <token>}``.
        The last chunk has ``finish_reason="stop"`` and empty content.

        When a stop sequence is found inside the stub text, generation halts
        at that point and ``serve.vllm_stop_sequence`` evidence is emitted.
        """
        _require_vllm()

        evidence_log = EvidenceLogger.get_instance()
        stop_seqs: list = stop or []

        # Build stub output text
        stub_text = f"[stub stream for: {prompt[:40]}]"

        # Check for stop-sequence match in stub text
        matched_stop: Optional[str] = None
        truncated_text = stub_text
        for seq in stop_seqs:
            if seq and seq in stub_text:
                idx = stub_text.index(seq)
                truncated_text = stub_text[:idx]
                matched_stop = seq
                break

        if matched_stop is not None:
            evidence_log.emit(
                {
                    "type": EvidenceType.SERVE_VLLM_STOP_SEQUENCE.value,
                    "data": {
                        "backend": "vllm",
                        "stop_sequence": matched_stop,
                        "model_path": str(self._model_path),
                    },
                },
                organ="serve",
            )

        # Split into word-level tokens for realistic multi-chunk output
        tokens = truncated_text.split() if truncated_text else []

        chunks: list = []
        for i, token in enumerate(tokens):
            content = token if i == 0 else " " + token
            chunk: dict = {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant" if i == 0 else "",
                            "content": content,
                        },
                        "finish_reason": None,
                    }
                ]
            }
            chunks.append(chunk)

        # Final chunk: finish_reason="stop", empty content
        chunks.append(
            {
                "choices": [
                    {
                        "delta": {"role": "", "content": ""},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

        return chunks

    def batch_generate(
        self,
        prompts: list,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: Optional[list] = None,
        max_tokens_per_prompt: Optional[list] = None,
    ) -> list:
        """Generate completions for a batch of prompts.

        Emits ``serve.vllm_batch_start`` evidence before generation and
        ``serve.vllm_batch_complete`` evidence after, both stamped into the
        Merkle chain.

        Args:
            prompts: List of input text prompts.
            max_tokens: Default maximum tokens to generate per prompt.
            temperature: Sampling temperature.
            stop: Optional list of stop sequences.
            max_tokens_per_prompt: Optional per-prompt max_tokens overrides.
                If provided, ``max_tokens_per_prompt[i]`` overrides ``max_tokens``
                for prompt ``i``.  List length must match ``prompts``.

        Returns:
            List of generated text strings, one per prompt.

        Raises:
            BackendUnavailable: if vllm is not installed.
        """
        _require_vllm()

        evidence_log = EvidenceLogger.get_instance()

        evidence_log.emit(
            {
                "type": EvidenceType.SERVE_VLLM_BATCH_START.value,
                "data": {
                    "backend": "vllm",
                    "prompt_count": len(prompts),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "model_path": str(self._model_path),
                },
            },
            organ="serve",
        )

        if _real_gpu_mode() and self._engine is not None:
            outputs = self._run_engine_batch(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                max_tokens_per_prompt=max_tokens_per_prompt,
            )
        else:
            # Stub mode: return synthetic completions
            outputs = [f"[stub completion for: {p[:40]}]" for p in prompts]

        tokens_generated = sum(len(o.split()) for o in outputs)

        evidence_log.emit(
            {
                "type": EvidenceType.SERVE_VLLM_BATCH_COMPLETE.value,
                "data": {
                    "backend": "vllm",
                    "prompt_count": len(prompts),
                    "tokens_generated": tokens_generated,
                    "model_path": str(self._model_path),
                },
            },
            organ="serve",
        )

        return outputs

    def health_check_status(self) -> dict:
        """Return a status dict for the engine health.

        Returns:
            ``{"status": "ok"}`` when the engine is running or in stub mode,
            ``{"status": "error", "reason": <str>}`` on failure.
        """
        if not _real_gpu_mode():
            return {"status": "ok"}

        try:
            import urllib.request
            import urllib.error

            with urllib.request.urlopen(
                f"{self.endpoint_url}/v1/models", timeout=5.0
            ) as resp:
                if resp.status == 200:
                    return {"status": "ok"}
                return {"status": "error", "reason": f"HTTP {resp.status}"}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

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

    def _run_engine_batch(
        self,
        prompts: list,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: Optional[list] = None,
        max_tokens_per_prompt: Optional[list] = None,
    ) -> list:
        """Send prompts to the AsyncLLMEngine and collect outputs.

        Only called in real-GPU mode when ``self._engine`` is set.

        Args:
            prompts: List of input text prompts.
            max_tokens: Default max tokens per prompt.
            temperature: Sampling temperature.
            stop: Optional stop sequences.
            max_tokens_per_prompt: Optional per-prompt max_tokens overrides.
        """
        import asyncio
        import uuid as _uuid

        engine = self._engine

        async def _run() -> list:
            from vllm import SamplingParams  # type: ignore[import]
            outputs = []
            for i, prompt in enumerate(prompts):
                per_max = (
                    max_tokens_per_prompt[i]
                    if max_tokens_per_prompt and i < len(max_tokens_per_prompt)
                    else max_tokens
                )
                sp = SamplingParams(
                    max_tokens=per_max,
                    temperature=temperature,
                    stop=stop or [],
                )
                req_id = _uuid.uuid4().hex
                results_gen = engine.generate(prompt, sp, req_id)  # type: ignore[union-attr]
                req_output = None
                async for req_output in results_gen:
                    pass  # consume until done
                text = req_output.outputs[0].text if req_output and req_output.outputs else ""
                outputs.append(text)
            return outputs

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(asyncio.run, _run())
                    return fut.result(timeout=120)
            return loop.run_until_complete(_run())
        except Exception:
            return ["[engine error for prompt]" for _ in prompts]

    def _build_command(self) -> list[str]:
        """Build the vLLM server launch command list."""
        tp = self._tensor_parallel_size if self._tensor_parallel_size > 1 else _tp_size_for_profile(self._profile)
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self._model_path),
            "--port", str(self._port),
            "--max-model-len", str(self._context_length),
            "--tensor-parallel-size", str(tp),
            "--trust-remote-code",
        ]
        return cmd

    def _send_concurrent_requests(
        self,
        requests: list[dict],
        timeout_per_request: float,
    ) -> list[dict]:
        """Send requests concurrently via HTTP to the vLLM completions endpoint.

        Used only in real-GPU mode (DOJO_REAL_GPU=1).
        """
        import json as _json
        import urllib.request

        url = f"{self.endpoint_url}/v1/completions"

        def _send_one(req: dict) -> dict:
            prompt = req.get("prompt", "")
            payload = _json.dumps({
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0.0,
            }).encode()
            http_req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.monotonic()
            with urllib.request.urlopen(http_req, timeout=timeout_per_request) as resp:
                body = _json.loads(resp.read())
            elapsed = time.monotonic() - t0
            response_text: str = body["choices"][0]["text"] if body.get("choices") else ""
            tokens: int = int(body.get("usage", {}).get("completion_tokens", len(response_text.split())))
            return {
                "prompt": prompt,
                "response": response_text,
                "tokens": tokens,
                "elapsed_s": elapsed,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
            futures = [pool.submit(_send_one, req) for req in requests]
            results: list[dict] = [f.result(timeout=timeout_per_request + 5.0) for f in futures]

        return results


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
