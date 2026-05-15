"""OpenAI-compatible API server for DanteDojo.

Provides /v1/chat/completions and /v1/models endpoints compatible with
the OpenAI SDK. Proxies requests to the configured backend (llama.cpp or vLLM).
Emits constitutional evidence per request.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import ServeError

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Registry of loaded models: model_id -> metadata dict
_LOADED_MODELS: Dict[str, Dict[str, Any]] = {}

# Auto-start backend instance (set when DOJO_AUTO_START_BACKEND=1)
_backend_instance: Optional[Any] = None


# ---------------------------------------------------------------------------
# Optional FastAPI imports
# ---------------------------------------------------------------------------

try:
    import fastapi as _fastapi
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]


def _require_fastapi() -> None:
    """Raise ServeError if FastAPI is not installed."""
    if not _FASTAPI_AVAILABLE:
        raise ServeError(
            "fastapi is not installed. Install with: pip install fastapi uvicorn",
            reason="fastapi package not found",
        )


# ---------------------------------------------------------------------------
# Backend health / auto-start helpers
# ---------------------------------------------------------------------------

async def _is_backend_healthy(backend_url: str = "http://localhost:8080") -> bool:
    """Return True if the backend /health endpoint responds with HTTP 200."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{backend_url}/health")
            return resp.status_code == 200
    except Exception:
        return False


async def _ensure_backend_running(backend_url: str, model_id: str, port: int = 8080) -> None:
    """Auto-start LlamaCppBackend if DOJO_AUTO_START_BACKEND=1 and backend is unreachable."""
    global _backend_instance
    if await _is_backend_healthy(backend_url):
        return
    try:
        from pathlib import Path as _Path
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        _backend_instance = LlamaCppBackend(model_path=_Path(model_id), port=port)
        _backend_instance.start()
    except Exception:
        pass


def stop_serving() -> None:
    """Stop the module-level backend instance if one was started via auto-start."""
    global _backend_instance
    if _backend_instance is not None:
        try:
            _backend_instance.stop()
        except Exception:
            pass
        _backend_instance = None


# ---------------------------------------------------------------------------
# Request / response models (only valid when FastAPI is available)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    class ChatMessage(BaseModel):  # type: ignore[valid-type]
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):  # type: ignore[valid-type]
        model: str
        messages: List[ChatMessage]
        max_tokens: Optional[int] = 512
        temperature: Optional[float] = 0.7
        stream: Optional[bool] = False
        stop: Optional[List[str]] = None

else:
    # Minimal fallback so the module can be imported in FastAPI-less environments
    class ChatMessage:  # type: ignore[no-redef]
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    class ChatCompletionRequest:  # type: ignore[no-redef]
        def __init__(
            self,
            model: str,
            messages: List[ChatMessage],
            max_tokens: Optional[int] = 512,
            temperature: Optional[float] = 0.7,
            stream: Optional[bool] = False,
            stop: Optional[List[str]] = None,
        ) -> None:
            self.model = model
            self.messages = messages
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.stream = stream
            self.stop = stop


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(backend_url: str, model_id: str, passport_id: str = "") -> Any:
    """Create the FastAPI app configured for a specific backend.

    Args:
        backend_url: URL of the llama.cpp / vLLM backend (e.g. ``http://localhost:8080``).
        model_id: Model identifier returned in API responses.
        passport_id: Constitutional passport ID (empty = no passport).

    Returns:
        Configured :class:`fastapi.FastAPI` application.

    Raises:
        ServeError: If FastAPI is not installed.
    """
    _require_fastapi()

    app = _fastapi.FastAPI(title="DanteDojo OpenAI-Compatible API", version="1.0.0")
    logger = EvidenceLogger.get_instance()

    # Register this model in the loaded-models registry
    _LOADED_MODELS[model_id] = {
        "id": model_id,
        "object": "model",
        "owned_by": "dantedojo",
        "passport_id": passport_id,
    }

    # ------------------------------------------------------------------
    # Startup: auto-start backend if configured
    # ------------------------------------------------------------------

    import os as _os

    @app.on_event("startup")
    async def _startup() -> None:
        if _os.environ.get("DOJO_AUTO_START_BACKEND", "0") == "1":
            await _ensure_backend_running(backend_url, model_id)

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        backend_healthy = await _is_backend_healthy(backend_url)
        return {"status": "ok", "model": model_id, "backend_healthy": backend_healthy}

    # ------------------------------------------------------------------
    # GET /v1/models — lists all models in the registry
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": list(_LOADED_MODELS.values()),
        }

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        request_id = uuid.uuid4().hex
        start_time = time.monotonic()

        logger.emit(
            {
                "type": EvidenceType.SERVE_START.value,
                "data": {
                    "request_id": request_id,
                    "model": request.model,
                    "message_count": len(request.messages),
                    "stream": request.stream,
                },
            },
            organ="serve",
        )

        try:
            if request.stream:
                return StreamingResponse(
                    _stream_from_backend(
                        backend_url, request, request_id, logger, start_time
                    ),
                    media_type="text/event-stream",
                )
            else:
                response = await _complete_from_backend(backend_url, request)

                latency_ms = (time.monotonic() - start_time) * 1000
                logger.emit(
                    {
                        "type": EvidenceType.SERVE_READY.value,
                        "data": {
                            "request_id": request_id,
                            "latency_ms": round(latency_ms, 2),
                            "tokens_generated": response.get("usage", {}).get(
                                "completion_tokens", 0
                            ),
                        },
                    },
                    organ="serve",
                )

                return JSONResponse(response)
        except ServeError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


# ---------------------------------------------------------------------------
# Backend proxy helpers
# ---------------------------------------------------------------------------

async def _complete_from_backend(
    backend_url: str, request: "ChatCompletionRequest"
) -> dict:
    """Forward a non-streaming request to the backend and return the full response.

    Tries the llama.cpp ``/completion`` endpoint.  If the backend is unreachable
    a stub response is returned so tests and offline runs degrade gracefully.
    """
    import httpx

    prompt = _messages_to_prompt(request.messages)
    payload = {
        "prompt": prompt,
        "n_predict": request.max_tokens or 512,
        "temperature": request.temperature or 0.7,
        "stop": request.stop or [],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{backend_url}/completion", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", "")
        except httpx.ConnectError:
            content = "[DanteDojo stub: backend not running]"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(prompt.split()) + len(content.split()),
        },
    }


async def _stream_from_backend(
    backend_url: str,
    request: "ChatCompletionRequest",
    request_id: str,
    logger: EvidenceLogger,
    start_time: float,
) -> AsyncIterator[str]:
    """Stream tokens from the backend as SSE events.

    Yields ``data: <json>\\n\\n`` lines and a final ``data: [DONE]\\n\\n``.
    Falls back to a stub token when the backend is unavailable.
    """
    import json
    import httpx

    prompt = _messages_to_prompt(request.messages)
    payload = {
        "prompt": prompt,
        "n_predict": request.max_tokens or 512,
        "temperature": request.temperature or 0.7,
        "stream": True,
    }

    tokens_generated = 0
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", f"{backend_url}/completion", json=payload
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = json.loads(line[6:])
                        token = chunk_data.get("content", "")
                        tokens_generated += 1

                        chunk = {
                            "id": f"chatcmpl-{request_id[:8]}",
                            "object": "chat.completion.chunk",
                            "model": request.model,
                            "choices": [
                                {
                                    "delta": {"content": token},
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                        if chunk_data.get("stop", False):
                            break
    except httpx.ConnectError:
        # Backend unavailable — emit a single stub token
        yield (
            'data: {"id":"stub","object":"chat.completion.chunk",'
            '"choices":[{"delta":{"content":"[stub]"},"index":0,"finish_reason":"stop"}]}\n\n'
        )

    latency_ms = (time.monotonic() - start_time) * 1000
    logger.emit(
        {
            "type": EvidenceType.SERVE_READY.value,
            "data": {
                "request_id": request_id,
                "tokens_generated": tokens_generated,
                "latency_ms": round(latency_ms, 2),
            },
        },
        organ="serve",
    )

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Prompt formatter
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: "List[ChatMessage]") -> str:
    """Convert a list of OpenAI-format chat messages to a single prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|system|>\n{msg.content}")
        elif msg.role == "user":
            parts.append(f"<|user|>\n{msg.content}")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Uvicorn runner
# ---------------------------------------------------------------------------

def run_server(
    backend_url: str,
    model_id: str,
    port: int = 8080,
    host: str = "127.0.0.1",
    passport_id: str = "",
) -> None:
    """Run the OpenAI-compatible FastAPI server with uvicorn.

    Args:
        backend_url: URL of the llama.cpp / vLLM backend.
        model_id: Model identifier returned in API responses.
        port: Port to listen on.
        host: Host to bind to.
        passport_id: Constitutional passport ID.

    Raises:
        ServeError: If FastAPI or uvicorn are not installed.
    """
    _require_fastapi()
    try:
        import uvicorn
    except ImportError as exc:
        raise ServeError(
            "uvicorn is not installed. Install with: pip install uvicorn",
            reason="uvicorn package not found",
        ) from exc

    app = create_app(backend_url=backend_url, model_id=model_id, passport_id=passport_id)
    uvicorn.run(app, host=host, port=port)


__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "create_app",
    "run_server",
    "stop_serving",
]
