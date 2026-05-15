"""Unit tests for dojo/serve/openai_compat.py — d29 OpenAI-compatible API.

Covers:
1. App creation — create_app returns a FastAPI instance
2. GET /health — returns {"status": "ok"}
3. GET /v1/models — returns configured model in list
4. POST /v1/chat/completions (non-streaming) — OpenAI-format response
5. POST /v1/chat/completions (backend unavailable) — stub fallback, no crash
6. Evidence — SERVE_START and SERVE_READY emitted per request
7. POST /v1/chat/completions (streaming) — text/event-stream with data: lines
8. _messages_to_prompt — converts messages to prompt string
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.serve.openai_compat import (
    ChatMessage,
    ChatCompletionRequest,
    _messages_to_prompt,
    _LOADED_MODELS,
    create_app,
    stop_serving,
)

# ---------------------------------------------------------------------------
# Skip entire module if FastAPI is not installed
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE,
    reason="fastapi not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    _LOADED_MODELS.clear()
    yield
    EvidenceLogger.clear()
    _LOADED_MODELS.clear()


@pytest.fixture()
def app():
    return create_app(backend_url="http://localhost:9999", model_id="test-model")


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Helper: build a mock httpx response
# ---------------------------------------------------------------------------

def _mock_httpx_post(content: str = "Hello there") -> Any:
    """Return a mock that makes httpx.AsyncClient.post behave like llama.cpp /completion."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={"content": content})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


# ---------------------------------------------------------------------------
# 1. App creation
# ---------------------------------------------------------------------------

class TestCreateApp:
    def test_returns_fastapi_instance(self):
        import fastapi
        app = create_app(backend_url="http://localhost:9999", model_id="test")
        assert isinstance(app, fastapi.FastAPI)

    def test_returns_different_instances(self):
        a1 = create_app(backend_url="http://localhost:9999", model_id="m1")
        a2 = create_app(backend_url="http://localhost:9999", model_id="m2")
        assert a1 is not a2


# ---------------------------------------------------------------------------
# 2. GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_status_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_model_returned(self, client):
        resp = client.get("/health")
        assert resp.json()["model"] == "test-model"


# ---------------------------------------------------------------------------
# 3. GET /v1/models
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    def test_returns_list(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1

    def test_model_id_matches(self, client):
        resp = client.get("/v1/models")
        assert resp.json()["data"][0]["id"] == "test-model"

    def test_owned_by_dantedojo(self, client):
        resp = client.get("/v1/models")
        assert resp.json()["data"][0]["owned_by"] == "dantedojo"

    def test_passport_id_in_response(self):
        app = create_app(
            backend_url="http://localhost:9999",
            model_id="m",
            passport_id="pp-abc",
        )
        c = TestClient(app)
        resp = c.get("/v1/models")
        assert resp.json()["data"][0]["passport_id"] == "pp-abc"


# ---------------------------------------------------------------------------
# 4. POST /v1/chat/completions — non-streaming, backend available
# ---------------------------------------------------------------------------

class TestChatCompletionNonStreaming:
    def _post(self, client, messages=None, **extra):
        payload = {
            "model": "test-model",
            "messages": messages or [{"role": "user", "content": "Hello"}],
            **extra,
        }
        return client.post("/v1/chat/completions", json=payload)

    def test_returns_200(self, client):
        mock_client = _mock_httpx_post("Hi!")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = self._post(client)
        assert resp.status_code == 200

    def test_response_shape(self, client):
        mock_client = _mock_httpx_post("Hi!")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = self._post(client)
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert "choices" in body
        assert len(body["choices"]) == 1

    def test_choice_has_message(self, client):
        mock_client = _mock_httpx_post("Hello back")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = self._post(client)
        msg = resp.json()["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello back"

    def test_finish_reason_stop(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = self._post(client)
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_model_echoed(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = self._post(client)
        assert resp.json()["model"] == "test-model"

    def test_usage_present(self, client):
        mock_client = _mock_httpx_post("word")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = self._post(client)
        usage = resp.json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage


# ---------------------------------------------------------------------------
# 5. Backend unavailable — stub fallback
# ---------------------------------------------------------------------------

class TestBackendUnavailableStub:
    def test_returns_200_when_backend_down(self, client):
        import httpx
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200

    def test_stub_content_in_response(self, client):
        import httpx
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        content = resp.json()["choices"][0]["message"]["content"]
        assert "stub" in content.lower()


# ---------------------------------------------------------------------------
# 6. Evidence emission
# ---------------------------------------------------------------------------

class TestEvidenceEmission:
    def test_serve_start_emitted(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert "serve.start" in types

    def test_serve_ready_emitted(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert "serve.ready" in types

    def test_both_events_per_request(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert types.count("serve.start") >= 1
        assert types.count("serve.ready") >= 1


# ---------------------------------------------------------------------------
# 7. Streaming
# ---------------------------------------------------------------------------

def _mock_httpx_streaming_connect_error():
    """Return an httpx.AsyncClient mock where client.stream() raises ConnectError.

    The code uses ``async with client.stream(...) as resp``, so ``client.stream``
    must return an async context manager that raises on ``__aenter__``.
    """
    import httpx

    class _RaisingStreamCM:
        async def __aenter__(self):
            raise httpx.ConnectError("refused")

        async def __aexit__(self, *_):
            return False

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    # stream() is called as `async with client.stream(...)`, so it must be a
    # regular (non-async) callable that returns an async context manager.
    mock_client.stream = MagicMock(return_value=_RaisingStreamCM())
    return mock_client


class TestStreamingResponse:
    def _stream_payload(self):
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    def test_content_type_is_event_stream(self, client):
        mock_client = _mock_httpx_streaming_connect_error()
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = client.post("/v1/chat/completions", json=self._stream_payload())

        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_response_contains_data_lines(self, client):
        mock_client = _mock_httpx_streaming_connect_error()
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = client.post("/v1/chat/completions", json=self._stream_payload())

        assert "data: " in resp.text

    def test_response_ends_with_done(self, client):
        mock_client = _mock_httpx_streaming_connect_error()
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = client.post("/v1/chat/completions", json=self._stream_payload())

        assert "[DONE]" in resp.text


# ---------------------------------------------------------------------------
# 8. _messages_to_prompt
# ---------------------------------------------------------------------------

class TestMessagesToPrompt:
    def test_user_message_included(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        prompt = _messages_to_prompt(msgs)
        assert "Hello" in prompt

    def test_system_message_included(self):
        msgs = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hi"),
        ]
        prompt = _messages_to_prompt(msgs)
        assert "Be helpful" in prompt
        assert "Hi" in prompt

    def test_assistant_turn_included(self):
        msgs = [
            ChatMessage(role="user", content="Q"),
            ChatMessage(role="assistant", content="A"),
        ]
        prompt = _messages_to_prompt(msgs)
        assert "A" in prompt

    def test_ends_with_assistant_tag(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        prompt = _messages_to_prompt(msgs)
        assert prompt.endswith("<|assistant|>")

    def test_role_tags_present(self):
        msgs = [
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
        ]
        prompt = _messages_to_prompt(msgs)
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt


# ---------------------------------------------------------------------------
# 9. /health — degraded when backend unreachable
# ---------------------------------------------------------------------------

class TestHealthDegraded:
    def test_health_degraded_when_backend_down(self):
        """When the backend is unreachable, /health reports backend_healthy=False."""
        from unittest.mock import AsyncMock, patch
        app = create_app(backend_url="http://localhost:19999", model_id="test-model")
        c = TestClient(app, raise_server_exceptions=True)

        with patch(
            "dojo.serve.openai_compat._is_backend_healthy",
            new=AsyncMock(return_value=False),
        ):
            resp = c.get("/health")

        assert resp.status_code == 200
        assert resp.json()["backend_healthy"] is False

    def test_health_ok_when_backend_up(self):
        """When the backend is healthy, /health should return status=ok."""
        from unittest.mock import AsyncMock, patch
        app = create_app(backend_url="http://localhost:19999", model_id="test-model")
        c = TestClient(app, raise_server_exceptions=True)

        with patch(
            "dojo.serve.openai_compat._is_backend_healthy",
            new=AsyncMock(return_value=True),
        ):
            resp = c.get("/health")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 10. /v1/models — lists all registered models
# ---------------------------------------------------------------------------

class TestLoadedModelsRegistry:
    def test_multiple_models_registered(self):
        """Creating two apps registers two models in /v1/models."""
        a1 = create_app(backend_url="http://localhost:9999", model_id="model-a")
        a2 = create_app(backend_url="http://localhost:9999", model_id="model-b")
        # Query via either app — both share the same _LOADED_MODELS dict
        c = TestClient(a1)
        resp = c.get("/v1/models")
        ids = {m["id"] for m in resp.json()["data"]}
        assert "model-a" in ids
        assert "model-b" in ids

    def test_model_passport_id_stored(self):
        """passport_id is preserved in the loaded-models registry."""
        app = create_app(
            backend_url="http://localhost:9999",
            model_id="passport-model",
            passport_id="pp-reg-001",
        )
        c = TestClient(app)
        resp = c.get("/v1/models")
        data = resp.json()["data"]
        match = next((m for m in data if m["id"] == "passport-model"), None)
        assert match is not None
        assert match["passport_id"] == "pp-reg-001"


# ---------------------------------------------------------------------------
# 11. stop_serving — calls backend.stop()
# ---------------------------------------------------------------------------

class TestStopServing:
    def test_stop_serving_no_backend_no_error(self):
        """stop_serving() with no active backend must not raise."""
        stop_serving()  # should be a no-op

    def test_stop_serving_calls_backend_stop(self):
        """stop_serving() calls .stop() on _backend_instance if set."""
        from unittest.mock import MagicMock
        import dojo.serve.openai_compat as _compat

        mock_backend = MagicMock()
        _compat._backend_instance = mock_backend

        stop_serving()

        mock_backend.stop.assert_called_once()
        assert _compat._backend_instance is None

    def test_stop_serving_clears_instance(self):
        """After stop_serving(), _backend_instance must be None."""
        from unittest.mock import MagicMock
        import dojo.serve.openai_compat as _compat

        _compat._backend_instance = MagicMock()
        stop_serving()
        assert _compat._backend_instance is None
