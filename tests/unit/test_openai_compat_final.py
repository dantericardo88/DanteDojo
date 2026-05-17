"""Comprehensive tests for dojo/serve/openai_compat.py.

Covers 18+ scenarios: HTTP responses, SSE streaming, evidence emission,
validation, parameter handling, and model echoing.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.serve.openai_compat import (
    _LOADED_MODELS,
    ChatMessage,
    _messages_to_prompt,
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
# Helpers
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


def _mock_httpx_streaming_connect_error():
    """Return an httpx.AsyncClient mock where client.stream() raises ConnectError."""
    import httpx

    class _RaisingStreamCM:
        async def __aenter__(self):
            raise httpx.ConnectError("refused")

        async def __aexit__(self, *_):
            return False

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=_RaisingStreamCM())
    return mock_client


def _post_chat(client, messages=None, **extra):
    payload = {
        "model": "test-model",
        "messages": messages or [{"role": "user", "content": "Hello"}],
        **extra,
    }
    return client.post("/v1/chat/completions", json=payload)


# ---------------------------------------------------------------------------
# 1. POST /v1/chat/completions returns 200
# ---------------------------------------------------------------------------

class TestChatCompletionsStatus:
    def test_returns_200(self, client):
        mock_client = _mock_httpx_post("Hi!")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. Response has choices[0].message.content field
# ---------------------------------------------------------------------------

class TestChatCompletionsContent:
    def test_choices_message_content_field_exists(self, client):
        mock_client = _mock_httpx_post("Test response")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        body = resp.json()
        assert "choices" in body
        assert len(body["choices"]) >= 1
        assert "message" in body["choices"][0]
        assert "content" in body["choices"][0]["message"]

    def test_choices_message_content_value(self, client):
        mock_client = _mock_httpx_post("Test response")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        content = resp.json()["choices"][0]["message"]["content"]
        assert content == "Test response"


# ---------------------------------------------------------------------------
# 3. Response has usage.total_tokens field (integer)
# ---------------------------------------------------------------------------

class TestChatCompletionsUsage:
    def test_usage_total_tokens_is_integer(self, client):
        mock_client = _mock_httpx_post("one two three")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        usage = resp.json()["usage"]
        assert "total_tokens" in usage
        assert isinstance(usage["total_tokens"], int)

    def test_usage_total_tokens_positive(self, client):
        mock_client = _mock_httpx_post("word")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        assert resp.json()["usage"]["total_tokens"] > 0


# ---------------------------------------------------------------------------
# 4. stream=True returns 200 with SSE content-type
# ---------------------------------------------------------------------------

class TestStreamingStatus:
    def test_stream_returns_200(self, client):
        mock_client = _mock_httpx_streaming_connect_error()
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, stream=True)
        assert resp.status_code == 200

    def test_stream_content_type_is_event_stream(self, client):
        mock_client = _mock_httpx_streaming_connect_error()
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, stream=True)
        assert "text/event-stream" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 5. SSE response contains "data: " prefixed lines when streaming
# ---------------------------------------------------------------------------

class TestStreamingDataLines:
    def test_sse_response_contains_data_prefix(self, client):
        mock_client = _mock_httpx_streaming_connect_error()
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, stream=True)
        assert "data: " in resp.text

    def test_sse_stub_chunk_is_json(self, client):
        mock_client = _mock_httpx_streaming_connect_error()
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, stream=True)
        # Find a data: line that's not [DONE] and parse it
        data_lines = [
            line[6:] for line in resp.text.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        assert len(data_lines) >= 1
        chunk = json.loads(data_lines[0])
        assert "choices" in chunk or "id" in chunk


# ---------------------------------------------------------------------------
# 6. SSE response ends with data: [DONE] marker
# ---------------------------------------------------------------------------

class TestStreamingDone:
    def test_sse_ends_with_done(self, client):
        mock_client = _mock_httpx_streaming_connect_error()
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, stream=True)
        assert "data: [DONE]" in resp.text


# ---------------------------------------------------------------------------
# 7. GET /v1/models returns 200
# ---------------------------------------------------------------------------

class TestModelsEndpointStatus:
    def test_models_returns_200(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 8. /v1/models response has "data" list
# ---------------------------------------------------------------------------

class TestModelsEndpointShape:
    def test_models_has_data_list(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        assert "data" in body
        assert isinstance(body["data"], list)

    def test_models_data_contains_test_model(self, client):
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert "test-model" in ids

    def test_models_object_is_list(self, client):
        resp = client.get("/v1/models")
        assert resp.json()["object"] == "list"


# ---------------------------------------------------------------------------
# 9. GET /health returns 200 with status field
# ---------------------------------------------------------------------------

class TestHealthStatus:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert "status" in body

    def test_health_status_is_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 10. /health has backend_healthy field
# ---------------------------------------------------------------------------

class TestHealthBackendHealthy:
    def test_health_has_backend_healthy_field(self, client):
        resp = client.get("/health")
        assert "backend_healthy" in resp.json()

    def test_health_backend_healthy_is_bool(self, client):
        resp = client.get("/health")
        assert isinstance(resp.json()["backend_healthy"], bool)

    def test_health_backend_healthy_false_when_backend_down(self):
        app = create_app(backend_url="http://localhost:19999", model_id="x")
        c = TestClient(app, raise_server_exceptions=True)
        with patch(
            "dojo.serve.openai_compat._is_backend_healthy",
            new=AsyncMock(return_value=False),
        ):
            resp = c.get("/health")
        assert resp.json()["backend_healthy"] is False


# ---------------------------------------------------------------------------
# 11. Evidence is emitted after a chat completion request
# ---------------------------------------------------------------------------

class TestEvidenceEmittedAfterRequest:
    def test_evidence_emitted_after_completion(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            _post_chat(client)
        events = EvidenceLogger.get_events()
        assert len(events) >= 1

    def test_multiple_evidence_events_emitted(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            _post_chat(client)
        # The source emits at least SERVE_START + SERVE_READY + two canonical events = 4
        events = EvidenceLogger.get_events()
        assert len(events) >= 2


# ---------------------------------------------------------------------------
# 12. Evidence type contains "serve" in the type string
# ---------------------------------------------------------------------------

class TestEvidenceTypeContainsServe:
    def test_evidence_type_contains_serve(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            _post_chat(client)
        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert any("serve" in t for t in types), f"No serve event in {types}"

    def test_serve_start_type_emitted(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            _post_chat(client)
        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert "serve.start" in types


# ---------------------------------------------------------------------------
# 13. Evidence has model field or request_id field
# ---------------------------------------------------------------------------

class TestEvidenceFields:
    def test_evidence_has_request_id_or_model(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            _post_chat(client)
        events = EvidenceLogger.get_events()
        # At least one event's data should contain request_id or model
        found = any(
            "request_id" in e.get("data", {}) or "model" in e.get("data", {})
            for e in events
        )
        assert found, "No event has request_id or model in data"

    def test_evidence_serve_start_has_model(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            _post_chat(client)
        start_events = [e for e in EvidenceLogger.get_events() if e["type"] == "serve.start"]
        assert len(start_events) >= 1
        assert "model" in start_events[0]["data"]


# ---------------------------------------------------------------------------
# 14. Invalid/missing messages body returns 4xx
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_messages_returns_4xx(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},  # missing "messages"
        )
        assert resp.status_code in (400, 422)

    def test_empty_body_returns_4xx(self, client):
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code in (400, 422)

    def test_missing_model_returns_4xx(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 15. temperature parameter accepted without error
# ---------------------------------------------------------------------------

class TestTemperatureParameter:
    def test_temperature_zero_accepted(self, client):
        mock_client = _mock_httpx_post("cold")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, temperature=0.0)
        assert resp.status_code == 200

    def test_temperature_one_accepted(self, client):
        mock_client = _mock_httpx_post("hot")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, temperature=1.0)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 16. max_tokens parameter accepted without error
# ---------------------------------------------------------------------------

class TestMaxTokensParameter:
    def test_max_tokens_accepted(self, client):
        mock_client = _mock_httpx_post("short")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, max_tokens=64)
        assert resp.status_code == 200

    def test_max_tokens_large_accepted(self, client):
        mock_client = _mock_httpx_post("long")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, max_tokens=4096)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 17. system message in messages list is handled
# ---------------------------------------------------------------------------

class TestSystemMessage:
    def test_system_message_returns_200(self, client):
        mock_client = _mock_httpx_post("ok")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, messages=messages)
        assert resp.status_code == 200

    def test_system_message_included_in_prompt(self):
        msgs = [
            ChatMessage(role="system", content="Be concise"),
            ChatMessage(role="user", content="Hi"),
        ]
        prompt = _messages_to_prompt(msgs)
        assert "Be concise" in prompt
        assert "<|system|>" in prompt

    def test_multi_turn_with_system_handled(self, client):
        mock_client = _mock_httpx_post("multi-turn ok")
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First reply"},
            {"role": "user", "content": "Second message"},
        ]
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client, messages=messages)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 18. Response model field matches the model in the request
# ---------------------------------------------------------------------------

class TestResponseModelField:
    def test_response_model_matches_request(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        assert resp.json()["model"] == "test-model"

    def test_response_model_with_custom_model_id(self):
        app = create_app(backend_url="http://localhost:9999", model_id="my-custom-model")
        c = TestClient(app, raise_server_exceptions=True)
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = c.post(
                "/v1/chat/completions",
                json={
                    "model": "my-custom-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.json()["model"] == "my-custom-model"

    def test_response_object_type(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        assert resp.json()["object"] == "chat.completion"

    def test_response_has_id(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        assert "id" in resp.json()
        assert resp.json()["id"].startswith("chatcmpl-")


# ---------------------------------------------------------------------------
# Bonus: Backend stub fallback
# ---------------------------------------------------------------------------

class TestBackendFallback:
    def test_stub_fallback_returns_200(self, client):
        import httpx
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        assert resp.status_code == 200

    def test_stub_fallback_content_contains_stub(self, client):
        import httpx
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = _post_chat(client)
        content = resp.json()["choices"][0]["message"]["content"]
        assert "stub" in content.lower()


# ---------------------------------------------------------------------------
# Bonus: serve.request_complete canonical evidence
# ---------------------------------------------------------------------------

class TestCanonicalEvidence:
    def test_serve_request_complete_emitted(self, client):
        mock_client = _mock_httpx_post("ok")
        with patch("httpx.AsyncClient", return_value=mock_client):
            _post_chat(client)
        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert "serve.request_complete" in types

    def test_serve_request_complete_has_total_tokens(self, client):
        mock_client = _mock_httpx_post("word word")
        with patch("httpx.AsyncClient", return_value=mock_client):
            _post_chat(client)
        complete_events = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == "serve.request_complete"
        ]
        assert len(complete_events) >= 1
        assert "total_tokens" in complete_events[0]["data"]
