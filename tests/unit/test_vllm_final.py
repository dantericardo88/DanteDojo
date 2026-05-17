"""Final d31 vLLM backend tests — streaming, stop-sequence, batch max_tokens.

Covers:
1.  stop=["stub"] causes generation to halt (stub text contains "stub")
2.  stream=True returns multiple chunks
3.  First non-final chunk has role="assistant" in delta
4.  Last chunk before end has finish_reason="stop"
5.  Batch with 3 different max_tokens all receive responses
6.  serve.vllm_stop_sequence evidence emitted when stop sequence triggered
7.  Evidence includes which stop sequence was matched (stop_sequence field)
8.  temperature=0 produces deterministic output (same call → same result)
9.  Empty stop=[] works same as no stop sequences
10. generate() with stream=True returns iterable of chunks (list with dicts)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.types import HALProfile, HardwareTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hal(tier: HardwareTier = HardwareTier.TIER_3090) -> HALProfile:
    return HALProfile(
        tier=tier,
        gpu_name="Test GPU",
        vram_bytes=24 * 1024 ** 3,
        compute_capability="8.6",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="550.0",
        profile_sha256="a" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_backend(port: int = 8080) -> "VLLMBackend":
    from dojo.serve.vllm_backend import VLLMBackend
    return VLLMBackend(
        model_path=Path("/models/test-model"),
        profile=_make_hal(),
        port=port,
        context_length=4096,
    )


@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def inject_vllm_stub():
    """Inject a stub vllm module so BackendUnavailable is never raised."""
    mock_vllm = MagicMock()
    with patch.dict(sys.modules, {"vllm": mock_vllm}):
        yield mock_vllm


# ---------------------------------------------------------------------------
# 1. stop=["stub"] causes generation to halt before "stub" token
# ---------------------------------------------------------------------------

class TestStopSequenceHalt:
    def test_stop_sequence_truncates_output(self, monkeypatch):
        """stop=["stub"] should cause streaming to halt before the stop token.

        The stub text is '[stub stream for: ...]'; when stop=['stub'], the
        text is truncated to everything before the word 'stub', so the
        content chunks must not include 'stub'.
        """
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks = backend.generate("hello world", stream=True, stop=["stub"])
        # Collect all content from non-final chunks
        content = "".join(
            ch["choices"][0]["delta"]["content"]
            for ch in chunks
            if ch["choices"][0]["finish_reason"] is None
        )
        assert "stub" not in content, (
            f"Expected 'stub' to be truncated from output, got: {content!r}"
        )

    def test_stop_sequence_with_end_token_does_not_crash(self, monkeypatch):
        """stop=['<|end|>'] should complete without error (token not in stub)."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        # <|end|> is not in stub text — generation should produce full output
        chunks = backend.generate("hello", stream=True, stop=["<|end|>"])
        assert isinstance(chunks, list)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# 2. stream=True returns multiple chunks
# ---------------------------------------------------------------------------

class TestStreamingChunks:
    def test_streaming_returns_multiple_chunks(self, monkeypatch):
        """generate(stream=True) must return more than one chunk."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks = backend.generate("tell me a story", stream=True)
        assert isinstance(chunks, list)
        assert len(chunks) > 1, f"Expected >1 chunks, got {len(chunks)}"

    def test_all_chunks_are_dicts(self, monkeypatch):
        """Every chunk returned by streaming must be a dict."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks = backend.generate("test", stream=True)
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, dict), f"Chunk {i} is not a dict: {type(chunk)}"

    def test_all_chunks_have_choices_key(self, monkeypatch):
        """Every chunk must have a 'choices' key."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks = backend.generate("test prompt", stream=True)
        for i, chunk in enumerate(chunks):
            assert "choices" in chunk, f"Chunk {i} missing 'choices': {chunk}"

    def test_all_choices_have_delta(self, monkeypatch):
        """Every chunk's first choice must have a 'delta' key."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks = backend.generate("test", stream=True)
        for i, chunk in enumerate(chunks):
            choice = chunk["choices"][0]
            assert "delta" in choice, f"Chunk {i} choice missing 'delta': {choice}"

    def test_delta_has_content_field(self, monkeypatch):
        """Every delta must have a 'content' field."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks = backend.generate("test", stream=True)
        for i, chunk in enumerate(chunks):
            delta = chunk["choices"][0]["delta"]
            assert "content" in delta, f"Chunk {i} delta missing 'content': {delta}"


# ---------------------------------------------------------------------------
# 3. First chunk has role="assistant" delta
# ---------------------------------------------------------------------------

class TestFirstChunkRole:
    def test_first_chunk_has_assistant_role(self, monkeypatch):
        """The first content chunk must carry role='assistant' in its delta."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks = backend.generate("hello", stream=True)
        # Find first chunk with non-empty content
        first = next(
            (ch for ch in chunks if ch["choices"][0]["delta"].get("content")),
            None,
        )
        assert first is not None, "No chunk with content found"
        assert first["choices"][0]["delta"].get("role") == "assistant", (
            f"First content chunk role should be 'assistant', got: "
            f"{first['choices'][0]['delta'].get('role')!r}"
        )


# ---------------------------------------------------------------------------
# 4. Last chunk has finish_reason="stop"
# ---------------------------------------------------------------------------

class TestLastChunkFinishReason:
    def test_last_chunk_finish_reason_stop(self, monkeypatch):
        """The very last chunk must have finish_reason='stop'."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks = backend.generate("hello", stream=True)
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] == "stop", (
            f"Last chunk finish_reason should be 'stop', got: "
            f"{last['choices'][0]['finish_reason']!r}"
        )

    def test_non_final_chunks_have_null_finish_reason(self, monkeypatch):
        """All chunks except the last must have finish_reason=None."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks = backend.generate("hello world", stream=True)
        for i, chunk in enumerate(chunks[:-1]):
            fr = chunk["choices"][0]["finish_reason"]
            assert fr is None, (
                f"Chunk {i} (non-final) should have finish_reason=None, got {fr!r}"
            )


# ---------------------------------------------------------------------------
# 5. Batch with 3 different max_tokens all receive responses
# ---------------------------------------------------------------------------

class TestBatchDifferentMaxTokens:
    def test_batch_with_per_prompt_max_tokens(self, monkeypatch):
        """batch_generate with max_tokens_per_prompt gives a response for every prompt."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        prompts = ["short", "medium length prompt", "this is a longer prompt for testing"]
        results = backend.batch_generate(
            prompts,
            max_tokens_per_prompt=[16, 64, 256],
        )
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    def test_batch_all_results_are_strings(self, monkeypatch):
        """Every result in a batch must be a string."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        prompts = ["a", "b", "c"]
        results = backend.batch_generate(prompts, max_tokens_per_prompt=[10, 20, 30])
        for i, r in enumerate(results):
            assert isinstance(r, str), f"Result {i} is not a string: {type(r)}"

    def test_batch_three_prompts_default_max_tokens(self, monkeypatch):
        """batch_generate with 3 prompts and default max_tokens returns 3 results."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        results = backend.batch_generate(["p1", "p2", "p3"], max_tokens=128)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# 6 & 7. serve.vllm_stop_sequence evidence emitted with stop_sequence field
# ---------------------------------------------------------------------------

class TestStopSequenceEvidence:
    def test_stop_sequence_evidence_emitted(self, monkeypatch):
        """serve.vllm_stop_sequence evidence must be emitted when stop seq triggered."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        # "stub" appears in stub text "[stub stream for: ...]"
        backend.generate("hello", stream=True, stop=["stub"])

        types = [
            e["evidence"].get("type")
            for e in EvidenceLogger._events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "serve.vllm_stop_sequence" in types, (
            f"Expected 'serve.vllm_stop_sequence' in evidence, got: {types}"
        )

    def test_stop_sequence_evidence_has_stop_sequence_field(self, monkeypatch):
        """serve.vllm_stop_sequence evidence data must include 'stop_sequence' field."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        backend.generate("hello", stream=True, stop=["stub"])

        ev_data = next(
            (
                e["evidence"]["data"]
                for e in EvidenceLogger._events
                if isinstance(e.get("evidence"), dict)
                and e["evidence"].get("type") == "serve.vllm_stop_sequence"
            ),
            None,
        )
        assert ev_data is not None, "No serve.vllm_stop_sequence event found"
        assert "stop_sequence" in ev_data, (
            f"Evidence data missing 'stop_sequence' field: {ev_data}"
        )

    def test_stop_sequence_evidence_records_matched_token(self, monkeypatch):
        """The stop_sequence field must equal the matched stop token."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        backend.generate("hello", stream=True, stop=["stub"])

        ev_data = next(
            (
                e["evidence"]["data"]
                for e in EvidenceLogger._events
                if isinstance(e.get("evidence"), dict)
                and e["evidence"].get("type") == "serve.vllm_stop_sequence"
            ),
            None,
        )
        assert ev_data is not None
        assert ev_data["stop_sequence"] == "stub", (
            f"Expected stop_sequence='stub', got {ev_data['stop_sequence']!r}"
        )

    def test_no_stop_sequence_evidence_when_no_match(self, monkeypatch):
        """No serve.vllm_stop_sequence evidence when stop token not in output."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        backend.generate("hello", stream=True, stop=["<|end|>"])

        types = [
            e["evidence"].get("type")
            for e in EvidenceLogger._events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "serve.vllm_stop_sequence" not in types, (
            "Should NOT emit stop_sequence evidence when stop token not in output"
        )


# ---------------------------------------------------------------------------
# 8. temperature=0 produces deterministic output
# ---------------------------------------------------------------------------

class TestDeterministicOutput:
    def test_temperature_zero_deterministic(self, monkeypatch):
        """generate() with temperature=0.0 returns identical results on two calls."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        result1 = backend.generate("deterministic test", temperature=0.0)
        result2 = backend.generate("deterministic test", temperature=0.0)
        assert result1 == result2, (
            f"temperature=0.0 should be deterministic, got {result1!r} vs {result2!r}"
        )


# ---------------------------------------------------------------------------
# 9. Empty stop=[] works same as no stop sequences
# ---------------------------------------------------------------------------

class TestEmptyStopList:
    def test_empty_stop_list_no_truncation(self, monkeypatch):
        """stop=[] should produce the same output as stop=None."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        result_none = backend.generate("hello world", stop=None)
        result_empty = backend.generate("hello world", stop=[])
        assert result_none == result_empty, (
            f"stop=[] should behave like stop=None, got {result_none!r} vs {result_empty!r}"
        )

    def test_empty_stop_list_streaming_no_truncation(self, monkeypatch):
        """stream=True with stop=[] should produce same chunk count as stop=None."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        chunks_none = backend.generate("hello world", stream=True, stop=None)
        chunks_empty = backend.generate("hello world", stream=True, stop=[])
        assert len(chunks_none) == len(chunks_empty), (
            f"stop=[] should produce same chunks as stop=None: "
            f"{len(chunks_none)} vs {len(chunks_empty)}"
        )


# ---------------------------------------------------------------------------
# 10. generate() with stream=True returns iterable of chunk dicts
# ---------------------------------------------------------------------------

class TestStreamReturnType:
    def test_stream_true_returns_list(self, monkeypatch):
        """generate(stream=True) must return a list (iterable of chunks)."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        result = backend.generate("test", stream=True)
        assert isinstance(result, list), (
            f"stream=True should return list, got {type(result)}"
        )

    def test_stream_false_returns_string(self, monkeypatch):
        """generate(stream=False) must return a string."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        result = backend.generate("test", stream=False)
        assert isinstance(result, str), (
            f"stream=False should return str, got {type(result)}"
        )

    def test_stream_default_returns_string(self, monkeypatch):
        """generate() without stream kwarg must return a string (default behaviour)."""
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        backend = _make_backend()
        result = backend.generate("test")
        assert isinstance(result, str)
