"""Tests for d20: FixtureBasedPromptLoader (real fixture-based prompts).

Verifies JSONL loading, in-memory wrapping, batched iteration, stats,
and error paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dojo.idle.prompt_loader import FixtureBasedPromptLoader, PromptLoaderError


# ---------------------------------------------------------------------------
# from_list constructor
# ---------------------------------------------------------------------------

class TestFromList:
    def test_wraps_list_of_dicts(self):
        items = [{"prompt": "hi"}, {"prompt": "bye"}]
        loader = FixtureBasedPromptLoader.from_list(items)
        assert len(loader) == 2

    def test_empty_list_is_valid(self):
        loader = FixtureBasedPromptLoader.from_list([])
        assert len(loader) == 0

    def test_source_is_in_memory(self):
        loader = FixtureBasedPromptLoader.from_list([{"prompt": "hi"}])
        assert loader.source == "in_memory"

    def test_rejects_non_dict_entries(self):
        with pytest.raises(PromptLoaderError, match="not a dict"):
            FixtureBasedPromptLoader.from_list(["not a dict"])

    def test_rejects_missing_prompt_key(self):
        with pytest.raises(PromptLoaderError, match="missing required key"):
            FixtureBasedPromptLoader.from_list([{"text": "hi"}])


# ---------------------------------------------------------------------------
# from_jsonl constructor
# ---------------------------------------------------------------------------

class TestFromJsonl:
    def test_loads_jsonl_file(self, tmp_path: Path):
        p = tmp_path / "prompts.jsonl"
        p.write_text(
            json.dumps({"prompt": "a"}) + "\n" + json.dumps({"prompt": "b"}) + "\n",
            encoding="utf-8",
        )
        loader = FixtureBasedPromptLoader.from_jsonl(p)
        assert len(loader) == 2

    def test_skips_blank_lines(self, tmp_path: Path):
        p = tmp_path / "prompts.jsonl"
        p.write_text(
            json.dumps({"prompt": "a"}) + "\n\n" + json.dumps({"prompt": "b"}) + "\n",
            encoding="utf-8",
        )
        loader = FixtureBasedPromptLoader.from_jsonl(p)
        assert len(loader) == 2

    def test_records_source_path(self, tmp_path: Path):
        p = tmp_path / "prompts.jsonl"
        p.write_text(json.dumps({"prompt": "a"}) + "\n", encoding="utf-8")
        loader = FixtureBasedPromptLoader.from_jsonl(p)
        assert loader.source == str(p)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(PromptLoaderError, match="not found"):
            FixtureBasedPromptLoader.from_jsonl(tmp_path / "missing.jsonl")

    def test_invalid_json_raises_with_line_no(self, tmp_path: Path):
        p = tmp_path / "bad.jsonl"
        p.write_text("{\"prompt\":\"ok\"}\n{not_json}\n", encoding="utf-8")
        with pytest.raises(PromptLoaderError, match=":2"):
            FixtureBasedPromptLoader.from_jsonl(p)


# ---------------------------------------------------------------------------
# Iteration / indexing
# ---------------------------------------------------------------------------

class TestIteration:
    def test_iter_yields_all_prompts(self):
        items = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
        loader = FixtureBasedPromptLoader.from_list(items)
        assert list(loader) == items

    def test_index_access(self):
        loader = FixtureBasedPromptLoader.from_list([{"prompt": "a"}, {"prompt": "b"}])
        assert loader[1] == {"prompt": "b"}

    def test_batches_yields_correct_sizes(self):
        items = [{"prompt": str(i)} for i in range(10)]
        loader = FixtureBasedPromptLoader.from_list(items)
        batches = list(loader.batches(size=3))
        assert [len(b) for b in batches] == [3, 3, 3, 1]

    def test_batches_zero_size_raises(self):
        loader = FixtureBasedPromptLoader.from_list([{"prompt": "a"}])
        with pytest.raises(PromptLoaderError):
            list(loader.batches(size=0))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_for_empty_loader(self):
        loader = FixtureBasedPromptLoader.from_list([])
        s = loader.stats()
        assert s["count"] == 0
        assert s["mean_chars"] == 0

    def test_stats_count_matches_len(self):
        items = [{"prompt": "a"}, {"prompt": "abc"}]
        loader = FixtureBasedPromptLoader.from_list(items)
        assert loader.stats()["count"] == 2

    def test_max_chars_reflects_longest_prompt(self):
        items = [{"prompt": "short"}, {"prompt": "a" * 100}]
        loader = FixtureBasedPromptLoader.from_list(items)
        assert loader.stats()["max_chars"] == 100

    def test_stats_includes_source(self):
        loader = FixtureBasedPromptLoader.from_list([{"prompt": "a"}])
        assert loader.stats()["source"] == "in_memory"


# ---------------------------------------------------------------------------
# Evidence emission
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_evidence_emitted_on_first_iteration(self):
        from dojo.core.evidence import EvidenceLogger
        loader = FixtureBasedPromptLoader.from_list([{"prompt": "a"}])
        EvidenceLogger.clear()
        list(loader)
        events = EvidenceLogger.get_events()
        assert any(e.get("data", {}).get("phase") == "fixture_loaded" for e in events)

    def test_evidence_only_emitted_once(self):
        from dojo.core.evidence import EvidenceLogger
        loader = FixtureBasedPromptLoader.from_list([{"prompt": "a"}])
        EvidenceLogger.clear()
        list(loader)
        list(loader)  # second iteration
        events = [
            e for e in EvidenceLogger.get_events()
            if e.get("data", {}).get("phase") == "fixture_loaded"
        ]
        assert len(events) == 1
