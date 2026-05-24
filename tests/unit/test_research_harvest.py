"""Unit tests for dojo/research/ — claim extraction, mapping, stub generation."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _clear_evidence():
    from dojo.core.evidence import EvidenceLogger
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


class TestArxivCategoryFilter:
    """Verify the ML category filter is applied to all arXiv search queries."""

    def test_search_query_contains_category_filter(self):
        from unittest.mock import MagicMock, patch
        import urllib.parse

        captured_url = []

        def mock_urlopen(req, context=None, timeout=None):
            captured_url.append(req.full_url)
            # Return minimal valid Atom feed
            feed = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""
            mock_resp = MagicMock()
            mock_resp.read.return_value = feed
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from dojo.research.arxiv_scanner import search
            search("quantization", max_results=1)

        assert captured_url, "urlopen was not called"
        url = captured_url[0]
        decoded = urllib.parse.unquote(url)
        assert "cat:cs.LG" in decoded, (
            f"arXiv search URL must include ML category filter (cat:cs.LG), got: {decoded}"
        )


class TestClaimExtractor:
    """Verify rule-based claim extraction from abstracts."""

    def test_extracts_gptq_technique(self):
        from dojo.research.claim_extractor import extract_claims
        abstract = "We apply GPTQ to quantize LLaMA-7B to 4-bit, achieving 3.2× speedup."
        claims = extract_claims("2401.12345", "GPTQ Quantization", abstract)
        techniques = [c.technique for c in claims]
        assert "gptq" in techniques

    def test_extracts_fp8_technique(self):
        from dojo.research.claim_extractor import extract_claims
        abstract = "FP8 training on H100 reduces memory by 40% vs BF16 baseline."
        claims = extract_claims("2402.00001", "FP8 Training", abstract)
        techniques = [c.technique for c in claims]
        assert "fp8" in techniques

    def test_extracts_flash_attention(self):
        from dojo.research.claim_extractor import extract_claims
        abstract = "FlashAttention-2 achieves 2× speedup over standard attention with IO-awareness."
        claims = extract_claims("2307.08691", "FlashAttention-2", abstract)
        techniques = [c.technique for c in claims]
        assert "flash_attention" in techniques

    def test_detects_hardware(self):
        from dojo.research.claim_extractor import extract_claims
        abstract = "We benchmark LoRA on an H100 GPU cluster. Results show 35% less memory."
        claims = extract_claims("2401.99999", "LoRA on H100", abstract)
        assert any(c.hardware == "H100" for c in claims)

    def test_detects_improvement_pct(self):
        from dojo.research.claim_extractor import extract_claims
        abstract = "Our GPTQ approach is 25% faster than the BF16 baseline on RTX 4090."
        claims = extract_claims("2401.11111", "GPTQ Speed Test", abstract)
        assert claims, "Expected at least one claim from GPTQ abstract"
        assert any(abs(c.improvement_pct - 25.0) < 1.0 for c in claims)

    def test_no_techniques_returns_empty_when_ollama_unavailable(self):
        from dojo.research.claim_extractor import extract_claims
        abstract = "We study the social impact of AI on employment markets."
        # Patch the Ollama fallback so this test is deterministic even when Ollama is running.
        with patch("dojo.research.claim_extractor._extract_via_ollama", return_value=[]):
            claims = extract_claims("2401.00000", "AI Ethics", abstract)
        assert claims == []


class TestDimensionMapper:
    """Verify technique → dimension mapping."""

    def test_gptq_maps_to_d22(self):
        from dojo.research.claim_extractor import CapabilityClaim
        from dojo.research.dimension_mapper import map_claims
        claim = CapabilityClaim(
            paper_id="test", title="t", technique="gptq",
            metric="tps", value=200.0, baseline_value=100.0,
            improvement_pct=100.0, hardware="A100",
        )
        result = map_claims([claim])
        assert "d22" in result[0].dimension_ids

    def test_awq_maps_to_d23(self):
        from dojo.research.claim_extractor import CapabilityClaim
        from dojo.research.dimension_mapper import map_claims
        claim = CapabilityClaim(
            paper_id="test", title="t", technique="awq",
            metric="ppl", value=8.0, baseline_value=10.0,
            improvement_pct=20.0, hardware="RTX40xx",
        )
        result = map_claims([claim])
        assert "d23" in result[0].dimension_ids

    def test_fp8_maps_to_d10(self):
        from dojo.research.claim_extractor import CapabilityClaim
        from dojo.research.dimension_mapper import map_claims
        claim = CapabilityClaim(
            paper_id="test", title="t", technique="fp8",
            metric="tokens_per_sec", value=1500.0, baseline_value=1000.0,
            improvement_pct=50.0, hardware="H100",
        )
        result = map_claims([claim])
        assert "d10" in result[0].dimension_ids

    def test_lora_maps_to_d17(self):
        from dojo.research.claim_extractor import CapabilityClaim
        from dojo.research.dimension_mapper import map_claims
        claim = CapabilityClaim(
            paper_id="test", title="t", technique="lora",
            metric="quality_score", value=85.0, baseline_value=80.0,
            improvement_pct=6.25, hardware="A100",
        )
        result = map_claims([claim])
        assert "d17" in result[0].dimension_ids

    def test_unknown_technique_falls_back(self):
        from dojo.research.claim_extractor import CapabilityClaim
        from dojo.research.dimension_mapper import map_claims
        claim = CapabilityClaim(
            paper_id="test", title="t", technique="totally_new_technique",
            metric="tps", value=100.0, baseline_value=50.0,
            improvement_pct=100.0, hardware="H100",
        )
        result = map_claims([claim])
        assert len(result[0].dimension_ids) > 0  # fallback dimension assigned


class TestStubGenerator:
    """Verify test stub generation creates valid Python files."""

    def _make_claim(self):
        from dojo.research.claim_extractor import CapabilityClaim
        return CapabilityClaim(
            paper_id="2401.12345v1",
            title="Fast Quantization on GPU",
            technique="gptq",
            metric="tokens_per_sec",
            value=1500.0,
            baseline_value=1000.0,
            improvement_pct=50.0,
            hardware="H100",
            dimension_ids=["d22"],
            raw_excerpt="We apply GPTQ to quantize LLaMA-7B.",
        )

    def test_generates_file(self, tmp_path):
        from dojo.research.test_stub_generator import generate_stub
        claim = self._make_claim()
        path = generate_stub(claim, tmp_path)
        assert path.exists()
        assert path.suffix == ".py"

    def test_stub_contains_paper_id(self, tmp_path):
        from dojo.research.test_stub_generator import generate_stub
        claim = self._make_claim()
        path = generate_stub(claim, tmp_path)
        content = path.read_text()
        assert claim.paper_id in content

    def test_stub_contains_technique(self, tmp_path):
        from dojo.research.test_stub_generator import generate_stub
        claim = self._make_claim()
        path = generate_stub(claim, tmp_path)
        content = path.read_text()
        assert claim.technique in content

    def test_stub_is_valid_python(self, tmp_path):
        from dojo.research.test_stub_generator import generate_stub
        import ast
        claim = self._make_claim()
        path = generate_stub(claim, tmp_path)
        source = path.read_text()
        ast.parse(source)   # raises SyntaxError if invalid


class TestHarvestLoop:
    """Verify harvest_loop orchestration with mocked arXiv."""

    def test_harvest_returns_result(self, tmp_path):
        from dojo.research.harvest_loop import run_harvest
        from dojo.research.arxiv_scanner import ArxivPaper

        mock_papers = [
            ArxivPaper(
                arxiv_id="2401.00001",
                title="GPTQ Quantization for LLMs",
                abstract="We apply GPTQ quantization and achieve 2× speedup on H100.",
                authors=["Alice"],
                published="2024-01-01",
                pdf_url="",
                categories=["cs.LG"],
            ),
        ]

        with patch("dojo.research.harvest_loop.search", return_value=mock_papers):
            result = run_harvest(
                queries=["quantization"],
                output_dir=tmp_path,
                max_results_per_query=5,
                emit_evidence=True,
            )

        assert result.papers_scanned >= 1
        assert result.claims_extracted >= 1
        assert result.stubs_generated >= 1

        from dojo.core.evidence import EvidenceLogger
        events = EvidenceLogger.get_events()
        event_types = [e.get("type", "") for e in events]
        assert "ARXIV_PAPER_SCANNED" in event_types
        assert "ARXIV_CLAIM_EXTRACTED" in event_types
        assert "ARXIV_TEST_STUB_GENERATED" in event_types

    def test_harvest_deduplicates_papers(self, tmp_path):
        from dojo.research.harvest_loop import run_harvest
        from dojo.research.arxiv_scanner import ArxivPaper

        same_paper = ArxivPaper(
            arxiv_id="2401.99999",
            title="AWQ on RTX 4090",
            abstract="AWQ quantization 3× speedup vs BF16 baseline on RTX 4090.",
            authors=["Bob"],
            published="2024-01-02",
            pdf_url="",
            categories=["cs.LG"],
        )

        with patch("dojo.research.harvest_loop.search", return_value=[same_paper]):
            result = run_harvest(
                queries=["awq", "quantization"],   # same paper returned for both
                output_dir=tmp_path,
                max_results_per_query=5,
                emit_evidence=False,
            )

        assert result.papers_scanned == 1   # deduplicated


class TestNoveltyScoring:
    """Verify novelty scoring assigns accurate scores during harvest."""

    def _make_paper(self, arxiv_id: str, technique: str, improvement_pct: float):
        from dojo.research.arxiv_scanner import ArxivPaper
        technique_word = technique.upper().replace("_", " ")
        abstract = (
            f"We apply {technique_word} and achieve {improvement_pct:.0f}% improvement on H100."
        )
        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=f"{technique_word} Paper",
            abstract=abstract,
            authors=["Test Author"],
            published="2024-01-01",
            pdf_url="",
            categories=["cs.LG"],
        )

    def test_first_claim_for_dimension_is_maximally_novel(self, tmp_path):
        from dojo.research.harvest_loop import run_harvest
        paper = self._make_paper("2401.11111", "gptq", 50.0)
        with patch("dojo.research.harvest_loop.search", return_value=[paper]):
            result = run_harvest(["quantization"], tmp_path, emit_evidence=False)
        gptq_claims = [c for c in result.claims if c.technique == "gptq"]
        assert gptq_claims, "Expected at least one GPTQ claim"
        assert gptq_claims[0].novelty_score == 1.0, (
            f"First claim for a dimension should be maximally novel (1.0), "
            f"got {gptq_claims[0].novelty_score}"
        )

    def test_duplicate_dimension_gets_lower_novelty(self, tmp_path):
        from dojo.research.harvest_loop import run_harvest
        from dojo.research.arxiv_scanner import ArxivPaper
        papers = [
            self._make_paper("2401.22222", "gptq", 30.0),
            self._make_paper("2401.33333", "gptq", 20.0),  # same dim, lower improvement
        ]
        with patch("dojo.research.harvest_loop.search", return_value=papers):
            result = run_harvest(["quantization"], tmp_path, emit_evidence=False)
        gptq_claims = [c for c in result.claims if c.technique == "gptq"]
        assert len(gptq_claims) >= 2, f"Expected 2 GPTQ claims, got {len(gptq_claims)}"
        novelties = [c.novelty_score for c in gptq_claims]
        # Second claim (lower improvement in same dim) gets a continuous score below 0.5
        non_first = [s for s in novelties if s < 1.0]
        assert non_first, f"Expected at least one non-first novelty score, got {novelties}"
        assert all(s < 0.5 for s in non_first), (
            f"Duplicate dim with lower improvement should score < 0.5, got {non_first}"
        )

    def test_higher_improvement_gets_high_novelty(self, tmp_path):
        from dojo.research.harvest_loop import run_harvest
        papers = [
            self._make_paper("2401.44444", "gptq", 10.0),   # first → 1.0
            self._make_paper("2401.55555", "gptq", 80.0),   # bigger improvement → > 0.5
        ]
        with patch("dojo.research.harvest_loop.search", return_value=papers):
            result = run_harvest(["quantization"], tmp_path, emit_evidence=False)
        gptq_claims = [c for c in result.claims if c.technique == "gptq"]
        assert len(gptq_claims) >= 2
        novelties = sorted([c.novelty_score for c in gptq_claims], reverse=True)
        # First = 1.0 (first for dimension), second > 0.5 (bigger improvement)
        assert novelties[0] == 1.0
        assert novelties[1] > 0.5, (
            f"Higher improvement in known dimension should get novelty_score > 0.5, got {novelties[1]}"
        )

    def test_equal_improvement_gets_midpoint_novelty(self, tmp_path):
        from dojo.research.harvest_loop import run_harvest
        papers = [
            self._make_paper("2401.66666", "fp8", 40.0),   # first → 1.0
            self._make_paper("2401.77777", "fp8", 40.0),   # same → ~0.5
        ]
        with patch("dojo.research.harvest_loop.search", return_value=papers):
            result = run_harvest(["quantization"], tmp_path, emit_evidence=False)
        fp8_claims = [c for c in result.claims if c.technique == "fp8"]
        assert len(fp8_claims) >= 2
        novelties = sorted([c.novelty_score for c in fp8_claims], reverse=True)
        # Second claim with same improvement should land near 0.5
        assert novelties[0] == 1.0
        assert abs(novelties[1] - 0.5) < 0.05, (
            f"Equal improvement should give novelty ~0.5, got {novelties[1]}"
        )


def _ollama_available() -> bool:
    """Check if Ollama is reachable at localhost:11434."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


class TestOllamaFallback:
    """Verify the Ollama LLM fallback for claim extraction."""

    def test_ollama_returns_empty_on_failure(self):
        from dojo.research.claim_extractor import _extract_via_ollama
        # Force a connection failure by pointing at an invalid port
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = _extract_via_ollama("2401.99999", "Test Paper", "Some abstract text.")
        assert result == [], "Ollama unavailable should return empty list, not raise"

    def test_ollama_parses_valid_json_response(self):
        from dojo.research.claim_extractor import _extract_via_ollama
        import json
        mock_body = json.dumps({
            "choices": [{
                "message": {
                    "content": '{"technique": "gptq", "metric": "tokens_per_sec", '
                               '"improvement_pct": 35.0, "hardware": "H100"}'
                }
            }]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            claims = _extract_via_ollama("2401.12345", "Fast GPTQ", "abstract")
        assert len(claims) == 1
        assert claims[0].technique == "gptq"
        assert claims[0].improvement_pct == 35.0
        assert claims[0].hardware == "H100"

    def test_ollama_returns_empty_when_technique_is_null(self):
        from dojo.research.claim_extractor import _extract_via_ollama
        import json
        mock_body = json.dumps({
            "choices": [{"message": {"content": '{"technique": null}'}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            claims = _extract_via_ollama("2401.00000", "AI Ethics", "No ML content here.")
        assert claims == []

    def test_ollama_returns_empty_on_malformed_json(self):
        from dojo.research.claim_extractor import _extract_via_ollama
        import json
        mock_body = json.dumps({
            "choices": [{"message": {"content": "I cannot extract any claims from this abstract."}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            claims = _extract_via_ollama("2401.00000", "Test", "abstract")
        assert claims == []

    @pytest.mark.skipif(not _ollama_available(), reason="requires Ollama at localhost:11434")
    def test_live_ollama_extracts_or_returns_empty(self):
        from dojo.research.claim_extractor import _extract_via_ollama
        abstract = (
            "We present a novel method for weight pruning that reduces model size "
            "by 60% while maintaining 98% of baseline accuracy on ImageNet."
        )
        claims = _extract_via_ollama("live_test", "Weight Pruning Paper", abstract)
        # Either extracts a claim or returns [] — both are valid; should not raise
        assert isinstance(claims, list)
