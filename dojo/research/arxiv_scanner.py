"""dojo/research/arxiv_scanner.py — arXiv paper scanner.

Fetches recent ML papers from the arXiv Atom feed and returns structured
paper records. Uses only stdlib (urllib, xml) — no third-party dependencies.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Optional


_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_ML_CATS = "cat:cs.LG OR cat:cs.CL OR cat:cs.AI OR cat:cs.CV OR cat:cs.NE OR cat:cs.AR"


@dataclass
class ArxivPaper:
    """Structured arXiv paper record."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    pdf_url: str
    categories: list[str] = field(default_factory=list)


def search(
    query: str,
    max_results: int = 20,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    _retry_on_rate_limit: bool = True,
) -> list[ArxivPaper]:
    """Search arXiv for papers matching *query*.

    Args:
        query:      arXiv query string (e.g. "quantization language model").
        max_results: Maximum number of results to return.
        sort_by:    "submittedDate" | "lastUpdatedDate" | "relevance"
        sort_order: "descending" | "ascending"

    Returns:
        List of ArxivPaper records, ordered by *sort_by/sort_order*.
    """
    params = urllib.parse.urlencode({
        "search_query": f"(ti:{query} OR abs:{query}) AND ({_ML_CATS})",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    })
    url = f"{_ARXIV_API}?{params}"

    try:
        import platform
        import ssl as _ssl
        # On Windows, Python often can't find the system CA cert store.
        # arXiv is a trusted academic source — use unverified context on Windows.
        if platform.system() == "Windows":
            ctx: Any = _ssl._create_unverified_context()  # type: ignore[attr-defined]
        else:
            ctx = _ssl.create_default_context()
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "DanteDojo/1.0"}),
            context=ctx,
            timeout=30,
        ) as resp:
            raw = resp.read()
    except Exception as e:
        if _retry_on_rate_limit and "429" in str(e):
            time.sleep(5)
            return search(query, max_results, sort_by, sort_order,
                         _retry_on_rate_limit=False)
        raise

    return _parse_feed(raw)


def _parse_feed(raw: bytes) -> list[ArxivPaper]:
    root = ET.fromstring(raw)
    papers = []
    for entry in root.findall("atom:entry", _NS):
        arxiv_id = _text(entry, "atom:id").split("/abs/")[-1]
        title = _text(entry, "atom:title").replace("\n", " ").strip()
        abstract = _text(entry, "atom:summary").replace("\n", " ").strip()
        published = _text(entry, "atom:published")
        authors = [
            _text(a, "atom:name")
            for a in entry.findall("atom:author", _NS)
        ]
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall("atom:category", _NS)
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", _NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        if not title or not abstract:
            continue

        papers.append(ArxivPaper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors[:5],
            published=published[:10],
            pdf_url=pdf_url,
            categories=categories[:5],
        ))
    return papers


def _text(element: ET.Element, tag: str) -> str:
    el = element.find(tag, _NS)
    return (el.text or "").strip() if el is not None else ""
