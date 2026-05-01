"""
node_scholar_search.py — Trusted Scholarly API Fallback

Fires only when local Qdrant retrieval is insufficient after the CRAG
rewrite cycle. Queries two trusted, peer-reviewed sources:

    1. Semantic Scholar Graph API  (api.semanticscholar.org)  — free tier
    2. arXiv API                   (export.arxiv.org)         — free, open

No raw web search is ever performed. Every result is a real academic paper.

Returned docs are shaped identically to Qdrant retrieval results so they
flow naturally into grade_documents and generate without modification.

The `is_fallback: True` metadata flag lets node_generator append an ingest
suggestion so the user knows how to get deeper, fully-cited answers.
"""

import re
import time
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import (
    ARCHITECTURE_NAME,
    SCHOLAR_FALLBACK_MAX_RESULTS,
    ARXIV_FALLBACK_MAX_RESULTS,
)
from state_agentic import AgenticGraphState

# ── API Endpoints ────────────────────────────────────────────────────────────
_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_ARXIV_URL = "http://export.arxiv.org/api/query"

# arXiv recommends >= 3 second delay between requests for the free tier
_ARXIV_POLITE_DELAY_SECONDS = 3.0

# HTTP request timeout
_REQUEST_TIMEOUT = 15


# ── Semantic Scholar ─────────────────────────────────────────────────────────

def _fetch_semantic_scholar(query: str, limit: int) -> list[dict]:
    """
    Query the Semantic Scholar Graph API (free, unauthenticated).

    Fields requested: title, abstract, authors, year, venue, citationCount.
    Returns a list of paper dicts, filtered to those with non-empty abstracts.
    """
    params = urllib.parse.urlencode({
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,venue,citationCount",
    })
    url = f"{_SEMANTIC_SCHOLAR_URL}?{params}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AcademicRAG-AgenticScholar/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        papers = []
        for paper in data.get("data", []):
            title = (paper.get("title") or "").strip()
            abstract = (paper.get("abstract") or "").strip()
            if not title or not abstract:
                continue

            authors = ", ".join(
                a.get("name", "")
                for a in (paper.get("authors") or [])[:3]
            )
            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": str(paper.get("year") or "Unknown"),
                "venue": (paper.get("venue") or "Unknown Venue").strip() or "Unknown Venue",
                "citation_count": paper.get("citationCount"),
                "source": "Semantic Scholar",
            })

        print(f"[{ARCHITECTURE_NAME}] Semantic Scholar → {len(papers)} papers with abstracts.")
        return papers

    except Exception as exc:
        print(f"[{ARCHITECTURE_NAME}] Semantic Scholar API error: {exc}")
        return []


# ── arXiv ────────────────────────────────────────────────────────────────────

def _fetch_arxiv(query: str, limit: int) -> list[dict]:
    """
    Query the arXiv Atom API (free, open access).

    Returns a list of paper dicts, filtered to those with non-empty summaries.
    Applies a polite delay before the request as recommended by arXiv.
    """
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{_ARXIV_URL}?{params}"

    # Polite delay before hitting arXiv
    time.sleep(_ARXIV_POLITE_DELAY_SECONDS)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AcademicRAG-AgenticScholar/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(content)

        papers = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)

            title = (title_el.text or "").strip() if title_el is not None else ""
            abstract = (summary_el.text or "").strip() if summary_el is not None else ""

            if not title or not abstract:
                continue

            name_els = [
                a.find("atom:name", ns)
                for a in entry.findall("atom:author", ns)[:3]
            ]
            authors = ", ".join(
                el.text for el in name_els if el is not None and el.text
            )

            published_el = entry.find("atom:published", ns)
            year = published_el.text[:4] if published_el is not None else "Unknown"

            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "venue": "arXiv",
                "citation_count": None,
                "source": "arXiv",
            })

        print(f"[{ARCHITECTURE_NAME}] arXiv → {len(papers)} papers with abstracts.")
        return papers

    except Exception as exc:
        print(f"[{ARCHITECTURE_NAME}] arXiv API error: {exc}")
        return []


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate_papers(papers: list[dict]) -> list[dict]:
    """
    Remove near-duplicate papers by normalising the title to a 60-char key.
    Semantic Scholar results are prioritised (they appear first).
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for p in papers:
        key = re.sub(r"[^a-z0-9]", "", p["title"].lower())[:60]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ── Doc Conversion ─────────────────────────────────────────────────────────────

def _paper_to_doc(paper: dict, idx: int) -> dict:
    """
    Convert a scholarly paper dict into the same shape as a Qdrant
    retrieval result so it flows cleanly into grade_documents and generate.

    The `is_fallback: True` flag lets the generator add an ingest suggestion.
    """
    source_label = (
        f"[{paper['source']}] {paper['title']} ({paper['year']})"
    )

    citation_note = (
        f" | Citations: {paper['citation_count']}"
        if paper.get("citation_count") is not None
        else ""
    )

    text = (
        f"Title: {paper['title']}\n"
        f"Authors: {paper['authors']}\n"
        f"Year: {paper['year']} | Venue: {paper['venue']}{citation_note}\n\n"
        f"Abstract: {paper['abstract']}"
    )

    return {
        "text": text,
        "metadata": {
            "source_file": source_label,
            "section_header": "Abstract",
            "page_number": None,
            "chunk_index": idx,
            "content_type": "scholar_abstract",
            "has_table": False,
            "has_image_description": False,
            "continued_from_previous_page": False,
            "previous_page_number": None,
            "is_fallback": True,
            "scholar_source": paper["source"],
        },
        "score": 0.0,
    }


# ── Main Node ─────────────────────────────────────────────────────────────────

def scholar_search(state: AgenticGraphState) -> dict:
    """
    Scholarly fallback node.

    Triggered only when local Qdrant retrieval is insufficient even after
    the CRAG query-rewrite cycle. Queries Semantic Scholar and arXiv using
    the original user query, deduplicates results, and injects them directly
    into `graded_docs` so generation can proceed without re-grading.

    Only trusted, peer-reviewed academic APIs are used. No open web search.
    """
    query = state["original_query"]

    print(f"\n[{ARCHITECTURE_NAME}] ── Scholarly Fallback Activated ──────────────────")
    print(f"[{ARCHITECTURE_NAME}] Local retrieval was insufficient after rewrite.")
    print(f"[{ARCHITECTURE_NAME}] Querying trusted scholarly APIs for: '{query}'")
    print(f"[{ARCHITECTURE_NAME}] Sources: Semantic Scholar (free tier) + arXiv")

    # Query both APIs
    ss_papers = _fetch_semantic_scholar(query, SCHOLAR_FALLBACK_MAX_RESULTS)
    arxiv_papers = _fetch_arxiv(query, ARXIV_FALLBACK_MAX_RESULTS)

    all_papers = ss_papers + arxiv_papers
    all_papers = _deduplicate_papers(all_papers)

    print(
        f"[{ARCHITECTURE_NAME}] Fallback total: {len(ss_papers)} SS + "
        f"{len(arxiv_papers)} arXiv → {len(all_papers)} unique papers."
    )

    scholar_docs = [_paper_to_doc(p, idx) for idx, p in enumerate(all_papers)]

    if not scholar_docs:
        print(f"[{ARCHITECTURE_NAME}] Scholarly fallback returned no results.")
    else:
        print(f"[{ARCHITECTURE_NAME}] Fallback docs ready for generation.")

    return {
        "scholar_docs": scholar_docs,
        "scholar_used": True,
        # Inject directly into graded_docs so generate can proceed.
        # We skip re-grading for fallback docs to avoid latency.
        "graded_docs": scholar_docs,
    }
