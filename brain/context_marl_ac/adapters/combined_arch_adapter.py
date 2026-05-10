"""
brain/context_marl_ac/adapters/combined_arch_adapter.py
---------------------------------------------------------
Optional wrapper around the existing brain/final_arch/ combined/reliability-
enhanced pipeline.  Useful for baseline comparison or fallback routing.

Exposed API
-----------
    run_combined_pipeline(query)
        -> dict with keys: answer, citations, latency_sec, generation

This adapter is OPTIONAL — the MARL system does not depend on it during
normal operation.  It exists so the new architecture can call the combined
pipeline as a reference or as a fallback.

In dry-run mode it returns a stub result without touching the graph.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_BRAIN_ROOT     = Path(__file__).resolve().parents[2]
_FINAL_ARCH_DIR = _BRAIN_ROOT / "final_arch"

for _p in [str(_BRAIN_ROOT), str(_FINAL_ARCH_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    from context_marl_ac.config import DRY_RUN
except ImportError:
    _MARL_ROOT = Path(__file__).resolve().parents[1]
    if str(_MARL_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(_MARL_ROOT.parent))
    from context_marl_ac.config import DRY_RUN


# ---------------------------------------------------------------------------
# Lazy import of the combined pipeline graph
# ---------------------------------------------------------------------------
_combined_graph    = None
_combined_loaded   = False


def _ensure_combined_loaded() -> None:
    global _combined_graph, _combined_loaded
    if _combined_loaded:
        return
    try:
        from graph import build_graph  # brain/final_arch/graph.py
        _combined_graph  = build_graph()
        _combined_loaded = True
    except Exception as exc:
        raise ImportError(
            f"[combined_arch_adapter] Failed to build the combined LangGraph pipeline.\n"
            f"Make sure brain/final_arch/ is importable and Groq/Qdrant are reachable.\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_combined_pipeline(query: str) -> Dict[str, Any]:
    """
    Run the full combined / reliability-enhanced RAG pipeline on a query.

    The pipeline internally handles:
      retrieve → rerank → evaluate → (rewrite → retrieve → rerank → select) →
      grade → generate → audit

    Parameters
    ----------
    query : str — user question.

    Returns
    -------
    dict with keys:
        answer       : str
        citations    : List[dict]  (source_file, page_number, section_header, excerpt)
        generation   : str  (alias for answer)
        latency_sec  : float
        status       : "ok" | "error"
        error        : str | None
    """
    if DRY_RUN:
        return {
            "answer":      f"[DRY-RUN] Combined pipeline answer for: {query}",
            "generation":  f"[DRY-RUN] Combined pipeline answer for: {query}",
            "citations":   [],
            "latency_sec": 0.0,
            "status":      "ok",
            "error":       None,
        }

    _ensure_combined_loaded()

    initial_state: Dict[str, Any] = {
        "original_query": query,
        "search_query":   query,
    }

    t0 = time.perf_counter()
    try:
        final_state = _combined_graph.invoke(initial_state)
        latency = round(time.perf_counter() - t0, 4)

        answer = str(final_state.get("generation", "")).strip()

        # Build citation dicts from graded_docs if available
        graded_docs: List[Dict[str, Any]] = final_state.get("graded_docs", [])
        citations: List[Dict[str, Any]] = []
        for doc in graded_docs:
            meta = doc.get("metadata", {})
            citations.append({
                "source_file":    str(meta.get("source_file", "Unknown")),
                "page_number":    meta.get("page_number", ""),
                "section_header": str(meta.get("section_header", "")),
                "excerpt":        str(doc.get("text", ""))[:300],
                "content_type":   str(meta.get("content_type", "text")),
            })

        return {
            "answer":      answer,
            "generation":  answer,
            "citations":   citations,
            "latency_sec": latency,
            "status":      "ok",
            "error":       None,
        }

    except Exception as exc:
        latency = round(time.perf_counter() - t0, 4)
        return {
            "answer":      "",
            "generation":  "",
            "citations":   [],
            "latency_sec": latency,
            "status":      "error",
            "error":       str(exc),
        }
