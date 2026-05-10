"""
brain/context_marl_ac/adapters/llm_adapter.py
-----------------------------------------------
Wraps brain/final_arch LLM nodes to expose clean functions
needed by RewriterAgent, GraderAgent, GeneratorAgent, and VerifierAgent.

Exposed API
-----------
    rewrite_query(query, mode)                    -> str
    grade_chunks(query, chunks, mode)             -> List[dict]  (filtered)
    generate_answer(query, evidence_pack, mode)   -> str
    verify_answer(query, answer, evidence_pack)   -> dict

All functions accept the new evidence_pack format (list of dicts with
chunk_id, source, page, section, text, retrieval_score, grade, citation_id)
and convert internally to the legacy GraphState / docs format expected
by the existing brain/final_arch nodes.

In dry-run mode all functions return deterministic stub values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_BRAIN_ROOT = Path(__file__).resolve().parents[2]
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
# Internal format conversion helpers
# ---------------------------------------------------------------------------

def _evidence_pack_to_docs(evidence_pack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert evidence_pack items → legacy {text, metadata, score} format
    expected by the existing brain/final_arch LLM nodes.
    """
    docs = []
    for item in evidence_pack:
        docs.append({
            "text": item.get("text", ""),
            "metadata": {
                "source_file":    item.get("source", "Unknown Source"),
                "page_number":    item.get("page", "Unknown Page"),
                "section_header": item.get("section", "Unknown Section"),
                "content_type":   "text",
            },
            "score": item.get("retrieval_score", 0.0),
        })
    return docs


def _chunks_to_docs(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw retriever chunks (already in legacy format) to legacy format.
    This is a no-op but kept explicit for clarity.
    """
    return chunks


# ---------------------------------------------------------------------------
# Lazy import helpers for real LLM nodes
# ---------------------------------------------------------------------------
_llm_loaded = False
_rewrite_fn = None
_grade_fn   = None
_gen_fn     = None
_verify_fn  = None


def _ensure_llm_loaded() -> None:
    global _llm_loaded, _rewrite_fn, _grade_fn, _gen_fn, _verify_fn
    if _llm_loaded:
        return
    try:
        from node_rewriter import rewrite_query as _rw
        from node_grader import grade_documents as _gd
        from node_generator import generate as _gen
        from claim_verifier import verify_claims as _vc

        _rewrite_fn = _rw
        _grade_fn   = _gd
        _gen_fn     = _gen
        _verify_fn  = _vc
        _llm_loaded = True

    except Exception as exc:
        raise ImportError(
            f"[llm_adapter] Failed to load brain/final_arch LLM nodes.\n"
            f"Make sure GROQ_API_KEY is set and brain/final_arch/ is importable.\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Rewrite modes → prompt hints passed via GraphState fields
# ---------------------------------------------------------------------------
_REWRITE_MODE_MAP: Dict[str, str] = {
    "no_rewrite":         "no_rewrite",
    "simple_rewrite":     "simple",
    "keyword_rewrite":    "keyword",
    "expanded_rewrite":   "expanded",
    "multi_query_rewrite": "multi",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_query(query: str, mode: str = "simple_rewrite") -> str:
    """
    Rewrite the query for better retrieval.

    Parameters
    ----------
    query : str
        Original user query.
    mode : str
        One of the REWRITER_ACTIONS defined in schemas/actions.py.

    Returns
    -------
    str
        Rewritten query string.
    """
    if DRY_RUN:
        return f"{query} [rewritten:{mode}]"

    if mode == "no_rewrite":
        return query

    _ensure_llm_loaded()

    # Build a mock GraphState-compatible dict
    mock_state: Dict[str, Any] = {
        "original_query":   query,
        "search_query":     query,
        "weak_signal_docs": [],
        "crag_retries":     0,
    }

    result = _rewrite_fn(mock_state)
    return result.get("search_query", query)


def grade_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    mode: str = "medium_filter",
) -> List[Dict[str, Any]]:
    """
    Grade / filter retrieved chunks by relevance to the query.

    Parameters
    ----------
    query  : str
    chunks : list of raw retriever chunk dicts {text, metadata, score}
    mode   : grader action string — controls how aggressively to filter.

    Returns
    -------
    list of chunk dicts that passed the grade filter.
    """
    if DRY_RUN:
        if mode == "keep_all":
            return chunks
        # Stub: keep chunks with score >= 0.5
        return [c for c in chunks if c.get("score", 1.0) >= 0.5] or chunks

    if mode == "keep_all":
        return chunks

    _ensure_llm_loaded()

    mock_state: Dict[str, Any] = {
        "original_query": query,
        "candidate_docs": chunks,
    }
    result = _grade_fn(mock_state)
    graded = result.get("graded_docs", chunks)

    # Apply additional threshold tightening based on mode
    if mode == "strict_filter":
        # Keep only chunks the grader approved with relatively high scores
        top_n = max(1, len(graded) // 2)
        graded = sorted(graded, key=lambda d: d.get("score", 0), reverse=True)[:top_n]
    elif mode == "loose_filter":
        # Accept all grader-approved chunks
        pass
    elif mode == "rerank_only":
        # Sort by score but do not drop any grader-approved chunks
        graded = sorted(graded, key=lambda d: d.get("score", 0), reverse=True)

    return graded


def generate_answer(
    query: str,
    evidence_pack: List[Dict[str, Any]],
    mode: str = "generate_answer",
) -> str:
    """
    Generate an answer grounded in the provided evidence pack.

    Parameters
    ----------
    query         : str
    evidence_pack : list of evidence_pack dicts (new format) OR raw chunk dicts.
    mode          : generator action string.

    Returns
    -------
    str — generated answer.
    """
    if DRY_RUN:
        sources = ", ".join(
            set(
                item.get("source", item.get("metadata", {}).get("source_file", "?"))
                for item in evidence_pack
            )
        )
        return (
            f"[DRY-RUN ANSWER] Based on the retrieved evidence ({sources}), "
            f"the answer to '{query}' is a placeholder response generated in dry-run mode."
        )

    if mode == "abstain_request_more_evidence":
        return "I do not have sufficient evidence to answer this question. More retrieval is needed."

    _ensure_llm_loaded()

    # Convert evidence_pack to legacy doc format
    docs = _evidence_pack_to_docs(evidence_pack) if _is_evidence_pack(evidence_pack) else evidence_pack

    auditor_feedback = ""
    if mode == "regenerate":
        auditor_feedback = "Please regenerate the answer with more specificity."

    mock_state: Dict[str, Any] = {
        "original_query":   query,
        "graded_docs":      docs,
        "auditor_feedback": auditor_feedback,
        "verify_retries":   1 if mode == "regenerate" else 0,
    }
    result = _gen_fn(mock_state)
    return result.get("generation", "")


def verify_answer(
    query: str,
    answer: str,
    evidence_pack: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify the generated answer claim-by-claim against the evidence pack.

    Parameters
    ----------
    query         : str
    answer        : str — generated answer to verify.
    evidence_pack : list of evidence_pack dicts or raw chunk dicts.

    Returns
    -------
    dict with keys:
        decision         : "PASS" | "FAIL"
        overall_feedback : str
        claims           : list[dict]  — per-claim verification records
    """
    if DRY_RUN:
        return {
            "decision": "PASS",
            "overall_feedback": "[DRY-RUN] All claims accepted in dry-run mode.",
            "claims": [
                {
                    "claim_id": 1,
                    "claim_text": answer[:80] + "..." if len(answer) > 80 else answer,
                    "supported": True,
                    "feedback": "supported (dry-run)",
                    "supporting_doc_ids": [1],
                    "supporting_sources": [],
                }
            ],
        }

    _ensure_llm_loaded()

    docs = _evidence_pack_to_docs(evidence_pack) if _is_evidence_pack(evidence_pack) else evidence_pack
    return _verify_fn(query, answer, docs)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_evidence_pack(items: List[Dict[str, Any]]) -> bool:
    """
    Heuristic: evidence_pack dicts have 'chunk_id' or 'retrieval_score'.
    Raw chunks have 'metadata' and 'score'.
    """
    if not items:
        return False
    first = items[0]
    return "chunk_id" in first or "retrieval_score" in first
