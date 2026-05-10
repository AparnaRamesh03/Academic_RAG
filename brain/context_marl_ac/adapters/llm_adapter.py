"""
brain/context_marl_ac/adapters/llm_adapter.py
-----------------------------------------------
Wraps brain/final_arch LLM nodes to expose clean functions
needed by RewriterAgent, GraderAgent, GeneratorAgent, and VerifierAgent.

Returns (result, token_count) for all LLM-backed calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
    import context_marl_ac.config as cfg
except ImportError:
    _MARL_ROOT = Path(__file__).resolve().parents[1]
    if str(_MARL_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(_MARL_ROOT.parent))
    import context_marl_ac.config as cfg


# ---------------------------------------------------------------------------
# Internal format conversion helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Fallback token estimation: ~4 characters per token."""
    if not text: return 0
    return max(1, len(text) // 4)

def _extract_tokens(result: Any, fallback_text: str = "") -> int:
    """Extract token usage from LangChain result if possible."""
    if hasattr(result, "response_metadata"):
        usage = result.response_metadata.get("token_usage", {})
        if usage:
            return usage.get("total_tokens", 0)
    
    # Check if result is a dict containing a response object
    if isinstance(result, dict):
        # Existing legacy nodes often return just a dict with the answer string.
        # They don't expose the raw response. We estimate.
        pass
        
    return _estimate_tokens(fallback_text)

def _evidence_pack_to_docs(evidence_pack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for idx, item in enumerate(evidence_pack):
        docs.append({
            "text": item.get("text", ""),
            "metadata": {
                "source_file":    item.get("source", "Unknown Source"),
                "page_number":    item.get("page", "Unknown Page"),
                "section_header": item.get("section", "Unknown Section"),
                "citation_id":    item.get("citation_id", f"[{idx + 1}]"),
                "content_type":   "text",
                "excerpt":        item.get("text", "")[:300],
            },
            "score": item.get("retrieval_score", 0.0),
        })
    return docs


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
        raise ImportError(f"[llm_adapter] Failed to load brain/final_arch nodes: {exc}")


# ---------------------------------------------------------------------------
# Public API (Returns (Result, Tokens))
# ---------------------------------------------------------------------------

def rewrite_query(query: str, mode: str = "simple_rewrite") -> Tuple[str, int]:
    if cfg.DRY_RUN:
        return f"{query} [rewritten:{mode}]", 0

    if mode == "no_rewrite":
        return query, 0

    _ensure_llm_loaded()
    mock_state = {"original_query": query, "search_query": query, "weak_signal_docs": [], "crag_retries": 0}
    
    result = _rewrite_fn(mock_state)
    rewritten = result.get("search_query", query)
    # Estimation as legacy node doesn't return raw response
    return rewritten, _estimate_tokens(query + rewritten)


def grade_chunks(query: str, chunks: List[Dict[str, Any]], mode: str = "medium_filter") -> Tuple[List[Dict[str, Any]], int]:
    if cfg.DRY_RUN:
        if mode == "keep_all": return chunks, 0
        if mode == "strict_filter": return [c for c in chunks if c.get("score", 0) > 0.85], 0
        return [c for c in chunks if c.get("score", 0) > 0.75], 0

    if mode == "keep_all":
        return chunks, 0

    _ensure_llm_loaded()
    mock_state = {"original_query": query, "retrieved_docs": chunks}
    
    result = _grade_fn(mock_state)
    filtered = result.get("retrieved_docs", [])
    # Estimation: query + all chunk texts
    chunk_text = "".join([c.get("text", "") for c in chunks])
    return filtered, _estimate_tokens(query + chunk_text)


def generate_answer(query: str, evidence_pack: List[Dict[str, Any]], mode: str = "generate_answer") -> Tuple[str, int]:
    if cfg.DRY_RUN:
        if mode == "abstain_request_more_evidence":
            return "I don't have enough information to answer this question.", 0
        text_snippet = evidence_pack[0].get('text', '')[:100] if evidence_pack else ""
        return f"[DRY-RUN] Answer about {text_snippet}", 0

    _ensure_llm_loaded()
    docs = _evidence_pack_to_docs(evidence_pack)
    mock_state = {"original_query": query, "search_query": query, "graded_docs": docs, "final_answer": ""}

    result = _gen_fn(mock_state)
    
    # Extraction logic
    extracted = ""
    if result and isinstance(result, str): extracted = result
    elif isinstance(result, dict):
        for key in ["final_answer", "answer", "response", "generation", "output", "result"]:
            val = result.get(key)
            if val:
                if hasattr(val, "content"): extracted = str(val.content)
                else: extracted = str(val)
                break
        if not extracted:
            msgs = result.get("messages", [])
            if msgs and hasattr(msgs[-1], "content"): extracted = str(msgs[-1].content)
            
    if not extracted and isinstance(result, (list, tuple)) and len(result) > 0:
        if hasattr(result[0], "content"): extracted = str(result[0].content)
        elif isinstance(result[0], str): extracted = result[0]

    # Token usage
    evidence_text = "".join([e.get("text", "") for e in evidence_pack])
    tokens = _estimate_tokens(query + evidence_text + extracted)
    
    return extracted, tokens


def verify_answer(query: str, answer: str, evidence_pack: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], int]:
    if cfg.DRY_RUN:
        return {"decision": "PASS", "reason": "Dry-run", "verified_claims": []}, 0

    _ensure_llm_loaded()
    docs = _evidence_pack_to_docs(evidence_pack)
    
    # verify_claims(query, answer, docs)
    result = _verify_fn(query, answer, docs)

    formatted = {
        "decision":        result.get("decision", "FAIL"),
        "reason":          result.get("overall_feedback", ""),
        "verified_claims": result.get("claims", []),
    }
    
    evidence_text = "".join([e.get("text", "") for e in evidence_pack])
    tokens = _estimate_tokens(query + answer + evidence_text)
    
    return formatted, tokens
