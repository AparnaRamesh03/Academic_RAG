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
    import context_marl_ac.config as cfg
except ImportError:
    _MARL_ROOT = Path(__file__).resolve().parents[1]
    if str(_MARL_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(_MARL_ROOT.parent))
    import context_marl_ac.config as cfg


# ---------------------------------------------------------------------------
# Internal format conversion helpers
# ---------------------------------------------------------------------------

def _evidence_pack_to_docs(evidence_pack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert evidence_pack items → legacy {text, metadata, score} format
    expected by the existing brain/final_arch LLM nodes.
    """
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
                "excerpt":        item.get("text", "")[:300], # Excerpt for verifier/generator
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
        raise ImportError(
            f"[llm_adapter] Failed to load brain/final_arch LLM nodes.\n"
            f"Make sure Groq is reachable and brain/final_arch/ is on sys.path.\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_query(query: str, mode: str = "simple_rewrite") -> str:
    if cfg.DRY_RUN:
        return f"{query} [rewritten:{mode}]"

    if mode == "no_rewrite":
        return query

    _ensure_llm_loaded()

    mock_state: Dict[str, Any] = {
        "original_query":   query,
        "search_query":     query,
        "weak_signal_docs": [],
        "crag_retries":     0,
    }

    result = _rewrite_fn(mock_state)
    return result.get("search_query", query)


def grade_chunks(query: str, chunks: List[Dict[str, Any]], mode: str = "medium_filter") -> List[Dict[str, Any]]:
    if cfg.DRY_RUN:
        if mode == "keep_all":
            return chunks
        if mode == "strict_filter":
            return [c for c in chunks if c.get("score", 0) > 0.85]
        return [c for c in chunks if c.get("score", 0) > 0.75]

    if mode == "keep_all":
        return chunks

    _ensure_llm_loaded()

    mock_state: Dict[str, Any] = {
        "original_query": query,
        "retrieved_docs": chunks,
    }

    result = _grade_fn(mock_state)
    return result.get("retrieved_docs", [])


def generate_answer(query: str, evidence_pack: List[Dict[str, Any]], mode: str = "generate_answer") -> str:
    if cfg.DRY_RUN:
        if mode == "abstain_request_more_evidence":
            return "I don't have enough information to answer this question."
        
        text_snippet = evidence_pack[0].get('text', 'No text available')[:100] if evidence_pack else "no information available"
        refs = ", ".join([e.get("citation_id", "[?]") for e in evidence_pack]) if evidence_pack else "none"
        
        if mode == "generate_short_answer":
            return f"[DRY-RUN SHORT ANSWER] Factual summary: {text_snippet[:50]}. Citations: {refs}."
        elif mode == "generate_with_strict_citations":
            return f"[DRY-RUN STRICT ANSWER] Detailed finding: {text_snippet}. Citations: {refs}."
        else:
            return f"[DRY-RUN ANSWER] The evidence suggests that {text_snippet}. This is supported by citations: {refs}."

    _ensure_llm_loaded()

    docs = _evidence_pack_to_docs(evidence_pack)

    mock_state: Dict[str, Any] = {
        "original_query":   query,
        "search_query":     query,
        "graded_docs":      docs,
        "final_answer":     "",
    }

    result = _gen_fn(mock_state)
    
    # --- DETAILED DEBUG LOGGING ---
    res_type = str(type(result))
    res_keys = []
    if isinstance(result, dict):
        res_keys = list(result.keys())
    
    preview = ""
    if isinstance(result, str): preview = result[:200]
    elif isinstance(result, dict): preview = str(result)[:200]
    
    print(f"\n[llm_adapter] DEBUG GENERATION:")
    print(f"  -> Type: {res_type}")
    print(f"  -> Keys: {res_keys}")
    print(f"  -> Preview: {preview}")
    # ------------------------------

    # Robust Answer Extraction
    if result is None: return ""
    if isinstance(result, str): return result
    
    if isinstance(result, dict):
        # 1. Try common keys
        for key in ["final_answer", "answer", "response", "generation", "output", "result"]:
            val = result.get(key)
            if val:
                if hasattr(val, "content"): return str(val.content)
                return str(val)
        
        # 2. Try LangChain messages list
        msgs = result.get("messages", [])
        if msgs and hasattr(msgs[-1], "content"):
            return str(msgs[-1].content)
            
    # 3. Handle tuples/lists
    if isinstance(result, (list, tuple)) and len(result) > 0:
        if isinstance(result[0], str): return result[0]
        if hasattr(result[0], "content"): return str(result[0].content)

    return ""


def verify_answer(query: str, answer: str, evidence_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
    if cfg.DRY_RUN:
        if "abstain" in answer.lower() or not evidence_pack:
            return {"decision": "FAIL", "reason": "Insufficient evidence", "verified_claims": []}
        
        import random
        if random.random() > 0.4:
            return {"decision": "PASS", "reason": "Dry-run pass", "verified_claims": []}
        else:
            return {"decision": "FAIL", "reason": "Dry-run failure", "verified_claims": []}

    _ensure_llm_loaded()

    docs = _evidence_pack_to_docs(evidence_pack)

    # FIXED: verify_claims expects positional arguments (query, answer, docs)
    # and returns {decision, overall_feedback, claims}
    result = _verify_fn(query, answer, docs)

    return {
        "decision":        result.get("decision", "FAIL"),
        "reason":          result.get("overall_feedback", ""),
        "verified_claims": result.get("claims", []),
    }
