"""
brain/context_marl_ac/adapters/llm_adapter.py
-----------------------------------------------
Wraps brain/final_arch LLM nodes to expose clean functions
needed by RewriterAgent, GraderAgent, GeneratorAgent, and VerifierAgent.

Returns (result, token_count) for all LLM-backed calls.
"""

from __future__ import annotations

import sys
import time
import re
import random
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

def _extract_retry_after_seconds(exc_msg: str) -> float:
    ms_match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*ms", exc_msg, re.IGNORECASE)
    if ms_match:
        return max(0.5, float(ms_match.group(1)) / 1000.0)
    sec_match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*s", exc_msg, re.IGNORECASE)
    if sec_match:
        return max(0.5, float(sec_match.group(1)))
    return 2.0

def _call_with_retry(fn):
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "rate limit" in exc_msg or "429" in exc_msg or "rate_limit_exceeded" in exc_msg:
                wait_sec = _extract_retry_after_seconds(exc_msg)
                wait_sec += random.uniform(0.1, 0.5)
                print(f"[llm_adapter] Rate limit hit. Retrying in {wait_sec:.2f}s (Attempt {attempt}/{max_attempts})...")
                time.sleep(wait_sec)
            else:
                raise
    raise Exception(f"Failed after {max_attempts} retries due to rate limits.")

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
    
    result = _call_with_retry(lambda: _rewrite_fn(mock_state))
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
    
    result = _call_with_retry(lambda: _grade_fn(mock_state))
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

    result = _call_with_retry(lambda: _gen_fn(mock_state))
    
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
    """
    Verify the generated answer against selected evidence.

    Important:
    The legacy final_arch claim verifier can sometimes return malformed JSON.
    This wrapper must never crash evaluation/training. If parsing fails,
    return a controlled FAIL result so the episode can terminate normally.
    """
    if cfg.DRY_RUN:
        return {
            "decision": "PASS",
            "reason": "Dry-run verification passed.",
            "verified_claims": [],
        }, 0

    if not answer or not answer.strip():
        return {
            "decision": "FAIL",
            "reason": "Empty answer; verification skipped.",
            "verified_claims": [],
            "verifier_error": "empty_answer",
        }, 0

    if not evidence_pack:
        return {
            "decision": "FAIL",
            "reason": "No evidence provided for verification.",
            "verified_claims": [],
            "verifier_error": "missing_evidence",
        }, _estimate_tokens(query + answer)

    _ensure_llm_loaded()
    docs = _evidence_pack_to_docs(evidence_pack)

    evidence_text = "".join([e.get("text", "") for e in evidence_pack])
    tokens = _estimate_tokens(query + answer + evidence_text)

    try:
        result = _call_with_retry(lambda: _verify_fn(query, answer, docs))

        if not isinstance(result, dict):
            return {
                "decision": "FAIL",
                "reason": f"Verifier returned unexpected type: {type(result)}",
                "verified_claims": [],
                "verifier_error": "unexpected_verifier_return_type",
            }, tokens

        claims = (
            result.get("claims")
            or result.get("verified_claims")
            or []
        )

        decision = str(result.get("decision", "")).upper()
        if decision not in {"PASS", "FAIL"}:
            # Fallback: infer from claims if possible.
            if claims:
                unsupported = [
                    c for c in claims
                    if not (
                        c.get("supported") is True
                        or str(c.get("decision", "")).upper() == "PASS"
                        or str(c.get("status", "")).upper() == "SUPPORTED"
                        or str(c.get("support_status", "")).upper() == "SUPPORTED"
                    )
                ]
                decision = "PASS" if not unsupported else "FAIL"
            else:
                decision = "FAIL"

        formatted = {
            "decision": decision,
            "reason": (
                result.get("overall_feedback")
                or result.get("reason")
                or result.get("feedback")
                or ""
            ),
            "verified_claims": claims,
        }

        return formatted, tokens

    except Exception as exc:
        # Do not crash evaluation/training because the verifier returned malformed JSON.
        return {
            "decision": "FAIL",
            "reason": f"Verifier failed or returned unparseable output: {type(exc).__name__}: {exc}",
            "verified_claims": [],
            "verifier_error": "claim_verifier_exception",
        }, tokens
