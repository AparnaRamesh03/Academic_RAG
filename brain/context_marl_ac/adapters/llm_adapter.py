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
        if mode in ("keep_all", "rerank_only"):
            return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True), 0
        if mode == "strict_filter":
            return [c for c in chunks if c.get("score", 0) > 0.85], 0
        if mode == "medium_filter":
            return [c for c in chunks if c.get("score", 0) > 0.75], 0
        return [c for c in chunks if c.get("score", 0) > 0.60], 0

    if mode in ("keep_all", "rerank_only"):
        return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True), 0

    _ensure_llm_loaded()
    # node_grader reads "candidate_docs" and returns "graded_docs".
    mock_state = {"original_query": query, "candidate_docs": chunks}
    result = _call_with_retry(lambda: _grade_fn(mock_state))
    filtered = result.get("graded_docs", [])
    if not filtered:
        # Fallback: grader returned nothing — keep top-3 by score.
        filtered = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:3]
    chunk_text = "".join([c.get("text", "") for c in chunks])
    return filtered, _estimate_tokens(query + chunk_text)


def _build_generation_prompt(query: str, docs: List[Dict[str, Any]]) -> str:
    """Replicate node_generator context-block format for temperature-aware calls."""
    blocks = []
    for idx, doc in enumerate(docs):
        meta   = doc.get("metadata", {})
        source  = meta.get("source_file",    "Unknown Source")
        section = meta.get("section_header", "Unknown Section")
        page    = meta.get("page_number",    "Unknown Page")
        score   = doc.get("score", None)
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        blocks.append(
            f"DOCUMENT [{idx + 1}]\nSource: {source}\nSection: {section}\n"
            f"Page: {page}\nRetrieval Score: {score_s}\nText:\n{doc.get('text', '')}"
        )
    context = "\n\n---\n\n".join(blocks)
    return (
        f"You are an expert academic question-answering assistant.\n\n"
        f"Your task is to answer the user's question using ONLY the retrieved documents below.\n\n"
        f"Retrieved Documents:\n---\n{context}\n---\n\n"
        f"User Question:\n{query}\n\n"
        f"Rules:\n"
        f"1. Use ONLY the retrieved documents. Do not use outside knowledge.\n"
        f"2. Prefer evidence most specific to the question.\n"
        f"3. Start with the direct answer immediately.\n"
        f"4. Keep the answer concise and evaluation-friendly.\n"
        f"5. Do NOT include inline bracket citations in the answer text.\n"
        f"Now answer the question."
    )


def _generate_with_temperature(
    query: str,
    evidence_pack: List[Dict[str, Any]],
    temperature: float,
    max_tokens: Optional[int] = None,
) -> Tuple[str, int]:
    """Direct generation bypassing the module-level LLM to use a custom temperature."""
    from langchain_core.messages import HumanMessage
    import sys
    from pathlib import Path
    _fa = _BRAIN_ROOT / "final_arch"
    if str(_fa) not in sys.path:
        sys.path.insert(0, str(_fa))
    from llm_config import build_groq_llm

    from langchain_groq import ChatGroq
    import os
    _model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    chat_kwargs: Dict[str, Any] = {"model": _model, "temperature": float(temperature)}
    if max_tokens is not None:
        chat_kwargs["max_tokens"] = int(max_tokens)
    temp_llm = ChatGroq(**chat_kwargs)

    docs    = _evidence_pack_to_docs(evidence_pack)
    prompt  = _build_generation_prompt(query, docs)

    def _call():
        return temp_llm.invoke([HumanMessage(content=prompt)])

    response = _call_with_retry(_call)
    answer   = response.content.strip() if hasattr(response, "content") else str(response)
    tokens   = _estimate_tokens(query + answer)
    return answer, tokens


def generate_answer(
    query: str,
    evidence_pack: List[Dict[str, Any]],
    mode: str = "generate_answer",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, int]:
    if cfg.DRY_RUN:
        if mode == "abstain_request_more_evidence":
            return "I don't have enough information to answer this question.", 0
        text_snippet = evidence_pack[0].get("text", "")[:100] if evidence_pack else ""
        return f"[DRY-RUN] Answer about {text_snippet}", 0

    # Temperature-aware path: bypass cached node_generator and use custom LLM.
    if temperature is not None:
        return _generate_with_temperature(query, evidence_pack, temperature, max_tokens)

    _ensure_llm_loaded()
    docs = _evidence_pack_to_docs(evidence_pack)
    mock_state = {"original_query": query, "search_query": query, "graded_docs": docs, "final_answer": ""}

    result = _call_with_retry(lambda: _gen_fn(mock_state))

    extracted = ""
    if result and isinstance(result, str):
        extracted = result
    elif isinstance(result, dict):
        for key in ["generation", "final_answer", "answer", "response", "output", "result"]:
            val = result.get(key)
            if val:
                extracted = str(val.content) if hasattr(val, "content") else str(val)
                break
        if not extracted:
            msgs = result.get("messages", [])
            if msgs and hasattr(msgs[-1], "content"):
                extracted = str(msgs[-1].content)
    if not extracted and isinstance(result, (list, tuple)) and result:
        extracted = str(result[0].content) if hasattr(result[0], "content") else str(result[0])

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
