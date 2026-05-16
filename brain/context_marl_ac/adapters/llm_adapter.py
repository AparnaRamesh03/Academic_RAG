"""
brain/context_marl_ac/adapters/llm_adapter.py
---------------------------------------------
LLM adapter for Context MARL Actor-Critic RAG.

This file wraps the existing final_arch rewriter/grader/generator nodes, but
uses a lightweight single-call verifier for MARL training/evaluation.

Why:
- final_arch/claim_verifier.py may internally call Groq multiple times.
- During MARL training this creates repeated verifier logs and high TPM usage.
- This lightweight verifier performs one strict JSON verification call per
  MARL verifier step.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_BRAIN_ROOT = Path(__file__).resolve().parents[2]
_FINAL_ARCH_DIR = _BRAIN_ROOT / "final_arch"

for _p in [str(_BRAIN_ROOT), str(_FINAL_ARCH_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import context_marl_ac.config as cfg
except ImportError:
    _MARL_ROOT = Path(__file__).resolve().parents[1]
    if str(_MARL_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(_MARL_ROOT.parent))
    import context_marl_ac.config as cfg


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


GROQ_RETRY_MAX_ATTEMPTS = int(_cfg("GROQ_RETRY_MAX_ATTEMPTS", 6))
GROQ_RETRY_BASE_SLEEP_SECONDS = float(_cfg("GROQ_RETRY_BASE_SLEEP_SECONDS", 2.0))
GROQ_RETRY_MAX_SLEEP_SECONDS = float(_cfg("GROQ_RETRY_MAX_SLEEP_SECONDS", 30.0))
GROQ_MIN_SECONDS_BETWEEN_CALLS = float(_cfg("GROQ_MIN_SECONDS_BETWEEN_CALLS", 1.25))

LLM_MAX_EVIDENCE_DOCS = int(_cfg("LLM_MAX_EVIDENCE_DOCS", 6))
LLM_MAX_EVIDENCE_CHARS_PER_DOC = int(_cfg("LLM_MAX_EVIDENCE_CHARS_PER_DOC", 900))

LLM_MAX_GRADER_CHUNKS = int(_cfg("LLM_MAX_GRADER_CHUNKS", 16))
LLM_MAX_GRADER_CHARS_PER_CHUNK = int(_cfg("LLM_MAX_GRADER_CHARS_PER_CHUNK", 800))

LIGHTWEIGHT_VERIFIER_MODEL = str(
    _cfg(
        "LIGHTWEIGHT_VERIFIER_MODEL",
        os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    )
)


_last_llm_call_time = 0.0


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""

    text = str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + " ..."


def _compact_evidence_pack(evidence_pack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compacted = []

    for item in evidence_pack[:LLM_MAX_EVIDENCE_DOCS]:
        new_item = dict(item)
        new_item["text"] = _truncate_text(
            item.get("text", ""),
            LLM_MAX_EVIDENCE_CHARS_PER_DOC,
        )
        compacted.append(new_item)

    return compacted


def _compact_chunks_for_grading(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compacted = []

    for chunk in chunks[:LLM_MAX_GRADER_CHUNKS]:
        new_chunk = dict(chunk)

        new_chunk["text"] = _truncate_text(
            chunk.get("text", ""),
            LLM_MAX_GRADER_CHARS_PER_CHUNK,
        )

        metadata = dict(new_chunk.get("metadata", {}))

        if "text" in metadata:
            metadata["text"] = _truncate_text(
                metadata.get("text", ""),
                LLM_MAX_GRADER_CHARS_PER_CHUNK,
            )

        new_chunk["metadata"] = metadata
        compacted.append(new_chunk)

    return compacted


def _evidence_pack_to_docs(evidence_pack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []

    for idx, item in enumerate(evidence_pack):
        text = item.get("text", "")

        docs.append(
            {
                "text": text,
                "metadata": {
                    "source_file": item.get("source", "Unknown Source"),
                    "page_number": item.get("page", "Unknown Page"),
                    "section_header": item.get("section", "Unknown Section"),
                    "citation_id": item.get("citation_id", f"[{idx + 1}]"),
                    "content_type": "text",
                    "excerpt": text[:300],
                },
                "score": item.get("retrieval_score", 0.0),
            }
        )

    return docs


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    return (
        "ratelimit" in name
        or "rate limit" in msg
        or "rate_limit_exceeded" in msg
        or "429" in msg
        or "tokens per minute" in msg
        or "tpm" in msg
    )


def _extract_retry_after_seconds(exc: Exception) -> float:
    msg = str(exc)

    ms_match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*ms",
        msg,
        re.IGNORECASE,
    )
    if ms_match:
        return max(1.0, float(ms_match.group(1)) / 1000.0)

    sec_match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*s",
        msg,
        re.IGNORECASE,
    )
    if sec_match:
        return max(1.0, float(sec_match.group(1)))

    return 0.0


def _pace_llm_calls() -> None:
    global _last_llm_call_time

    now = time.time()
    elapsed = now - _last_llm_call_time
    wait = GROQ_MIN_SECONDS_BETWEEN_CALLS - elapsed

    if wait > 0:
        time.sleep(wait)

    _last_llm_call_time = time.time()


def _call_with_retry(fn: Callable[[], Any], label: str = "llm_call") -> Any:
    last_exc = None

    for attempt in range(1, GROQ_RETRY_MAX_ATTEMPTS + 1):
        try:
            _pace_llm_calls()
            return fn()

        except Exception as exc:
            last_exc = exc

            if not _is_rate_limit_error(exc):
                raise

            suggested_wait = _extract_retry_after_seconds(exc)
            exponential_wait = GROQ_RETRY_BASE_SLEEP_SECONDS * (2 ** (attempt - 1))
            wait = max(suggested_wait, exponential_wait)
            wait = min(wait, GROQ_RETRY_MAX_SLEEP_SECONDS)
            wait += random.uniform(0.0, 0.5)

            print(
                f"[llm_adapter] Rate limit during {label}. "
                f"Retry {attempt}/{GROQ_RETRY_MAX_ATTEMPTS} after {wait:.2f}s."
            )

            time.sleep(wait)

    raise last_exc


# ---------------------------------------------------------------------------
# Load existing final_arch nodes for rewrite / grade / generate only
# ---------------------------------------------------------------------------

_llm_loaded = False
_rewrite_fn = None
_grade_fn = None
_gen_fn = None


def _ensure_llm_loaded() -> None:
    global _llm_loaded, _rewrite_fn, _grade_fn, _gen_fn

    if _llm_loaded:
        return

    try:
        from node_rewriter import rewrite_query as _rw
        from node_grader import grade_documents as _gd
        from node_generator import generate as _gen

        _rewrite_fn = _rw
        _grade_fn = _gd
        _gen_fn = _gen
        _llm_loaded = True

    except Exception as exc:
        raise ImportError(f"[llm_adapter] Failed to load brain/final_arch nodes: {exc}")


# ---------------------------------------------------------------------------
# Public adapter functions
# ---------------------------------------------------------------------------

def rewrite_query(query: str, mode: str = "simple_rewrite") -> Tuple[str, int]:
    if cfg.DRY_RUN:
        return f"{query} [rewritten:{mode}]", 0

    if mode == "no_rewrite":
        return query, 0

    _ensure_llm_loaded()

    mock_state = {
        "original_query": query,
        "search_query": query,
        "weak_signal_docs": [],
        "crag_retries": 0,
    }

    result = _call_with_retry(
        lambda: _rewrite_fn(mock_state),
        label=f"rewrite_query:{mode}",
    )

    rewritten = result.get("search_query", query) if isinstance(result, dict) else query

    return rewritten, _estimate_tokens(query + rewritten)


def grade_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    mode: str = "medium_filter",
) -> Tuple[List[Dict[str, Any]], int]:
    if cfg.DRY_RUN:
        if mode == "keep_all":
            return chunks, 0
        if mode == "strict_filter":
            return [c for c in chunks if c.get("score", 0) > 0.85], 0
        return [c for c in chunks if c.get("score", 0) > 0.75], 0

    if mode == "keep_all":
        return chunks, 0

    _ensure_llm_loaded()

    compact_chunks = _compact_chunks_for_grading(chunks)

    mock_state = {
        "original_query": query,
        "retrieved_docs": compact_chunks,
    }

    result = _call_with_retry(
        lambda: _grade_fn(mock_state),
        label=f"grade_chunks:{mode}",
    )

    filtered = result.get("retrieved_docs", []) if isinstance(result, dict) else []

    chunk_text = "".join([c.get("text", "") for c in compact_chunks])
    return filtered, _estimate_tokens(query + chunk_text)


def generate_answer(
    query: str,
    evidence_pack: List[Dict[str, Any]],
    mode: str = "generate_answer",
    choices: Dict[str, Any] = None,  # New: ARC support
) -> Tuple[str, int]:
    if cfg.DRY_RUN:
        if mode == "abstain_request_more_evidence":
            return "Insufficient information.", 0

        text_snippet = evidence_pack[0].get("text", "")[:100] if evidence_pack else ""
        return f"[DRY-RUN] Answer about {text_snippet}", 0

    _ensure_llm_loaded()

    compact_evidence = _compact_evidence_pack(evidence_pack)
    docs = _evidence_pack_to_docs(compact_evidence)

    mock_state = {
        "original_query": query,
        "search_query": query,
        "graded_docs": docs,
        "final_answer": "",
        "choices": choices,
    }

    result = _call_with_retry(
        lambda: _gen_fn(mock_state),
        label=f"generate_answer:{mode}",
    )

    extracted = ""

    if result and isinstance(result, str):
        extracted = result

    elif isinstance(result, dict):
        for key in [
            "final_answer",
            "answer",
            "response",
            "generation",
            "output",
            "result",
        ]:
            val = result.get(key)

            if val:
                if hasattr(val, "content"):
                    extracted = str(val.content)
                else:
                    extracted = str(val)
                break

        if not extracted:
            msgs = result.get("messages", [])
            if msgs and hasattr(msgs[-1], "content"):
                extracted = str(msgs[-1].content)

    if not extracted and isinstance(result, (list, tuple)) and len(result) > 0:
        if hasattr(result[0], "content"):
            extracted = str(result[0].content)
        elif isinstance(result[0], str):
            extracted = result[0]

    evidence_text = "".join([e.get("text", "") for e in compact_evidence])
    tokens = _estimate_tokens(query + evidence_text + extracted)

    return extracted, tokens


# ---------------------------------------------------------------------------
# Lightweight single-call verifier
# ---------------------------------------------------------------------------

_lightweight_verifier_llm = None


def _get_lightweight_verifier_llm():
    """
    Lazily create a ChatGroq instance.

    Supports both newer and older langchain_groq constructor signatures.
    """
    global _lightweight_verifier_llm

    if _lightweight_verifier_llm is not None:
        return _lightweight_verifier_llm

    try:
        from langchain_groq import ChatGroq

        try:
            _lightweight_verifier_llm = ChatGroq(
                model=LIGHTWEIGHT_VERIFIER_MODEL,
                temperature=0,
            )
        except TypeError:
            _lightweight_verifier_llm = ChatGroq(
                model_name=LIGHTWEIGHT_VERIFIER_MODEL,
                temperature=0,
            )

        return _lightweight_verifier_llm

    except Exception as exc:
        raise ImportError(f"[llm_adapter] Failed to create lightweight verifier LLM: {exc}")


def _build_verifier_evidence_text(evidence_pack: List[Dict[str, Any]]) -> str:
    compact_evidence = _compact_evidence_pack(evidence_pack)
    blocks = []

    for idx, item in enumerate(compact_evidence, start=1):
        source = item.get("source", "Unknown Source")
        page = item.get("page", "Unknown Page")
        section = item.get("section", "Unknown Section")
        citation_id = item.get("citation_id", f"[{idx}]")
        text = item.get("text", "")

        blocks.append(
            f"Document {idx}\n"
            f"Citation ID: {citation_id}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Section: {section}\n"
            f"Text:\n{text}"
        )

    return "\n\n---\n\n".join(blocks)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract JSON from a model response.

    Handles:
    - raw JSON
    - ```json fenced blocks
    - extra text around the JSON object
    """
    if not text:
        raise ValueError("Empty verifier response.")

    cleaned = text.strip()

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)


def _normalize_verified_claims(raw_claims: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_claims, list):
        return []

    normalized = []

    for idx, claim in enumerate(raw_claims, start=1):
        if not isinstance(claim, dict):
            continue

        supported = bool(claim.get("supported", False))

        doc_ids = claim.get("supporting_doc_ids", [])
        if not isinstance(doc_ids, list):
            doc_ids = []

        normalized.append(
            {
                "claim_id": int(claim.get("claim_id", idx)),
                "claim_text": str(claim.get("claim_text", "")).strip(),
                "supported": supported,
                "feedback": str(claim.get("feedback", "")).strip(),
                "supporting_doc_ids": doc_ids,
                "supporting_sources": claim.get("supporting_sources", []),
            }
        )

    return normalized


def _fallback_verification_result(reason: str) -> Dict[str, Any]:
    return {
        "decision": "FAIL",
        "overall_feedback": reason,
        "claims": [
            {
                "claim_id": 1,
                "claim_text": "Verification could not be completed reliably.",
                "supported": False,
                "feedback": reason,
                "supporting_doc_ids": [],
                "supporting_sources": [],
            }
        ],
    }


def _single_call_verify(query: str, answer: str, evidence_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
    llm = _get_lightweight_verifier_llm()

    evidence_text = _build_verifier_evidence_text(evidence_pack)

    prompt = f"""
You are a strict but practical evidence verifier for a RAG system.

Task:
Check whether the answer is supported by the provided evidence.

Rules:
1. Use only the provided evidence.
2. Do not use outside knowledge.
3. Ignore purely stylistic wording differences.
4. Mark a claim as supported if the evidence clearly implies it.
5. Mark a claim as unsupported if it contradicts the evidence or is not present.
6. Do not create claims from headings like "Key differences:".
7. Keep the number of claims small. Prefer 1-5 important factual claims.
8. Return JSON only. No markdown.

Question:
{query}

Answer:
{answer}

Evidence:
{evidence_text}

Return this exact JSON shape:
{{
  "decision": "PASS" or "FAIL",
  "overall_feedback": "brief reason",
  "claims": [
    {{
      "claim_id": 1,
      "claim_text": "claim text",
      "supported": true or false,
      "feedback": "supported or reason unsupported",
      "supporting_doc_ids": [1]
    }}
  ]
}}

Decision rule:
- PASS if the primary answer is factually correct and supported by the evidence.
- Do not reject an answer solely because a minor 'bridge' claim is implied rather than explicitly stated, as long as the final conclusion is solid.
- FAIL if the answer contradicts the evidence or makes a major factual error that is not supported.
""".strip()

    try:
        from langchain_core.messages import HumanMessage

        response = _call_with_retry(
            lambda: llm.invoke([HumanMessage(content=prompt)]),
            label="lightweight_verify_answer",
        )

        content = response.content if hasattr(response, "content") else str(response)
        parsed = _extract_json_object(content)

        decision = str(parsed.get("decision", "FAIL")).upper()
        if decision not in {"PASS", "FAIL"}:
            decision = "FAIL"

        claims = _normalize_verified_claims(parsed.get("claims", []))

        if not claims:
            return _fallback_verification_result(
                "Lightweight verifier returned no claim-level checks."
            )

        # Make final decision consistent with claim support.
        if any(not c.get("supported", False) for c in claims):
            decision = "FAIL"

        return {
            "decision": decision,
            "overall_feedback": str(
                parsed.get("overall_feedback", parsed.get("reason", ""))
            ).strip(),
            "claims": claims,
        }

    except Exception as exc:
        print(
            f"[llm_adapter] Lightweight verification failed safely: "
            f"{type(exc).__name__}: {exc}"
        )
        return _fallback_verification_result(
            f"Lightweight verification failed safely: {type(exc).__name__}"
        )


def verify_answer(
    query: str,
    answer: str,
    evidence_pack: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int]:
    """
    Single-call verifier used by MARL.

    Output format matches the rest of the code:
    {
        "decision": "PASS" or "FAIL",
        "reason": "...",
        "verified_claims": [...]
    }
    """
    if cfg.DRY_RUN:
        return {
            "decision": "PASS",
            "reason": "Dry-run",
            "verified_claims": [],
        }, 0

    compact_evidence = _compact_evidence_pack(evidence_pack)

    result = _single_call_verify(query, answer, compact_evidence)

    formatted = {
        "decision": result.get("decision", "FAIL"),
        "reason": result.get("overall_feedback", ""),
        "verified_claims": result.get("claims", []),
    }

    evidence_text = "".join([e.get("text", "") for e in compact_evidence])
    tokens = _estimate_tokens(query + answer + evidence_text)

    return formatted, tokens