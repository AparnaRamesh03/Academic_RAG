from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent
FINAL_COMBINED_DIR = BRAIN_DIR / "final_combined"

for path in [str(CURRENT_DIR), str(BRAIN_DIR), str(FINAL_COMBINED_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState
from config import GRADE_TOP_K, WEAK_SIGNAL_TOP_K, MAX_REWRITE_ROUNDS
from node_grader import grade_documents
from agent_protocol import extract_json_block, build_agent_update
from query_targeting import (
    infer_target_sources,
    is_comparison_query,
    is_underspecified_superlative_query,
    pick_balanced_docs,
    source_distribution,
    source_distribution_text,
)

llm = build_groq_llm(temperature=0.0)


def _build_doc_summary(docs: List[Dict[str, Any]], label: str) -> str:
    if not docs:
        return f"{label}: no documents"

    blocks = []
    for idx, doc in enumerate(docs[:6], start=1):
        meta = doc.get("metadata", {}) or {}
        text = str(doc.get("text", "")).replace("\n", " ")
        blocks.append(
            f"{label} DOC [{idx}]\n"
            f"source={meta.get('source_file', 'Unknown')}\n"
            f"section={meta.get('section_header', 'Unknown')}\n"
            f"content_type={meta.get('content_type', 'text')}\n"
            f"score={doc.get('score', 'n/a')}\n"
            f"rerank_score={doc.get('rerank_score', 'n/a')}\n"
            f"text={text[:260]}"
        )
    return "\n\n".join(blocks)


def _normalize_doc_ids(raw_ids: Any, max_doc_id: int) -> List[int]:
    if not isinstance(raw_ids, list):
        return []

    cleaned = []
    for value in raw_ids:
        try:
            doc_id = int(value)
        except Exception:
            continue
        if 1 <= doc_id <= max_doc_id and doc_id not in cleaned:
            cleaned.append(doc_id)
    return cleaned


def _grade_selected_docs(state: GraphState, selected_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = {**state, "candidate_docs": selected_docs}
    graded_updates = grade_documents(merged)
    return graded_updates.get("graded_docs", []) or selected_docs


def evidence_agent(state: GraphState):
    query = state["original_query"]
    current_search_query = state.get("search_query", query)
    retrieved_docs = state.get("retrieved_docs", []) or []
    crag_retries = int(state.get("crag_retries", 0))

    target_sources = infer_target_sources(query)
    comparison_query = is_comparison_query(query)
    underspecified_superlative = is_underspecified_superlative_query(query)

    dist = source_distribution(retrieved_docs, top_n=5)
    dist_text = source_distribution_text(retrieved_docs, top_n=5)
    distinct_top_sources = len(dist)
    mixed_domain_evidence = underspecified_superlative and distinct_top_sources >= 2

    # ---- Deterministic hardening for comparison queries ----------------------
    if comparison_query and target_sources:
        balanced_docs = pick_balanced_docs(
            retrieved_docs,
            target_sources,
            per_source=2,
            max_total=GRADE_TOP_K,
        )

        covered_sources = {
            ((doc.get("metadata") or {}).get("source_file"))
            for doc in balanced_docs
            if ((doc.get("metadata") or {}).get("source_file"))
        }

        expected_coverage = min(2, len(target_sources))

        if len(covered_sources.intersection(set(target_sources))) >= expected_coverage:
            graded_docs = _grade_selected_docs(state, balanced_docs)

            return build_agent_update(
                state,
                agent_name="evidence_agent",
                next_action="answer_agent",
                decision_payload={
                    "decision": "USE_CURRENT_RESULTS",
                    "evidence_gap_reason": "",
                    "selected_doc_ids": [],
                    "target_sources": target_sources,
                    "mixed_domain_evidence": False,
                    "rationale": "Balanced evidence was found across the requested comparison targets.",
                },
                note="Balanced evidence was found across the requested comparison targets.",
                extra_updates={
                    "candidate_docs": balanced_docs,
                    "graded_docs": graded_docs,
                    "evidence_gap_reason": "",
                    "mixed_domain_evidence": False,
                    "evidence_source_distribution": dist,
                    "citations_pass": True,
                },
            )

        if crag_retries < MAX_REWRITE_ROUNDS:
            weak_signal_docs = balanced_docs[:WEAK_SIGNAL_TOP_K] if balanced_docs else retrieved_docs[:WEAK_SIGNAL_TOP_K]
            return build_agent_update(
                state,
                agent_name="evidence_agent",
                next_action="rewrite_agent",
                decision_payload={
                    "decision": "NEED_REWRITE",
                    "evidence_gap_reason": "missing_target_source_coverage",
                    "selected_doc_ids": [],
                    "target_sources": target_sources,
                    "mixed_domain_evidence": False,
                    "rationale": "Comparison question does not yet cover all requested target sources.",
                },
                note="Comparison question does not yet cover all requested target sources.",
                extra_updates={
                    "weak_signal_docs": weak_signal_docs,
                    "evidence_gap_reason": "missing_target_source_coverage",
                    "mixed_domain_evidence": False,
                    "evidence_source_distribution": dist,
                    "citations_pass": False,
                },
            )

        selected_docs = balanced_docs[: max(1, min(len(balanced_docs), GRADE_TOP_K))]
        graded_docs = _grade_selected_docs(state, selected_docs)

        return build_agent_update(
            state,
            agent_name="evidence_agent",
            next_action="answer_agent",
            decision_payload={
                "decision": "USE_CURRENT_RESULTS",
                "evidence_gap_reason": "missing_target_source_coverage",
                "selected_doc_ids": [],
                "target_sources": target_sources,
                "mixed_domain_evidence": False,
                "rationale": "Rewrite budget exhausted, so one-sided but relevant evidence is being passed forward cautiously.",
            },
            note="Rewrite budget exhausted, so one-sided but relevant evidence is being passed forward cautiously.",
            status="degraded",
            extra_updates={
                "candidate_docs": selected_docs,
                "graded_docs": graded_docs,
                "evidence_gap_reason": "missing_target_source_coverage",
                "mixed_domain_evidence": False,
                "evidence_source_distribution": dist,
                "citations_pass": True,
            },
        )

    # ---- Deterministic hardening for underspecified superlative questions ----
    if mixed_domain_evidence and crag_retries < MAX_REWRITE_ROUNDS:
        return build_agent_update(
            state,
            agent_name="evidence_agent",
            next_action="rewrite_agent",
            decision_payload={
                "decision": "NEED_REWRITE",
                "evidence_gap_reason": "underspecified_mixed_domain_retrieval",
                "selected_doc_ids": [],
                "mixed_domain_evidence": True,
                "rationale": f"Top evidence spans mixed sources/domains: {dist_text}",
            },
            note=f"Top evidence spans mixed sources/domains: {dist_text}",
            extra_updates={
                "weak_signal_docs": retrieved_docs[:WEAK_SIGNAL_TOP_K],
                "evidence_gap_reason": "underspecified_mixed_domain_retrieval",
                "mixed_domain_evidence": True,
                "evidence_source_distribution": dist,
                "citations_pass": False,
            },
        )

    current_docs_summary = _build_doc_summary(retrieved_docs, "CURRENT")

    prompt = f"""You are the Evidence Agent in a hierarchical multi-agent academic QA system.

Your job is to decide whether the current evidence is enough to move forward, whether a rewrite is needed,
or whether the system should stop.

User question:
{query}

Current search query:
{current_search_query}

Rewrite attempts so far:
{crag_retries}

Comparison query:
{comparison_query}

Underspecified superlative query:
{underspecified_superlative}

Target source files inferred from question:
{target_sources if target_sources else "none"}

Top source distribution:
{dist_text}

Current evidence snapshot:
{current_docs_summary}

Choose exactly one decision:
1. NEED_REWRITE
2. USE_CURRENT_RESULTS
3. STOP_NO_PROGRESS

Also provide:
- evidence_gap_reason: short label
- selected_doc_ids: which CURRENT docs are best to keep

Return ONLY valid JSON in this schema:
{{
  "decision": "NEED_REWRITE or USE_CURRENT_RESULTS or STOP_NO_PROGRESS",
  "evidence_gap_reason": "short label",
  "selected_doc_ids": [1, 2],
  "rationale": "short explanation"
}}

Guidelines:
- Prefer USE_CURRENT_RESULTS if there is any meaningful grounded evidence.
- For comparison questions, partial coverage is still enough to continue if one side is clearly present.
- If the top evidence spans mixed domains for an underspecified question, prefer NEED_REWRITE or cautious forward handoff.
- Do not answer the question.
"""

    print("\n[Multi-Agent] Evidence agent reasoning...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = extract_json_block(response.content, default={})

    decision = str(parsed.get("decision", "USE_CURRENT_RESULTS")).strip().upper()
    evidence_gap_reason = str(parsed.get("evidence_gap_reason", "unspecified")).strip()
    rationale = str(parsed.get("rationale", "No rationale provided.")).strip()
    selected_doc_ids = _normalize_doc_ids(parsed.get("selected_doc_ids", []), len(retrieved_docs))

    if decision == "NEED_REWRITE" and crag_retries < MAX_REWRITE_ROUNDS:
        weak_signal_docs = retrieved_docs[:WEAK_SIGNAL_TOP_K]
        return build_agent_update(
            state,
            agent_name="evidence_agent",
            next_action="rewrite_agent",
            decision_payload={
                "decision": decision,
                "evidence_gap_reason": evidence_gap_reason,
                "selected_doc_ids": selected_doc_ids,
                "target_sources": target_sources,
                "mixed_domain_evidence": mixed_domain_evidence,
                "rationale": rationale,
            },
            note=rationale,
            extra_updates={
                "weak_signal_docs": weak_signal_docs,
                "evidence_gap_reason": evidence_gap_reason,
                "mixed_domain_evidence": mixed_domain_evidence,
                "evidence_source_distribution": dist,
                "citations_pass": False,
            },
        )

    if decision == "NEED_REWRITE" and crag_retries >= MAX_REWRITE_ROUNDS:
        decision = "USE_CURRENT_RESULTS"
        rationale = f"{rationale} Rewrite budget exhausted, so current evidence is being passed forward."

    if decision == "USE_CURRENT_RESULTS":
        if not selected_doc_ids:
            selected_doc_ids = list(range(1, min(len(retrieved_docs), GRADE_TOP_K) + 1))

        selected_docs = [retrieved_docs[i - 1] for i in selected_doc_ids if 1 <= i <= len(retrieved_docs)]
        if not selected_docs:
            selected_docs = retrieved_docs[:GRADE_TOP_K]

        graded_docs = _grade_selected_docs(state, selected_docs)

        return build_agent_update(
            state,
            agent_name="evidence_agent",
            next_action="answer_agent",
            decision_payload={
                "decision": decision,
                "evidence_gap_reason": evidence_gap_reason,
                "selected_doc_ids": selected_doc_ids,
                "target_sources": target_sources,
                "mixed_domain_evidence": mixed_domain_evidence,
                "rationale": rationale,
            },
            note=rationale,
            extra_updates={
                "candidate_docs": selected_docs,
                "graded_docs": graded_docs,
                "evidence_gap_reason": evidence_gap_reason,
                "mixed_domain_evidence": mixed_domain_evidence,
                "evidence_source_distribution": dist,
                "citations_pass": True,
            },
        )

    if not retrieved_docs:
        stop_note = rationale or "No useful evidence available."
        return build_agent_update(
            state,
            agent_name="evidence_agent",
            next_action="finish",
            decision_payload={
                "decision": "STOP_NO_PROGRESS",
                "evidence_gap_reason": evidence_gap_reason,
                "selected_doc_ids": [],
                "target_sources": target_sources,
                "mixed_domain_evidence": mixed_domain_evidence,
                "rationale": stop_note,
            },
            note=stop_note,
            status="degraded",
            extra_updates={
                "evidence_gap_reason": evidence_gap_reason,
                "mixed_domain_evidence": mixed_domain_evidence,
                "evidence_source_distribution": dist,
                "citations_pass": False,
            },
        )

    selected_docs = retrieved_docs[: min(len(retrieved_docs), GRADE_TOP_K)]
    graded_docs = _grade_selected_docs(state, selected_docs)
    fallback_note = rationale or "Evidence was weak, but current evidence is still being passed forward cautiously."

    return build_agent_update(
        state,
        agent_name="evidence_agent",
        next_action="answer_agent",
        decision_payload={
            "decision": "USE_CURRENT_RESULTS",
            "evidence_gap_reason": evidence_gap_reason,
            "selected_doc_ids": list(range(1, len(selected_docs) + 1)),
            "target_sources": target_sources,
            "mixed_domain_evidence": mixed_domain_evidence,
            "rationale": fallback_note,
        },
        note=fallback_note,
        status="degraded",
        extra_updates={
            "candidate_docs": selected_docs,
            "graded_docs": graded_docs,
            "evidence_gap_reason": evidence_gap_reason,
            "mixed_domain_evidence": mixed_domain_evidence,
            "evidence_source_distribution": dist,
            "citations_pass": True,
        },
    )