from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

from langchain_core.messages import HumanMessage

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent

for path in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState
from agent_protocol import extract_json_block, build_agent_update
from query_targeting import (
    infer_target_entities,
    infer_target_sources,
    is_comparison_query,
    enforce_target_entities,
)

llm = build_groq_llm(temperature=0.0)


def _summarize_weak_docs(docs: List[Dict]) -> str:
    if not docs:
        return "No weak or partial evidence available."

    blocks = []
    for idx, doc in enumerate(docs[:3], start=1):
        meta = doc.get("metadata", {}) or {}
        blocks.append(
            f"[{idx}] "
            f"source={meta.get('source_file', 'Unknown')} | "
            f"section={meta.get('section_header', 'Unknown')} | "
            f"text={str(doc.get('text', ''))[:220]}"
        )
    return "\n".join(blocks)


def rewrite_agent(state: GraphState):
    original_query = state["original_query"]
    current_search_query = state.get("search_query", original_query)
    weak_docs = state.get("weak_signal_docs", []) or []
    retries = int(state.get("crag_retries", 0))

    weak_context = _summarize_weak_docs(weak_docs)
    target_sources = infer_target_sources(original_query)
    target_entities = infer_target_entities(original_query)
    comparison_query = is_comparison_query(original_query)

    prompt = f"""You are the Rewrite Agent in a multi-agent academic QA system.

Your job is to improve retrieval when the current query appears weak or incomplete.

Original user question:
{original_query}

Current search query:
{current_search_query}

Weak / partial evidence:
{weak_context}

Rewrite attempts so far:
{retries}

Target entities explicitly inferred from the question:
{target_entities if target_entities else "none"}

Target source files inferred from the question:
{target_sources if target_sources else "none"}

Choose a rewrite type:
- comparison_entity_focused
- entity_focused
- keyword_dense
- figure_table_focused
- synthesis_focused
- none

Return ONLY valid JSON:
{{
  "decision": "REWRITE or KEEP_CURRENT",
  "rewrite_type": "one label from above",
  "rewritten_query": "short retrieval query",
  "rationale": "short explanation"
}}

Rules:
- Preserve user intent exactly.
- Make the query more retrieval-friendly.
- Keep the rewritten query short and keyword-rich.
- If the question is a comparison, preserve the compared entities explicitly.
- If explicit paper/model names appear in the original query, keep them in the rewritten query.
- Do not answer the question.
"""

    print("\n[Multi-Agent] Rewrite agent reasoning...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = extract_json_block(response.content, default={})

    decision = str(parsed.get("decision", "KEEP_CURRENT")).strip().upper()
    rewrite_type = str(parsed.get("rewrite_type", "none")).strip()
    rewritten_query = " ".join(str(parsed.get("rewritten_query", current_search_query)).split())
    rationale = str(parsed.get("rationale", "No rationale provided.")).strip()

    if not rewritten_query:
        rewritten_query = current_search_query

    if comparison_query and target_entities:
        rewritten_query = enforce_target_entities(rewritten_query, target_entities)
        if decision != "REWRITE":
            decision = "REWRITE"
        if rewrite_type == "none":
            rewrite_type = "comparison_entity_focused"

    new_query = rewritten_query if decision == "REWRITE" else current_search_query

    return build_agent_update(
        state,
        agent_name="rewrite_agent",
        next_action="retriever_agent",
        decision_payload={
            "decision": decision,
            "rewrite_type": rewrite_type,
            "rewritten_query": new_query,
            "target_entities": target_entities,
            "target_sources": target_sources,
            "rationale": rationale,
        },
        note=rationale,
        extra_updates={
            "search_query": new_query,
            "crag_retries": retries + 1,
            "retrieved_docs": [],
            "candidate_docs": [],
            "graded_docs": [],
            "weak_signal_docs": [],
            "generation": "",
            "citations_pass": False,
            "rewrite_type": rewrite_type,
        },
    )