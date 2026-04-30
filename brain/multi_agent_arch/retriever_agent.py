from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent
FINAL_COMBINED_DIR = BRAIN_DIR / "final_arch"

for path in [str(CURRENT_DIR), str(BRAIN_DIR), str(FINAL_COMBINED_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState
from retriever_shared import retrieve_and_store
from reranker_shared import rerank_retrieved_docs
from config import RERANK_INPUT_TOP_K, RERANK_OUTPUT_TOP_K, MAX_REWRITE_ROUNDS
from agent_protocol import extract_json_block, build_agent_update

llm = build_groq_llm(temperature=0.0)


def _summarize_retrieval(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "No retrieval results available yet."

    blocks = []
    for idx, doc in enumerate(docs[:5], start=1):
        meta = doc.get("metadata", {}) or {}
        blocks.append(
            f"[{idx}] "
            f"source={meta.get('source_file', 'Unknown')} | "
            f"section={meta.get('section_header', 'Unknown')} | "
            f"content_type={meta.get('content_type', 'text')} | "
            f"score={doc.get('score', 'n/a')} | "
            f"rerank_score={doc.get('rerank_score', 'n/a')}"
        )
    return "\n".join(blocks)


def retriever_agent(state: GraphState):
    query = state["original_query"]
    search_query = state.get("search_query", query)
    existing_docs = state.get("retrieved_docs", []) or []
    crag_retries = int(state.get("crag_retries", 0))
    last_action = str(state.get("last_action", ""))

    # Force one real retrieval immediately after rewrite.
    if last_action == "rewrite_agent":
        print("\n[Multi-Agent] Retriever agent forcing retrieval after rewrite...")

        retrieval_updates = retrieve_and_store(state)
        merged = {**state, **retrieval_updates}

        rerank_updates = rerank_retrieved_docs(
            merged,
            input_top_k=RERANK_INPUT_TOP_K,
            output_top_k=RERANK_OUTPUT_TOP_K,
        )

        return build_agent_update(
            state,
            agent_name="retriever_agent",
            next_action="evidence_agent",
            decision_payload={
                "decision": "RETRIEVE_AND_RERANK",
                "retrieval_strategy": "forced_after_rewrite",
                "rationale": "A rewritten query was just produced, so retrieval must run now.",
            },
            note="Forced retrieval after rewrite.",
            extra_updates={
                **retrieval_updates,
                **rerank_updates,
                "retrieval_strategy": "forced_after_rewrite",
            },
        )

    retrieval_snapshot = _summarize_retrieval(existing_docs)

    prompt = f"""You are the Retriever Agent in a multi-agent academic QA system.

Your job is to decide the best retrieval-oriented next move.

User question:
{query}

Current search query:
{search_query}

Rewrite attempts so far:
{crag_retries}

Current retrieval snapshot:
{retrieval_snapshot}

Choose exactly one decision:
1. RETRIEVE_AND_RERANK
2. REQUEST_REWRITE
3. USE_EXISTING_RESULTS

Guidelines:
- For cross-paper comparisons, prefer broader coverage of both papers.
- For figure/table grounded questions, pay attention to whether structured chunks may be needed.
- If no retrieval results exist yet, choose RETRIEVE_AND_RERANK.
- If rewrite attempts have already reached the limit, do NOT choose REQUEST_REWRITE.
- Do not answer the user question.

Return ONLY valid JSON:
{{
  "decision": "RETRIEVE_AND_RERANK or REQUEST_REWRITE or USE_EXISTING_RESULTS",
  "retrieval_strategy": "short label",
  "rationale": "short explanation"
}}
"""

    print("\n[Multi-Agent] Retriever agent reasoning...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = extract_json_block(response.content)

    decision = str(parsed.get("decision", "RETRIEVE_AND_RERANK")).strip().upper()
    retrieval_strategy = str(parsed.get("retrieval_strategy", "default")).strip()
    rationale = str(parsed.get("rationale", "No rationale provided.")).strip()

    if crag_retries >= MAX_REWRITE_ROUNDS and decision == "REQUEST_REWRITE":
        decision = "USE_EXISTING_RESULTS" if existing_docs else "RETRIEVE_AND_RERANK"
        rationale = f"{rationale} Rewrite budget already exhausted, so rewrite request was overridden."

    if decision == "REQUEST_REWRITE":
        return build_agent_update(
            state,
            agent_name="retriever_agent",
            next_action="rewrite_agent",
            decision_payload={
                "decision": decision,
                "retrieval_strategy": retrieval_strategy,
                "rationale": rationale,
            },
            note=rationale,
            status="degraded",
            extra_updates={
                "retrieval_strategy": retrieval_strategy,
            },
        )

    if decision == "USE_EXISTING_RESULTS" and existing_docs:
        return build_agent_update(
            state,
            agent_name="retriever_agent",
            next_action="evidence_agent",
            decision_payload={
                "decision": decision,
                "retrieval_strategy": retrieval_strategy,
                "rationale": rationale,
            },
            note=rationale,
            extra_updates={
                "retrieval_strategy": retrieval_strategy,
            },
        )

    retrieval_updates = retrieve_and_store(state)
    merged = {**state, **retrieval_updates}

    rerank_updates = rerank_retrieved_docs(
        merged,
        input_top_k=RERANK_INPUT_TOP_K,
        output_top_k=RERANK_OUTPUT_TOP_K,
    )

    return build_agent_update(
        state,
        agent_name="retriever_agent",
        next_action="evidence_agent",
        decision_payload={
            "decision": "RETRIEVE_AND_RERANK",
            "retrieval_strategy": retrieval_strategy,
            "rationale": rationale,
        },
        note=rationale,
        extra_updates={
            **retrieval_updates,
            **rerank_updates,
            "retrieval_strategy": retrieval_strategy,
        },
    )