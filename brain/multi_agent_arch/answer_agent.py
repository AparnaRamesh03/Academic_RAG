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
from query_targeting import is_comparison_query, is_underspecified_superlative_query

llm = build_groq_llm(temperature=0.0)


def _build_context_blocks(docs: List[Dict]) -> str:
    blocks = []

    for idx, doc in enumerate(docs[:8], start=1):
        metadata = doc.get("metadata", {}) or {}
        score = doc.get("rerank_score", doc.get("score", None))
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"

        block = (
            f"DOCUMENT [{idx}]\n"
            f"Source: {metadata.get('source_file', 'Unknown Source')}\n"
            f"Section: {metadata.get('section_header', 'Unknown Section')}\n"
            f"Page: {metadata.get('page_number', 'Unknown Page')}\n"
            f"Content Type: {metadata.get('content_type', 'text')}\n"
            f"Score: {score_text}\n"
            f"Text:\n{doc.get('text', '')}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def answer_agent(state: GraphState):
    query = state["original_query"]
    selected_docs = state.get("graded_docs", []) or []
    auditor_feedback = str(state.get("auditor_feedback", "") or "").strip()
    mixed_domain_evidence = bool(state.get("mixed_domain_evidence", False))
    source_dist = state.get("evidence_source_distribution", {}) or {}
    evidence_gap_reason = str(state.get("evidence_gap_reason", "") or "").strip()

    comparison_query = is_comparison_query(query)
    underspecified_superlative = is_underspecified_superlative_query(query)

    if not selected_docs:
        return build_agent_update(
            state,
            agent_name="answer_agent",
            next_action="verification_agent",
            decision_payload={
                "answer_strategy": "cautious_partial",
                "rationale": "No grounded evidence was available, so the answer must remain cautious.",
            },
            note="No grounded evidence was available, so the answer remains cautious.",
            status="degraded",
            extra_updates={
                "generation": "I could not find enough grounded evidence in the retrieved papers to answer this question.",
                "answer_strategy": "cautious_partial",
            },
        )

    context = _build_context_blocks(selected_docs)

    # Determine if a strategy should be forced based on evidence signals.
    # These cases previously returned hardcoded static strings; we now force the
    # right strategy label and let the LLM generate a grounded answer from the
    # actual retrieved docs instead.
    forced_strategy: str | None = None
    forced_status: str = "ok"

    if mixed_domain_evidence and underspecified_superlative:
        forced_strategy = "scoped_comparison"
        forced_status = "degraded"
    elif comparison_query and evidence_gap_reason == "missing_target_source_coverage":
        forced_strategy = "cautious_partial"
        forced_status = "degraded"

    if forced_strategy:
        answer_strategy = forced_strategy
        rationale = "Strategy forced by evidence signal (mixed-domain or partial-coverage)."
    else:
        strategy_prompt = f"""You are the Answer Agent in a hierarchical multi-agent academic QA system.

User question:
{query}

You will answer only from the evidence below.

Evidence snapshot:
{context}

Choose the best answer strategy label from:
- direct_fact
- comparison
- synthesis
- figure_grounded
- cautious_partial
- scoped_comparison

Return ONLY valid JSON:
{{
  "answer_strategy": "one label",
  "rationale": "short explanation"
}}
"""

        print("\n[Multi-Agent] Answer agent planning...")
        print(f"  -> Groq model: {GROQ_MODEL}")

        strategy_response = llm.invoke([HumanMessage(content=strategy_prompt)])
        strategy_parsed = extract_json_block(strategy_response.content, default={})

        answer_strategy = str(strategy_parsed.get("answer_strategy", "synthesis")).strip()
        rationale = str(strategy_parsed.get("rationale", "No rationale provided.")).strip()

    feedback_block = ""
    if auditor_feedback:
        feedback_block = f"""
Verification feedback from the previous answer:
{auditor_feedback}

Retry rule:
- Remove unsupported or incomplete claims.
- Be more question-specific.
- Do not keep broad background that is not required.
"""

    mixed_domain_block = ""
    if mixed_domain_evidence:
        mixed_domain_block = f"""
Mixed-domain evidence signal:
- The retrieved evidence spans multiple domains/sources: {source_dist}

Special rule:
- Do NOT declare a single global winner or say one architecture is best overall unless the evidence directly compares them in the same task/domain.
- Prefer a scoped answer that explains which architecture is efficient for which setting.
- Ground every claim in the documents above — do not use your training knowledge.
"""

    partial_coverage_block = ""
    if comparison_query and evidence_gap_reason == "missing_target_source_coverage":
        partial_coverage_block = """
Partial-coverage signal:
- The retrieved evidence only covers one side of the requested comparison.
- Summarise what the available evidence says about the covered side.
- Explicitly state that the evidence for the other side is not available in the retrieved documents.
- Do NOT invent or infer details about the missing side from your training knowledge.
"""

    generation_prompt = f"""You are the Answer Agent in a hierarchical multi-agent academic QA system.

Your answer strategy is:
{answer_strategy}

User question:
{query}

Evidence:
---
{context}
---
{feedback_block}
{mixed_domain_block}
{partial_coverage_block}

Rules:
1. Use ONLY the evidence above — do not use external or training knowledge.
2. Start with the direct answer immediately.
3. Do NOT use inline citations in the answer text.
4. Keep the answer concise and grounded.
5. For comparison questions, explicitly cover both sides if supported by evidence.
6. If the evidence is partial, answer cautiously and explicitly acknowledge the gap.
7. If evidence spans different domains, do not force a single overall winner.

Now write the final answer.
"""

    generation_response = llm.invoke([HumanMessage(content=generation_prompt)])
    answer = generation_response.content.strip()

    print(f"  -> answer_strategy: {answer_strategy}")

    return build_agent_update(
        state,
        agent_name="answer_agent",
        next_action="verification_agent",
        decision_payload={
            "answer_strategy": answer_strategy,
            "rationale": rationale,
        },
        note=rationale,
        status=forced_status if forced_strategy else "ok",
        extra_updates={
            "generation": answer,
            "answer_strategy": answer_strategy,
        },
    )