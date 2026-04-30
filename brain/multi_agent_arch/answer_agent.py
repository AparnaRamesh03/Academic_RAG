from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent

for path in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState
from agent_protocol import extract_json_block, build_agent_update

llm = build_groq_llm(temperature=0.0)


def _build_context_blocks(docs: List[Dict[str, Any]]) -> str:
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

Return ONLY valid JSON:
{{
  "answer_strategy": "one label",
  "rationale": "short explanation"
}}
"""

    print("\n[Multi-Agent] Answer agent planning...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    strategy_response = llm.invoke([HumanMessage(content=strategy_prompt)])
    strategy_parsed = extract_json_block(strategy_response.content)

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

Rules:
1. Use ONLY the evidence above.
2. Start with the direct answer immediately.
3. Do NOT use inline citations in the answer text.
4. Keep the answer concise and grounded.
5. For comparison questions, explicitly cover both sides if supported.
6. If the evidence is partial, answer cautiously and do not invent missing parts.

Now write the final answer.
"""

    generation_response = llm.invoke([HumanMessage(content=generation_prompt)])
    answer = generation_response.content.strip()

    return build_agent_update(
        state,
        agent_name="answer_agent",
        next_action="verification_agent",
        decision_payload={
            "answer_strategy": answer_strategy,
            "rationale": rationale,
        },
        note=rationale,
        extra_updates={
            "generation": answer,
            "answer_strategy": answer_strategy,
        },
    )