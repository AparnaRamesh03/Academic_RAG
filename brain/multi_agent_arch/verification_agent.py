from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

from langchain_core.messages import HumanMessage

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent
FINAL_COMBINED_DIR = BRAIN_DIR / "final_combined"

for path in [str(CURRENT_DIR), str(BRAIN_DIR), str(FINAL_COMBINED_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState
from config import MAX_AUDIT_RETRIES, MAX_REWRITE_ROUNDS
from claim_verifier import verify_claims
from agent_protocol import extract_json_block, build_agent_update
from query_targeting import is_comparison_query

llm = build_groq_llm(temperature=0.0)


def _summarize_claims(claims: List[Dict]) -> str:
    if not claims:
        return "No claim verification details available."

    blocks = []
    for claim in claims:
        blocks.append(
            f"Claim {claim.get('claim_id')} | "
            f"supported={claim.get('supported')} | "
            f"text={claim.get('claim_text', '')} | "
            f"feedback={claim.get('feedback', '')}"
        )
    return "\n".join(blocks)


def _answer_has_honest_limitation_language(answer: str) -> bool:
    text = answer.lower()
    patterns = [
        "the provided evidence does not",
        "the evidence does not",
        "cannot be fully made",
        "cannot be made based on the given information",
        "cannot be fully answered",
        "not fully supported",
        "not enough evidence",
        "insufficient evidence",
        "does not discuss",
        "missing evidence",
        "based on the available evidence",
        "from the available evidence",
        "the available evidence",
        "does not conclusively determine",
        "direct comparison cannot",
        "limiting a direct comparison",
    ]
    return any(p in text for p in patterns)


def _is_meta_limitation_claim(claim_text: str) -> bool:
    text = (claim_text or "").lower()
    markers = [
        "provided evidence does not",
        "the evidence does not",
        "cannot be made",
        "not directly compared",
        "does not conclusively determine",
        "limiting a direct comparison",
        "cannot be fully made",
    ]
    return any(m in text for m in markers)


def verification_agent(state: GraphState):
    query = state["original_query"]
    answer = str(state.get("generation", "") or "").strip()
    selected_docs = state.get("graded_docs", []) or []
    verify_retries = int(state.get("verify_retries", 0))
    crag_retries = int(state.get("crag_retries", 0))
    comparison_query = is_comparison_query(query)

    print("\n[Multi-Agent] Verification agent reasoning...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    if not answer or not selected_docs:
        return build_agent_update(
            state,
            agent_name="verification_agent",
            next_action="finish",
            decision_payload={
                "decision": "STOP",
                "verification_outcome": "empty_or_ungrounded",
                "rationale": "The answer is empty or lacks grounded evidence.",
            },
            note="The answer is empty or lacks grounded evidence.",
            status="degraded",
            extra_updates={
                "citations_pass": False,
                "auditor_feedback": "The answer is empty or lacks grounded evidence.",
                "claim_verification": [],
                "verification_outcome": "empty_or_ungrounded",
                "verify_retries": verify_retries + 1,
            },
        )

    verification = verify_claims(
        query=query,
        answer=answer,
        docs=selected_docs,
    )

    decision = verification.get("decision", "FAIL")
    overall_feedback = verification.get(
        "overall_feedback",
        "The answer contains unsupported or insufficiently grounded claims.",
    )
    claim_details = verification.get("claims", [])

    unsupported_claims = [c for c in claim_details if not c.get("supported", False)]
    supported_claims = [c for c in claim_details if c.get("supported", False)]

    if decision == "PASS" and not unsupported_claims:
        return build_agent_update(
            state,
            agent_name="verification_agent",
            next_action="finish",
            decision_payload={
                "decision": "PASS",
                "verification_outcome": "pass",
                "rationale": "All claims are supported.",
            },
            note="All claims are supported.",
            extra_updates={
                "citations_pass": True,
                "auditor_feedback": "",
                "claim_verification": claim_details,
                "verification_outcome": "pass",
            },
        )

    claims_summary = _summarize_claims(claim_details)
    honest_limitation = _answer_has_honest_limitation_language(answer)
    unsupported_meta_only = (
        bool(unsupported_claims)
        and all(_is_meta_limitation_claim(c.get("claim_text", "")) for c in unsupported_claims)
    )

    # Deterministic guardrail: grounded but incomplete answers should not be treated like hallucinations.
    if honest_limitation and supported_claims and (unsupported_meta_only or (comparison_query and len(unsupported_claims) <= 1)):
        return build_agent_update(
            state,
            agent_name="verification_agent",
            next_action="finish",
            decision_payload={
                "decision": "PARTIAL_PASS",
                "verification_outcome": "grounded_incomplete",
                "rationale": "The answer is grounded and honest about missing evidence.",
            },
            note="The answer is grounded and honest about missing evidence.",
            status="degraded",
            extra_updates={
                "citations_pass": True,
                "auditor_feedback": overall_feedback,
                "claim_verification": claim_details,
                "verification_outcome": "grounded_incomplete",
            },
        )

    prompt = f"""You are the Verification Agent in a hierarchical multi-agent academic QA system.

Your job is to decide whether the current answer should:
- PASS
- PARTIAL_PASS
- REGENERATE
- REQUEST_REWRITE
- STOP

User question:
{query}

Current answer:
{answer}

Current evidence-backed claim check:
Decision: {decision}
Overall feedback: {overall_feedback}

Claim details:
{claims_summary}

Extra signal:
- answer_contains_honest_limitation_language = {honest_limitation}
- unsupported_claim_count = {len(unsupported_claims)}
- supported_claim_count = {len(supported_claims)}
- verify_retries = {verify_retries}
- rewrite_retries = {crag_retries}
- comparison_query = {comparison_query}

Decision meanings:
- PASS: all important claims are supported
- PARTIAL_PASS: answer is grounded and honest, explicitly states missing evidence or limitations, and should be accepted as a cautious partial answer
- REGENERATE: current evidence is mostly enough, but the wording/claim structure should be improved
- REQUEST_REWRITE: retrieval coverage is missing and another retrieval-focused rewrite may help
- STOP: repeated retries are unlikely to improve the result

Important:
- PARTIAL_PASS is allowed when the answer is honest about what the evidence does and does not support.
- Do NOT treat an honest limitation statement as hallucination.
- Prefer PARTIAL_PASS over STOP when the answer is grounded but incomplete.

Return ONLY valid JSON:
{{
  "decision": "PASS or PARTIAL_PASS or REGENERATE or REQUEST_REWRITE or STOP",
  "verification_outcome": "pass or grounded_incomplete or unsupported or needs_revision",
  "rationale": "short explanation"
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = extract_json_block(response.content, default={})

    next_decision = str(parsed.get("decision", "REGENERATE")).strip().upper()
    verification_outcome = str(parsed.get("verification_outcome", "needs_revision")).strip() or "needs_revision"
    rationale = str(parsed.get("rationale", "No rationale provided.")).strip()

    if next_decision not in {"PASS", "PARTIAL_PASS", "REGENERATE", "REQUEST_REWRITE", "STOP"}:
        next_decision = "REGENERATE"
        verification_outcome = "needs_revision"

    if (
        next_decision in {"STOP", "REGENERATE"}
        and honest_limitation
        and len(supported_claims) >= 1
        and len(unsupported_claims) <= 1
    ):
        next_decision = "PARTIAL_PASS"
        verification_outcome = "grounded_incomplete"
        rationale = (
            "The answer is grounded and explicitly states the missing evidence, "
            "so it should be accepted as an honest partial answer."
        )

    if next_decision == "PASS":
        return build_agent_update(
            state,
            agent_name="verification_agent",
            next_action="finish",
            decision_payload={
                "decision": "PASS",
                "verification_outcome": "pass",
                "rationale": rationale or "All claims are supported.",
            },
            note=rationale or "All claims are supported.",
            extra_updates={
                "citations_pass": True,
                "auditor_feedback": "",
                "claim_verification": claim_details,
                "verification_outcome": "pass",
            },
        )

    if next_decision == "PARTIAL_PASS":
        return build_agent_update(
            state,
            agent_name="verification_agent",
            next_action="finish",
            decision_payload={
                "decision": "PARTIAL_PASS",
                "verification_outcome": verification_outcome,
                "rationale": rationale,
            },
            note=rationale,
            status="degraded",
            extra_updates={
                "citations_pass": True,
                "auditor_feedback": overall_feedback,
                "claim_verification": claim_details,
                "verification_outcome": verification_outcome,
            },
        )

    if next_decision == "REQUEST_REWRITE" and crag_retries < MAX_REWRITE_ROUNDS:
        return build_agent_update(
            state,
            agent_name="verification_agent",
            next_action="rewrite_agent",
            decision_payload={
                "decision": "REQUEST_REWRITE",
                "verification_outcome": verification_outcome,
                "rationale": rationale,
            },
            note=rationale,
            status="degraded",
            extra_updates={
                "citations_pass": False,
                "auditor_feedback": overall_feedback,
                "claim_verification": claim_details,
                "verification_outcome": verification_outcome,
                "verify_retries": verify_retries + 1,
            },
        )

    if next_decision == "REGENERATE" and verify_retries < MAX_AUDIT_RETRIES:
        return build_agent_update(
            state,
            agent_name="verification_agent",
            next_action="answer_agent",
            decision_payload={
                "decision": "REGENERATE",
                "verification_outcome": verification_outcome,
                "rationale": rationale,
            },
            note=rationale,
            status="degraded",
            extra_updates={
                "citations_pass": False,
                "auditor_feedback": overall_feedback,
                "claim_verification": claim_details,
                "verification_outcome": verification_outcome,
                "verify_retries": verify_retries + 1,
            },
        )

    return build_agent_update(
        state,
        agent_name="verification_agent",
        next_action="finish",
        decision_payload={
            "decision": "STOP",
            "verification_outcome": "unsupported" if verification_outcome == "needs_revision" else verification_outcome,
            "rationale": rationale,
        },
        note=rationale,
        status="degraded",
        extra_updates={
            "citations_pass": False,
            "auditor_feedback": overall_feedback,
            "claim_verification": claim_details,
            "verification_outcome": "unsupported" if verification_outcome == "needs_revision" else verification_outcome,
            "verify_retries": verify_retries + 1,
        },
    )