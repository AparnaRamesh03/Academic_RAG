from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from state_shared import GraphState
from config import MAX_STEPS, MAX_REWRITE_ROUNDS, MAX_AUDIT_RETRIES, MIN_CONFIDENCE_TO_STOP


def estimate_confidence(state: GraphState) -> float:
    """
    Very simple first-pass confidence estimate.

    This is intentionally heuristic for Phase 2.
    Later phases can replace this with stronger logic or learned control.
    """
    claim_verification = state.get("claim_verification", []) or []
    graded_docs = state.get("graded_docs", []) or []
    citations_pass = bool(state.get("citations_pass", False))

    if claim_verification:
        total = len(claim_verification)
        supported = sum(1 for c in claim_verification if bool(c.get("supported", False)))
        return supported / total if total else 0.0

    if citations_pass and graded_docs:
        return 0.75

    if graded_docs:
        return 0.55

    if state.get("candidate_docs"):
        return 0.35

    if state.get("retrieved_docs"):
        return 0.20

    return 0.0


def build_stop_reason(state: GraphState) -> str:
    if state.get("citations_pass", False) and (state.get("claim_verification") or state.get("generation")):
        return "grounded_answer_ready"

    if state.get("generation") and state.get("verify_retries", 0) > MAX_AUDIT_RETRIES:
        return "audit_retry_limit_reached"

    if state.get("step_count", 0) >= MAX_STEPS:
        return "max_steps_reached"

    if not state.get("retrieved_docs") and state.get("step_count", 0) > 0:
        return "no_retrieval_progress"

    return "supervisor_stopped"


def supervisor_step(state: GraphState):
    """
    Supervisor bookkeeping node.
    It does not execute worker logic. It only updates supervisor-facing fields.
    """
    step_count = int(state.get("step_count", 0))
    confidence = estimate_confidence(state)

    done = False
    stop_reason = state.get("stop_reason", "")

    if step_count >= MAX_STEPS:
        done = True
        stop_reason = "max_steps_reached"
    elif state.get("citations_pass", False) and confidence >= MIN_CONFIDENCE_TO_STOP:
        done = True
        stop_reason = "grounded_answer_ready"
    elif state.get("generation") and state.get("verify_retries", 0) > MAX_AUDIT_RETRIES:
        done = True
        stop_reason = "audit_retry_limit_reached"

    return {
        "confidence": confidence,
        "done": done,
        "stop_reason": stop_reason,
    }


def choose_next_action(state: GraphState) -> str:
    """
    Decide which worker node should act next.
    """
    if state.get("done", False):
        return "finish"

    if state.get("step_count", 0) >= MAX_STEPS:
        return "finish"

    last_action = state.get("last_action", "")
    retrieved_docs = state.get("retrieved_docs", []) or []
    candidate_docs = state.get("candidate_docs", []) or []
    graded_docs = state.get("graded_docs", []) or []
    generation = (state.get("generation", "") or "").strip()

    # Start of episode
    if not last_action and not retrieved_docs and not generation:
        return "retrieve_original"

    # Original retrieval path
    if last_action == "retrieve_original":
        return "rerank_original"

    if last_action == "rerank_original":
        return "evaluate_retrieval"

    if last_action == "evaluate_retrieval":
        if state.get("citations_pass", False):
            return "grade_documents"

        if state.get("crag_retries", 0) < MAX_REWRITE_ROUNDS:
            return "rewrite_query"

        return "grade_documents"

    # Rewrite path
    if last_action == "rewrite_query":
        return "retrieve_rewritten"

    if last_action == "retrieve_rewritten":
        return "rerank_rewritten"

    if last_action == "rerank_rewritten":
        return "select_best_context"

    if last_action == "select_best_context":
        return "grade_documents"

    # Final evidence -> answer -> audit
    if last_action == "grade_documents":
        return "generate"

    if last_action == "generate":
        return "audit_answer"

    if last_action == "audit_answer":
        if state.get("citations_pass", False):
            return "finish"

        if state.get("verify_retries", 0) <= MAX_AUDIT_RETRIES:
            return "generate"

        return "finish"

    # Fallback rules
    if graded_docs and not generation:
        return "generate"

    if candidate_docs and not graded_docs:
        return "grade_documents"

    if generation:
        return "audit_answer"

    if retrieved_docs and not candidate_docs:
        return "evaluate_retrieval"

    return "finish"


def finish_step(state: GraphState):
    """
    Final state update before END.
    """
    return {
        "done": True,
        "stop_reason": build_stop_reason(state),
        "confidence": estimate_confidence(state),
    }