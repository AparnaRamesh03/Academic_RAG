from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent

for path in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from state_shared import GraphState
from config import MAX_STEPS, MAX_AUDIT_RETRIES, MIN_CONFIDENCE_TO_STOP, MAX_REWRITE_ROUNDS
from agent_protocol import normalize_next_action, append_agent_trace

VALID_NEXT_AGENTS = {
    "retriever_agent",
    "rewrite_agent",
    "evidence_agent",
    "answer_agent",
    "verification_agent",
    "finish",
}


def _update_supervisor_status(
    state: GraphState,
    *,
    status: str,
    next_action: str,
    note: str,
):
    agent_status = dict(state.get("agent_status", {}) or {})
    agent_notes = dict(state.get("agent_notes", {}) or {})

    agent_status["supervisor"] = {
        "status": status,
        "next_action": normalize_next_action(next_action, fallback="finish"),
    }
    agent_notes["supervisor"] = note

    return {
        "active_agent": "supervisor",
        "agent_status": agent_status,
        "agent_notes": agent_notes,
    }


def detect_agent_loop(state: GraphState) -> bool:
    history = list(state.get("action_history", []) or [])

    if len(history) >= 6 and history[-6:-3] == history[-3:]:
        return True

    return False


def estimate_confidence(state: GraphState) -> float:
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
    verification_outcome = str(state.get("verification_outcome", "") or "").strip()

    if state.get("citations_pass", False) and verification_outcome in {"pass", "grounded_incomplete"}:
        return "grounded_answer_ready"

    if state.get("citations_pass", False) and (state.get("claim_verification") or state.get("generation")):
        return "grounded_answer_ready"

    if state.get("generation") and state.get("verify_retries", 0) > MAX_AUDIT_RETRIES:
        return "audit_retry_limit_reached"

    if detect_agent_loop(state):
        return "agent_loop_detected"

    if state.get("step_count", 0) >= MAX_STEPS:
        return "max_steps_reached"

    if not state.get("retrieved_docs") and state.get("step_count", 0) > 0:
        return "no_retrieval_progress"

    return "supervisor_stopped"


def supervisor_step(state: GraphState):
    step_count = int(state.get("step_count", 0))
    confidence = estimate_confidence(state)

    done = False
    stop_reason = state.get("stop_reason", "")

    if detect_agent_loop(state):
        done = True
        stop_reason = "agent_loop_detected"
    elif step_count >= MAX_STEPS:
        done = True
        stop_reason = "max_steps_reached"
    elif state.get("citations_pass", False) and confidence >= MIN_CONFIDENCE_TO_STOP:
        done = True
        stop_reason = "grounded_answer_ready"
    elif state.get("generation") and state.get("verify_retries", 0) > MAX_AUDIT_RETRIES:
        done = True
        stop_reason = "audit_retry_limit_reached"

    status_note = "Evaluating next agent."
    status_value = "ok" if not done else "degraded"

    return {
        "confidence": confidence,
        "done": done,
        "stop_reason": stop_reason,
        **_update_supervisor_status(
            state,
            status=status_value,
            next_action="finish" if done else "finish",
            note=status_note if not done else stop_reason,
        ),
    }


def choose_next_agent(state: GraphState) -> str:
    if state.get("done", False):
        return "finish"

    if detect_agent_loop(state):
        return "finish"

    if state.get("step_count", 0) >= MAX_STEPS:
        return "finish"

    last_action = str(state.get("last_action", "") or "")
    agent_status = dict(state.get("agent_status", {}) or {})

    if last_action and last_action in agent_status:
        routed = normalize_next_action(agent_status[last_action].get("next_action"), fallback="")
        if routed in VALID_NEXT_AGENTS:
            return routed

    recommended = normalize_next_action(state.get("next_action_recommendation"), fallback="")
    if recommended in VALID_NEXT_AGENTS:
        return recommended

    retrieved_docs = state.get("retrieved_docs", []) or []
    graded_docs = state.get("graded_docs", []) or []
    generation = (state.get("generation", "") or "").strip()

    if not last_action and not retrieved_docs and not generation:
        return "retriever_agent"

    if last_action == "retriever_agent":
        return "evidence_agent"

    if last_action == "rewrite_agent":
        return "retriever_agent"

    if last_action == "evidence_agent":
        if graded_docs:
            return "answer_agent"
        if state.get("crag_retries", 0) < MAX_REWRITE_ROUNDS:
            return "rewrite_agent"
        return "finish"

    if last_action == "answer_agent":
        return "verification_agent"

    if last_action == "verification_agent":
        if state.get("citations_pass", False):
            return "finish"
        if state.get("verify_retries", 0) <= MAX_AUDIT_RETRIES:
            return "answer_agent"
        return "finish"

    if graded_docs and not generation:
        return "answer_agent"

    if retrieved_docs and not graded_docs:
        return "evidence_agent"

    if generation:
        return "verification_agent"

    return "finish"


def finish_step(state: GraphState):
    stop_reason = build_stop_reason(state)
    confidence = estimate_confidence(state)

    agent_status = dict(state.get("agent_status", {}) or {})
    agent_notes = dict(state.get("agent_notes", {}) or {})
    agent_status["supervisor"] = {
        "status": "ok" if state.get("citations_pass", False) else "degraded",
        "next_action": "finish",
    }
    agent_notes["supervisor"] = stop_reason

    trace = append_agent_trace(
        state,
        agent_name="supervisor",
        next_action="finish",
        status="ok" if state.get("citations_pass", False) else "degraded",
        summary=stop_reason,
    )

    return {
        "done": True,
        "stop_reason": stop_reason,
        "confidence": confidence,
        "active_agent": "supervisor",
        "agent_status": agent_status,
        "agent_notes": agent_notes,
        "agent_trace": trace,
        "next_action_recommendation": "finish",
    }