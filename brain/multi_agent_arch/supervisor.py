from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent

for path in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from state_shared import GraphState
from config import (
    MAX_STEPS,
    MAX_AUDIT_RETRIES,
    MIN_CONFIDENCE_TO_STOP,
    MAX_REWRITE_ROUNDS,
    RL_ENABLED,
    RL_EPSILON,
    RL_ALPHA,
    RL_POLICY_PATH,
)
from agent_protocol import normalize_next_action, append_agent_trace
from rl_policy import RLPolicy
from rl_state_features import extract_state_key

# ── Module-level policy singleton ─────────────────────────────────────────────
# Loaded once at import time; persists across requests within the same process.
_policy = RLPolicy(
    policy_path=RL_POLICY_PATH,
    epsilon=RL_EPSILON,
    alpha=RL_ALPHA,
) if RL_ENABLED else None

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

    # Structural loop: same 3-step sequence repeated back-to-back.
    if len(history) >= 6 and history[-6:-3] == history[-3:]:
        return True

    # Semantic loop: stuck on the same evidence gap after exhausting the rewrite budget.
    # If we've called evidence_agent at least twice and still have the same gap reason
    # with no rewrites left, further iterations won't make progress.
    crag_retries = int(state.get("crag_retries", 0))
    evidence_gap_reason = str(state.get("evidence_gap_reason", "") or "")
    evidence_calls = history.count("evidence_agent")

    stuck_gap_reasons = {
        "missing_target_source_coverage",
        "underspecified_mixed_domain_retrieval",
    }

    if (
        evidence_calls >= 2
        and crag_retries >= MAX_REWRITE_ROUNDS
        and evidence_gap_reason in stuck_gap_reasons
    ):
        return True

    return False


def estimate_confidence(state: GraphState) -> float:
    claim_verification = state.get("claim_verification", []) or []
    graded_docs = state.get("graded_docs", []) or []
    citations_pass = bool(state.get("citations_pass", False))
    crag_retries = int(state.get("crag_retries", 0))
    verify_retries = int(state.get("verify_retries", 0))

    # Best signal: claim-level verification ratio.
    if claim_verification:
        total = len(claim_verification)
        supported = sum(1 for c in claim_verification if bool(c.get("supported", False)))
        base = supported / total if total else 0.0

    # Second signal: citations passed + graded docs exist — use rerank scores to
    # interpolate between 0.55 and 0.82 instead of a flat 0.75 bucket.
    elif citations_pass and graded_docs:
        scores = [
            float(doc.get("rerank_score") or doc.get("score") or 0.0)
            for doc in graded_docs
            if isinstance(doc.get("rerank_score") or doc.get("score"), (int, float))
        ]
        if scores:
            mean_score = sum(scores) / len(scores)
            # Clamp mean_score contribution to [0, 0.27] so total stays ≤ 0.82.
            base = 0.55 + min(0.27, max(0.0, mean_score) * 0.27)
        else:
            base = 0.65

    elif graded_docs:
        base = 0.45

    elif state.get("candidate_docs"):
        base = 0.30

    elif state.get("retrieved_docs"):
        base = 0.15

    else:
        base = 0.0

    # Penalise for effort: each rewrite and audit retry reduces confidence slightly.
    penalty = 0.05 * crag_retries + 0.05 * verify_retries
    return round(max(0.0, min(1.0, base - penalty)), 4)


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
    elif (
        state.get("citations_pass", False)
        and confidence >= MIN_CONFIDENCE_TO_STOP
        and bool(str(state.get("generation", "") or "").strip())
    ):
        # Only stop when an answer has actually been generated — evidence_agent
        # also sets citations_pass=True (meaning "good evidence found, proceed"),
        # and we must not confuse that with a fully verified answer.
        done = True
        stop_reason = "grounded_answer_ready"
    elif state.get("generation") and state.get("verify_retries", 0) > MAX_AUDIT_RETRIES:
        done = True
        stop_reason = "audit_retry_limit_reached"

    status_note = "Evaluating next agent."
    status_value = "ok" if not done else "degraded"

    # ── RL routing decision ───────────────────────────────────────────────────
    # Make the decision *before* returning so we can write it into state AND
    # record the transition for end-of-episode learning.
    rl_next_action: str | None = None
    rl_transitions = list(state.get("rl_transitions", []) or [])

    if not done and _policy is not None:
        import os
        is_training = os.getenv("RL_TRAINING_MODE") == "1"
        state_key = extract_state_key({**state, "confidence": confidence})
        rl_next_action = _policy.act(state_key, explore=is_training)
        if rl_next_action is not None and rl_next_action in VALID_NEXT_AGENTS:
            print(f"[RL] Override: {rl_next_action}  (key={state_key})")
    else:
        state_key = None
        rl_next_action = None

    supervisor_update = _update_supervisor_status(
        state,
        status=status_value,
        next_action="finish" if done else "finish",
        note=status_note if not done else stop_reason,
    )

    result = {
        "confidence": confidence,
        "done": done,
        "stop_reason": stop_reason,
        "rl_transitions": rl_transitions,
        # Pending key consumed by _run_agent to record the (state, action) transition.
        "rl_pending_state_key": str(state_key) if state_key is not None else "",
        **supervisor_update,
    }

    # Let RL override the routing recommendation when it has a preference.
    if rl_next_action is not None and rl_next_action in VALID_NEXT_AGENTS:
        result["next_action_recommendation"] = rl_next_action

    return result


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