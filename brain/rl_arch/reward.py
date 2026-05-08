from __future__ import annotations

from typing import Dict, Tuple

try:
    from .config import (
        REWARD_GROUNDED_PASS,
        REWARD_GROUNDED_INCOMPLETE,
        REWARD_CITATIONS_PASS_BONUS,
        PENALTY_UNSUPPORTED,
        PENALTY_NEEDS_REVISION,
        PENALTY_EMPTY_ANSWER,
        PENALTY_PER_STEP,
        PENALTY_REWRITE_ACTION,
        PENALTY_VERIFY_ACTION,
        BONUS_EARLY_STOP_GOOD,
        PENALTY_BAD_STOP,
        PENALTY_MAX_STEP_TERMINATION,
        PENALTY_AUDIT_RETRY_LIMIT,
        REWARD_EVIDENCE_IMPROVEMENT,
        EVIDENCE_IMPROVEMENT_DELTA,
    )
except ImportError:
    from config import (
        REWARD_GROUNDED_PASS,
        REWARD_GROUNDED_INCOMPLETE,
        REWARD_CITATIONS_PASS_BONUS,
        PENALTY_UNSUPPORTED,
        PENALTY_NEEDS_REVISION,
        PENALTY_EMPTY_ANSWER,
        PENALTY_PER_STEP,
        PENALTY_REWRITE_ACTION,
        PENALTY_VERIFY_ACTION,
        BONUS_EARLY_STOP_GOOD,
        PENALTY_BAD_STOP,
        PENALTY_MAX_STEP_TERMINATION,
        PENALTY_AUDIT_RETRY_LIMIT,
        REWARD_EVIDENCE_IMPROVEMENT,
        EVIDENCE_IMPROVEMENT_DELTA,
    )


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _top_score(state: Dict, key: str) -> float:
    docs = state.get("retrieved_docs", []) or []
    if not docs:
        return 0.0

    vals = []
    for doc in docs:
        raw = doc.get(key, None)
        if raw is None:
            raw = (doc.get("metadata", {}) or {}).get(key, None)
        vals.append(_safe_float(raw, 0.0))

    return max(vals) if vals else 0.0


def _doc_count(state: Dict, field: str) -> int:
    docs = state.get(field, []) or []
    return len(docs)


def _has_answer(state: Dict) -> bool:
    generation = str(state.get("generation", "") or "").strip()
    return len(generation) > 0


def _verification_outcome(state: Dict) -> str:
    return str(state.get("verification_outcome", "") or "").strip().lower()


def _stop_reason(state: Dict) -> str:
    return str(state.get("stop_reason", "") or "").strip().lower()


def _evidence_quality_score(state: Dict) -> float:
    """
    Compact heuristic used only for reward shaping.
    This is not the RL state itself.
    """
    top_retrieval = _top_score(state, "score")
    top_rerank = _top_score(state, "rerank_score")
    graded_count = _doc_count(state, "graded_docs")
    candidate_count = _doc_count(state, "candidate_docs")
    mixed_domain = 1.0 if bool(state.get("mixed_domain_evidence", False)) else 0.0

    score = (
        0.35 * top_retrieval
        + 0.35 * top_rerank
        + 0.15 * min(graded_count, 5) / 5.0
        + 0.15 * min(candidate_count, 5) / 5.0
        - 0.10 * mixed_domain
    )
    return score


def get_terminal_reward(final_state: Dict) -> Tuple[float, Dict]:
    """
    Reward applied at the end of an episode.
    """
    reward = 0.0
    details = {}

    outcome = _verification_outcome(final_state)
    stop_reason = _stop_reason(final_state)
    citations_pass = bool(final_state.get("citations_pass", False))
    has_answer = _has_answer(final_state)

    if outcome == "pass":
        reward += REWARD_GROUNDED_PASS
        details["grounded_pass"] = REWARD_GROUNDED_PASS

    elif outcome == "grounded_incomplete":
        reward += REWARD_GROUNDED_INCOMPLETE
        details["grounded_incomplete"] = REWARD_GROUNDED_INCOMPLETE

    elif outcome == "unsupported":
        reward += PENALTY_UNSUPPORTED
        details["unsupported"] = PENALTY_UNSUPPORTED

    elif outcome == "needs_revision":
        reward += PENALTY_NEEDS_REVISION
        details["needs_revision"] = PENALTY_NEEDS_REVISION

    if citations_pass:
        reward += REWARD_CITATIONS_PASS_BONUS
        details["citations_bonus"] = REWARD_CITATIONS_PASS_BONUS

    if not has_answer:
        reward += PENALTY_EMPTY_ANSWER
        details["empty_answer"] = PENALTY_EMPTY_ANSWER

    if stop_reason == "grounded_answer_ready":
        reward += BONUS_EARLY_STOP_GOOD
        details["good_stop"] = BONUS_EARLY_STOP_GOOD

    if stop_reason == "max_steps_reached":
        reward += PENALTY_MAX_STEP_TERMINATION
        details["max_steps"] = PENALTY_MAX_STEP_TERMINATION

    if stop_reason == "audit_retry_limit_reached":
        reward += PENALTY_AUDIT_RETRY_LIMIT
        details["audit_retry_limit"] = PENALTY_AUDIT_RETRY_LIMIT

    return reward, details


def get_step_reward(
    prev_state: Dict,
    action: str,
    next_state: Dict,
) -> Tuple[float, Dict]:
    """
    Reward applied for a single transition.
    This includes:
    - small per-step penalty
    - action-specific penalties
    - evidence-improvement bonus
    - bad-stop penalty if stop is chosen prematurely
    """
    reward = 0.0
    details = {}

    # Every extra step costs something.
    reward += PENALTY_PER_STEP
    details["step_penalty"] = PENALTY_PER_STEP

    if action == "rewrite_query":
        reward += PENALTY_REWRITE_ACTION
        details["rewrite_penalty"] = PENALTY_REWRITE_ACTION

    if action == "verify":
        reward += PENALTY_VERIFY_ACTION
        details["verify_penalty"] = PENALTY_VERIFY_ACTION

    # Evidence improvement shaping.
    prev_quality = _evidence_quality_score(prev_state)
    next_quality = _evidence_quality_score(next_state)
    improvement = next_quality - prev_quality

    if improvement >= EVIDENCE_IMPROVEMENT_DELTA:
        reward += REWARD_EVIDENCE_IMPROVEMENT
        details["evidence_improvement"] = REWARD_EVIDENCE_IMPROVEMENT
        details["evidence_delta"] = improvement
    else:
        details["evidence_delta"] = improvement

    # Penalize obviously bad early stopping.
    if action == "stop":
        next_outcome = _verification_outcome(next_state)
        next_has_answer = _has_answer(next_state)
        next_has_retrieval = len(next_state.get("retrieved_docs", []) or []) > 0

        bad_stop = (
            (not next_has_answer and next_outcome == "")
            or (not next_has_retrieval)
            or (next_outcome in {"unsupported", "needs_revision"})
        )

        if bad_stop:
            reward += PENALTY_BAD_STOP
            details["bad_stop"] = PENALTY_BAD_STOP

    return reward, details


def compute_transition_reward(
    prev_state: Dict,
    action: str,
    next_state: Dict,
) -> Tuple[float, Dict]:
    """
    Main helper used later by trajectory logging and RL training.
    """
    reward, details = get_step_reward(prev_state, action, next_state)

    if bool(next_state.get("done", False)):
        terminal_reward, terminal_details = get_terminal_reward(next_state)
        reward += terminal_reward
        details["terminal"] = terminal_details

    return reward, details


if __name__ == "__main__":
    prev_state = {
        "retrieved_docs": [
            {"score": 0.40, "rerank_score": 0.05, "metadata": {"source_file": "A.pdf"}},
        ],
        "candidate_docs": [],
        "graded_docs": [],
        "generation": "",
        "mixed_domain_evidence": True,
        "done": False,
        "verification_outcome": "",
        "citations_pass": False,
    }

    next_state = {
        "retrieved_docs": [
            {"score": 0.82, "rerank_score": 0.31, "metadata": {"source_file": "A.pdf"}},
            {"score": 0.70, "rerank_score": 0.12, "metadata": {"source_file": "B.pdf"}},
        ],
        "candidate_docs": [
            {"score": 0.82, "rerank_score": 0.31, "metadata": {"source_file": "A.pdf"}},
        ],
        "graded_docs": [],
        "generation": "",
        "mixed_domain_evidence": False,
        "done": False,
        "verification_outcome": "",
        "citations_pass": False,
    }

    reward, details = compute_transition_reward(prev_state, "rewrite_query", next_state)
    print("Transition reward:", reward)
    print("Details:", details)

    final_state = {
        "retrieved_docs": next_state["retrieved_docs"],
        "candidate_docs": next_state["candidate_docs"],
        "graded_docs": next_state["candidate_docs"],
        "generation": "A grounded answer.",
        "mixed_domain_evidence": False,
        "done": True,
        "verification_outcome": "grounded_incomplete",
        "citations_pass": True,
        "stop_reason": "grounded_answer_ready",
    }

    reward2, details2 = compute_transition_reward(next_state, "stop", final_state)
    print("\nTerminal transition reward:", reward2)
    print("Details:", details2)