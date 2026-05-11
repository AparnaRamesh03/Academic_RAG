"""
brain/context_marl_ac/marl/reward.py
------------------------------------
Cooperative reward function for Supervisor-Guided Fully Free MARL RAG.

This reward is designed for the fully free version where:
    Supervisor Actor chooses the next agent.
    Selected Agent Actor chooses the action.

It adds shaping penalties for:
    - repeated same agent/action without progress
    - repeated rewriter calls
    - generator skipping the grader on complex questions
    - generate_short_answer on complex questions
    - invalid actions
    - timeout / no answer / unsupported claims
"""

from typing import Dict, Tuple, List, Any

from context_marl_ac.config import (
    W_ANSWER_QUALITY,
    W_CITATION_SUPPORT,
    W_VERIFICATION_PASS,
    W_RETRIEVAL_F1,
    W_LATENCY_COST,
    W_STEP_COST,
    PENALTY_HALLUCINATION,
    PENALTY_UNSUPPORTED_CLAIM,
    PENALTY_REPEATED_ACTION,
    PENALTY_INVALID_ACTION,
    PENALTY_NO_ANSWER,
    PENALTY_MAX_STEPS,
)

import context_marl_ac.config as cfg
from context_marl_ac.schemas.context_state import ContextState


COMPLEX_QUERY_TYPES = {
    "conceptual",
    "comparison",
    "multi_hop",
    "section_specific",
    "summarization",
}


def _get_penalty(name: str, default: float) -> float:
    """
    Read optional penalty from config.py, with fallback.
    Keeps this file safe even if config constants are not added yet.
    """
    return float(getattr(cfg, name, default))


def _last_action_pair(state: ContextState) -> Dict[str, str]:
    if not state.previous_actions:
        return {}
    return state.previous_actions[-1]


def _previous_action_pair(state: ContextState) -> Dict[str, str]:
    if len(state.previous_actions) < 2:
        return {}
    return state.previous_actions[-2]


def _same_agent_repeated(state: ContextState) -> bool:
    """
    True if the same agent acted in the current and immediately previous step.
    Reward is calculated after the current action has been recorded, so we
    compare previous_actions[-1] and previous_actions[-2].
    """
    if len(state.previous_actions) < 2:
        return False

    cur = state.previous_actions[-1]
    prev = state.previous_actions[-2]
    return cur.get("agent") == prev.get("agent")


def _same_agent_same_action_repeated(state: ContextState) -> bool:
    if len(state.previous_actions) < 2:
        return False

    cur = state.previous_actions[-1]
    prev = state.previous_actions[-2]

    return (
        cur.get("agent") == prev.get("agent")
        and cur.get("action") == prev.get("action")
    )


def _is_complex_query(state: ContextState) -> bool:
    return state.query_type in COMPLEX_QUERY_TYPES or state.query_complexity == "high"


def _has_grader_been_used(state: ContextState) -> bool:
    return state.action_count_for("grader") > 0


def _current_agent(state: ContextState) -> str:
    return _last_action_pair(state).get("agent", "")


def _current_action(state: ContextState, fallback_action_name: str) -> str:
    return _last_action_pair(state).get("action", fallback_action_name)


def _retrieved_source_f1(state: ContextState) -> Tuple[float, Dict[str, float]]:
    """
    Source-level retrieval score using expected source files.
    """
    components: Dict[str, float] = {}

    if not state.expected_sources or not state.retrieved_chunks:
        return 0.0, components

    retrieved_sources = {
        c.get("metadata", {}).get("source_file")
        for c in state.retrieved_chunks
        if c.get("metadata", {}).get("source_file")
    }

    expected_sources = set(state.expected_sources)
    intersection = retrieved_sources.intersection(expected_sources)

    hit = 1.0 if intersection else 0.0
    precision = len(intersection) / len(retrieved_sources) if retrieved_sources else 0.0
    recall = len(intersection) / len(expected_sources) if expected_sources else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    components["source_hit_at_k"] = hit
    components["source_precision_at_k"] = precision
    components["source_recall_at_k"] = recall
    components["source_f1_at_k"] = f1

    return f1, components


def _citation_source_accuracy(state: ContextState) -> float:
    if not state.citation_candidates or not state.expected_sources:
        return 0.0

    citation_sources = {
        c.get("source_file")
        for c in state.citation_candidates
        if c.get("source_file")
    }

    expected_sources = set(state.expected_sources)

    if not citation_sources:
        return 0.0

    return len(citation_sources.intersection(expected_sources)) / len(citation_sources)


def _terminal_answer_quality(state: ContextState, gold_answer: str) -> float:
    """
    Lightweight answer-quality fallback.

    For now:
    - dry-run answer receives 0.85
    - accepted real answer receives 1.0
    - generated but rejected answer receives 0.0
    """
    if not state.generated_answer.strip():
        return 0.0

    if "DRY-RUN" in state.generated_answer:
        return 0.85

    if state.final_status == "accepted":
        return float(state.verification_result.get("quality_score", 1.0))

    return 0.0


def calculate_reward(
    state: ContextState,
    action_name: str,
    is_terminal: bool,
    gold_answer: str = "",
    gold_chunks: list = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculates the shared cooperative reward for the current step.

    Returns:
        (total_reward, reward_components_dict)
    """
    if gold_chunks is None:
        gold_chunks = []

    components: Dict[str, float] = {}
    reward = 0.0

    current_agent = _current_agent(state)
    current_action = _current_action(state, action_name)

    # ------------------------------------------------------------------
    # 1. Base efficiency costs
    # ------------------------------------------------------------------
    step_cost = float(W_STEP_COST)
    latency_cost = float(state.latency_so_far) * float(W_LATENCY_COST) / 10.0

    reward -= step_cost
    reward -= latency_cost

    components["step_cost"] = -step_cost
    components["latency_cost"] = -latency_cost

    # ------------------------------------------------------------------
    # 2. Invalid action penalty
    # ------------------------------------------------------------------
    if current_action.startswith("INVALID") or current_agent.startswith("INVALID"):
        reward += PENALTY_INVALID_ACTION
        components["penalty_invalid_action"] = PENALTY_INVALID_ACTION

    # ------------------------------------------------------------------
    # 3. Repeated-agent / repeated-action penalties
    # ------------------------------------------------------------------
    repeated_agent_penalty = _get_penalty(
        "PENALTY_REPEATED_AGENT",
        PENALTY_REPEATED_ACTION,
    )

    repeated_same_action_penalty = _get_penalty(
        "PENALTY_REPEATED_SAME_ACTION",
        PENALTY_REPEATED_ACTION * 1.5,
    )

    if _same_agent_repeated(state):
        reward += repeated_agent_penalty
        components["penalty_repeated_agent"] = repeated_agent_penalty

    if _same_agent_same_action_repeated(state):
        reward += repeated_same_action_penalty
        components["penalty_repeated_same_action"] = repeated_same_action_penalty

    # Stronger penalty for the exact bad pattern we observed:
    # rewriter -> rewriter before retrieval.
    if (
        current_agent == "rewriter"
        and _same_agent_repeated(state)
        and not state.retrieved_chunks
    ):
        penalty = _get_penalty("PENALTY_REWRITE_REPEAT_BEFORE_RETRIEVAL", -0.08)
        reward += penalty
        components["penalty_rewrite_repeat_before_retrieval"] = penalty

    # ------------------------------------------------------------------
    # 4. Penalize skipping grader on complex questions
    # ------------------------------------------------------------------
    # This does NOT force the grader. It just teaches the supervisor that
    # complex questions usually benefit from evidence grading.
    if (
        current_agent == "generator"
        and _is_complex_query(state)
        and not _has_grader_been_used(state)
    ):
        penalty = _get_penalty("PENALTY_SKIP_GRADER_COMPLEX", -0.08)
        reward += penalty
        components["penalty_skip_grader_complex"] = penalty

    # Small positive shaping reward when complex questions use grader before generator.
    if (
        current_agent == "generator"
        and _is_complex_query(state)
        and _has_grader_been_used(state)
    ):
        bonus = float(getattr(cfg, "BONUS_GRADER_USED_COMPLEX", 0.03))
        reward += bonus
        components["bonus_grader_used_complex"] = bonus

    # ------------------------------------------------------------------
    # 5. Penalize short-answer mode for complex questions
    # ------------------------------------------------------------------
    if current_agent == "generator" and current_action == "generate_short_answer":
        if _is_complex_query(state):
            penalty = _get_penalty("PENALTY_SHORT_ANSWER_COMPLEX", -0.08)
            reward += penalty
            components["penalty_short_answer_complex"] = penalty

    # Extra penalty if the generated answer is very short for complex questions.
    if (
        current_agent == "generator"
        and _is_complex_query(state)
        and state.generated_answer.strip()
        and len(state.generated_answer.strip()) < 180
    ):
        penalty = _get_penalty("PENALTY_TOO_SHORT_COMPLEX_ANSWER", -0.05)
        reward += penalty
        components["penalty_too_short_complex_answer"] = penalty

    # ------------------------------------------------------------------
    # 6. Penalize verifying weak/empty answer
    # ------------------------------------------------------------------
    if current_agent == "verifier" and not state.generated_answer.strip():
        reward += PENALTY_NO_ANSWER
        components["penalty_verify_no_answer"] = PENALTY_NO_ANSWER

    # ------------------------------------------------------------------
    # 7. Terminal rewards and penalties
    # ------------------------------------------------------------------
    if is_terminal:
        # A. Answer quality
        answer_quality = _terminal_answer_quality(state, gold_answer)
        answer_quality_reward = W_ANSWER_QUALITY * answer_quality
        reward += answer_quality_reward
        components["answer_quality"] = float(answer_quality_reward)

        # B. Citation support
        citation_reward = W_CITATION_SUPPORT * state.citation_support_rate
        reward += citation_reward
        components["citation_support"] = float(citation_reward)

        # C. Citation source accuracy
        citation_accuracy = _citation_source_accuracy(state)
        if citation_accuracy > 0.0:
            citation_accuracy_reward = 0.10 * citation_accuracy
            reward += citation_accuracy_reward
            components["citation_source_accuracy"] = citation_accuracy
            components["citation_source_accuracy_reward"] = citation_accuracy_reward

        # D. Verification status
        if state.final_status == "accepted":
            reward += W_VERIFICATION_PASS
            components["verification_pass"] = W_VERIFICATION_PASS

        elif state.final_status == "rejected":
            reward += PENALTY_HALLUCINATION
            components["penalty_hallucination"] = PENALTY_HALLUCINATION

        elif state.final_status == "abstained":
            reward += PENALTY_NO_ANSWER
            components["penalty_abstained"] = PENALTY_NO_ANSWER

        elif state.final_status == "timeout":
            reward += PENALTY_MAX_STEPS
            components["penalty_timeout"] = PENALTY_MAX_STEPS

        elif state.final_status == "error":
            reward += PENALTY_INVALID_ACTION
            components["penalty_error"] = PENALTY_INVALID_ACTION

        # E. Retrieval source F1
        source_f1, source_components = _retrieved_source_f1(state)
        components.update(source_components)

        retrieval_reward = W_RETRIEVAL_F1 * source_f1
        reward += retrieval_reward
        components["retrieval_f1"] = float(retrieval_reward)

        # F. Unsupported claims
        if len(state.unsupported_claims) > 0:
            unsupported_penalty = PENALTY_UNSUPPORTED_CLAIM * len(state.unsupported_claims)
            reward += unsupported_penalty
            components["penalty_unsupported"] = unsupported_penalty

        # G. No answer
        if not state.generated_answer.strip():
            reward += PENALTY_NO_ANSWER
            components["penalty_no_answer"] = PENALTY_NO_ANSWER

        # H. Terminal short-answer penalty for complex accepted/rejected answers.
        if (
            _is_complex_query(state)
            and state.generated_answer.strip()
            and len(state.generated_answer.strip()) < 180
        ):
            penalty = _get_penalty("PENALTY_TOO_SHORT_COMPLEX_ANSWER_TERMINAL", -0.05)
            reward += penalty
            components["penalty_too_short_complex_answer_terminal"] = penalty

    return round(reward, 4), components