"""
brain/context_marl_ac/marl/action_masking.py
--------------------------------------------
Stage-based action masking for Context MARL Actor-Critic RAG.

This is NOT a fixed pipeline. It is constrained MARL:
- The supervisor still chooses between valid agents.
- Specialist actors still choose valid actions.
- The mask removes invalid / degenerate RAG actions.

Stages:
    START
    RETRIEVED
    GRADED
    GENERATED
    VERIFIED_FAIL
    RECOVERY_REQUESTED
    TERMINAL
"""

from typing import List, Set

import context_marl_ac.config as cfg
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.schemas.actions import AGENT_ACTIONS, AGENT_NAMES


TERMINAL_STATUSES = {
    "accepted",
    "rejected",
    "abstained",
    "generation_failed",
    "timeout",
    "error",
}


COMPLEX_QUERY_TYPES = {
    "conceptual",
    "comparison",
    "multi_hop",
    "section_specific",
    "summarization",
}


RECOVERY_ACTION_TO_AGENT = {
    "request_more_retrieval": "retriever",
    "request_regeneration": "generator",
    "request_rewrite": "rewriter",
}


# ---------------------------------------------------------------------------
# Basic state helpers
# ---------------------------------------------------------------------------

def _last_entry(state: ContextState):
    return state.previous_actions[-1] if state.previous_actions else None


def _last_agent(state: ContextState) -> str:
    entry = _last_entry(state)
    return entry.get("agent", "") if entry else ""


def _last_action(state: ContextState) -> str:
    entry = _last_entry(state)
    return entry.get("action", "") if entry else ""


def _has_retrieved(state: ContextState) -> bool:
    return bool(state.retrieved_chunks)


def _has_selected_evidence(state: ContextState) -> bool:
    return bool(state.selected_evidence)


def _has_answer(state: ContextState) -> bool:
    return bool((state.generated_answer or "").strip())


def _verification_decision(state: ContextState) -> str:
    if not state.verification_result:
        return ""
    return str(state.verification_result.get("decision", "")).upper()


def _verification_passed(state: ContextState) -> bool:
    return _verification_decision(state) == "PASS"


def _verification_failed(state: ContextState) -> bool:
    return _verification_decision(state) == "FAIL"


def _retry_budget_left(state: ContextState) -> bool:
    max_retries = getattr(cfg, "MAX_VERIFICATION_RETRIES", 1)
    return state.retry_count < max_retries


def _retrieval_budget_left(state: ContextState) -> bool:
    max_retrievals = getattr(cfg, "MAX_RETRIEVAL_RETRIES", 3) + 1
    return state.action_count_for("retriever") < max_retrievals


def _rewrite_budget_left(state: ContextState) -> bool:
    max_rewrites = getattr(cfg, "MAX_REWRITES", 2)
    return state.action_count_for("rewriter") < max_rewrites


def _is_terminal(state: ContextState) -> bool:
    return bool(state.done) or state.final_status in TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# Grader/retrieval tracking
# ---------------------------------------------------------------------------

def _last_retriever_index(state: ContextState) -> int:
    idx = -1
    for i, entry in enumerate(state.previous_actions):
        if entry.get("agent") == "retriever":
            idx = i
    return idx


def _last_grader_index(state: ContextState) -> int:
    idx = -1
    for i, entry in enumerate(state.previous_actions):
        if entry.get("agent") == "grader":
            idx = i
    return idx


def _grader_used_since_last_retrieval(state: ContextState) -> bool:
    return _last_grader_index(state) > _last_retriever_index(state)


def _grader_allowed(state: ContextState) -> bool:
    return _has_retrieved(state) and not _grader_used_since_last_retrieval(state)


# ---------------------------------------------------------------------------
# Query type / source-diversity helpers
# ---------------------------------------------------------------------------

def _is_complex_query(state: ContextState) -> bool:
    return (
        state.query_type in COMPLEX_QUERY_TYPES
        or state.query_complexity == "high"
    )


def _requires_full_answer(state: ContextState) -> bool:
    q = (state.original_query or state.user_query or "").lower()

    trigger_phrases = [
        "how ",
        "why ",
        "different",
        "compare",
        "comparison",
        "relationship",
        "alternative",
        "propose",
        "argues",
        "main contributions",
        "distinguish",
        "explain",
        "and how",
        "whereas",
        "contrast",
        "between",
    ]

    if any(phrase in q for phrase in trigger_phrases):
        return True

    if state.query_type in COMPLEX_QUERY_TYPES:
        return True

    if state.query_complexity == "high":
        return True

    return False


def _source_from_item(item: dict) -> str:
    metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
    return (
        item.get("source")
        or item.get("source_file")
        or metadata.get("source_file")
        or ""
    )


def _selected_source_count(state: ContextState) -> int:
    return len({_source_from_item(ev) for ev in state.selected_evidence if _source_from_item(ev)})


def _retrieved_source_count(state: ContextState) -> int:
    return len({_source_from_item(ch) for ch in state.retrieved_chunks if _source_from_item(ch)})


def _comparison_signal_present(state: ContextState) -> bool:
    q = (state.original_query or state.user_query or "").lower()

    signals = [
        "different from",
        "how is that different",
        "how are they different",
        "how do they differ",
        "how does it differ",
        "compare",
        "comparison",
        "contrast",
        "whereas",
        "versus",
        " vs ",
        "between",
        "distinguish between",
    ]

    return any(signal in q for signal in signals)


def _known_entity_groups(state: ContextState) -> Set[str]:
    q = (state.original_query or state.user_query or "").lower()
    groups = set()

    if (
        "bert" in q
        or "masked language model" in q
        or "masked language modeling" in q
        or "mlm" in q
        or "next sentence prediction" in q
        or "nsp" in q
        or "bidirectional pre-training" in q
    ):
        groups.add("bert")

    if (
        "attention is all you need" in q
        or (
            "transformer" in q
            and (
                "translation" in q
                or "sequence transduction" in q
                or "machine translation" in q
                or "recurrence" in q
                or "wmt" in q
                or "english-to-german" in q
                or "english-to-french" in q
                or "proposes the transformer" in q
                or "propose the transformer" in q
            )
        )
    ):
        groups.add("transformer_translation")

    if (
        "tabnet" in q
        or (
            "tabular" in q
            and (
                "tree-based" in q
                or "decision tree" in q
                or "sequential attention" in q
                or "feature selection" in q
            )
        )
    ):
        groups.add("tabnet")

    if "resnet" in q or "residual network" in q or "residual learning" in q:
        groups.add("resnet")

    if "vgg" in q:
        groups.add("vgg")

    if (
        "rag survey" in q
        or "naive rag" in q
        or "advanced rag" in q
        or "modular rag" in q
    ):
        groups.add("rag_survey")

    return groups


def _needs_source_diversity(state: ContextState) -> bool:
    groups = _known_entity_groups(state)

    if len(groups) < 2:
        return False

    if _comparison_signal_present(state):
        return True

    if state.query_type in {"comparison", "multi_hop"}:
        return True

    return False


def _source_diversity_retry_available(state: ContextState) -> bool:
    max_diversity_retrievals = getattr(cfg, "MAX_SOURCE_DIVERSITY_RETRIEVALS", 2)

    return (
        _needs_source_diversity(state)
        and _selected_source_count(state) < 2
        and _retrieval_budget_left(state)
        and state.action_count_for("retriever") < max_diversity_retrievals
    )


# ---------------------------------------------------------------------------
# Stage detection
# ---------------------------------------------------------------------------

def _recovery_requested_agent(state: ContextState) -> str:
    if _last_agent(state) != "verifier":
        return ""

    return RECOVERY_ACTION_TO_AGENT.get(_last_action(state), "")


def get_stage(state: ContextState) -> str:
    """
    Returns the current stage name.

    Order matters.
    """
    if _is_terminal(state):
        return "TERMINAL"

    recovery_agent = _recovery_requested_agent(state)
    if recovery_agent:
        return "RECOVERY_REQUESTED"

    if _verification_failed(state):
        if _retry_budget_left(state):
            return "VERIFIED_FAIL"
        return "TERMINAL"

    if _verification_passed(state):
        return "TERMINAL"

    if _has_answer(state) and not state.verification_result:
        return "GENERATED"

    if _has_selected_evidence(state) and _grader_used_since_last_retrieval(state):
        return "GRADED"

    if _has_retrieved(state):
        return "RETRIEVED"

    return "START"


# ---------------------------------------------------------------------------
# Agent masks
# ---------------------------------------------------------------------------

def get_valid_agents(state: ContextState) -> List[str]:
    stage = get_stage(state)

    if stage == "TERMINAL":
        return []

    if state.num_steps >= cfg.MAX_STEPS_PER_EPISODE:
        return []

    if stage == "START":
        return ["retriever"]

    if stage == "RECOVERY_REQUESTED":
        forced = _recovery_requested_agent(state)
        return [forced] if forced else []

    if stage == "RETRIEVED":
        valid = []

        if _source_diversity_retry_available(state):
            valid.append("retriever")

        if _grader_allowed(state):
            valid.append("grader")

        return _dedupe_valid_agents(valid)

    if stage == "GRADED":
        valid = []

        if _source_diversity_retry_available(state):
            valid.append("retriever")
        else:
            valid.append("generator")

        return _dedupe_valid_agents(valid)

    if stage == "GENERATED":
        return ["verifier"]

    if stage == "VERIFIED_FAIL":
        return ["verifier"]

    return []


def _dedupe_valid_agents(valid: List[str]) -> List[str]:
    result = []
    seen = set()

    for agent in valid:
        if agent in AGENT_NAMES and agent not in seen:
            result.append(agent)
            seen.add(agent)

    return result


def get_agent_mask(state: ContextState) -> List[int]:
    valid = set(get_valid_agents(state))
    return [1 if agent in valid else 0 for agent in AGENT_NAMES]


# ---------------------------------------------------------------------------
# Action masks
# ---------------------------------------------------------------------------

def get_valid_actions(agent: str, state: ContextState) -> List[str]:
    all_actions = AGENT_ACTIONS.get(agent, [])
    stage = get_stage(state)

    if stage == "TERMINAL":
        return []

    if state.num_steps >= cfg.MAX_STEPS_PER_EPISODE:
        return []

    valid_agents = set(get_valid_agents(state))
    if agent not in valid_agents:
        return []

    # ------------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------------
    if agent == "retriever":
        if not _retrieval_budget_left(state):
            return []

        if stage == "START":
            return _initial_retriever_actions(all_actions)

        if stage == "RECOVERY_REQUESTED":
            if _last_action(state) == "request_more_retrieval":
                return ["retrieve_more"] if "retrieve_more" in all_actions else []

            # request_rewrite is handled by rewriter, not retriever.
            return []

        if stage in {"RETRIEVED", "GRADED"}:
            if _source_diversity_retry_available(state):
                return ["retrieve_more"] if "retrieve_more" in all_actions else []

        return []

    # ------------------------------------------------------------------
    # Rewriter
    # ------------------------------------------------------------------
    if agent == "rewriter":
        if not _rewrite_budget_left(state):
            return []

        if stage == "RECOVERY_REQUESTED" and _last_action(state) == "request_rewrite":
            return [a for a in all_actions if a != "no_rewrite"]

        return []

    # ------------------------------------------------------------------
    # Grader
    # ------------------------------------------------------------------
    if agent == "grader":
        if stage != "RETRIEVED":
            return []

        if not _grader_allowed(state):
            return []

        actions = list(all_actions)

        # keep_all is unsafe for noisy or complex cases.
        if _is_complex_query(state) or _retrieved_source_count(state) > 1 or _needs_source_diversity(state):
            actions = [a for a in actions if a != "keep_all"]

        return actions

    # ------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------
    if agent == "generator":
        if stage not in {"GRADED", "RECOVERY_REQUESTED"}:
            return []

        if stage == "RECOVERY_REQUESTED":
            if _last_action(state) != "request_regeneration":
                return []

            actions = [
                a for a in all_actions
                if a != "abstain_request_more_evidence"
            ]

            # Prefer actual regeneration action if available.
            if "regenerate" in actions:
                return ["regenerate"]

            if _requires_full_answer(state):
                actions = [a for a in actions if a != "generate_short_answer"]

            return actions

        if not _has_selected_evidence(state):
            return (
                ["abstain_request_more_evidence"]
                if "abstain_request_more_evidence" in all_actions
                else []
            )

        actions = [
            a for a in all_actions
            if a not in {"abstain_request_more_evidence", "regenerate"}
        ]

        if _requires_full_answer(state):
            actions = [a for a in actions if a != "generate_short_answer"]

        return actions

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------
    if agent == "verifier":
        if stage == "GENERATED":
            return ["verify_answer"] if "verify_answer" in all_actions else []

        if stage == "VERIFIED_FAIL":
            if not _retry_budget_left(state):
                return []

            return [
                a for a in all_actions
                if a in {
                    "request_regeneration",
                    "request_more_retrieval",
                    "request_rewrite",
                }
            ]

        return []

    return []


def _initial_retriever_actions(all_actions: List[str]) -> List[str]:
    """
    Prefer the strongest initial retrieval action.
    """
    if "hybrid_rerank" in all_actions:
        return ["hybrid_rerank"]

    if "hybrid_retrieve" in all_actions:
        return ["hybrid_retrieve"]

    return [a for a in all_actions if a != "retrieve_more"]


def get_action_mask(agent: str, state: ContextState) -> List[int]:
    all_actions = AGENT_ACTIONS.get(agent, [])
    valid_names = set(get_valid_actions(agent, state))
    return [1 if action in valid_names else 0 for action in all_actions]