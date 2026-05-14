"""
brain/context_marl_ac/marl/action_masking.py
--------------------------------------------
Action masking for the Context-Engineered MARL Actor-Critic RAG system.

Main path:
    retriever -> grader -> generator -> verifier.verify_answer

Recovery paths after verification failure:
    verifier.request_regeneration -> generator -> verifier
    verifier.request_more_retrieval -> retriever.retrieve_more -> grader -> generator -> verifier
    verifier.request_rewrite -> rewriter -> retriever.hybrid_rerank -> grader -> generator -> verifier

The design is stage-gated, not fully free-routing. This keeps training stable while still
allowing adaptive recovery after failed verification.
"""

from typing import List

from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.schemas.actions import AGENT_ACTIONS


MAX_RECOVERY_ATTEMPTS = 2


TERMINAL_STATUSES = {
    "accepted",
    "rejected",
    "abstained",
    "generation_failed",
    "timeout",
    "error",
}


def _non_retrieve_more(actions: List[str]) -> List[str]:
    return [a for a in actions if a != "retrieve_more"]


def _default_retrieval_action(actions: List[str]) -> List[str]:
    """
    Allow the retriever to pick its retrieval strategy. The MADDPG-style
    continuous-control architecture relies on the retriever actor being able
    to choose dense / sparse / hybrid / hybrid_rerank on the main retrieval
    paths — restricting this to a single fixed strategy makes the actor's
    continuous parameters meaningless.

    `retrieve_more` is explicitly excluded here because it is a recovery-only
    action that requires existing retrieved chunks.
    """
    return _non_retrieve_more(actions)


def _generation_actions_without_abstain(actions: List[str]) -> List[str]:
    """
    If evidence exists, do not allow the generator to abstain randomly.
    Also block regenerate except when explicitly requested by verifier.
    """
    blocked = {"abstain_request_more_evidence", "regenerate"}
    return [a for a in actions if a not in blocked]


def _generation_actions_for_regeneration(actions: List[str]) -> List[str]:
    """
    After verifier asks for regeneration, generation actions are allowed.
    Abstain is still blocked if evidence exists.
    """
    return [a for a in actions if a != "abstain_request_more_evidence"]


def _can_request_recovery(state: ContextState) -> bool:
    """
    Used before choosing a recovery request.
    """
    return state.retry_count < MAX_RECOVERY_ATTEMPTS


def _can_continue_recovery(state: ContextState) -> bool:
    """
    Used after a recovery request was already chosen and retry_count was incremented.
    """
    return state.retry_count <= MAX_RECOVERY_ATTEMPTS


def _verification_failed(state: ContextState) -> bool:
    if not state.verification_result:
        return False

    return state.verification_result.get("decision") == "FAIL"


def get_valid_actions(agent: str, state: ContextState) -> List[str]:
    """
    Return valid action names for a given agent in the current state.
    """
    all_actions = AGENT_ACTIONS.get(agent, [])
    valid: List[str] = []

    last_entry = state.previous_actions[-1] if state.previous_actions else None
    last_agent = last_entry["agent"] if last_entry else None
    last_action = last_entry["action"] if last_entry else None

    has_retrieved = bool(state.retrieved_chunks)
    has_selected_evidence = bool(state.selected_evidence)
    has_answer = bool(state.generated_answer.strip())
    has_verification = bool(state.verification_result)

    if state.done or state.final_status in TERMINAL_STATUSES:
        return []

    # If an answer exists and has not been verified, force verifier.verify_answer.
    if has_answer and not has_verification:
        if agent == "verifier" and "verify_answer" in all_actions:
            return ["verify_answer"]
        return []

    # ------------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------------
    if agent == "retriever":
        # Main first step: force hybrid_rerank so reranking is always used.
        if last_agent is None:
            valid = _default_retrieval_action(all_actions)

        # After query rewrite, retrieve again using reranking.
        elif last_agent == "rewriter":
            valid = _default_retrieval_action(all_actions)

        # Recovery path: verifier explicitly requested more evidence.
        elif last_agent == "verifier" and last_action == "request_more_retrieval":
            if "retrieve_more" in all_actions and _can_continue_recovery(state):
                valid = ["retrieve_more"]
            else:
                valid = []

        # Do not allow automatic consecutive retrieval.
        elif last_agent == "retriever":
            valid = []

    # ------------------------------------------------------------------
    # Grader
    # ------------------------------------------------------------------
    elif agent == "grader":
        if last_agent == "retriever" and has_retrieved:
            valid = all_actions

    # ------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------
    elif agent == "generator":
        # Main path after grading.
        if last_agent == "grader":
            if has_selected_evidence:
                valid = _generation_actions_without_abstain(all_actions)
            else:
                valid = ["abstain_request_more_evidence"]

        # Recovery path: verifier explicitly requested regeneration.
        elif last_agent == "verifier" and last_action == "request_regeneration":
            if has_selected_evidence and _can_continue_recovery(state):
                valid = _generation_actions_for_regeneration(all_actions)
            else:
                valid = ["abstain_request_more_evidence"]

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------
    elif agent == "verifier":
        # Normal verification immediately after generator.
        if last_agent == "generator" and has_answer and not has_verification:
            if "verify_answer" in all_actions:
                valid = ["verify_answer"]

        # After a failed verification, allow exactly one recovery choice.
        elif (
            last_agent == "verifier"
            and last_action == "verify_answer"
            and has_verification
            and _verification_failed(state)
        ):
            if _can_request_recovery(state):
                valid = [
                    a for a in all_actions
                    if a in {
                        "request_regeneration",
                        "request_more_retrieval",
                        "request_rewrite",
                    }
                ]
            else:
                valid = []

    # ------------------------------------------------------------------
    # Rewriter
    # ------------------------------------------------------------------
    elif agent == "rewriter":
    # Rewriter is only used as a recovery action after verifier failure.
        # If verifier requested rewrite, do not allow no_rewrite.
        if last_agent == "verifier" and last_action == "request_rewrite":
            if _can_continue_recovery(state):
                valid = [a for a in all_actions if a != "no_rewrite"]
            else:
                valid = []

    return valid


def get_action_mask(agent: str, state: ContextState) -> List[int]:
    """
    Return binary mask aligned to the agent's full action list.
    """
    all_actions = AGENT_ACTIONS.get(agent, [])
    valid_names = set(get_valid_actions(agent, state))

    return [1 if action in valid_names else 0 for action in all_actions]