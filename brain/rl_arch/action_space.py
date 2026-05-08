from __future__ import annotations

from typing import Dict, List, Tuple

try:
    from .config import (
        BASE_ACTIONS,
        OPTIONAL_ACTIONS,
        ENABLE_GRADE_DOCS_ACTION,
        MAX_RL_CONTROLLER_STEPS,
        MAX_REWRITE_ACTIONS,
        MAX_VERIFY_ACTIONS,
    )
except ImportError:
    from config import (
        BASE_ACTIONS,
        OPTIONAL_ACTIONS,
        ENABLE_GRADE_DOCS_ACTION,
        MAX_RL_CONTROLLER_STEPS,
        MAX_REWRITE_ACTIONS,
        MAX_VERIFY_ACTIONS,
    )

RLAction = str


def get_action_list() -> List[RLAction]:
    actions = list(BASE_ACTIONS)
    if ENABLE_GRADE_DOCS_ACTION:
        actions.extend(OPTIONAL_ACTIONS)
    return actions


ACTION_LIST: List[RLAction] = get_action_list()
ACTION_TO_ID: Dict[RLAction, int] = {name: idx for idx, name in enumerate(ACTION_LIST)}
ID_TO_ACTION: Dict[int, RLAction] = {idx: name for name, idx in ACTION_TO_ID.items()}


def action_to_id(action: RLAction) -> int:
    if action not in ACTION_TO_ID:
        raise ValueError(f"Unknown RL action: {action}")
    return ACTION_TO_ID[action]


def id_to_action(action_id: int) -> RLAction:
    if action_id not in ID_TO_ACTION:
        raise ValueError(f"Unknown RL action id: {action_id}")
    return ID_TO_ACTION[action_id]


def get_valid_actions(state: Dict) -> List[RLAction]:
    """
    Return the list of currently valid controller actions.

    This is NOT the learned policy.
    This is only a safety / legality mask so RL cannot choose nonsense actions.
    """
    valid: List[RLAction] = []

    step_count = int(state.get("step_count", 0) or 0)
    crag_retries = int(state.get("crag_retries", 0) or 0)
    verify_retries = int(state.get("verify_retries", 0) or 0)

    retrieved_docs = state.get("retrieved_docs", []) or []
    candidate_docs = state.get("candidate_docs", []) or []
    graded_docs = state.get("graded_docs", []) or []
    generation = str(state.get("generation", "") or "").strip()

    done = bool(state.get("done", False))
    if done:
        return ["stop"]

    # Global hard stop fallback
    if step_count >= MAX_RL_CONTROLLER_STEPS:
        return ["stop"]

    has_retrieval = len(retrieved_docs) > 0
    has_candidates = len(candidate_docs) > 0
    has_graded = len(graded_docs) > 0
    has_answer = len(generation) > 0

    # retrieve
    # Allowed at the beginning or when answer not yet available.
    if not has_answer:
        valid.append("retrieve")

    # rewrite_query
    # Only makes sense after at least one retrieval attempt and before rewrite budget is exhausted.
    if has_retrieval and crag_retries < MAX_REWRITE_ACTIONS and not has_answer:
        valid.append("rewrite_query")

    # grade_docs
    # Optional later; disabled for now.
    if ENABLE_GRADE_DOCS_ACTION and has_retrieval and not has_graded:
        valid.append("grade_docs")

    # answer
    # Only allowed when some evidence exists.
    if has_graded or has_candidates or has_retrieval:
        valid.append("answer")

    # verify
    # Only allowed if an answer already exists and verify budget remains.
    if has_answer and verify_retries < MAX_VERIFY_ACTIONS:
        valid.append("verify")

    # stop
    # Always allow stop after some work has happened.
    if step_count > 0:
        valid.append("stop")

    # Final fallback: if nothing else is valid, stop.
    if not valid:
        valid = ["stop"]

    # Remove duplicates while preserving order.
    deduped: List[RLAction] = []
    seen = set()
    for action in valid:
        if action not in seen:
            deduped.append(action)
            seen.add(action)

    return deduped


def get_valid_action_mask(state: Dict) -> List[int]:
    """
    Returns a binary mask aligned with ACTION_LIST.
    1 means valid, 0 means invalid.
    """
    valid_actions = set(get_valid_actions(state))
    return [1 if action in valid_actions else 0 for action in ACTION_LIST]


def summarize_action_space() -> Dict:
    return {
        "actions": ACTION_LIST,
        "action_to_id": ACTION_TO_ID,
        "id_to_action": ID_TO_ACTION,
        "grade_docs_enabled": ENABLE_GRADE_DOCS_ACTION,
        "max_controller_steps": MAX_RL_CONTROLLER_STEPS,
        "max_rewrite_actions": MAX_REWRITE_ACTIONS,
        "max_verify_actions": MAX_VERIFY_ACTIONS,
    }


if __name__ == "__main__":
    example_state = {
        "step_count": 0,
        "crag_retries": 0,
        "verify_retries": 0,
        "retrieved_docs": [],
        "candidate_docs": [],
        "graded_docs": [],
        "generation": "",
        "done": False,
    }

    print("Action summary:")
    print(summarize_action_space())
    print("\nValid actions for example state:")
    print(get_valid_actions(example_state))
    print("Mask:")
    print(get_valid_action_mask(example_state))