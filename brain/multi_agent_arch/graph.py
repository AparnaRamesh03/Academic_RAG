from pathlib import Path
import sys
import time
from typing import Callable, Dict, Any

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent

for path in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from langgraph.graph import StateGraph, START, END

from state_shared import GraphState
from supervisor import supervisor_step, choose_next_agent, finish_step, _policy
from retriever_agent import retriever_agent
from rewrite_agent import rewrite_agent
from evidence_agent import evidence_agent
from answer_agent import answer_agent
from verification_agent import verification_agent
from rl_reward import compute_episode_reward
from config import RL_ENABLED


def _run_agent(
    state: GraphState,
    action_name: str,
    agent_fn: Callable[[GraphState], Dict[str, Any]],
):
    start = time.perf_counter()
    updates = agent_fn(state)
    elapsed = time.perf_counter() - start

    history = list(state.get("action_history", []))
    history.append(action_name)

    # Record the RL transition: pair the pending state_key (written by
    # supervisor_step) with the action that was actually executed.
    # This captures rule-based actions too, not just RL overrides.
    rl_transitions = list(state.get("rl_transitions", []) or [])
    pending_key = state.get("rl_pending_state_key")
    if pending_key:
        rl_transitions = rl_transitions + [[pending_key, action_name]]

    return {
        **updates,
        "last_action": action_name,
        "step_count": int(state.get("step_count", 0)) + 1,
        "action_history": history,
        "latency_so_far": float(state.get("latency_so_far", 0.0)) + elapsed,
        "rl_transitions": rl_transitions,
        "rl_pending_state_key": "",  # consume the key
    }


def supervisor_node(state: GraphState):
    return supervisor_step(state)


def _rl_finish_node(state: GraphState):
    """
    Wraps finish_step with a policy update.

    The episode's rl_transitions (accumulated in GraphState throughout the run)
    are used to compute a Monte Carlo reward update after the episode completes.
    """
    updates = finish_step(state)

    if RL_ENABLED and _policy is not None:
        # Merge final state with the updates so reward sees the definitive values.
        final_state = {**state, **updates}
        reward = compute_episode_reward(final_state)
        raw_transitions = list(final_state.get("rl_transitions", []) or [])
        if raw_transitions:
            _policy.update_from_state_transitions(raw_transitions, reward)
        else:
            print("[RL] No transitions recorded this episode — skipping update.")

    return updates


def retriever_agent_node(state: GraphState):
    return _run_agent(state, "retriever_agent", retriever_agent)


def rewrite_agent_node(state: GraphState):
    return _run_agent(state, "rewrite_agent", rewrite_agent)


def evidence_agent_node(state: GraphState):
    return _run_agent(state, "evidence_agent", evidence_agent)


def answer_agent_node(state: GraphState):
    return _run_agent(state, "answer_agent", answer_agent)


def verification_agent_node(state: GraphState):
    return _run_agent(state, "verification_agent", verification_agent)


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("finish", _rl_finish_node)

    workflow.add_node("retriever_agent", retriever_agent_node)
    workflow.add_node("rewrite_agent", rewrite_agent_node)
    workflow.add_node("evidence_agent", evidence_agent_node)
    workflow.add_node("answer_agent", answer_agent_node)
    workflow.add_node("verification_agent", verification_agent_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        choose_next_agent,
        {
            "retriever_agent": "retriever_agent",
            "rewrite_agent": "rewrite_agent",
            "evidence_agent": "evidence_agent",
            "answer_agent": "answer_agent",
            "verification_agent": "verification_agent",
            "finish": "finish",
        },
    )

    workflow.add_edge("retriever_agent", "supervisor")
    workflow.add_edge("rewrite_agent", "supervisor")
    workflow.add_edge("evidence_agent", "supervisor")
    workflow.add_edge("answer_agent", "supervisor")
    workflow.add_edge("verification_agent", "supervisor")

    workflow.add_edge("finish", END)

    return workflow.compile()