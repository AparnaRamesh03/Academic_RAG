"""
brain/context_marl_ac/marl/action_masking.py
--------------------------------------------
Logic for generating valid action masks for each agent.
Enforces deterministic stage-based flow to prevent infinite loops.
"""

from typing import List
import numpy as np
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.schemas.actions import AGENT_ACTIONS

def get_valid_actions(agent: str, state: ContextState) -> List[str]:
    """
    Returns a list of valid action names for a given agent in the current state.
    Enforces a logical sequence: Rewriter -> Retriever -> Grader -> Generator -> Verifier.
    """
    all_actions = AGENT_ACTIONS.get(agent, [])
    valid = []
    
    # Identify the last agent that acted
    last_entry = state.previous_actions[-1] if state.previous_actions else None
    last_agent = last_entry["agent"] if last_entry else None
    last_action = last_entry["action"] if last_entry else None

    # Forced Verifier: if we have an answer but haven't verified it yet
    # This prevents the agent from looping on retrieval/rewriting after generating.
    has_answer = bool(state.generated_answer.strip())
    has_verification = bool(state.verification_result)
    
    if has_answer and not has_verification:
        if agent == "verifier":
            return all_actions
        else:
            return [] # Force Verifier next

    if agent == "rewriter":
        # Rewriter acts first or when explicitly requested by verifier
        if last_agent is None:
            valid = all_actions
        elif last_agent == "verifier" and last_action == "request_rewrite":
            valid = all_actions
                
    elif agent == "retriever":
        # Retriever acts after rewriter or when requested by verifier/generator
        if last_agent == "rewriter":
            # Initial retrieval
            valid = [a for a in all_actions if a != "retrieve_more"]
        elif last_agent == "verifier" and last_action == "request_more_retrieval":
            valid = all_actions
        elif last_agent == "generator" and last_action == "abstain_request_more_evidence":
            valid = all_actions
        elif last_agent == "retriever":
            # Continuous retrieval gating:
            # Only allow retrieve_more if confidence is low (stub: mean score < 0.7)
            # and we haven't done it too many times consecutively.
            avg_score = np.mean(state.retrieval_scores) if state.retrieval_scores else 0.0
            if state.retrieved_chunks and avg_score < 0.7 and state.action_count_for("retriever") < 2:
                valid = ["retrieve_more"]
                
    elif agent == "grader":
        # Grader acts after retriever
        if last_agent == "retriever":
            valid = all_actions
            
    elif agent == "generator":
        # Generator acts after grader, or after retriever if grader was skipped, or after verifier retry
        if last_agent == "grader":
            valid = [a for a in all_actions if a != "regenerate"]
        elif last_agent == "verifier" and last_action == "request_regeneration":
            valid = all_actions
        elif last_agent == "retriever" and not state.graded_chunks:
            valid = [a for a in all_actions if a != "regenerate"]
                
    elif agent == "verifier":
        # Verifier acts after generator (handled by forced logic above mostly)
        if last_agent == "generator":
            valid = all_actions
            
    return valid

def get_action_mask(agent: str, state: ContextState) -> List[int]:
    """
    Returns a binary mask [1, 0, ...] aligned to the agent's full action list.
    """
    all_actions = AGENT_ACTIONS.get(agent, [])
    valid_names = set(get_valid_actions(agent, state))
    
    return [1 if a in valid_names else 0 for a in all_actions]
