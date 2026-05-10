"""
brain/context_marl_ac/marl/action_masking.py
--------------------------------------------
Logic for generating valid action masks for each agent.
"""

from typing import List
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.schemas.actions import AGENT_ACTIONS

def get_valid_actions(agent: str, state: ContextState) -> List[str]:
    """
    Returns a list of valid action names for a given agent in the current state.
    """
    all_actions = AGENT_ACTIONS.get(agent, [])
    valid = []
    
    if agent == "retriever":
        # Always allowed to retrieve
        for a in all_actions:
            if a == "retrieve_more":
                if state.retrieved_chunks:
                    valid.append(a)
            else:
                valid.append(a)
                
    elif agent == "rewriter":
        # Always allowed to rewrite
        valid = all_actions
        
    elif agent == "grader":
        # Only if we have chunks to grade
        if state.retrieved_chunks:
            valid = all_actions
        else:
            # If no chunks, only 'keep_all' is technically safe (but useless)
            # or maybe nothing. Let's say must have chunks.
            valid = ["keep_all"] if "keep_all" in all_actions else []
            
    elif agent == "generator":
        # Only if we have evidence or chunks
        if state.selected_evidence or state.retrieved_chunks:
            valid = all_actions
        else:
            # Can only abstain if no evidence
            if "abstain_request_more_evidence" in all_actions:
                valid = ["abstain_request_more_evidence"]
                
    elif agent == "verifier":
        # Only if we have an answer to verify
        if state.generated_answer.strip():
            valid = all_actions
        else:
            # Cannot verify nothing. 
            valid = []
            
    return valid

def get_action_mask(agent: str, state: ContextState) -> List[int]:
    """
    Returns a binary mask [1, 0, ...] aligned to the agent's full action list.
    """
    all_actions = AGENT_ACTIONS.get(agent, [])
    valid_names = set(get_valid_actions(agent, state))
    
    return [1 if a in valid_names else 0 for a in all_actions]
