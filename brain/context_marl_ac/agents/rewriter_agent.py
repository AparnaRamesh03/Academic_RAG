"""
brain/context_marl_ac/agents/rewriter_agent.py
----------------------------------------------
Agent responsible for query reformulation.
"""

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import rewrite_query

class RewriterAgent(BaseAgent):
    def __init__(self):
        super().__init__("rewriter")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        if action_name == "no_rewrite":
            # No LLM call here
            state.record_action(self.name, action_name)
            state.update_latency()
            return state
            
        # Use LLM to rewrite
        rewritten = rewrite_query(state.user_query, mode=action_name)
        
        # Update state for logging and next steps
        state.rewritten_query = rewritten
        state.user_query = rewritten
        
        self.log_action(state, action_name)
        return state
