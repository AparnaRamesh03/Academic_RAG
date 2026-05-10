"""
brain/context_marl_ac/agents/generator_agent.py
-----------------------------------------------
Agent responsible for generating the answer.
"""

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import generate_answer
from context_marl_ac.adapters.citation_adapter import build_citations
from context_marl_ac.context_engineering.citation_context_builder import reassign_citation_ids

class GeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("generator")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        if not state.selected_evidence and action_name != "abstain_request_more_evidence":
             action_name = "abstain_request_more_evidence"

        if action_name == "abstain_request_more_evidence":
            state.generated_answer = "I don't have enough information to answer this question."
            state.final_status = "abstained"
            self.log_action(state, action_name)
            state.num_llm_calls -= 1
            return state

        # 1. Prepare evidence (ensure sequential IDs)
        state.selected_evidence = reassign_citation_ids(state.selected_evidence)
        
        # 2. Generate answer
        answer = generate_answer(state.user_query, state.selected_evidence, mode=action_name)
        state.generated_answer = answer
        
        # 3. Build citation candidates
        state.citation_candidates = build_citations(state.selected_evidence)
        
        self.log_action(state, action_name)
        return state
