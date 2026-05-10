"""
brain/context_marl_ac/agents/grader_agent.py
--------------------------------------------
Agent responsible for filtering and relevance grading.
"""

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import grade_chunks

class GraderAgent(BaseAgent):
    def __init__(self):
        super().__init__("grader")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        if not state.retrieved_chunks:
            state.record_action(self.name, "skipped_no_chunks")
            return state
            
        # Filter chunks
        filtered_chunks = grade_chunks(state.user_query, state.retrieved_chunks, mode=action_name)
        
        # Update graded chunks
        state.graded_chunks = filtered_chunks
        
        # Update selected evidence based on graded chunks
        from context_marl_ac.context_engineering.evidence_pack_builder import build_evidence_pack
        state.selected_evidence = build_evidence_pack(filtered_chunks)
        
        self.log_action(state, action_name)
        if action_name == "keep_all":
            state.num_llm_calls -= 1 # No LLM used for keep_all
            
        return state
