"""
brain/context_marl_ac/agents/verifier_agent.py
----------------------------------------------
Agent responsible for verification and termination.
"""

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import verify_answer
from context_marl_ac.adapters.citation_adapter import compute_citation_support, detect_unsupported_claims

class VerifierAgent(BaseAgent):
    def __init__(self):
        super().__init__("verifier")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        if not state.generated_answer or state.generated_answer.strip() == "":
            state.record_action(self.name, "skipped_no_answer")
            state.final_status = "generation_failed"
            state.done = True
            return state

        # 1. Run LLM verification
        verification, tokens = verify_answer(state.user_query, state.generated_answer, state.selected_evidence)
        state.token_usage += tokens
        state.verification_result = verification
        
        # 2. Compute citation metrics
        state.citation_support_rate = compute_citation_support(
            state.generated_answer, 
            state.citation_candidates, 
            state.selected_evidence
        )
        state.unsupported_claims = detect_unsupported_claims(
            state.generated_answer, 
            state.selected_evidence
        )
        
        # 3. Handle terminal decision
        if action_name == "verify_answer":
            if verification.get("decision") == "PASS":
                state.final_status = "accepted"
            else:
                state.final_status = "rejected"
            state.done = True
        else:
            state.record_action(self.name, action_name)
            self.update_latency(state)
            
        return state
