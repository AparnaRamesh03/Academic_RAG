"""
brain/context_marl_ac/agents/generator_agent.py
-----------------------------------------------
Answer generation agent.

Important behavior:
- Any new generation clears old verification.
- Regeneration uses the same selected evidence but invalidates old verifier state.
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
            state.done = True
            self.log_action(state, action_name)
            state.num_llm_calls -= 1
            return state

        state.verification_result = {}
        state.unsupported_claims = []
        state.citation_support_rate = 0.0

        state.selected_evidence = reassign_citation_ids(state.selected_evidence)

        answer, tokens = generate_answer(
            state.user_query,
            state.selected_evidence,
            mode=action_name,
        )

        state.token_usage += tokens

        if not answer or answer.strip() == "":
            print(f"Warning: Generator produced empty answer for {state.question_id}")
            state.generated_answer = ""
            state.final_status = "generation_failed"
            state.done = True
        else:
            state.generated_answer = answer
            state.final_status = "pending"
            state.done = False

        state.citation_candidates = build_citations(state.selected_evidence)

        self.log_action(state, action_name)
        return state