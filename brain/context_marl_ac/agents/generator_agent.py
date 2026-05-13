"""
brain/context_marl_ac/agents/generator_agent.py
-----------------------------------------------
Agent responsible for generating the answer.
"""

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import generate_answer
from context_marl_ac.adapters.citation_adapter import build_citations
from context_marl_ac.context_engineering.citation_context_builder import (
    reassign_citation_ids,
)


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

        # Prepare evidence with sequential citation IDs.
        state.selected_evidence = reassign_citation_ids(state.selected_evidence)

        # MADDPG continuous params: temperature and max_tokens for generation.
        p = state.maddpg_params or {}
        temperature = p.get("temperature", None)  # None → adapter default
        max_tokens  = p.get("max_tokens", None)   # None → adapter default

        # Use the original user question, not the rewritten retrieval query.
        answer, tokens = generate_answer(
            state.original_query,
            state.selected_evidence,
            mode=action_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        state.token_usage += tokens

        if not answer or answer.strip() == "":
            print(f"Warning: Generator produced empty answer for {state.question_id}")
            state.generated_answer = ""
            state.final_status = "generation_failed"
            state.done = True
        else:
            state.generated_answer = answer
            if state.final_status == "generation_failed":
                state.final_status = "pending"

        state.citation_candidates = build_citations(state.selected_evidence)

        self.log_action(state, action_name)
        return state