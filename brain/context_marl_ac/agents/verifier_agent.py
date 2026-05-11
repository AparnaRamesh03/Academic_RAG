"""
brain/context_marl_ac/agents/verifier_agent.py
----------------------------------------------
Verifier and recovery-decision agent.

Important behavior:
- verify_answer performs actual LLM verification.
- If PASS: terminal accepted.
- If FAIL and retry budget remains: non-terminal VERIFIED_FAIL stage.
- Recovery actions do not call the LLM verifier again.
"""

import context_marl_ac.config as cfg
from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import verify_answer
from context_marl_ac.adapters.citation_adapter import (
    compute_citation_support,
    detect_unsupported_claims,
)


class VerifierAgent(BaseAgent):
    def __init__(self):
        super().__init__("verifier")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        if action_name in {
            "request_regeneration",
            "request_more_retrieval",
            "request_rewrite",
        }:
            state.retry_count += 1
            state.final_status = "pending"
            state.done = False

            self.log_action(state, action_name)

            # Recovery-choice action itself does not call Groq.
            state.num_llm_calls -= 1
            return state

        if action_name != "verify_answer":
            raise ValueError(f"Unknown action {action_name} for {self.name}")

        if not state.generated_answer or state.generated_answer.strip() == "":
            state.record_action(self.name, "skipped_no_answer")
            state.final_status = "generation_failed"
            state.done = True
            state.update_latency()
            return state

        verification, tokens = verify_answer(
            state.user_query,
            state.generated_answer,
            state.selected_evidence,
        )

        state.token_usage += tokens
        state.verification_result = verification

        state.citation_support_rate = compute_citation_support(
            state.generated_answer,
            state.citation_candidates,
            state.selected_evidence,
        )

        state.unsupported_claims = detect_unsupported_claims(
            state.generated_answer,
            state.selected_evidence,
        )

        decision = str(verification.get("decision", "FAIL")).upper()

        if decision == "PASS":
            state.final_status = "accepted"
            state.done = True
        else:
            max_retries = getattr(cfg, "MAX_VERIFICATION_RETRIES", 1)

            if state.retry_count < max_retries:
                state.final_status = "pending"
                state.done = False
            else:
                state.final_status = "rejected"
                state.done = True

        self.log_action(state, action_name)
        return state