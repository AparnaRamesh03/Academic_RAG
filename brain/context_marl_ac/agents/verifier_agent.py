"""
brain/context_marl_ac/agents/verifier_agent.py
----------------------------------------------
Agent responsible for answer verification and recovery control.
"""

from typing import Any, Dict, List

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import verify_answer


MAX_RECOVERY_ATTEMPTS = 2


class VerifierAgent(BaseAgent):
    def __init__(self):
        super().__init__("verifier")

    def _claim_supported(self, claim: Dict[str, Any]) -> bool:
        """
        Robust support check because the legacy verifier can use different keys.
        """
        if claim.get("supported") is True:
            return True

        for key in ["decision", "status", "support_status", "verdict"]:
            value = str(claim.get(key, "")).upper()
            if value in {"PASS", "SUPPORTED", "SUPPORTS", "TRUE"}:
                return True

        return False

    def _claim_text(self, claim: Dict[str, Any]) -> str:
        return (
            claim.get("claim_text")
            or claim.get("claim")
            or claim.get("text")
            or str(claim)
        )

    def _update_support_metrics_from_verification(
        self,
        state: ContextState,
        verification: Dict[str, Any],
    ) -> None:
        """
        Compute citation support from the already-computed verifier output.

        Do not call compute_citation_support() or detect_unsupported_claims()
        here because those call the claim verifier again.
        """
        claims: List[Dict[str, Any]] = (
            verification.get("verified_claims")
            or verification.get("claims")
            or []
        )

        if not claims:
            if verification.get("decision") == "PASS":
                state.citation_support_rate = 1.0
                state.unsupported_claims = []
            else:
                state.citation_support_rate = 0.0
                state.unsupported_claims = (
                    [state.generated_answer[:500]]
                    if state.generated_answer.strip()
                    else []
                )
            return

        supported_claims = [claim for claim in claims if self._claim_supported(claim)]
        unsupported_claims = [
            claim for claim in claims
            if not self._claim_supported(claim)
        ]

        state.citation_support_rate = round(len(supported_claims) / len(claims), 4)
        state.unsupported_claims = [
            self._claim_text(claim)
            for claim in unsupported_claims
        ]

    def _clear_for_regeneration(self, state: ContextState) -> None:
        """
        Regenerate using the same selected evidence.
        """
        state.generated_answer = ""
        state.verification_result = {}
        state.unsupported_claims = []
        state.citation_support_rate = 0.0
        state.final_status = "pending"
        state.done = False

    def _clear_for_more_retrieval(self, state: ContextState) -> None:
        """
        Retrieve more evidence, then grade again, then generate again.

        Keep existing retrieved/selected evidence so retrieve_more can use the
        previous context and append/expand if the retriever implementation does that.
        """
        state.generated_answer = ""
        state.verification_result = {}
        state.unsupported_claims = []
        state.citation_support_rate = 0.0
        state.final_status = "pending"
        state.done = False

    def _clear_for_rewrite(self, state: ContextState) -> None:
        """
        Rewrite query and start retrieval/evidence selection again.
        """
        state.generated_answer = ""
        state.verification_result = {}
        state.unsupported_claims = []
        state.citation_support_rate = 0.0

        state.retrieved_chunks = []
        state.graded_chunks = []
        state.selected_evidence = []
        state.citation_candidates = []

        state.final_status = "pending"
        state.done = False

    def _handle_recovery_action(self, state: ContextState, action_name: str) -> ContextState:
        """
        Recovery actions are non-LLM control actions.
        They should not be counted as LLM calls.
        """
        state.record_action(self.name, action_name)
        state.retry_count += 1

        if action_name == "request_regeneration":
            self._clear_for_regeneration(state)

        elif action_name == "request_more_retrieval":
            self._clear_for_more_retrieval(state)

        elif action_name == "request_rewrite":
            self._clear_for_rewrite(state)

        else:
            state.verification_result = {
                "decision": "FAIL",
                "reason": f"Unknown recovery action: {action_name}",
                "verified_claims": [],
            }
            state.final_status = "error"
            state.done = True

        state.update_latency()
        return state

    def act(self, state: ContextState, action_name: str) -> ContextState:
        # Recovery actions after failed verification.
        if action_name in {
            "request_regeneration",
            "request_more_retrieval",
            "request_rewrite",
        }:
            return self._handle_recovery_action(state, action_name)

        if action_name != "verify_answer":
            state.record_action(self.name, action_name)
            state.verification_result = {
                "decision": "FAIL",
                "reason": f"Unknown verifier action: {action_name}",
                "verified_claims": [],
            }
            state.final_status = "error"
            state.done = True
            state.update_latency()
            return state

        if not state.generated_answer or state.generated_answer.strip() == "":
            state.record_action(self.name, "skipped_no_answer")
            state.verification_result = {
                "decision": "FAIL",
                "reason": "No generated answer to verify.",
                "verified_claims": [],
            }
            state.final_status = "generation_failed"
            state.done = True
            state.update_latency()
            return state

        # Verify against original user question, not rewritten retrieval query.
        verification, tokens = verify_answer(
            state.original_query,
            state.generated_answer,
            state.selected_evidence,
        )

        state.token_usage += tokens
        state.verification_result = verification

        self._update_support_metrics_from_verification(state, verification)

        # MADDPG support_threshold: override LLM decision with citation-rate check.
        p = state.maddpg_params or {}
        support_threshold = p.get("support_threshold", None)
        if support_threshold is not None and state.citation_support_rate > 0:
            overridden = "PASS" if state.citation_support_rate >= float(support_threshold) else "FAIL"
            state.verification_result = dict(state.verification_result)
            state.verification_result["decision"] = overridden
            verification = state.verification_result

        # verify_answer is an LLM-backed call.
        self.log_action(state, action_name)

        if verification.get("decision") == "PASS":
            state.final_status = "accepted"
            state.done = True
            return state

        # Verification failed.
        # If recovery attempts remain, keep episode alive so the verifier can choose a recovery action next.
        if state.retry_count < MAX_RECOVERY_ATTEMPTS:
            state.final_status = "pending"
            state.done = False
        else:
            state.final_status = "rejected"
            state.done = True

        return state