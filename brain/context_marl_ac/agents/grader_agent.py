"""
brain/context_marl_ac/agents/grader_agent.py
--------------------------------------------
Agent responsible for filtering and relevance grading.
"""

from typing import List, Dict, Any
from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import grade_chunks
from context_marl_ac.context_engineering.evidence_pack_builder import build_evidence_pack

class GraderAgent(BaseAgent):
    def __init__(self):
        super().__init__("grader")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        if not state.retrieved_chunks:
            state.record_action(self.name, "skipped_no_chunks")
            return state

        # MADDPG continuous params for post-grading filtering.
        p = state.maddpg_params or {}
        keep_ratio          = float(p.get("evidence_keep_ratio", 1.0))
        relevance_threshold = float(p.get("relevance_threshold", 0.0))

        # 1. Run Filtering/Grading
        filtered_chunks, tokens = grade_chunks(state.user_query, state.retrieved_chunks, mode=action_name)
        state.token_usage += tokens

        # 2. Fallback Logic
        fallback_used = False
        if not filtered_chunks and state.retrieved_chunks:
            sorted_chunks = sorted(state.retrieved_chunks, key=lambda x: x.get("score", 0.0), reverse=True)
            filtered_chunks = sorted_chunks[:3]
            fallback_used = True

        # 3. MADDPG relevance_threshold: drop chunks below the threshold score
        # before applying keep-ratio. Only meaningful when chunks have scores.
        if relevance_threshold > 0.0 and filtered_chunks:
            scored = [c for c in filtered_chunks if float(c.get("score", 0.0)) >= relevance_threshold]
            if scored:
                filtered_chunks = scored

        # 4. MADDPG evidence_keep_ratio: trim to top-N fraction after LLM grading.
        if keep_ratio < 1.0 and filtered_chunks:
            n_keep = max(1, int(len(filtered_chunks) * keep_ratio))
            filtered_chunks = sorted(filtered_chunks, key=lambda x: x.get("score", 0.0), reverse=True)[:n_keep]

        # 4. Update State
        state.graded_chunks = filtered_chunks
        state.selected_evidence = build_evidence_pack(filtered_chunks)

        # 5. Detailed Logging
        scores = [float(c.get("score", 0.0)) for c in filtered_chunks]
        threshold = 0.85 if action_name == "strict_filter" else (0.75 if action_name == "medium_filter" else 0.0)

        state.grader_output = {
            "grader_action": action_name,
            "num_retrieved_before_grading": len(state.retrieved_chunks),
            "num_selected_after_grading": len(filtered_chunks),
            "selected_chunk_scores": scores,
            "filter_threshold": threshold,
            "fallback_used": fallback_used,
            "evidence_keep_ratio": keep_ratio,
        }

        self.log_action(state, action_name)
        if action_name in ("keep_all", "rerank_only"):
            state.num_llm_calls -= 1

        return state
