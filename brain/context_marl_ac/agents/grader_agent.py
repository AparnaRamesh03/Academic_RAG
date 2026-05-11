"""
brain/context_marl_ac/agents/grader_agent.py
--------------------------------------------
Evidence grading/filtering agent.

Important behavior:
- Grader runs once per retrieval set.
- keep_all is bounded.
- rerank_only is local and does not call Groq.
- Any grading change clears stale answer/verification.
"""

from typing import List, Dict, Any

import context_marl_ac.config as cfg
from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import grade_chunks
from context_marl_ac.context_engineering.evidence_pack_builder import build_evidence_pack


def _sort_by_score(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        chunks,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )


def _source_file(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    return (
        chunk.get("source")
        or chunk.get("source_file")
        or metadata.get("source_file")
        or ""
    )


def _source_count(chunks: List[Dict[str, Any]]) -> int:
    return len({_source_file(c) for c in chunks if _source_file(c)})


def _is_complex_query(state: ContextState) -> bool:
    return (
        state.query_type in {
            "conceptual",
            "comparison",
            "multi_hop",
            "section_specific",
            "summarization",
        }
        or state.query_complexity == "high"
    )


def _clear_downstream_answer_state(state: ContextState) -> None:
    state.generated_answer = ""
    state.citation_candidates = []
    state.verification_result = {}
    state.unsupported_claims = []
    state.citation_support_rate = 0.0


class GraderAgent(BaseAgent):
    def __init__(self):
        super().__init__("grader")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        if not state.retrieved_chunks:
            state.record_action(self.name, "skipped_no_chunks")
            state.update_latency()
            return state

        fallback_used = False
        tokens = 0

        sorted_chunks = _sort_by_score(state.retrieved_chunks)

        if action_name == "keep_all":
            max_docs = getattr(cfg, "MAX_KEEP_ALL_DOCS", 4)

            if _is_complex_query(state) or _source_count(state.retrieved_chunks) > 1:
                max_docs = min(max_docs, 3)

            filtered_chunks = sorted_chunks[:max_docs]

        elif action_name == "rerank_only":
            max_docs = getattr(cfg, "MAX_RERANK_ONLY_DOCS", 5)
            filtered_chunks = sorted_chunks[:max_docs]

        else:
            filtered_chunks, tokens = grade_chunks(
                state.user_query,
                state.retrieved_chunks,
                mode=action_name,
            )

            state.token_usage += tokens

            if not filtered_chunks and state.retrieved_chunks:
                filtered_chunks = sorted_chunks[:3]
                fallback_used = True

        state.graded_chunks = filtered_chunks
        state.selected_evidence = build_evidence_pack(filtered_chunks)

        _clear_downstream_answer_state(state)

        scores = [float(c.get("score", 0.0)) for c in filtered_chunks]

        if action_name == "strict_filter":
            threshold = 0.85
        elif action_name == "medium_filter":
            threshold = 0.75
        elif action_name == "loose_filter":
            threshold = 0.0
        else:
            threshold = 0.0

        state.grader_output = {
            "grader_action": action_name,
            "num_retrieved_before_grading": len(state.retrieved_chunks),
            "num_selected_after_grading": len(filtered_chunks),
            "selected_chunk_scores": scores,
            "selected_sources": sorted(
                list({_source_file(c) for c in filtered_chunks if _source_file(c)})
            ),
            "filter_threshold": threshold,
            "fallback_used": fallback_used,
        }

        self.log_action(state, action_name)

        if action_name in {"keep_all", "rerank_only"}:
            state.num_llm_calls -= 1

        return state