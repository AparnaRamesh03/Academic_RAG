"""
brain/context_marl_ac/agents/retriever_agent.py
-----------------------------------------------
Agent responsible for interacting with the retrieval system.
"""

from typing import List, Dict, Any

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.retriever_adapter import (
    retrieve_hybrid, retrieve_dense, retrieve_sparse,
    retrieve_hybrid_rerank, retrieve_more
)
from context_marl_ac.context_engineering.evidence_pack_builder import build_evidence_pack


def _enforce_source_diversity(
    chunks: List[Dict[str, Any]],
    diversity: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Cap chunks per source_file so that one document does not dominate retrieval.

    `diversity` is in [0, 1]:
      - 0.0–0.7: no capping (default behaviour during exploration)
      - 1.0:     at most 1 chunk per source
      - in between: linearly interpolated cap

    Threshold was raised from 0.5 to 0.7 so the OU exploration noise (σ=0.15
    around raw≈0 → u≈0.5) does NOT trigger capping by default. Previously this
    caused the trained actor to converge on a 1-chunk-per-question policy that
    looked great on verifier-pass but under-cited the gold answer.
    """
    if not chunks:
        return chunks
    if diversity < 0.7:
        return chunks
    span = max(1, top_k)
    # Remap diversity in [0.7, 1.0] to a cap in [span, 1].
    t = (diversity - 0.7) / 0.3
    cap = max(1, int(round(span * (1.0 - t) + 1.0 * t)))
    counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for c in chunks:
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        src = meta.get("source_file") or c.get("source_file") or ""
        counts[src] = counts.get(src, 0) + 1
        if counts[src] <= cap:
            out.append(c)
    return out if out else chunks

class RetrieverAgent(BaseAgent):
    def __init__(self):
        super().__init__("retriever")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        # Get query (might be original or rewritten)
        query = state.user_query

        # MADDPG continuous params override top_k when available.
        p = state.maddpg_params or {}
        raw_top_k = p.get("top_k", 8)
        top_k = max(1, int(round(float(raw_top_k))))
        source_diversity = float(p.get("source_diversity", 0.0))

        if action_name == "dense_retrieve":
            chunks = retrieve_dense(query, top_k)
        elif action_name == "sparse_retrieve":
            chunks = retrieve_sparse(query, top_k)
        elif action_name == "hybrid_retrieve":
            chunks = retrieve_hybrid(query, top_k)
        elif action_name == "hybrid_rerank":
            chunks = retrieve_hybrid_rerank(query, top_k)
        elif action_name == "retrieve_more":
            chunks = retrieve_more(query, state.retrieved_chunks, top_k)
            # Append instead of replace for retrieve_more
            state.retrieved_chunks.extend(chunks)
            state.retrieval_scores.extend([c.get("score", 0.0) for c in chunks])
            state.selected_evidence.extend(build_evidence_pack(chunks))
            self.log_action(state, action_name)
            # Retrieval doesn't use LLM (usually), so adjust count
            state.num_llm_calls -= 1
            return state
        else:
            raise ValueError(f"Unknown action {action_name} for {self.name}")

        # MADDPG source_diversity: when high, post-filter to cap chunks per source.
        if source_diversity >= 0.5 and chunks:
            chunks = _enforce_source_diversity(chunks, source_diversity, top_k)

        # Update state for standard retrieval actions
        state.retrieved_chunks = chunks
        state.retrieval_scores = [c.get("score", 0.0) for c in chunks]
        state.selected_evidence = build_evidence_pack(chunks)
        
        self.log_action(state, action_name)
        # Retrieval usually doesn't use LLM unless it's an embedding model, 
        # but in our architecture, LLM calls refer to Groq usage.
        state.num_llm_calls -= 1 
        
        return state
