"""
brain/context_marl_ac/agents/retriever_agent.py
-----------------------------------------------
Agent responsible for interacting with the retrieval system.
"""

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.retriever_adapter import (
    retrieve_hybrid, retrieve_dense, retrieve_sparse, 
    retrieve_hybrid_rerank, retrieve_more
)
from context_marl_ac.context_engineering.evidence_pack_builder import build_evidence_pack

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

        # Update state for standard retrieval actions
        state.retrieved_chunks = chunks
        state.retrieval_scores = [c.get("score", 0.0) for c in chunks]
        state.selected_evidence = build_evidence_pack(chunks)
        
        self.log_action(state, action_name)
        # Retrieval usually doesn't use LLM unless it's an embedding model, 
        # but in our architecture, LLM calls refer to Groq usage.
        state.num_llm_calls -= 1 
        
        return state
