"""
brain/context_marl_ac/agents/retriever_agent.py
-----------------------------------------------
Retrieval agent.

Fixes:
- Mandatory initial retrieval can still use source/entity query hints.
- Retrieval clears stale grading/generation/verification state.
"""

from typing import List

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.retriever_adapter import (
    retrieve_hybrid,
    retrieve_dense,
    retrieve_sparse,
    retrieve_hybrid_rerank,
    retrieve_more,
)
from context_marl_ac.context_engineering.evidence_pack_builder import build_evidence_pack


def _infer_retrieval_terms(query: str) -> List[str]:
    q = (query or "").lower()
    terms: List[str] = []

    if (
        "masked language modeling" in q
        or "masked language model" in q
        or "mlm" in q
        or "next sentence prediction" in q
        or "nsp" in q
        or "bidirectional pre-training" in q
        or "bidirectional transformer" in q
    ):
        terms.extend(
            [
                "BERT",
                "BERT.pdf",
                "masked language model",
                "next sentence prediction",
                "bidirectional pre-training",
            ]
        )

    if (
        "attention is all you need" in q
        or (
            "transformer" in q
            and (
                "translation" in q
                or "sequence transduction" in q
                or "machine translation" in q
                or "recurrence" in q
                or "wmt" in q
                or "english-to-german" in q
                or "english-to-french" in q
                or "proposes the transformer" in q
                or "propose the transformer" in q
            )
        )
    ):
        terms.extend(
            [
                "Attention Is All You Need",
                "AttentionIsAllYouNeed.pdf",
                "Transformer",
                "sequence transduction",
                "machine translation",
            ]
        )

    if (
        "tabnet" in q
        or (
            "tabular" in q
            and (
                "tree-based" in q
                or "decision tree" in q
                or "feature selection" in q
                or "sequential attention" in q
                or "alternative" in q
            )
        )
    ):
        terms.extend(
            [
                "TabNet",
                "TabNet.pdf",
                "tabular data",
                "sequential attention",
                "instance-wise feature selection",
            ]
        )

    if (
        "resnet" in q
        or "residual network" in q
        or "residual learning" in q
    ):
        terms.extend(
            [
                "ResNet",
                "residual network",
                "deep residual learning",
            ]
        )

    if "vgg" in q:
        terms.append("VGG")

    if (
        "rag survey" in q
        or "naive rag" in q
        or "advanced rag" in q
        or "modular rag" in q
        or "retrieval augmented generation survey" in q
    ):
        terms.extend(
            [
                "RAG survey",
                "Naive RAG",
                "Advanced RAG",
                "Modular RAG",
            ]
        )

    deduped = []
    seen = set()

    for term in terms:
        key = term.lower()
        if key not in seen:
            deduped.append(term)
            seen.add(key)

    return deduped


def _expand_query(query: str) -> str:
    terms = _infer_retrieval_terms(query)

    if not terms:
        return query

    lowered = query.lower()
    missing = [term for term in terms if term.lower() not in lowered]

    if not missing:
        return query

    return query.strip() + " " + " ".join(missing)


def _clear_downstream_state(state: ContextState) -> None:
    state.graded_chunks = []
    state.selected_evidence = []
    state.generated_answer = ""
    state.citation_candidates = []
    state.verification_result = {}
    state.unsupported_claims = []
    state.citation_support_rate = 0.0
    state.grader_output = {}


class RetrieverAgent(BaseAgent):
    def __init__(self):
        super().__init__("retriever")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        base_query = state.rewritten_query or state.original_query or state.user_query
        query = _expand_query(base_query)

        top_k = 8

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

            _clear_downstream_state(state)

            state.retrieved_chunks.extend(chunks)
            state.retrieval_scores = [
                c.get("score", 0.0) for c in state.retrieved_chunks
            ]
            state.selected_evidence = build_evidence_pack(state.retrieved_chunks)

            self.log_action(state, action_name)

            # Retrieval itself is not a Groq LLM call.
            state.num_llm_calls -= 1
            return state

        else:
            raise ValueError(f"Unknown action {action_name} for {self.name}")

        _clear_downstream_state(state)

        state.retrieved_chunks = chunks
        state.retrieval_scores = [c.get("score", 0.0) for c in chunks]
        state.selected_evidence = build_evidence_pack(chunks)

        self.log_action(state, action_name)

        # Retrieval itself is not a Groq LLM call.
        state.num_llm_calls -= 1

        return state