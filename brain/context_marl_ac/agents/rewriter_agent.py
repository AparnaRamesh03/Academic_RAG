"""
brain/context_marl_ac/agents/rewriter_agent.py
----------------------------------------------
Query rewriting agent.

Used only in recovery:
    verifier FAIL -> request_rewrite -> rewriter -> retriever
"""

from typing import List

from context_marl_ac.agents.base_agent import BaseAgent
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.adapters.llm_adapter import rewrite_query


PROTECTED_TERMS = [
    "BERT",
    "Transformer",
    "Attention Is All You Need",
    "TabNet",
    "ResNet",
    "VGG",
    "RAG",
    "Naive RAG",
    "Advanced RAG",
    "Modular RAG",
    "Kachru",
    "Norway",
    "English",
    "WMT 2014",
    "BLEU",
]


def _terms_present(text: str) -> List[str]:
    found = []
    lowered = (text or "").lower()

    for term in PROTECTED_TERMS:
        if term.lower() in lowered:
            found.append(term)

    return found


def _infer_required_terms(query: str) -> List[str]:
    q = (query or "").lower()
    inferred = []

    if (
        "masked language modeling" in q
        or "masked language model" in q
        or "mlm" in q
        or "next sentence prediction" in q
        or "nsp" in q
        or "bidirectional pre-training" in q
        or "bidirectional transformer" in q
    ):
        inferred.extend(
            [
                "BERT",
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
                or "recurrence" in q
                or "machine translation" in q
                or "english-to-german" in q
                or "english-to-french" in q
                or "wmt" in q
            )
        )
    ):
        inferred.extend(
            [
                "Attention Is All You Need",
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
        inferred.extend(
            [
                "TabNet",
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
        inferred.extend(
            [
                "ResNet",
                "residual network",
                "deep residual learning",
            ]
        )

    if "vgg" in q:
        inferred.append("VGG")

    if (
        "rag survey" in q
        or "naive rag" in q
        or "advanced rag" in q
        or "modular rag" in q
        or "retrieval augmented generation survey" in q
    ):
        inferred.extend(
            [
                "RAG survey",
                "Naive RAG",
                "Advanced RAG",
                "Modular RAG",
            ]
        )

    deduped = []
    seen = set()

    for term in inferred:
        key = term.lower()
        if key not in seen:
            deduped.append(term)
            seen.add(key)

    return deduped


def _append_missing_terms(original_query: str, rewritten_query: str) -> str:
    original_terms = _terms_present(original_query)
    inferred_terms = _infer_required_terms(original_query)

    required_terms = []
    seen = set()

    for term in original_terms + inferred_terms:
        key = term.lower()
        if key not in seen:
            required_terms.append(term)
            seen.add(key)

    rewritten = (rewritten_query or "").strip()
    rewritten_lower = rewritten.lower()

    missing = [
        term for term in required_terms
        if term.lower() not in rewritten_lower
    ]

    if not missing:
        return rewritten

    return rewritten + " " + " ".join(missing)


def _clear_after_rewrite(state: ContextState) -> None:
    state.retrieved_chunks = []
    state.retrieval_scores = []
    state.graded_chunks = []
    state.selected_evidence = []
    state.generated_answer = ""
    state.citation_candidates = []
    state.verification_result = {}
    state.unsupported_claims = []
    state.citation_support_rate = 0.0
    state.grader_output = {}


class RewriterAgent(BaseAgent):
    def __init__(self):
        super().__init__("rewriter")

    def act(self, state: ContextState, action_name: str) -> ContextState:
        if action_name == "no_rewrite":
            state.record_action(self.name, action_name)
            state.update_latency()
            return state

        base_query = state.original_query or state.user_query

        rewritten, tokens = rewrite_query(base_query, mode=action_name)
        rewritten = _append_missing_terms(base_query, rewritten)

        state.rewritten_query = rewritten
        state.user_query = rewritten
        state.token_usage += tokens

        _clear_after_rewrite(state)

        self.log_action(state, action_name)
        return state