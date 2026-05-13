# reranker_shared.py
import os
from typing import List, Dict, Any

from FlagEmbedding import FlagReranker

from state_shared import GraphState

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

_reranker = None


def _get_reranker() -> FlagReranker:
    """
    Lazy-load the reranker so startup does not fail before first use.
    """
    global _reranker

    if _reranker is None:
        print(f"[*] Loading reranker model: {RERANKER_MODEL} ...")
        _reranker = FlagReranker(
            RERANKER_MODEL,
            use_fp16=False,   # safer for Mac / CPU
        )
        print("[OK] Reranker loaded.")

    return _reranker


def rerank_docs(
    query: str,
    docs: List[Dict[str, Any]],
    input_top_k: int,
    output_top_k: int,
) -> List[Dict[str, Any]]:
    """
    Rerank the first `input_top_k` retrieved docs against the ORIGINAL user question.
    Returns only the top `output_top_k` docs sorted by rerank score.
    """
    if not docs:
        return []

    candidates = docs[:input_top_k]
    if not candidates:
        return []

    pairs = [[query, doc.get("text", "")] for doc in candidates]

    reranker = _get_reranker()
    scores = reranker.compute_score(
        pairs,
        batch_size=8,
        max_length=1024,
    )

    # normalize single-value edge case
    if not isinstance(scores, list):
        scores = [scores]

    rescored_docs = []
    for doc, score in zip(candidates, scores):
        new_doc = dict(doc)
        new_doc["rerank_score"] = float(score)
        rescored_docs.append(new_doc)

    rescored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)
    return rescored_docs[:output_top_k]


def rerank_retrieved_docs(state: GraphState, input_top_k: int, output_top_k: int):
    """
    Generic rerank node helper.
    Uses the ORIGINAL question for final relevance alignment.
    Overwrites retrieved_docs with reranked ordering.
    """
    original_query = state["original_query"]
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        print("[Reranker] No retrieved docs found. Skipping rerank.")
        return {"retrieved_docs": []}

    print(
        f"\n[Reranker] Reranking top {min(len(retrieved_docs), input_top_k)} "
        f"retrieved docs -> keeping top {output_top_k}..."
    )

    reranked_docs = rerank_docs(
        query=original_query,
        docs=retrieved_docs,
        input_top_k=input_top_k,
        output_top_k=output_top_k,
    )

    print(f"[Reranker] Kept {len(reranked_docs)} reranked docs.")
    for idx, doc in enumerate(reranked_docs[:10]):
        print(
            f"  -> Doc {idx + 1}: retrieval_score="
            f"{doc.get('score', 'n/a')} | rerank_score={doc.get('rerank_score', 'n/a'):.4f}"
        )

    return {"retrieved_docs": reranked_docs}