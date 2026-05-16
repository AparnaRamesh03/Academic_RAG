import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel

from state_shared import GraphState
from qdrant_config import (
    QDRANT_URL,
    COLLECTION_NAME,
    DENSE_EMBED_MODEL,
    SPARSE_EMBED_MODEL,
    DENSE_PREFETCH_LIMIT,
    SPARSE_PREFETCH_LIMIT,
    FINAL_FUSION_LIMIT,
)

load_dotenv()

client = QdrantClient(url=QDRANT_URL)

dense_model = BGEM3FlagModel(
    DENSE_EMBED_MODEL,
    use_fp16=False,
)

sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBED_MODEL)


def _get_dense_query_embedding(query: str) -> list[float]:
    """
    Generate a dense embedding for the query using local BGE-M3.
    """
    output = dense_model.encode(
        [query],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    dense_vec = output["dense_vecs"][0]
    return dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)


def _get_sparse_query_embedding(query: str):
    """
    Generate a sparse embedding for the query using Qdrant/bm25 via FastEmbed.
    """
    return list(sparse_model.embed([query]))[0]


def retrieve_docs(query: str) -> list[dict]:
    """
    Shared hybrid retrieval:
    dense BGE-M3 + sparse BM25 + Qdrant RRF fusion.

    Returns the full ranked retrieval list in original order.
    """
    # Dynamically resolve collection name to support benchmark switching
    from qdrant_config import COLLECTION_NAME as DEFAULT_COLLECTION
    collection_name = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)

    dense_vec = _get_dense_query_embedding(query)
    sparse_vec = _get_sparse_query_embedding(query)

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=DENSE_PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=SPARSE_PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=FINAL_FUSION_LIMIT,
        with_payload=True,
    )

    retrieved_docs = []
    for point in results.points:
        retrieved_docs.append(
            {
                "text": point.payload.get("text", ""),
                "metadata": point.payload,
                "score": point.score,
            }
        )

    return retrieved_docs


def retrieve_and_store(state: GraphState):
    """
    Shared retriever node for simple baseline or other architectures.

    It only retrieves and stores the full ranked result list.
    Architecture-specific folders can decide later how to use it.
    """
    query = state["search_query"]
    print(f"\n[Shared Retriever] Retrieving context for query: '{query}'")

    retrieved_docs = retrieve_docs(query)

    print(f"[Shared Retriever] Retrieved {len(retrieved_docs)} docs.")
    for idx, doc in enumerate(retrieved_docs[:10]):
        print(f"  -> Doc {idx + 1}: score={doc['score']:.4f}")

    return {
        "retrieved_docs": retrieved_docs
    }