import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel

from state import GraphState

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "academic_papers"

DENSE_EMBED_MODEL = "BAAI/bge-m3"
SPARSE_EMBED_MODEL = "Qdrant/bm25"

# Retrieval settings
DENSE_PREFETCH_LIMIT = 10
SPARSE_PREFETCH_LIMIT = 10
FINAL_FUSION_LIMIT = 15

# NOTE:
# This keeps your current score-based split logic unchanged.
# We can improve that later by removing the brittle hard threshold.
CANDIDATE_SCORE_THRESHOLD = 0.4

# ── Clients / Models ────────────────────────────────────────────────────────
client = QdrantClient(url=QDRANT_URL)

# Dense query encoder
dense_model = BGEM3FlagModel(
    DENSE_EMBED_MODEL,
    use_fp16=False,   # safer default
)

# Sparse query encoder
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


def retrieve_and_split(state: GraphState):
    """
    Node 1: Hybrid Retrieval & Splitter

    Steps:
        1. Dense query embedding with BGE-M3
        2. Sparse query embedding with Qdrant/bm25
        3. Qdrant hybrid retrieval using Reciprocal Rank Fusion (RRF)
        4. Split results into:
            - candidate_docs
            - weak_signal_docs

    Current split logic:
        - score > CANDIDATE_SCORE_THRESHOLD => candidate_docs
        - otherwise => weak_signal_docs
    """
    query = state["search_query"]
    print(f"\n[Node 1] Retrieving context for query: '{query}'")

    # 1. Dense and Sparse Query Embeddings
    dense_vec = _get_dense_query_embedding(query)
    sparse_vec = _get_sparse_query_embedding(query)

    # 2. Qdrant Hybrid Search (RRF)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
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

    candidate_docs = []
    weak_signal_docs = []

    # 3. Splitting Logic
    print("[Node 1] Evaluating RRF Scores:")
    for idx, point in enumerate(results.points):
        doc = {
            "text": point.payload.get("text", ""),
            "metadata": point.payload,
            "score": point.score,
        }

        print(f"  -> Doc {idx + 1}: score={point.score:.4f}")

        if point.score > CANDIDATE_SCORE_THRESHOLD:
            candidate_docs.append(doc)
        else:
            weak_signal_docs.append(doc)

    print(
        f"[Node 1] Retrieved {len(results.points)} docs "
        f"({len(candidate_docs)} candidates, {len(weak_signal_docs)} weak signals)."
    )

    return {
        "candidate_docs": candidate_docs,
        "weak_signal_docs": weak_signal_docs,
    }