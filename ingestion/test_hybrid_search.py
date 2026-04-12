"""
test_hybrid_search.py — Interactive Hybrid Search Test Tool

Tests both dense-only, sparse-only, and hybrid (RRF) retrieval from Qdrant.
Run from the ingestion/ directory with the venv activated.

Usage:
    python test_hybrid_search.py
"""

import ollama
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "academic_papers"
DENSE_MODEL      = "nomic-embed-text"
SPARSE_MODEL     = "prithivida/Splade_PP_en_v1"
TOP_K            = 5  # number of results to return

# ── Init ──────────────────────────────────────────────────────────────────────
print("[*] Initialising clients...")
client = QdrantClient(url=QDRANT_URL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
print("[✓] Ready.\n")


def embed_query(query: str):
    """Generate dense and sparse embeddings for a query."""
    dense = ollama.embeddings(model=DENSE_MODEL, prompt=query)["embedding"]
    sparse = list(sparse_model.embed([query]))[0]
    return dense, sparse


def hybrid_search(query: str):
    """
    Hybrid search using Reciprocal Rank Fusion (RRF) of dense + sparse results.
    This is the recommended approach for Qdrant hybrid retrieval.
    """
    dense_vec, sparse_vec = embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            # Dense retrieval — semantic similarity
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=TOP_K * 2,  # fetch extra candidates for RRF fusion
            ),
            # Sparse retrieval — keyword/BM25-style match
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=TOP_K * 2,
            ),
        ],
        # Fuse both result sets using Reciprocal Rank Fusion
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=TOP_K,
        with_payload=True,
    )
    return results.points


def dense_only_search(query: str):
    """Dense-only semantic search (for comparison)."""
    dense_vec, _ = embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vec,
        using="dense",
        limit=TOP_K,
        with_payload=True,
    )
    return results.points


def sparse_only_search(query: str):
    """Sparse-only keyword search (for comparison)."""
    _, sparse_vec = embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist(),
        ),
        using="sparse",
        limit=TOP_K,
        with_payload=True,
    )
    return results.points


def print_results(results, mode: str):
    """Print retrieved chunks in a readable format."""
    print(f"\n{'='*65}")
    print(f"  [{mode}] — Top {len(results)} Results")
    print(f"{'='*65}")
    for i, point in enumerate(results, 1):
        p = point.payload
        print(f"\n  [{i}] Score: {point.score:.4f}")
        print(f"       Source:  {p.get('source_file', 'N/A')}")
        print(f"       Section: {p.get('section_header', 'N/A')}")
        print(f"       Page:    {p.get('page_number', 'N/A')}")
        print(f"       Type:    {p.get('content_type', 'text')}")
        print(f"       Text:    {p.get('text', '')[:200]}...")
    print()


# ── Interactive Loop ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  Qdrant Hybrid Search Test — Academic RAG System")
    print(f"  Collection: '{COLLECTION_NAME}' | Top-K: {TOP_K}")
    print("="*65)
    print("\nModes:")
    print("  [1] Hybrid (RRF — recommended)")
    print("  [2] Dense only (semantic)")
    print("  [3] Sparse only (keyword)")
    print("  [q] Quit\n")

    while True:
        mode_input = input("Select mode (1/2/3/q): ").strip()
        if mode_input == "q":
            print("Goodbye!")
            break
        if mode_input not in ("1", "2", "3"):
            print("Invalid choice. Try again.")
            continue

        query = input("Enter your query: ").strip()
        if not query:
            continue

        print(f"\n[*] Searching for: \"{query}\"")

        if mode_input == "1":
            results = hybrid_search(query)
            print_results(results, "HYBRID (RRF)")
        elif mode_input == "2":
            results = dense_only_search(query)
            print_results(results, "DENSE ONLY")
        else:
            results = sparse_only_search(query)
            print_results(results, "SPARSE ONLY")
