"""
indexer.py — Qdrant Hybrid Indexing (Dense + Sparse)

Pipeline:
    1. Dense Embeddings: local Ollama (nomic-embed-text)
    2. Sparse Embeddings: fastembed (prithivida/Splade_PP_en_v1)
    3. Collection Setup: Hybrid (Cosine + Sparse)
    4. Upsert: Rich metadata + both vectors
"""

import os
import uuid
import ollama
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

# ── Configuration ───────────────────────────────────────────────────────────
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "academic_papers"
DENSE_MODEL = "nomic-embed-text"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

# ── Clients ──────────────────────────────────────────────────────────────────
_client = QdrantClient(url=QDRANT_URL)
_sparse_embedding_model = SparseTextEmbedding(model_name=SPARSE_MODEL)


def _setup_collection():
    """Ensure the Qdrant collection exists with proper hybrid vector configurations."""
    if not _client.collection_exists(COLLECTION_NAME):
        print(f"[*] Creating Qdrant collection: '{COLLECTION_NAME}'...")
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=768,  # nomic-embed-text dimension
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True,
                    )
                )
            },
        )
        print(f"[✓] Collection '{COLLECTION_NAME}' ready.")


def _get_dense_embeddings(texts: list[str]) -> list[list[float]]:
    """Batch embed texts using local Ollama nomic-embed-text."""
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=DENSE_MODEL, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings


def index_chunks(chunks: list[dict]):
    """
    Perform hybrid indexing on a list of chunks.
    
    Args:
        chunks: List of dictionaries, each with 'text' and 'metadata'.
    """
    if not chunks:
        print("[!] No chunks to index.")
        return

    _setup_collection()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"[*] Generating dense embeddings ({DENSE_MODEL})...")
    dense_vecs = _get_dense_embeddings(texts)

    print(f"[*] Generating sparse embeddings ({SPARSE_MODEL})...")
    # fastembed sparse returns a generator of sparse vectors
    sparse_vecs = list(_sparse_embedding_model.embed(texts))

    points = []
    for i, (text, meta, dense, sparse) in enumerate(zip(texts, metadatas, dense_vecs, sparse_vecs)):
        # Generate a stable UUID based on source file and index to prevent duplicate indexing
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{meta['source_file']}_{meta['chunk_index']}"))
        
        # Add original text to payload for retrieval
        payload = {**meta, "text": text}

        points.append(
            models.PointStruct(
                id=point_id,
                payload=payload,
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
            )
        )

    print(f"[*] Upserting {len(points)} points to Qdrant...")
    _client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )
    print(f"[✓] Hybrid indexing complete.")
