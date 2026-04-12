"""
indexer.py — Qdrant Hybrid Indexing (Dense + Sparse)

Chosen stack:
    1. Dense Embeddings: local BGE-M3
    2. Sparse Embeddings: Qdrant/bm25 via FastEmbed
    3. Hybrid Qdrant collection: dense + sparse
    4. Old chunks for the same source_file are replaced safely
    5. New chunks are upserted with both vector types

Important:
    - If you are migrating from nomic-embed-text to BGE-M3,
      recreate the Qdrant collection before re-indexing.
"""

import os
import uuid
import traceback
from typing import List, Dict, Any

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel

# ── Configuration ───────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "academic_papers"

DENSE_MODEL = "BAAI/bge-m3"
SPARSE_MODEL = "Qdrant/bm25"

# BGE-M3 dense vector size
DENSE_VECTOR_SIZE = 1024

# ── Clients / Models ────────────────────────────────────────────────────────
_client = QdrantClient(url=QDRANT_URL)

_dense_embedding_model = None
_sparse_embedding_model = None


def _ensure_models_loaded() -> None:
    """
    Lazy-load embedding models so import-time failures do not break the watcher.
    """
    global _dense_embedding_model, _sparse_embedding_model

    if _dense_embedding_model is None:
        try:
            print(f"[*] Loading dense model: {DENSE_MODEL} ...")
            _dense_embedding_model = BGEM3FlagModel(
                DENSE_MODEL,
                use_fp16=False,   # safer default; set True only if your hardware supports it
            )
            print("[✓] Dense model loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load dense model '{DENSE_MODEL}': {e}")

    if _sparse_embedding_model is None:
        try:
            print(f"[*] Loading sparse model: {SPARSE_MODEL} ...")
            _sparse_embedding_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
            print("[✓] Sparse model loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load sparse model '{SPARSE_MODEL}': {e}")


def _setup_collection() -> None:
    """
    Ensure the Qdrant collection exists with the correct hybrid config.

    If the collection already exists, validate that its schema matches the
    current indexing setup:
        - dense vector name: "dense"
        - dense size: 1024
        - dense distance: COSINE
        - sparse vector name: "sparse"
        - sparse modifier: IDF
    """
    try:
        if not _client.collection_exists(COLLECTION_NAME):
            print(f"[*] Creating Qdrant collection: '{COLLECTION_NAME}'...")

            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=DENSE_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                        index=models.SparseIndexParams(on_disk=True),
                    )
                },
            )

            print(f"[✓] Collection '{COLLECTION_NAME}' created.")
            return

        # Collection exists -> validate schema
        info = _client.get_collection(COLLECTION_NAME)
        config = info.config.params

        # ---- Validate dense vector config ----
        vectors_cfg = config.vectors

        if not isinstance(vectors_cfg, dict) or "dense" not in vectors_cfg:
            raise RuntimeError(
                "Existing collection is missing named dense vector 'dense'."
            )

        dense_cfg = vectors_cfg["dense"]

        if dense_cfg.size != DENSE_VECTOR_SIZE:
            raise RuntimeError(
                f"Existing collection dense size mismatch: "
                f"expected {DENSE_VECTOR_SIZE}, found {dense_cfg.size}."
            )

        if dense_cfg.distance != models.Distance.COSINE:
            raise RuntimeError(
                f"Existing collection dense distance mismatch: "
                f"expected COSINE, found {dense_cfg.distance}."
            )

        # ---- Validate sparse vector config ----
        sparse_cfg_map = getattr(config, "sparse_vectors", None)

        if not sparse_cfg_map or "sparse" not in sparse_cfg_map:
            raise RuntimeError(
                "Existing collection is missing named sparse vector 'sparse'."
            )

        sparse_cfg = sparse_cfg_map["sparse"]

        if sparse_cfg.modifier != models.Modifier.IDF:
            raise RuntimeError(
                f"Existing collection sparse modifier mismatch: "
                f"expected IDF, found {sparse_cfg.modifier}."
            )

        print(f"[✓] Existing collection '{COLLECTION_NAME}' schema is valid.")

    except Exception as e:
        raise RuntimeError(
            "Qdrant collection setup/validation failed. "
            "If you recently changed embedding models or vector config, "
            "delete and recreate the collection. "
            f"Details: {e}"
        )

def _build_source_filter(source_file: str) -> models.Filter:
    """
    Build a Qdrant filter for all chunks belonging to one source file.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key="source_file",
                match=models.MatchValue(value=source_file),
            )
        ]
    )


def _get_existing_point_ids_for_source(source_file: str) -> List[Any]:
    """
    Fetch all existing Qdrant point IDs for one source file.

    This is used so we can:
        1. upsert new points first
        2. delete only stale old point IDs afterward

    That is safer than deleting everything first.
    """
    point_ids: List[Any] = []
    next_page_offset = None

    while True:
        records, next_page_offset = _client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=_build_source_filter(source_file),
            limit=256,
            offset=next_page_offset,
            with_payload=False,
            with_vectors=False,
        )

        for rec in records:
            point_ids.append(rec.id)

        if next_page_offset is None:
            break

    return point_ids


def _delete_point_ids(point_ids: List[Any]) -> None:
    """
    Delete a list of point IDs from Qdrant.
    """
    if not point_ids:
        return

    _client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids),
        wait=True,
    )


def _get_dense_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate dense embeddings using local BGE-M3.
    """
    if not texts:
        return []

    _ensure_models_loaded()

    output = _dense_embedding_model.encode(
        texts,
        batch_size=8,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    dense_vecs = output["dense_vecs"]

    return [
        vec.tolist() if hasattr(vec, "tolist") else list(vec)
        for vec in dense_vecs
    ]


def _get_sparse_embeddings(texts: List[str]):
    """
    Generate sparse embeddings using FastEmbed BM25.
    """
    if not texts:
        return []

    _ensure_models_loaded()
    return list(_sparse_embedding_model.embed(texts))


def index_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Perform hybrid indexing on a list of chunks.

    Expected input:
        chunks = [
            {
                "text": "...",
                "embedding_text": "...",   # optional; preferred for embeddings
                "metadata": {
                    "source_file": "...",
                    "chunk_index": 0,
                    ...
                }
            },
            ...
        ]
    """
    if not chunks:
        print("[!] No chunks to index.")
        return

    # Validate source file consistency first
    try:
        source_files = {chunk["metadata"]["source_file"] for chunk in chunks}
    except Exception as e:
        raise RuntimeError(f"Chunk validation failed: missing metadata/source_file ({e})")

    if len(source_files) != 1:
        raise ValueError(
            "index_chunks() expects chunks from exactly one source file at a time."
        )

    source_file = next(iter(source_files))

    # Keep payload text and embedding text separate
    try:
        texts = [chunk["text"] for chunk in chunks]
        embedding_texts = [chunk.get("embedding_text", chunk["text"]) for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
    except Exception as e:
        raise RuntimeError(f"Chunk preparation failed for '{source_file}': {e}")

    # 1. Ensure collection exists
    _setup_collection()

    # 2. Read existing point IDs for this source first
    try:
        existing_ids = set(_get_existing_point_ids_for_source(source_file))
        print(f"[*] Found {len(existing_ids)} existing point(s) for '{source_file}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch existing Qdrant point IDs for '{source_file}': {e}")

    # 3. Generate embeddings before deleting anything
    print(f"[*] Generating dense embeddings ({DENSE_MODEL})...")
    try:
        dense_embeddings = _get_dense_embeddings(embedding_texts)
    except Exception as e:
        raise RuntimeError(f"Dense embedding generation failed for '{source_file}': {e}")

    print(f"[*] Generating sparse embeddings ({SPARSE_MODEL})...")
    try:
        sparse_embeddings = _get_sparse_embeddings(embedding_texts)
    except Exception as e:
        raise RuntimeError(f"Sparse embedding generation failed for '{source_file}': {e}")

    # 4. Validate embedding counts
    if not (len(texts) == len(dense_embeddings) == len(sparse_embeddings)):
        raise RuntimeError(
            f"Embedding count mismatch for '{source_file}': "
            f"texts={len(texts)}, dense={len(dense_embeddings)}, sparse={len(sparse_embeddings)}"
        )

    # 5. Build points and identify stale IDs
    points: List[models.PointStruct] = []
    new_ids = set()

    try:
        for text, metadata, dense_vec, sparse_vec in zip(
            texts, metadatas, dense_embeddings, sparse_embeddings
        ):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{metadata['source_file']}_{metadata['chunk_index']}"
                )
            )
            new_ids.add(point_id)

            payload = {
                **metadata,
                "text": text,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    payload=payload,
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                )
            )
    except Exception as e:
        raise RuntimeError(f"Failed building Qdrant points for '{source_file}': {e}")

    stale_ids = list(existing_ids - new_ids)

    # 6. Upsert new points first
    try:
        print(f"[*] Upserting {len(points)} chunk(s) into Qdrant for '{source_file}'...")
        _client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
        print(f"[✓] Upsert completed for '{source_file}'.")
    except Exception as e:
        raise RuntimeError(f"Qdrant upsert failed for '{source_file}': {e}")

    # 7. Delete stale point IDs only after successful upsert
    try:
        if stale_ids:
            print(f"[*] Removing {len(stale_ids)} stale point(s) for '{source_file}'...")
            _delete_point_ids(stale_ids)
            print(f"[✓] Stale points removed for '{source_file}'.")
    except Exception as e:
        # Do not fail the whole indexing after successful upsert.
        # Worst case: old stale chunks remain temporarily.
        print(f"[WARN] Upsert succeeded, but stale-point cleanup failed for '{source_file}': {e}")
        traceback.print_exc()

    print(f"[✓] Successfully indexed {len(points)} chunks for '{source_file}'.")


if __name__ == "__main__":
    print("indexer.py is a utility module. Import and call index_chunks(chunks).")