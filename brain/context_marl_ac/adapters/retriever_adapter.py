"""
brain/context_marl_ac/adapters/retriever_adapter.py
-----------------------------------------------------
Wraps brain/retriever_shared.py to expose retrieval functions
needed by RetrieverAgent.

Exposed API
-----------
    retrieve_hybrid(query, top_k)          – hybrid RRF (dense + sparse)
    retrieve_dense(query, top_k)           – dense-only (BGE-M3)
    retrieve_sparse(query, top_k)          – sparse-only (BM25)
    retrieve_hybrid_rerank(query, top_k)   – hybrid + CrossEncoder rerank
    retrieve_more(query, current_chunks, top_k) – hybrid, dedup existing

Each function returns List[dict] with keys:
    text, metadata, score
where metadata has: source_file, page_number, section_header, content_type.

In dry-run mode (config.DRY_RUN=True) all functions return small dummy
chunks without touching Qdrant or loading embedding models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Ensure brain/ is on sys.path so we can import shared brain utilities.
# adapters/ → context_marl_ac/ → brain/
# ---------------------------------------------------------------------------
_BRAIN_ROOT = Path(__file__).resolve().parents[2]
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ---------------------------------------------------------------------------
# Config (DRY_RUN flag lives here)
# ---------------------------------------------------------------------------
try:
    import context_marl_ac.config as cfg
except ImportError:
    # Fallback when running as a standalone script
    _MARL_ROOT = Path(__file__).resolve().parents[1]
    if str(_MARL_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(_MARL_ROOT.parent))
    import context_marl_ac.config as cfg


# ---------------------------------------------------------------------------
# Dry-run stubs
# ---------------------------------------------------------------------------
_DRY_RUN_CHUNKS: List[Dict[str, Any]] = [
    {
        "text": (
            "Transformer models introduced by Vaswani et al. rely on "
            "self-attention mechanisms to process sequences in parallel, "
            "achieving state-of-the-art results on machine translation tasks."
        ),
        "metadata": {
            "source_file": "AttentionIsAllYouNeed.pdf",
            "page_number": 3,
            "section_header": "Model Architecture",
            "content_type": "text",
        },
        "score": 0.91,
    },
    {
        "text": (
            "BERT uses a masked language modelling objective to pre-train "
            "deep bidirectional representations from unlabelled text, enabling "
            "fine-tuning across a wide range of NLP tasks."
        ),
        "metadata": {
            "source_file": "BERT_Devlin2019.pdf",
            "page_number": 1,
            "section_header": "Introduction",
            "content_type": "text",
        },
        "score": 0.85,
    },
    {
        "text": (
            "Retrieval-Augmented Generation (RAG) combines parametric memory "
            "stored in model weights with non-parametric memory accessed via "
            "a dense retriever over a document index."
        ),
        "metadata": {
            "source_file": "RAG_Lewis2020.pdf",
            "page_number": 2,
            "section_header": "Background",
            "content_type": "text",
        },
        "score": 0.79,
    },
]


def _dry_run_chunks(top_k: int) -> List[Dict[str, Any]]:
    return _DRY_RUN_CHUNKS[:top_k]


# ---------------------------------------------------------------------------
# Real retriever imports (lazy, so they are only loaded when cfg.DRY_RUN=False)
# ---------------------------------------------------------------------------
_retriever_loaded = False
_retrieve_docs_fn = None
_dense_embed_fn = None
_sparse_embed_fn = None
_qdrant_client = None
_qdrant_models = None
_reranker_fn = None

# Qdrant config values
_COLLECTION_NAME = None
_DENSE_PREFETCH_LIMIT = 20
_SPARSE_PREFETCH_LIMIT = 20
_FINAL_FUSION_LIMIT = 20


def _ensure_retriever_loaded() -> None:
    global _retriever_loaded, _retrieve_docs_fn, _dense_embed_fn
    global _sparse_embed_fn, _qdrant_client, _qdrant_models
    global _COLLECTION_NAME, _DENSE_PREFETCH_LIMIT, _SPARSE_PREFETCH_LIMIT
    global _FINAL_FUSION_LIMIT, _reranker_fn

    if _retriever_loaded:
        return

    try:
        import retriever_shared as _rs
        from qdrant_client import models as _qdrant_models_mod
        from qdrant_config import (
            COLLECTION_NAME,
            DENSE_PREFETCH_LIMIT,
            SPARSE_PREFETCH_LIMIT,
            FINAL_FUSION_LIMIT,
        )

        _retrieve_docs_fn = _rs.retrieve_docs
        _dense_embed_fn   = _rs._get_dense_query_embedding   # noqa: SLF001
        _sparse_embed_fn  = _rs._get_sparse_query_embedding  # noqa: SLF001
        _qdrant_client    = _rs.client
        _qdrant_models    = _qdrant_models_mod

        _COLLECTION_NAME       = COLLECTION_NAME
        _DENSE_PREFETCH_LIMIT  = DENSE_PREFETCH_LIMIT
        _SPARSE_PREFETCH_LIMIT = SPARSE_PREFETCH_LIMIT
        _FINAL_FUSION_LIMIT    = FINAL_FUSION_LIMIT

        _retriever_loaded = True

    except Exception as exc:
        raise ImportError(
            f"[retriever_adapter] Failed to load brain/retriever_shared.py.\n"
            f"Make sure Qdrant is running and brain/ is on sys.path.\n"
            f"Original error: {exc}"
        ) from exc


def _ensure_reranker_loaded() -> None:
    global _reranker_fn
    if _reranker_fn is not None:
        return
    try:
        _FINAL_ARCH = _BRAIN_ROOT / "final_arch"
        if str(_FINAL_ARCH) not in sys.path:
            sys.path.insert(0, str(_FINAL_ARCH))
        from reranker_shared import rerank_retrieved_docs
        _reranker_fn = rerank_retrieved_docs
    except Exception as exc:
        raise ImportError(
            f"[retriever_adapter] Failed to load brain/final_arch/reranker_shared.py.\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_hybrid(query: str, top_k: int = cfg.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Hybrid RRF retrieval: BGE-M3 dense + BM25 sparse fused via Qdrant RRF.
    Returns up to top_k chunks sorted by RRF score (highest first).
    """
    if cfg.DRY_RUN:
        return _dry_run_chunks(top_k)
    _ensure_retriever_loaded()
    docs = _retrieve_docs_fn(query)
    return docs[:top_k]


def retrieve_dense(query: str, top_k: int = cfg.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Dense-only retrieval using BGE-M3 embeddings via Qdrant ANN search.
    """
    if cfg.DRY_RUN:
        return _dry_run_chunks(top_k)
    _ensure_retriever_loaded()

    dense_vec = _dense_embed_fn(query)
    results = _qdrant_client.query_points(
        collection_name=_COLLECTION_NAME,
        query=dense_vec,
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": p.payload.get("text", ""),
            "metadata": p.payload,
            "score": p.score,
        }
        for p in results.points
    ]


def retrieve_sparse(query: str, top_k: int = cfg.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Sparse-only retrieval using BM25 (FastEmbed) via Qdrant sparse vector search.
    """
    if cfg.DRY_RUN:
        return _dry_run_chunks(top_k)
    _ensure_retriever_loaded()

    sparse_vec = _sparse_embed_fn(query)
    results = _qdrant_client.query_points(
        collection_name=_COLLECTION_NAME,
        query=_qdrant_models.SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist(),
        ),
        using="sparse",
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": p.payload.get("text", ""),
            "metadata": p.payload,
            "score": p.score,
        }
        for p in results.points
    ]


def retrieve_hybrid_rerank(query: str, top_k: int = cfg.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Hybrid RRF retrieval followed by CrossEncoder reranking.
    Fetches RERANK_INPUT_TOP_K candidates, reranks, returns top_k.
    """
    if cfg.DRY_RUN:
        return _dry_run_chunks(top_k)
    _ensure_retriever_loaded()
    _ensure_reranker_loaded()

    # Fetch a wider candidate set for reranking
    from context_marl_ac.config import RERANK_INPUT_TOP_K, RERANK_OUTPUT_TOP_K
    candidates = _retrieve_docs_fn(query)[:RERANK_INPUT_TOP_K]

    # Build a minimal GraphState-compatible dict for reranker_shared
    mock_state: Dict[str, Any] = {
        "original_query":   query,
        "search_query":     query,
        "retrieved_docs":   candidates,
    }
    updated = _reranker_fn(
        mock_state,
        input_top_k=RERANK_INPUT_TOP_K,
        output_top_k=min(RERANK_OUTPUT_TOP_K, top_k),
    )
    reranked = updated.get("retrieved_docs", candidates)
    return reranked[:top_k]


def retrieve_more(
    query: str,
    current_chunks: List[Dict[str, Any]],
    top_k: int = cfg.DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve additional chunks not already present in current_chunks.
    Uses text content as the deduplication key.
    """
    if cfg.DRY_RUN:
        # Return a slightly different dummy chunk in dry-run
        extra = {
            "text": (
                "GPT-4 is a large multimodal model that accepts image and text "
                "inputs and produces text outputs, achieving human-level performance "
                "on various professional and academic benchmarks."
            ),
            "metadata": {
                "source_file": "GPT4_Technical_Report.pdf",
                "page_number": 1,
                "section_header": "Abstract",
                "content_type": "text",
            },
            "score": 0.72,
        }
        return [extra][:top_k]

    _ensure_retriever_loaded()

    existing_texts = {c.get("text", "").strip() for c in current_chunks}
    candidates = _retrieve_docs_fn(query)
    new_chunks = [
        doc for doc in candidates
        if doc.get("text", "").strip() not in existing_texts
    ]
    return new_chunks[:top_k]
