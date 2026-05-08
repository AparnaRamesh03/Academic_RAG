from __future__ import annotations

from typing import Dict, List, Tuple
import math

try:
    from .config import (
        STATE_FEATURE_VERSION,
        MAX_DOC_COUNT_CLIP,
        MAX_LATENCY_CLIP,
        MAX_TEXT_LENGTH_CLIP,
    )
except ImportError:
    from config import (
        STATE_FEATURE_VERSION,
        MAX_DOC_COUNT_CLIP,
        MAX_LATENCY_CLIP,
        MAX_TEXT_LENGTH_CLIP,
    )


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_count(value: int, max_clip: int = MAX_DOC_COUNT_CLIP) -> float:
    value = max(0, min(value, max_clip))
    return float(value) / float(max_clip) if max_clip > 0 else 0.0


def _normalize_latency(value: float) -> float:
    value = _clip(value, 0.0, MAX_LATENCY_CLIP)
    return value / MAX_LATENCY_CLIP if MAX_LATENCY_CLIP > 0 else 0.0


def _normalize_text_len(value: int) -> float:
    value = max(0, min(value, MAX_TEXT_LENGTH_CLIP))
    return float(value) / float(MAX_TEXT_LENGTH_CLIP) if MAX_TEXT_LENGTH_CLIP > 0 else 0.0


def _top_score(docs: List[Dict], key: str) -> float:
    if not docs:
        return 0.0

    values = []
    for doc in docs:
        raw = doc.get(key, None)
        if raw is None:
            raw = (doc.get("metadata", {}) or {}).get(key, None)
        values.append(_safe_float(raw, 0.0))

    if not values:
        return 0.0
    return max(values)


def _mean_top_k_score(docs: List[Dict], key: str, k: int = 3) -> float:
    if not docs:
        return 0.0

    values = []
    for doc in docs:
        raw = doc.get(key, None)
        if raw is None:
            raw = (doc.get("metadata", {}) or {}).get(key, None)
        values.append(_safe_float(raw, 0.0))

    values.sort(reverse=True)
    topk = values[:k]
    if not topk:
        return 0.0
    return sum(topk) / len(topk)


def _count_unique_sources(docs: List[Dict]) -> int:
    sources = set()
    for doc in docs:
        meta = doc.get("metadata", {}) or {}
        source = meta.get("source_file", None)
        if source:
            sources.add(str(source))
    return len(sources)


def _claim_counts(claim_verification: List[Dict]) -> Tuple[int, int]:
    supported = 0
    unsupported = 0

    for item in claim_verification or []:
        if bool(item.get("supported", False)):
            supported += 1
        else:
            unsupported += 1

    return supported, unsupported


def _verification_outcome_flags(state: Dict) -> Dict[str, float]:
    outcome = str(state.get("verification_outcome", "") or "").strip().lower()

    return {
        "verification_pass": 1.0 if outcome == "pass" else 0.0,
        "verification_grounded_incomplete": 1.0 if outcome == "grounded_incomplete" else 0.0,
        "verification_unsupported": 1.0 if outcome == "unsupported" else 0.0,
        "verification_needs_revision": 1.0 if outcome == "needs_revision" else 0.0,
    }


def _evidence_gap_flags(state: Dict) -> Dict[str, float]:
    gap = str(state.get("evidence_gap_reason", "") or "").strip().lower()

    return {
        "gap_missing_target_source_coverage": 1.0 if gap == "missing_target_source_coverage" else 0.0,
        "gap_mixed_domain": 1.0 if gap == "underspecified_mixed_domain_retrieval" else 0.0,
        "gap_other": 1.0 if gap not in {"", "missing_target_source_coverage", "underspecified_mixed_domain_retrieval"} else 0.0,
    }


def encode_state(state: Dict) -> Dict[str, float]:
    """
    Convert the current graph state into a compact fixed numeric feature dict.
    This is the state seen by the RL controller.
    """

    retrieved_docs = state.get("retrieved_docs", []) or []
    candidate_docs = state.get("candidate_docs", []) or []
    graded_docs = state.get("graded_docs", []) or []
    claim_verification = state.get("claim_verification", []) or []
    generation = str(state.get("generation", "") or "")

    supported_claims, unsupported_claims = _claim_counts(claim_verification)

    features: Dict[str, float] = {
        # ---- controller progress ----
        "step_count_norm": _normalize_count(_safe_int(state.get("step_count", 0)), max_clip=20),
        "crag_retries_norm": _normalize_count(_safe_int(state.get("crag_retries", 0)), max_clip=5),
        "verify_retries_norm": _normalize_count(_safe_int(state.get("verify_retries", 0)), max_clip=5),
        "latency_norm": _normalize_latency(_safe_float(state.get("latency_so_far", 0.0), 0.0)),

        # ---- binary workflow state ----
        "has_retrieval": 1.0 if len(retrieved_docs) > 0 else 0.0,
        "has_candidates": 1.0 if len(candidate_docs) > 0 else 0.0,
        "has_graded_docs": 1.0 if len(graded_docs) > 0 else 0.0,
        "has_answer": 1.0 if len(generation.strip()) > 0 else 0.0,
        "done_flag": 1.0 if bool(state.get("done", False)) else 0.0,
        "citations_pass_flag": 1.0 if bool(state.get("citations_pass", False)) else 0.0,
        "mixed_domain_flag": 1.0 if bool(state.get("mixed_domain_evidence", False)) else 0.0,

        # ---- document counts ----
        "retrieved_doc_count_norm": _normalize_count(len(retrieved_docs)),
        "candidate_doc_count_norm": _normalize_count(len(candidate_docs)),
        "graded_doc_count_norm": _normalize_count(len(graded_docs)),

        # ---- source diversity ----
        "retrieved_unique_sources_norm": _normalize_count(_count_unique_sources(retrieved_docs), max_clip=10),
        "candidate_unique_sources_norm": _normalize_count(_count_unique_sources(candidate_docs), max_clip=10),
        "graded_unique_sources_norm": _normalize_count(_count_unique_sources(graded_docs), max_clip=10),

        # ---- retrieval quality ----
        "top_retrieval_score": _top_score(retrieved_docs, "score"),
        "mean_top3_retrieval_score": _mean_top_k_score(retrieved_docs, "score", k=3),
        "top_rerank_score": _top_score(retrieved_docs, "rerank_score"),
        "mean_top3_rerank_score": _mean_top_k_score(retrieved_docs, "rerank_score", k=3),

        # ---- answer / verification signals ----
        "answer_length_norm": _normalize_text_len(len(generation)),
        "supported_claims_norm": _normalize_count(supported_claims, max_clip=10),
        "unsupported_claims_norm": _normalize_count(unsupported_claims, max_clip=10),

        # ---- confidence ----
        "controller_confidence": _clip(_safe_float(state.get("confidence", 0.0), 0.0), 0.0, 1.0),
    }

    features.update(_verification_outcome_flags(state))
    features.update(_evidence_gap_flags(state))

    return features


def get_feature_names() -> List[str]:
    return list(encode_state({}).keys())


def encode_state_vector(state: Dict) -> List[float]:
    features = encode_state(state)
    return [features[name] for name in features.keys()]


def summarize_encoded_state(state: Dict) -> Dict:
    features = encode_state(state)
    return {
        "feature_version": STATE_FEATURE_VERSION,
        "feature_count": len(features),
        "feature_names": list(features.keys()),
        "feature_values": features,
    }


if __name__ == "__main__":
    example_state = {
        "step_count": 2,
        "crag_retries": 1,
        "verify_retries": 0,
        "latency_so_far": 14.2,
        "retrieved_docs": [
            {
                "score": 0.83,
                "rerank_score": 0.29,
                "metadata": {"source_file": "AttentionIsAllYouNeed.pdf"},
            },
            {
                "score": 0.61,
                "rerank_score": 0.11,
                "metadata": {"source_file": "TabNet.pdf"},
            },
        ],
        "candidate_docs": [],
        "graded_docs": [],
        "generation": "",
        "citations_pass": False,
        "mixed_domain_evidence": True,
        "verification_outcome": "",
        "evidence_gap_reason": "underspecified_mixed_domain_retrieval",
        "claim_verification": [],
        "confidence": 0.45,
        "done": False,
    }

    summary = summarize_encoded_state(example_state)
    print("Feature version:", summary["feature_version"])
    print("Feature count:", summary["feature_count"])
    print("Feature names:")
    for name in summary["feature_names"]:
        print(" -", name)

    print("\nEncoded values:")
    for k, v in summary["feature_values"].items():
        print(f"{k}: {v:.4f}")