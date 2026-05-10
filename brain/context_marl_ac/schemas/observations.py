"""
brain/context_marl_ac/schemas/observations.py
-----------------------------------------------
Per-agent local observation vectors.

Each agent only sees a small, relevant subset of the global ContextState.
This enforces the MARL constraint that agents have partial observability
(even though the centralized critic sees the full global state).

Design
------
- Each `get_<agent>_observation(state)` function returns a List[float].
- Feature names are also exposed as `<AGENT>_OBS_NAMES` for logging.
- Observation dimensions are kept small and consistent so that actor
  networks have a stable, well-specified input.

All values are normalised to [0, 1] or are binary flags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from context_marl_ac.schemas.context_state import ContextState

# Clip constants (match rl_arch/state_encoder.py conventions)
_MAX_CHUNKS  = 20
_MAX_STEPS   = 10
_MAX_LLM     = 15
_MAX_LATENCY = 60.0   # seconds
_MAX_RETRIES = 5


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _norm(v: float | int, max_val: float) -> float:
    return _clip01(float(v) / max_val) if max_val > 0 else 0.0


def _query_type_id(qt: str) -> float:
    types = ["factual", "conceptual", "comparison",
             "section_specific", "multi_hop", "definition", "summarization"]
    idx = types.index(qt) if qt in types else 0
    return _norm(idx, len(types) - 1)


def _complexity_id(c: str) -> float:
    mapping = {"low": 0.0, "medium": 0.5, "high": 1.0}
    return mapping.get(c, 0.5)


# ════════════════════════════════════════════════════════════════════════════
# Retriever observation
# — what retriever needs to decide how to search
# ════════════════════════════════════════════════════════════════════════════
RETRIEVER_OBS_NAMES: List[str] = [
    "query_type_id",
    "query_complexity_id",
    "requires_multiple_sources",
    "num_retrieved_chunks_norm",
    "top_retrieval_score",
    "mean_retrieval_score",
    "retry_count_norm",
    "num_steps_norm",
    "latency_norm",
    "has_graded_chunks",
]
RETRIEVER_OBS_DIM: int = len(RETRIEVER_OBS_NAMES)


def get_retriever_observation(state: "ContextState") -> List[float]:
    chunks  = state.retrieved_chunks or []
    scores  = [c.get("score", 0.0) for c in chunks]
    top_s   = max(scores) if scores else 0.0
    mean_s  = sum(scores) / len(scores) if scores else 0.0
    return [
        _query_type_id(state.query_type),
        _complexity_id(state.query_complexity),
        1.0 if state.requires_multiple_sources else 0.0,
        _norm(len(chunks), _MAX_CHUNKS),
        _clip01(top_s),
        _clip01(mean_s),
        _norm(state.retry_count, _MAX_RETRIES),
        _norm(state.num_steps, _MAX_STEPS),
        _norm(state.latency_so_far, _MAX_LATENCY),
        1.0 if len(state.graded_chunks) > 0 else 0.0,
    ]


# ════════════════════════════════════════════════════════════════════════════
# Rewriter observation
# — what rewriter needs to decide whether and how to rewrite
# ════════════════════════════════════════════════════════════════════════════
REWRITER_OBS_NAMES: List[str] = [
    "query_type_id",
    "query_complexity_id",
    "num_retrieved_chunks_norm",
    "top_retrieval_score",
    "mean_retrieval_score",
    "num_graded_chunks_norm",
    "graded_relevance_ratio",
    "retry_count_norm",
    "num_steps_norm",
    "rewriter_action_count_norm",
]
REWRITER_OBS_DIM: int = len(REWRITER_OBS_NAMES)


def get_rewriter_observation(state: "ContextState") -> List[float]:
    chunks  = state.retrieved_chunks or []
    graded  = state.graded_chunks or []
    scores  = [c.get("score", 0.0) for c in chunks]
    top_s   = max(scores) if scores else 0.0
    mean_s  = sum(scores) / len(scores) if scores else 0.0
    rel_r   = _norm(len(graded), max(len(chunks), 1))
    rw_cnt  = state.action_count_for("rewriter")
    return [
        _query_type_id(state.query_type),
        _complexity_id(state.query_complexity),
        _norm(len(chunks), _MAX_CHUNKS),
        _clip01(top_s),
        _clip01(mean_s),
        _norm(len(graded), _MAX_CHUNKS),
        rel_r,
        _norm(state.retry_count, _MAX_RETRIES),
        _norm(state.num_steps, _MAX_STEPS),
        _norm(rw_cnt, _MAX_RETRIES),
    ]


# ════════════════════════════════════════════════════════════════════════════
# Grader observation
# — what grader needs to decide how tightly to filter
# ════════════════════════════════════════════════════════════════════════════
GRADER_OBS_NAMES: List[str] = [
    "query_type_id",
    "requires_strict_citation",
    "num_retrieved_chunks_norm",
    "top_retrieval_score",
    "mean_retrieval_score",
    "num_graded_chunks_norm",
    "selected_evidence_count_norm",
    "retry_count_norm",
    "num_steps_norm",
    "has_answer",
]
GRADER_OBS_DIM: int = len(GRADER_OBS_NAMES)


def get_grader_observation(state: "ContextState") -> List[float]:
    chunks  = state.retrieved_chunks or []
    graded  = state.graded_chunks or []
    scores  = [c.get("score", 0.0) for c in chunks]
    top_s   = max(scores) if scores else 0.0
    mean_s  = sum(scores) / len(scores) if scores else 0.0
    return [
        _query_type_id(state.query_type),
        1.0 if state.requires_strict_citation else 0.0,
        _norm(len(chunks), _MAX_CHUNKS),
        _clip01(top_s),
        _clip01(mean_s),
        _norm(len(graded), _MAX_CHUNKS),
        _norm(len(state.selected_evidence), _MAX_CHUNKS),
        _norm(state.retry_count, _MAX_RETRIES),
        _norm(state.num_steps, _MAX_STEPS),
        1.0 if state.generated_answer.strip() else 0.0,
    ]


# ════════════════════════════════════════════════════════════════════════════
# Generator observation
# — what generator needs to decide how to formulate the answer
# ════════════════════════════════════════════════════════════════════════════
GENERATOR_OBS_NAMES: List[str] = [
    "query_type_id",
    "query_complexity_id",
    "requires_strict_citation",
    "selected_evidence_count_norm",
    "citation_support_rate",
    "num_llm_calls_norm",
    "num_steps_norm",
    "has_answer",
    "verification_failed",
    "generator_action_count_norm",
]
GENERATOR_OBS_DIM: int = len(GENERATOR_OBS_NAMES)


def get_generator_observation(state: "ContextState") -> List[float]:
    ver_failed = (
        state.verification_result.get("decision", "") == "FAIL"
        if state.verification_result else False
    )
    gen_cnt = state.action_count_for("generator")
    return [
        _query_type_id(state.query_type),
        _complexity_id(state.query_complexity),
        1.0 if state.requires_strict_citation else 0.0,
        _norm(len(state.selected_evidence), _MAX_CHUNKS),
        _clip01(state.citation_support_rate),
        _norm(state.num_llm_calls, _MAX_LLM),
        _norm(state.num_steps, _MAX_STEPS),
        1.0 if state.generated_answer.strip() else 0.0,
        1.0 if ver_failed else 0.0,
        _norm(gen_cnt, _MAX_RETRIES),
    ]


# ════════════════════════════════════════════════════════════════════════════
# Verifier observation
# — what verifier needs to decide whether to accept or request changes
# ════════════════════════════════════════════════════════════════════════════
VERIFIER_OBS_NAMES: List[str] = [
    "query_type_id",
    "requires_strict_citation",
    "has_answer",
    "citation_support_rate",
    "unsupported_claim_count_norm",
    "verification_failed",
    "num_llm_calls_norm",
    "num_steps_norm",
    "retry_count_norm",
    "verifier_action_count_norm",
]
VERIFIER_OBS_DIM: int = len(VERIFIER_OBS_NAMES)


def get_verifier_observation(state: "ContextState") -> List[float]:
    ver_failed = (
        state.verification_result.get("decision", "") == "FAIL"
        if state.verification_result else False
    )
    ver_cnt = state.action_count_for("verifier")
    return [
        _query_type_id(state.query_type),
        1.0 if state.requires_strict_citation else 0.0,
        1.0 if state.generated_answer.strip() else 0.0,
        _clip01(state.citation_support_rate),
        _norm(len(state.unsupported_claims), 10),
        1.0 if ver_failed else 0.0,
        _norm(state.num_llm_calls, _MAX_LLM),
        _norm(state.num_steps, _MAX_STEPS),
        _norm(state.retry_count, _MAX_RETRIES),
        _norm(ver_cnt, _MAX_RETRIES),
    ]


# ════════════════════════════════════════════════════════════════════════════
# Registry — makes it easy to look up by agent name
# ════════════════════════════════════════════════════════════════════════════
OBS_FN: Dict[str, callable] = {
    "retriever": get_retriever_observation,
    "rewriter":  get_rewriter_observation,
    "grader":    get_grader_observation,
    "generator": get_generator_observation,
    "verifier":  get_verifier_observation,
}

OBS_NAMES: Dict[str, List[str]] = {
    "retriever": RETRIEVER_OBS_NAMES,
    "rewriter":  REWRITER_OBS_NAMES,
    "grader":    GRADER_OBS_NAMES,
    "generator": GENERATOR_OBS_NAMES,
    "verifier":  VERIFIER_OBS_NAMES,
}

OBS_DIM: Dict[str, int] = {
    "retriever": RETRIEVER_OBS_DIM,
    "rewriter":  REWRITER_OBS_DIM,
    "grader":    GRADER_OBS_DIM,
    "generator": GENERATOR_OBS_DIM,
    "verifier":  VERIFIER_OBS_DIM,
}


def get_observation(agent: str, state: "ContextState") -> List[float]:
    """Dispatch to the correct per-agent observation function."""
    if agent not in OBS_FN:
        raise ValueError(f"Unknown agent '{agent}'. Valid: {list(OBS_FN.keys())}")
    return OBS_FN[agent](state)
