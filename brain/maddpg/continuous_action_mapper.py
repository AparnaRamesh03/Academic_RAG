"""
brain/maddpg/continuous_action_mapper.py
----------------------------------------------------------
Maps MADDPG actor outputs in [-1, 1] to:
  1. Real RAG execution parameters (numeric).
  2. Discrete action names within the current valid action set.

Param-wiring contract (see docs/MADDPG_EXTENSION.md for the full table):

| Agent     | Param                 | Used in execution? | Where                                                        |
|-----------|-----------------------|--------------------|--------------------------------------------------------------|
| retriever | top_k                 | yes                | retriever_agent.py `act()` -> retrieve_*(query, top_k)       |
| retriever | dense_sparse_weight   | yes (selector)     | _sel_retriever() picks dense/sparse/hybrid                   |
| retriever | rerank_threshold      | yes (selector)     | _sel_retriever() picks hybrid_rerank when >= 0.5             |
| retriever | source_diversity      | yes (post-filter)  | retriever_agent.py post-filters retrieved chunks             |
| rewriter  | rewrite_strength      | yes (selector)     | _sel_rewriter() picks aggressive vs simple rewrite mode      |
| rewriter  | query_expansion_weight| yes (selector)     | _sel_rewriter() picks expanded vs multi_query rewrite        |
| grader    | evidence_keep_ratio   | yes [0.5, 1.0]     | grader_agent.py trims filtered chunks to top-N fraction      |
| grader    | relevance_threshold   | yes (post-filter)  | grader_agent.py drops chunks with score < threshold          |
| grader    | strictness_score      | yes (selector)     | _sel_grader() picks strict/medium/loose filter mode          |
| generator | temperature           | yes                | generator_agent.py -> generate_answer(temperature=...)       |
| generator | max_tokens            | yes                | generator_agent.py -> generate_answer(max_tokens=...)        |
| generator | citation_strictness   | yes (selector)     | _sel_generator() picks generate_with_strict_citations >=0.65 |
| generator | answer_detail_level   | yes                | generator_agent.py scales max_tokens by detail level         |
| verifier  | support_threshold     | yes                | verifier_agent.py overrides PASS/FAIL with citation rate     |

All values are clamped. Safe defaults are used on any error.
"""
from typing import Any, Dict, List, Optional

import numpy as np

# ── Per-agent continuous action dimensions ─────────────────────────────────────
AGENT_ACTION_DIMS: Dict[str, int] = {
    "retriever": 4,   # dense_sparse_weight, top_k_norm, rerank_threshold, source_diversity
    "rewriter":  2,   # rewrite_strength, query_expansion_weight
    "grader":    3,   # relevance_threshold, evidence_keep_ratio, strictness_score
    "generator": 4,   # temperature, citation_strictness, max_tokens_norm, answer_detail_level
    "verifier":  1,   # support_threshold
}

AGENT_PARAM_NAMES: Dict[str, List[str]] = {
    "retriever": ["dense_sparse_weight", "top_k_norm", "rerank_threshold", "source_diversity"],
    "rewriter":  ["rewrite_strength", "query_expansion_weight"],
    "grader":    ["relevance_threshold", "evidence_keep_ratio", "strictness_score"],
    "generator": ["temperature", "citation_strictness", "max_tokens_norm", "answer_detail_level"],
    "verifier":  ["support_threshold"],
}

# Ordered list used to build/parse joint action vectors consistently.
ORDERED_AGENTS: List[str] = ["retriever", "rewriter", "grader", "generator", "verifier"]
JOINT_ACTION_DIM: int = sum(AGENT_ACTION_DIMS[n] for n in ORDERED_AGENTS)  # 14

# ── Safe defaults ──────────────────────────────────────────────────────────────
AGENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "retriever": {"dense_sparse_weight": 0.5, "top_k": 7,
                  "rerank_threshold": 0.5, "source_diversity": 0.5},
    "rewriter":  {"rewrite_strength": 0.5, "query_expansion_weight": 0.5},
    "grader":    {"relevance_threshold": 0.0, "evidence_keep_ratio": 0.75,
                  "strictness_score": 0.5},
    "generator": {"temperature": 0.3, "citation_strictness": 0.7,
                  "max_tokens": 512, "answer_detail_level": 0.5},
    "verifier":  {"support_threshold": 0.6},
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _u(x: float) -> float:
    """Map [-1, 1] → [0, 1]."""
    return (float(x) + 1.0) / 2.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe(raw: Optional[np.ndarray], idx: int, fallback: float) -> float:
    if raw is None or len(raw) <= idx:
        return fallback
    return float(raw[idx])


# ── Numeric param mappers ──────────────────────────────────────────────────────

def _map_retriever(raw: np.ndarray) -> Dict[str, Any]:
    r = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
    # top_k range 3..12 (was 5..30) — keeps Groq grading from blowing the rate
    # limit when the grader runs in LLM modes. Default raw≈0 → top_k≈7.
    return {
        "dense_sparse_weight":   _clamp(_u(_safe(r, 0, 0.0)), 0.0, 1.0),
        "top_k":                 int(_clamp(3 + _u(_safe(r, 1, 0.0)) * 9, 3, 12)),
        "rerank_threshold":      _clamp(_u(_safe(r, 2, 0.0)), 0.0, 1.0),
        "source_diversity":      _clamp(_u(_safe(r, 3, 0.0)), 0.0, 1.0),
    }


def _map_rewriter(raw: np.ndarray) -> Dict[str, Any]:
    r = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
    return {
        "rewrite_strength":       _clamp(_u(_safe(r, 0, 0.0)), 0.0, 1.0),
        "query_expansion_weight": _clamp(_u(_safe(r, 1, 0.0)), 0.0, 1.0),
    }


def _map_grader(raw: np.ndarray) -> Dict[str, Any]:
    r = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
    # evidence_keep_ratio range raised to [0.5, 1.0] (was [0.1, 1.0]) so the
    # actor cannot reward-hack the verifier by stripping evidence to ~1 chunk.
    # Maps raw [-1, 1] → [0.5, 1.0] linearly; default raw≈0 → ratio≈0.75.
    keep_ratio = _clamp(0.5 + 0.5 * _u(_safe(r, 1, 0.0)), 0.5, 1.0)
    return {
        "relevance_threshold": _clamp(_u(_safe(r, 0, 0.0)), 0.0, 1.0),
        "evidence_keep_ratio": keep_ratio,
        "strictness_score":    _clamp(_u(_safe(r, 2, 0.0)), 0.0, 1.0),
    }


def _map_generator(raw: np.ndarray) -> Dict[str, Any]:
    r = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
    tok_n = _u(_safe(r, 2, 0.0))
    detail = _clamp(_u(_safe(r, 3, 0.0)), 0.0, 1.0)
    # max_tokens range narrowed to 128..384 (was 128..1024) so generation stays
    # under the Groq free-tier TPM cap during exploration. The actor can still
    # push toward 384 when reward demands it.
    base_tokens = int(_clamp(128 + tok_n * 256, 128, 384))
    scaled = int(_clamp(base_tokens * (0.5 + detail), 128, 384))
    return {
        "temperature":         _clamp(_u(_safe(r, 0, -0.4)), 0.0, 1.0),
        "citation_strictness": _clamp(_u(_safe(r, 1, 0.0)), 0.0, 1.0),
        "max_tokens":          scaled,
        "answer_detail_level": detail,
    }


def _map_verifier(raw: np.ndarray) -> Dict[str, Any]:
    r = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
    return {
        "support_threshold": _clamp(_u(_safe(r, 0, 0.2)), 0.0, 1.0),
    }


_NUMERIC_MAPPERS = {
    "retriever": _map_retriever,
    "rewriter":  _map_rewriter,
    "grader":    _map_grader,
    "generator": _map_generator,
    "verifier":  _map_verifier,
}


def map_agent_params(agent_name: str, raw: Optional[np.ndarray]) -> Dict[str, Any]:
    """Map raw actor output → numeric RAG params. Returns safe defaults on error."""
    try:
        if raw is None:
            return dict(AGENT_DEFAULTS.get(agent_name, {}))
        return _NUMERIC_MAPPERS[agent_name](np.asarray(raw, dtype=np.float32))
    except Exception:
        return dict(AGENT_DEFAULTS.get(agent_name, {}))


# ── Continuous → discrete action selectors ────────────────────────────────────

def _sel_retriever(p: Dict[str, Any], valid: List[str]) -> str:
    alpha  = p.get("dense_sparse_weight", 0.5)
    rerank = p.get("rerank_threshold",    0.5)
    if rerank >= 0.5 and "hybrid_rerank"  in valid: return "hybrid_rerank"
    if alpha  >= 0.65 and "dense_retrieve" in valid: return "dense_retrieve"
    if alpha  <= 0.35 and "sparse_retrieve" in valid: return "sparse_retrieve"
    if "hybrid_retrieve" in valid:  return "hybrid_retrieve"
    if "retrieve_more"   in valid:  return "retrieve_more"
    return valid[0]


def _sel_rewriter(p: Dict[str, Any], valid: List[str]) -> str:
    s = p.get("rewrite_strength", 0.5)
    e = p.get("query_expansion_weight", 0.5)
    if s >= 0.7 and "multi_query_rewrite" in valid:  return "multi_query_rewrite"
    if e >= 0.6 and "expanded_rewrite"    in valid:  return "expanded_rewrite"
    if s >= 0.4 and "keyword_rewrite"     in valid:  return "keyword_rewrite"
    if "simple_rewrite" in valid:   return "simple_rewrite"
    if "no_rewrite"     in valid:   return "no_rewrite"
    return valid[0]


def _sel_grader(p: Dict[str, Any], valid: List[str]) -> str:
    # LLM-based grader modes (loose/medium/strict_filter) make one Groq call
    # per retrieved chunk. To avoid burning the rate limit during exploration,
    # the selector only escalates to an LLM mode when the actor has clearly
    # learned that strictness helps. Default raw≈0 + OU noise (σ=0.15) keeps
    # strictness in roughly [0.35, 0.65] — these thresholds force rerank_only
    # in that range and only switch to LLM modes when the actor pushes raw > 0.4.
    s = p.get("strictness_score",    0.5)
    r = p.get("evidence_keep_ratio", 0.7)
    if s >= 0.90 and "strict_filter" in valid: return "strict_filter"
    if s >= 0.80 and "medium_filter" in valid: return "medium_filter"
    if s >= 0.70 and "loose_filter"  in valid: return "loose_filter"
    if r >= 0.90 and "keep_all"      in valid: return "keep_all"
    if "rerank_only" in valid: return "rerank_only"
    return valid[0]


def _sel_generator(p: Dict[str, Any], valid: List[str]) -> str:
    # `generate_with_strict_citations` and `generate_answer` both consume
    # significantly more tokens than `generate_short_answer`. Thresholds are
    # tuned so the default (raw≈0 → c≈0.5, d≈0.5) plus OU noise (σ=0.15)
    # leaves short_answer as the dominant exploration mode; the actor escalates
    # only when raw output is decisively positive.
    c = p.get("citation_strictness", 0.5)
    d = p.get("answer_detail_level", 0.5)
    if c >= 0.85 and "generate_with_strict_citations" in valid:
        return "generate_with_strict_citations"
    if d >= 0.70 and "generate_answer"               in valid:
        return "generate_answer"
    if "generate_short_answer"                       in valid:
        return "generate_short_answer"
    if "abstain_request_more_evidence"               in valid:
        return "abstain_request_more_evidence"
    return valid[0]


def _sel_verifier(p: Dict[str, Any], valid: List[str]) -> str:
    if "verify_answer" in valid: return "verify_answer"
    sup = p.get("support_threshold", 0.6)
    if sup >= 0.55 and "request_regeneration"  in valid:
        return "request_regeneration"
    if sup < 0.55  and "request_more_retrieval" in valid:
        return "request_more_retrieval"
    if "request_rewrite" in valid: return "request_rewrite"
    return valid[0]


_DISCRETE_SELECTORS = {
    "retriever": _sel_retriever,
    "rewriter":  _sel_rewriter,
    "grader":    _sel_grader,
    "generator": _sel_generator,
    "verifier":  _sel_verifier,
}


def select_discrete_action(
    agent_name: str,
    params: Dict[str, Any],
    valid_actions: List[str],
) -> str:
    """Given mapped params and valid action names, pick the best discrete action."""
    if not valid_actions:
        raise ValueError(f"No valid actions for agent '{agent_name}'")
    fn = _DISCRETE_SELECTORS.get(agent_name)
    if fn is None:
        return valid_actions[0]
    return fn(params, valid_actions)


# ── Joint action vector (LEGACY) ──────────────────────────────────────────────

def build_joint_action_vector(
    agent_raw_actions: Dict[str, Optional[np.ndarray]],
) -> np.ndarray:
    """
    LEGACY: kept only so trajectories produced before the stage-conditioned
    refactor remain loadable. The stage-conditioned critic does NOT use this.
    """
    parts = []
    for name in ORDERED_AGENTS:
        dim = AGENT_ACTION_DIMS[name]
        raw = agent_raw_actions.get(name)
        if raw is not None:
            vec = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
            vec = vec[:dim]
            if len(vec) < dim:
                vec = np.pad(vec, (0, dim - len(vec)))
        else:
            vec = np.zeros(dim, dtype=np.float32)
        parts.append(vec)
    return np.concatenate(parts)
