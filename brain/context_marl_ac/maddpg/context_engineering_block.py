"""
brain/context_marl_ac/maddpg/context_engineering_block.py
----------------------------------------------------------
Builds a richer 20-dim state representation from ContextState.
Extends the base 14-dim global features with 6 additional features.

Extra features (dims 15-20):
  15. source_diversity       unique source files / total retrieved
  16. evidence_coverage      selected_evidence / retrieved_chunks
  17. step_fraction          num_steps / MAX_STEPS_PER_EPISODE
  18. llm_call_fraction      num_llm_calls / MAX_LLM_CALLS_PER_EPISODE
  19. query_length_norm      len(user_query) / 300, capped at 1
  20. requires_multiple_src  binary flag
"""
from typing import Any, List

from context_marl_ac.config import MAX_STEPS_PER_EPISODE, MAX_LLM_CALLS_PER_EPISODE
from context_marl_ac.context_engineering.feature_encoder import encode_features

CEB_STATE_DIM: int = 20   # 14 base + 6 extra


def _get(obj: Any, key: str, default: Any = 0) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def build_ceb_features(state: Any) -> List[float]:
    """
    Returns a 20-dim feature vector.
    Falls back to zeros for any missing field.
    """
    base = encode_features(state)  # 14-dim

    retrieved = _get(state, "retrieved_chunks", []) or []
    selected  = _get(state, "selected_evidence", []) or []
    num_steps = _get(state, "num_steps", 0)
    num_llm   = _get(state, "num_llm_calls", 0)
    query     = _get(state, "user_query", "") or ""
    rms       = _get(state, "requires_multiple_sources", False)

    # Source diversity: unique source_file values across retrieved chunks.
    sources = set()
    for c in retrieved:
        if isinstance(c, dict):
            meta = c.get("metadata", {})
            sf = (meta.get("source_file", "") if isinstance(meta, dict) else "")
            if sf:
                sources.add(sf)
    diversity = len(sources) / max(len(retrieved), 1)

    coverage   = len(selected) / max(len(retrieved), 1)
    step_frac  = min(num_steps / max(MAX_STEPS_PER_EPISODE,    1), 1.0)
    llm_frac   = min(num_llm   / max(MAX_LLM_CALLS_PER_EPISODE, 1), 1.0)
    q_len_norm = min(len(query) / 300.0, 1.0)
    rms_flag   = 1.0 if rms else 0.0

    return base + [diversity, coverage, step_frac, llm_frac, q_len_norm, rms_flag]
