"""
rl_state_features.py
--------------------
Feature extraction for the contextual bandit policy.

`extract_state_key(state)` returns a discrete tuple that serves as the Q-table
key for each routing decision the supervisor makes.

State space size:
  6 (last_action) × 2 × 2 × 2 (booleans) × 3 × 3 (retry bins) × 5 (query type)
  × 4 (confidence bin) = 8,640 possible states — tiny, fits in RAM and JSON.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent

for path in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from state_shared import GraphState
from query_targeting import is_comparison_query, is_underspecified_superlative_query

# ── Query type ────────────────────────────────────────────────────────────────

_FIGURE_MARKERS = ("figure", "table", "diagram", "chart", "graph", "fig.")
_DIRECT_FACT_STARTS = (
    "what is", "what are", "who is", "define", "explain what",
    "how does", "how do", "describe",
)


def classify_query_type(query: str) -> str:
    """Classify a query into one of five coarse categories."""
    q = (query or "").lower().strip()

    # Figure / table reference takes priority.
    if any(m in q for m in _FIGURE_MARKERS):
        return "figure"

    # Comparison queries (may overlap with superlative, so check first).
    if is_comparison_query(query):
        return "comparison"

    # Under-specified superlative ("which is best", "most efficient", …).
    if is_underspecified_superlative_query(query):
        return "superlative"

    # Simple direct-fact questions.
    if any(q.startswith(start) for start in _DIRECT_FACT_STARTS):
        return "direct_fact"

    return "other"


# ── Confidence binning ────────────────────────────────────────────────────────

def _confidence_bin(confidence: float) -> int:
    """Map a continuous confidence score into 4 ordinal bins."""
    if confidence < 0.30:
        return 0
    if confidence < 0.50:
        return 1
    if confidence < 0.70:
        return 2
    return 3


# ── Main feature extractor ────────────────────────────────────────────────────

def extract_state_key(state: GraphState) -> Tuple:
    """
    Return a discrete tuple that uniquely identifies the current routing state.

    The tuple is hashable and JSON-serialisable (via str()), making it suitable
    as a dictionary key for both the in-memory Q-table and the persisted JSON.
    """
    last_action = str(state.get("last_action", "") or "")
    has_graded_docs = bool(state.get("graded_docs"))
    has_generation = bool(str(state.get("generation", "") or "").strip())
    citations_pass = bool(state.get("citations_pass", False))
    crag_bin = min(int(state.get("crag_retries", 0) or 0), 2)
    verify_bin = min(int(state.get("verify_retries", 0) or 0), 2)
    query_type = classify_query_type(str(state.get("original_query", "") or ""))
    conf_bin = _confidence_bin(float(state.get("confidence", 0.0) or 0.0))

    return (
        last_action,
        has_graded_docs,
        has_generation,
        citations_pass,
        crag_bin,
        verify_bin,
        query_type,
        conf_bin,
    )
