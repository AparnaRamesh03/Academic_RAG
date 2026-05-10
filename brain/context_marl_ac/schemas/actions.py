"""
brain/context_marl_ac/schemas/actions.py
-----------------------------------------
Per-agent action space definitions for the MARL system.

Each agent has its own discrete action set.  Actions are represented as
strings so logs are human-readable.  Each agent also has its own
ACTION_TO_ID / ID_TO_ACTION mappings for neural network output indexing.

Usage
-----
    from context_marl_ac.schemas.actions import (
        RETRIEVER_ACTIONS,
        RETRIEVER_ACTION_TO_ID,
        RETRIEVER_ID_TO_ACTION,
        ...
    )
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Retriever agent actions
# ---------------------------------------------------------------------------
RETRIEVER_ACTIONS: List[str] = [
    "dense_retrieve",       # dense-only BGE-M3 retrieval
    "sparse_retrieve",      # sparse-only BM25 retrieval
    "hybrid_retrieve",      # RRF fusion of dense + sparse
    "hybrid_rerank",        # hybrid + CrossEncoder reranking
    "retrieve_more",        # additional retrieval excluding current chunks
]

RETRIEVER_ACTION_TO_ID: Dict[str, int] = {a: i for i, a in enumerate(RETRIEVER_ACTIONS)}
RETRIEVER_ID_TO_ACTION: Dict[int, str] = {i: a for a, i in RETRIEVER_ACTION_TO_ID.items()}


# ---------------------------------------------------------------------------
# Rewriter agent actions
# ---------------------------------------------------------------------------
REWRITER_ACTIONS: List[str] = [
    "no_rewrite",           # pass query through unchanged
    "simple_rewrite",       # basic paraphrase for retrieval
    "keyword_rewrite",      # keyword-heavy retrieval query
    "expanded_rewrite",     # expanded query with related terms
    "multi_query_rewrite",  # generate multiple sub-queries
]

REWRITER_ACTION_TO_ID: Dict[str, int] = {a: i for i, a in enumerate(REWRITER_ACTIONS)}
REWRITER_ID_TO_ACTION: Dict[int, str] = {i: a for a, i in REWRITER_ACTION_TO_ID.items()}


# ---------------------------------------------------------------------------
# Grader agent actions
# ---------------------------------------------------------------------------
GRADER_ACTIONS: List[str] = [
    "keep_all",             # accept all retrieved chunks without grading
    "loose_filter",         # LLM grading, keep borderline relevant docs
    "medium_filter",        # LLM grading, balanced recall/precision
    "strict_filter",        # LLM grading + score threshold, high precision
    "rerank_only",          # sort by score without dropping chunks
]

GRADER_ACTION_TO_ID: Dict[str, int] = {a: i for i, a in enumerate(GRADER_ACTIONS)}
GRADER_ID_TO_ACTION: Dict[int, str] = {i: a for a, i in GRADER_ACTION_TO_ID.items()}


# ---------------------------------------------------------------------------
# Generator agent actions
# ---------------------------------------------------------------------------
GENERATOR_ACTIONS: List[str] = [
    "generate_answer",                  # standard grounded answer
    "generate_with_strict_citations",   # answer with explicit citation markers
    "generate_short_answer",            # one-sentence factual answer
    "abstain_request_more_evidence",    # abstain and signal retrieval needed
    "regenerate",                       # retry generation with audit feedback
]

GENERATOR_ACTION_TO_ID: Dict[str, int] = {a: i for i, a in enumerate(GENERATOR_ACTIONS)}
GENERATOR_ID_TO_ACTION: Dict[int, str] = {i: a for a, i in GENERATOR_ACTION_TO_ID.items()}


# ---------------------------------------------------------------------------
# Verifier agent actions
# ---------------------------------------------------------------------------
VERIFIER_ACTIONS: List[str] = [
    "verify_answer",            # Check answer quality and grounding — ends episode
    "request_regeneration",     # ask generator to retry with feedback
    "request_more_retrieval",   # ask retriever to fetch more evidence
    "request_rewrite",          # ask rewriter to reformulate query
]

VERIFIER_ACTION_TO_ID: Dict[str, int] = {a: i for i, a in enumerate(VERIFIER_ACTIONS)}
VERIFIER_ID_TO_ACTION: Dict[int, str] = {i: a for a, i in VERIFIER_ACTION_TO_ID.items()}


# ---------------------------------------------------------------------------
# Convenience: all agents and their action lists
# ---------------------------------------------------------------------------
AGENT_NAMES: List[str] = [
    "retriever",
    "rewriter",
    "grader",
    "generator",
    "verifier",
]

AGENT_ACTIONS: Dict[str, List[str]] = {
    "retriever": RETRIEVER_ACTIONS,
    "rewriter":  REWRITER_ACTIONS,
    "grader":    GRADER_ACTIONS,
    "generator": GENERATOR_ACTIONS,
    "verifier":  VERIFIER_ACTIONS,
}

AGENT_ACTION_TO_ID: Dict[str, Dict[str, int]] = {
    "retriever": RETRIEVER_ACTION_TO_ID,
    "rewriter":  REWRITER_ACTION_TO_ID,
    "grader":    GRADER_ACTION_TO_ID,
    "generator": GENERATOR_ACTION_TO_ID,
    "verifier":  VERIFIER_ACTION_TO_ID,
}

AGENT_ID_TO_ACTION: Dict[str, Dict[int, str]] = {
    "retriever": RETRIEVER_ID_TO_ACTION,
    "rewriter":  REWRITER_ID_TO_ACTION,
    "grader":    GRADER_ID_TO_ACTION,
    "generator": GENERATOR_ID_TO_ACTION,
    "verifier":  VERIFIER_ID_TO_ACTION,
}


def num_actions(agent: str) -> int:
    """Return the number of discrete actions for a given agent."""
    return len(AGENT_ACTIONS[agent])


def action_name(agent: str, action_id: int) -> str:
    """Convert an action integer id back to its string name."""
    return AGENT_ID_TO_ACTION[agent][action_id]


def action_id(agent: str, action_name_str: str) -> int:
    """Convert an action string name to its integer id."""
    return AGENT_ACTION_TO_ID[agent][action_name_str]
