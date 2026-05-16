"""
brain/context_marl_ac/schemas/context_state.py
------------------------------------------------
ContextState dataclass — the central shared state for one MARL episode.

One ContextState instance is created per benchmark question (episode) and
mutated in-place as agents act.  It is the single source of truth that all
agents, the centralized critic, and the reward function read from.

Design choices
--------------
- Pure Python dataclass with field() defaults (no Pydantic runtime overhead).
- All list fields default to empty lists, not None, to avoid None checks.
- `to_dict()` produces a plain JSON-serializable dict for logging / saving.
- Gold evidence / gold answers are NEVER stored here — they live only in
  the reward function and the evaluator.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Query type and complexity enums (as string constants, not Enum, for
# simpler JSON serialisation and log readability)
# ---------------------------------------------------------------------------
QUERY_TYPES = [
    "factual",
    "conceptual",
    "comparison",
    "section_specific",
    "multi_hop",
    "definition",
    "summarization",
]

QUERY_COMPLEXITIES = ["low", "medium", "high"]

FINAL_STATUSES = [
    "pending",             # episode not yet resolved
    "accepted",            # verifier accepted the answer
    "rejected",            # verifier rejected the answer
    "abstained",           # generator abstained due to insufficient evidence
    "generation_failed",   # generator returned empty/invalid output
    "timeout",             # max steps or max LLM calls reached
    "error",               # unrecoverable execution error
]


# ---------------------------------------------------------------------------
# ContextState
# ---------------------------------------------------------------------------
@dataclass
class ContextState:
    """
    Mutable episode state shared by all MARL agents and the centralized critic.

    Fields are grouped by concern:
      - Query metadata
      - Retrieval
      - Grading / evidence selection
      - Citation
      - Generation
      - Verification
      - Episode bookkeeping
    """

    # ── Query metadata ──────────────────────────────────────────────────────
    question_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = "" # Fixed benchmark question
    user_query:  str = ""     # Current active query (may be rewritten)
    rewritten_query: str = ""  # Last rewrite result
    # List of filenames/identifiers expected for this question
    expected_sources: List[str] = field(default_factory=list)
    # Multiple choice options (for ARC/PopQA)
    choices: Optional[Dict[str, Any]] = None

    # Populated by query_analyzer
    query_type:                str  = "factual"   # one of QUERY_TYPES
    query_complexity:          str  = "medium"    # one of QUERY_COMPLEXITIES
    requires_multiple_sources: bool = False
    requires_strict_citation:  bool = False

    # ── Retrieval ───────────────────────────────────────────────────────────
    # Raw chunks: List[{text, metadata, score}]
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_scores: List[float]          = field(default_factory=list)

    # ── Grading / evidence selection ────────────────────────────────────────
    # Graded chunks after GraderAgent acts
    graded_chunks:   List[Dict[str, Any]] = field(default_factory=list)
    # Final evidence passed to generator: List[evidence_pack_item]
    selected_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # ── Citation ─────────────────────────────────────────────────────────────
    # Populated by GeneratorAgent after it calls citation_adapter.build_citations()
    # List[{source_file, page_number, section_header, excerpt, content_type}]
    citation_candidates: List[Dict[str, Any]] = field(default_factory=list)

    # ── Generation ──────────────────────────────────────────────────────────
    generated_answer: str = ""

    # ── Verification ────────────────────────────────────────────────────────
    # Raw output from llm_adapter.verify_answer()
    # {decision: PASS|FAIL, overall_feedback: str, claims: [...]}
    verification_result: Dict[str, Any] = field(default_factory=dict)

    # Computed from verification_result by VerifierAgent
    unsupported_claims:    List[str] = field(default_factory=list)
    citation_support_rate: float     = 0.0   # fraction of claims supported
    
    # Debug info from agents
    grader_output: Dict[str, Any] = field(default_factory=dict)

    # ── Episode bookkeeping ─────────────────────────────────────────────────
    # History of (agent, action) pairs this episode
    previous_actions: List[Dict[str, str]] = field(default_factory=list)
    retry_count:      int   = 0
    latency_so_far:   float = 0.0    # wall-clock seconds accumulated

    # LLM usage tracking
    token_usage:   int = 0   # approximate token count (if available)
    num_llm_calls: int = 0
    num_steps:     int = 0

    # Episode terminal state
    final_status: str  = "pending"   # one of FINAL_STATUSES
    done:         bool = False

    # Internal: episode start timestamp (for latency calculation)
    _start_time: float = field(default_factory=time.perf_counter, repr=False, compare=False)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def update_latency(self) -> None:
        """Refresh latency_so_far from wall clock."""
        self.latency_so_far = round(time.perf_counter() - self._start_time, 4)

    def record_action(self, agent: str, action: str) -> None:
        """Append one (agent, action) pair to previous_actions and bump num_steps."""
        self.previous_actions.append({"agent": agent, "action": action})
        self.num_steps += 1

    def last_action_for(self, agent: str) -> Optional[str]:
        """Return the most recent action taken by a given agent, or None."""
        for entry in reversed(self.previous_actions):
            if entry.get("agent") == agent:
                return entry.get("action")
        return None

    def action_count_for(self, agent: str) -> int:
        """Count how many times a given agent has acted this episode."""
        return sum(1 for e in self.previous_actions if e.get("agent") == agent)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable plain dict.  Excludes private fields.
        Useful for logging, trajectory storage, and result serialisation.
        """
        return {
            "question_id":               self.question_id,
            "user_query":                self.user_query,
            "expected_sources":          self.expected_sources,
            "query_type":                self.query_type,
            "query_complexity":          self.query_complexity,
            "requires_multiple_sources": self.requires_multiple_sources,
            "requires_strict_citation":  self.requires_strict_citation,
            # Retrieval
            "retrieved_chunks":          self.retrieved_chunks,
            "retrieval_scores":          self.retrieval_scores,
            # Grading
            "graded_chunks":             self.graded_chunks,
            "selected_evidence":         self.selected_evidence,
            # Citation
            "citation_candidates":       self.citation_candidates,
            # Generation
            "generated_answer":          self.generated_answer,
            # Verification
            "verification_result":       self.verification_result,
            "unsupported_claims":        self.unsupported_claims,
            "citation_support_rate":     self.citation_support_rate,
            # Bookkeeping
            "previous_actions":          self.previous_actions,
            "retry_count":               self.retry_count,
            "latency_so_far":            self.latency_so_far,
            "token_usage":               self.token_usage,
            "num_llm_calls":             self.num_llm_calls,
            "num_steps":                 self.num_steps,
            "final_status":              self.final_status,
            "done":                      self.done,
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        """Returns a dict with detailed agent outputs for debugging."""
        return {
            "question":                  self.original_query,
            "current_query":             self.user_query,
            "rewritten_query":           self.rewritten_query,
            "retrieved_chunk_ids":       [
                c.get("metadata", {}).get("chunk_id") or 
                c.get("id") or 
                f"{c.get('metadata', {}).get('source_file')}_p{c.get('metadata', {}).get('page_number')}_{i}"
                for i, c in enumerate(self.retrieved_chunks)
            ],
            "retrieved_chunk_preview":   [c.get("text", "")[:300] for c in self.retrieved_chunks],
            "retrieval_scores":          self.retrieval_scores,
            "selected_evidence_ids":     [e.get("chunk_id") for e in self.selected_evidence],
            "selected_evidence_preview": [e.get("text", "")[:300] for e in self.selected_evidence],
            "generated_answer_preview":  self.generated_answer[:300],
            "generated_answer_length":   len(self.generated_answer),
            "citations":                 self.citation_candidates,
            "verifier_decision":         self.verification_result.get("decision", "N/A"),
            "verifier_reason":           self.verification_result.get("reason", ""),
            "unsupported_claims":        self.unsupported_claims,
            "citation_support_rate":     self.citation_support_rate,
            "final_status":              self.final_status,
            "grader_output":             self.grader_output
        }

    @classmethod
    def from_question(cls, question_dict: Dict[str, Any], index: int = 1) -> "ContextState":
        """
        Convenience constructor: initialise a fresh ContextState from a
        benchmark question dict. Supports the standard_benchmark_v3 format.

        Mappings:
            question      -> user_query
            ground_truth  -> gold_answer (stored in evaluator/reward)
            source_file   -> expected_sources (list)
            category      -> query_type
            difficulty    -> query_complexity
        """
        q_id = question_dict.get("question_id")
        if not q_id:
            # Generate Q001, Q002, etc.
            q_id = f"Q{index:03d}"
            
        # Map source_file (can be string or list)
        src = question_dict.get("source_file", [])
        if isinstance(src, str):
            src = [src]

        q_text = str(question_dict.get("question", ""))
        return cls(
            question_id=str(q_id),
            original_query=q_text,
            user_query=q_text,
            expected_sources=src,
            query_type=str(question_dict.get("category", "factual")),
            query_complexity=str(question_dict.get("difficulty", "medium")),
        )
