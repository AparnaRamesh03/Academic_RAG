"""
brain/context_marl_ac/schemas/results_schema.py
-------------------------------------------------
Final evaluation output schema for the context_marl_ac architecture.

The `EvalResult` dataclass defines the per-question output row written to:
    results/final_eval/context_marl_ac_results.jsonl

The schema is a superset of the existing baseline schema produced by
brain/result_utils.build_success_result(), so it is backward-compatible.

New MARL-specific fields added on top of the baseline:
    retrieved_chunks, citations, verification_result,
    answer_quality, retrieval_precision_at_k, retrieval_recall_at_k,
    retrieval_f1_at_k, citation_support_rate, unsupported_claim_rate,
    hallucination_flag, verification_pass, num_steps, num_llm_calls,
    token_usage, trace

Helper
------
    to_baseline_dict(result)  — strips MARL-only fields to produce a
    dict compatible with evaluate_non_llm_metrics.py and other existing
    evaluation scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    """
    One evaluated question output row.

    Baseline-compatible fields  (match brain/result_utils.build_success_result)
    ---------------------------------------------------------------------------
    question        : str
    ground_truth    : str
    source_file     : str | None
    category        : str | None
    difficulty      : str | None
    architecture    : str
    answer          : str
    contexts        : List[str]
    latency_sec     : float
    status          : "ok" | "error"
    error           : str | None

    MARL-specific fields
    --------------------
    question_id           : str
    question_type         : str
    retrieved_chunks       : List[dict]
    citations              : List[dict]
    verification_result    : dict
    answer_quality         : float   (0.0 – 1.0, set by reward/evaluator)
    retrieval_precision_at_k : float
    retrieval_recall_at_k    : float
    retrieval_f1_at_k        : float
    citation_support_rate  : float
    unsupported_claim_rate : float
    hallucination_flag     : bool
    verification_pass      : bool
    num_steps              : int
    num_llm_calls          : int
    token_usage            : int
    trace                  : List[dict]  (step-level action trace)
    """

    # ── Baseline-compatible ─────────────────────────────────────────────────
    question:       str
    ground_truth:   str
    architecture:   str   = "context_marl_ac"
    answer:         str   = ""
    contexts:       List[str] = field(default_factory=list)
    latency_sec:    float = 0.0
    status:         str   = "ok"
    error:          Optional[str] = None
    source_file:    Optional[str] = None
    category:       Optional[str] = None
    difficulty:     Optional[str] = None

    # ── MARL-specific ───────────────────────────────────────────────────────
    question_id:             str   = ""
    question_type:           str   = "factual"
    retrieved_chunks:        List[Dict[str, Any]] = field(default_factory=list)
    citations:               List[Dict[str, Any]] = field(default_factory=list)
    verification_result:     Dict[str, Any]       = field(default_factory=dict)
    answer_quality:          float = 0.0
    retrieval_precision_at_k: float = 0.0
    retrieval_recall_at_k:    float = 0.0
    retrieval_f1_at_k:        float = 0.0
    citation_support_rate:   float = 0.0
    unsupported_claim_rate:  float = 0.0
    hallucination_flag:      bool  = False
    verification_pass:       bool  = False
    num_steps:               int   = 0
    num_llm_calls:           int   = 0
    token_usage:             int   = 0
    trace:                   List[Dict[str, Any]] = field(default_factory=list)

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Full JSON-serializable dict (all fields)."""
        return {
            # Baseline fields
            "question":       self.question,
            "ground_truth":   self.ground_truth,
            "source_file":    self.source_file,
            "category":       self.category,
            "difficulty":     self.difficulty,
            "architecture":   self.architecture,
            "answer":         self.answer,
            "contexts":       self.contexts,
            "latency_sec":    round(self.latency_sec, 4),
            "status":         self.status,
            "error":          self.error,
            # MARL-specific fields
            "question_id":               self.question_id,
            "question_type":             self.question_type,
            "retrieved_chunks":          self.retrieved_chunks,
            "citations":                 self.citations,
            "verification_result":       self.verification_result,
            "answer_quality":            round(self.answer_quality, 4),
            "retrieval_precision_at_k":  round(self.retrieval_precision_at_k, 4),
            "retrieval_recall_at_k":     round(self.retrieval_recall_at_k, 4),
            "retrieval_f1_at_k":         round(self.retrieval_f1_at_k, 4),
            "citation_support_rate":     round(self.citation_support_rate, 4),
            "unsupported_claim_rate":    round(self.unsupported_claim_rate, 4),
            "hallucination_flag":        self.hallucination_flag,
            "verification_pass":         self.verification_pass,
            "num_steps":                 self.num_steps,
            "num_llm_calls":             self.num_llm_calls,
            "token_usage":               self.token_usage,
            "trace":                     self.trace,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_baseline_dict(self) -> Dict[str, Any]:
        """
        Minimal dict compatible with brain/result_utils.build_success_result()
        and evaluate_non_llm_metrics.py.  Drops MARL-only fields.
        """
        return {
            "question":     self.question,
            "ground_truth": self.ground_truth,
            "source_file":  self.source_file,
            "category":     self.category,
            "difficulty":   self.difficulty,
            "architecture": self.architecture,
            "answer":       self.answer,
            "contexts":     self.contexts,
            "citations":    self.citations,
            "latency_sec":  round(self.latency_sec, 4),
            "status":       self.status,
            "error":        self.error,
        }

    @classmethod
    def from_error(
        cls,
        question_dict: Dict[str, Any],
        error_msg: str,
        latency_sec: float = 0.0,
    ) -> "EvalResult":
        """Convenience constructor for failed episodes."""
        return cls(
            question=str(question_dict.get("question", "")),
            ground_truth=str(question_dict.get("ground_truth", "")),
            source_file=question_dict.get("source_file"),
            category=question_dict.get("category"),
            difficulty=question_dict.get("difficulty"),
            question_id=str(question_dict.get("question_id", "")),
            status="error",
            error=error_msg,
            latency_sec=latency_sec,
        )
