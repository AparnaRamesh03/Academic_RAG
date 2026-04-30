from typing import List, Dict, Any, TypedDict


class GraphState(TypedDict):
    original_query: str
    search_query: str

    # Full ranked retrieval results in original order
    retrieved_docs: List[Dict[str, Any]]

    # Architecture-specific working sets
    candidate_docs: List[Dict[str, Any]]
    weak_signal_docs: List[Dict[str, Any]]
    graded_docs: List[Dict[str, Any]]

    # Final model output
    generation: str

    # Retry / audit bookkeeping
    crag_retries: int
    verify_retries: int
    citations_pass: bool
    auditor_feedback: str

    # --- RL Extended State ---
    step_count: int
    action_history: List[str]
    current_phase: str
    retrieval_rounds: int
    used_rewrite: bool
    used_grade: bool
    used_audit: bool
    top_retrieval_scores: List[float]
    num_distinct_sources: int
    question_category: str
    question_difficulty: str
    latency_so_far: float
    done: bool
    stop_reason: str
    current_action: str