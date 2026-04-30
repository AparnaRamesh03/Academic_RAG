from typing import List, Dict, Any, TypedDict


class GraphState(TypedDict, total=False):
    # Core query fields
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
    claim_verification: List[Dict[str, Any]]

    # Phase 2+: supervisor/autonomy fields
    step_count: int
    action_history: List[str]
    last_action: str
    done: bool
    stop_reason: str
    confidence: float
    latency_so_far: float