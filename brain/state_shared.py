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

    # Supervisor / autonomy fields
    step_count: int
    action_history: List[str]
    last_action: str
    done: bool
    stop_reason: str
    confidence: float
    latency_so_far: float

    # RL logging fields
    rl_episode_id: str
    rl_step_index: int

    # Optional multi-agent / analysis fields
    active_agent: str
    next_action_recommendation: str
    agent_notes: Dict[str, Any]
    agent_decisions: Dict[str, Any]
    agent_status: Dict[str, Any]
    agent_trace: List[Dict[str, Any]]

    # Role-specific reasoning fields
    retrieval_strategy: str
    rewrite_type: str
    evidence_gap_reason: str
    answer_strategy: str
    verification_outcome: str

    mixed_domain_evidence: bool
    evidence_source_distribution: Dict[str, Any]


    valid_actions: list[str]
    action_mask: list[int]
    controller_mode: str
    rule_action: str
    policy_action: str
    policy_confidence: float
    policy_probabilities: dict
    policy_loaded: bool
    policy_error: str
    chosen_action: str
    controller_source: str
    fallback_used: bool
    controller_decisions: list[dict]