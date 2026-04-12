from typing import List, Dict, Any, TypedDict

class GraphState(TypedDict):
    original_query: str
    search_query: str
    ablation_mode: str
    candidate_docs: List[Dict[str, Any]]
    weak_signal_docs: List[Dict[str, Any]]
    graded_docs: List[Dict[str, Any]]
    generation: str
    crag_retries: int
    verify_retries: int
    citations_pass: bool
    auditor_feedback: str
