from typing import List, Dict, Any, TypedDict


class AgenticGraphState(TypedDict):
    """
    Extended graph state for the agentic_scholar architecture.
    Adds scholar fallback tracking on top of the base GraphState fields.
    """
    # ── Core query fields ────────────────────────────────────────────────────
    original_query: str
    search_query: str

    # ── Retrieval working sets ───────────────────────────────────────────────
    retrieved_docs: List[Dict[str, Any]]
    candidate_docs: List[Dict[str, Any]]
    weak_signal_docs: List[Dict[str, Any]]
    graded_docs: List[Dict[str, Any]]

    # ── Generation output ────────────────────────────────────────────────────
    generation: str

    # ── Retry / audit bookkeeping ────────────────────────────────────────────
    crag_retries: int
    verify_retries: int
    citations_pass: bool
    auditor_feedback: str

    # ── Scholar fallback fields (new) ────────────────────────────────────────
    scholar_used: bool                   # True once the scholarly API fallback has fired
    scholar_docs: List[Dict[str, Any]]   # Raw docs returned by the scholarly APIs
