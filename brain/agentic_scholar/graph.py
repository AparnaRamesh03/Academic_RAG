"""
graph.py — Agentic Scholar LangGraph Pipeline

Extends final_arch with a trusted scholarly API fallback layer.

Flow:
    START
      └─► retrieve_original
            └─► evaluate_retrieval
                  ├─ SUFFICIENT ──────────────────────────────► grade_documents
                  └─ INSUFFICIENT ► rewrite_query
                                      └─► retrieve_rewritten
                                            └─► select_best_context
                                                  └─► grade_documents
                                                        ├─ docs found ──────────► generate
                                                        └─ empty & no scholar ► scholar_search
                                                                                    └─► generate
                                                                                          └─► audit_answer
                                                                                                ├─ PASS ► END
                                                                                                └─ FAIL ► generate (retry)

Key routing decisions:
    route_after_retrieval_eval  → SUFFICIENT / rewrite_query
    route_after_grading         → generate  / scholar_search  (new)
    route_after_audit           → end        / retry_generation
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langgraph.graph import StateGraph, START, END

from state_agentic import AgenticGraphState
from retriever_shared import retrieve_and_store
from node_retrieval_evaluator import evaluate_retrieval
from node_rewriter import rewrite_query
from node_context_selector import select_best_context
from node_grader import grade_documents
from node_generator import generate
from node_auditor import audit_answer
from node_scholar_search import scholar_search
from config import MAX_REWRITE_ROUNDS, MAX_AUDIT_RETRIES


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_after_retrieval_eval(state: AgenticGraphState) -> str:
    """
    After local retrieval evaluation:
    - SUFFICIENT  → go straight to grading
    - INSUFFICIENT → try a query rewrite first (up to MAX_REWRITE_ROUNDS)
    """
    if state.get("citations_pass", False):
        return "grade_documents"

    if state.get("crag_retries", 0) < MAX_REWRITE_ROUNDS:
        return "rewrite_query"

    return "grade_documents"


def route_after_grading(state: AgenticGraphState) -> str:
    """
    After local document grading:
    - If graded_docs is non-empty → proceed to generate (normal path)
    - If graded_docs is empty AND we haven't used the scholar fallback yet
      → trigger scholar_search as a last resort
    - If graded_docs is empty AND scholar fallback already fired
      → proceed to generate anyway (generator handles empty gracefully)
    """
    graded_docs = state.get("graded_docs", [])
    scholar_used = state.get("scholar_used", False)

    if not graded_docs and not scholar_used:
        return "scholar_search"

    return "generate"


def route_after_audit(state: AgenticGraphState) -> str:
    """
    After answer auditing:
    - PASS → done
    - FAIL + retries left → regenerate with auditor feedback
    - FAIL + out of retries → done (best effort)
    """
    if state.get("citations_pass", False):
        return "end"

    if state.get("verify_retries", 0) <= MAX_AUDIT_RETRIES:
        return "retry_generation"

    return "end"


# ── Graph Construction ────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(AgenticGraphState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("retrieve_original", retrieve_and_store)
    workflow.add_node("evaluate_retrieval", evaluate_retrieval)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve_rewritten", retrieve_and_store)
    workflow.add_node("select_best_context", select_best_context)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("scholar_search", scholar_search)   # ← new
    workflow.add_node("generate", generate)
    workflow.add_node("audit_answer", audit_answer)

    # ── Edges ──────────────────────────────────────────────────────────────────
    workflow.add_edge(START, "retrieve_original")
    workflow.add_edge("retrieve_original", "evaluate_retrieval")

    # After retrieval evaluation: sufficient → grade, else → rewrite
    workflow.add_conditional_edges(
        "evaluate_retrieval",
        route_after_retrieval_eval,
        {
            "grade_documents": "grade_documents",
            "rewrite_query": "rewrite_query",
        },
    )

    # CRAG rewrite path
    workflow.add_edge("rewrite_query", "retrieve_rewritten")
    workflow.add_edge("retrieve_rewritten", "select_best_context")
    workflow.add_edge("select_best_context", "grade_documents")

    # After grading: generate or fall back to scholarly APIs (new branch)
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "scholar_search": "scholar_search",
            "generate": "generate",
        },
    )

    # Scholar fallback injects graded_docs directly → skip re-grading
    workflow.add_edge("scholar_search", "generate")

    # Generation → audit
    workflow.add_edge("generate", "audit_answer")

    # After audit: done or retry generation
    workflow.add_conditional_edges(
        "audit_answer",
        route_after_audit,
        {
            "retry_generation": "generate",
            "end": END,
        },
    )

    return workflow.compile()
