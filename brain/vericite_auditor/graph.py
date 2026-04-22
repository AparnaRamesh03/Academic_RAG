from pathlib import Path
import sys

# Allow this folder to import shared files from ../
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langgraph.graph import StateGraph, START, END

from state_shared import GraphState
from retriever_shared import retrieve_and_store
from node_generator import generate
from node_auditor import audit_answer
from config import TOP_K, MAX_AUDIT_RETRIES


def select_top_k_docs(state: GraphState):
    """
    Use the simple baseline retrieval strategy:
    take top-k retrieved docs directly.
    """
    retrieved_docs = state.get("retrieved_docs", [])
    selected_docs = retrieved_docs[:TOP_K]

    print(
        f"\n[VeriCite Auditor] Selecting top {len(selected_docs)} "
        f"retrieved docs directly..."
    )

    return {
        "graded_docs": selected_docs
    }


def route_after_audit(state: GraphState):
    """
    If audit passes, stop.
    If audit fails and we still have retry budget, regenerate once.
    Otherwise stop with the last generated answer.
    """
    if state.get("citations_pass", False):
        return "end"

    if state.get("verify_retries", 0) <= MAX_AUDIT_RETRIES:
        return "retry_generation"

    return "end"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_and_store", retrieve_and_store)
    workflow.add_node("select_top_k_docs", select_top_k_docs)
    workflow.add_node("generate", generate)
    workflow.add_node("audit_answer", audit_answer)

    workflow.add_edge(START, "retrieve_and_store")
    workflow.add_edge("retrieve_and_store", "select_top_k_docs")
    workflow.add_edge("select_top_k_docs", "generate")
    workflow.add_edge("generate", "audit_answer")

    workflow.add_conditional_edges(
        "audit_answer",
        route_after_audit,
        {
            "retry_generation": "generate",
            "end": END,
        },
    )

    return workflow.compile()