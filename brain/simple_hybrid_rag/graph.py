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
from config import TOP_K


def select_top_k_docs(state: GraphState):
    """
    Simple Hybrid RAG:
    take the top-k retrieved docs directly, with no grader,
    no rewrite loop, and no auditor.
    """
    retrieved_docs = state.get("retrieved_docs", [])
    selected_docs = retrieved_docs[:TOP_K]

    print(
        f"\n[Simple Hybrid RAG] Selecting top {len(selected_docs)} "
        f"retrieved docs directly for generation..."
    )

    return {
        "graded_docs": selected_docs
    }


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_and_store", retrieve_and_store)
    workflow.add_node("select_top_k_docs", select_top_k_docs)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve_and_store")
    workflow.add_edge("retrieve_and_store", "select_top_k_docs")
    workflow.add_edge("select_top_k_docs", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()