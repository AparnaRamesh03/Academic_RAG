from pathlib import Path
import sys

# Allow this folder to import shared files from ../
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langgraph.graph import StateGraph, START, END

from state_shared import GraphState
from retriever_shared import retrieve_and_store
from node_grader import grade_documents
from node_generator import generate
from config import GRADE_TOP_K


def select_docs_for_grading(state: GraphState):
    """
    Take only the top few retrieved docs and pass those into the grader.
    """
    retrieved_docs = state.get("retrieved_docs", [])
    candidate_docs = retrieved_docs[:GRADE_TOP_K]

    print(
        f"\n[Self-RAG Grader] Selecting top {len(candidate_docs)} "
        f"retrieved docs for grading..."
    )

    return {"candidate_docs": candidate_docs}


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_and_store", retrieve_and_store)
    workflow.add_node("select_docs_for_grading", select_docs_for_grading)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve_and_store")
    workflow.add_edge("retrieve_and_store", "select_docs_for_grading")
    workflow.add_edge("select_docs_for_grading", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()