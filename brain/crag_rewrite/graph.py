from pathlib import Path
import sys

# Allow this folder to import shared files from ../
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langgraph.graph import StateGraph, START, END

from state_shared import GraphState
from retriever_shared import retrieve_and_store
from node_retrieval_evaluator import evaluate_retrieval
from node_rewriter import rewrite_query
from node_context_selector import select_best_context
from node_generator import generate


def route_after_retrieval_eval(state: GraphState):
    """
    If the original retrieval is sufficient, answer directly.
    Otherwise go into the rewrite path.
    """
    if state.get("citations_pass", False):
        return "generate"
    return "rewrite_query"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_original", retrieve_and_store)
    workflow.add_node("evaluate_retrieval", evaluate_retrieval)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve_rewritten", retrieve_and_store)
    workflow.add_node("select_best_context", select_best_context)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve_original")
    workflow.add_edge("retrieve_original", "evaluate_retrieval")

    workflow.add_conditional_edges(
        "evaluate_retrieval",
        route_after_retrieval_eval,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve_rewritten")
    workflow.add_edge("retrieve_rewritten", "select_best_context")
    workflow.add_edge("select_best_context", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()