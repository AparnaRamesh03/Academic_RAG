from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langgraph.graph import StateGraph, START, END

from state_shared import GraphState
from retriever_shared import retrieve_and_store
from node_rewriter import rewrite_query
from node_context_selector import select_best_context
from node_grader import grade_documents
from node_generator import generate
from node_auditor import audit_answer
from controller import controller_node

def route_from_controller(state: GraphState):
    action = state.get("current_action", "STOP")
    if action == "RETRIEVE":
        return "retrieve"
    elif action == "REWRITE_QUERY":
        return "rewrite_query"
    elif action == "SELECT_CONTEXT":
        return "select_best_context"
    elif action == "GRADE_DOCS":
        return "grade_documents"
    elif action == "GENERATE":
        return "generate"
    elif action == "AUDIT":
        return "audit_answer"
    else:
        return "end"

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("controller", controller_node)
    workflow.add_node("retrieve", retrieve_and_store)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("select_best_context", select_best_context)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("audit_answer", audit_answer)

    workflow.add_edge(START, "controller")

    workflow.add_conditional_edges(
        "controller",
        route_from_controller,
        {
            "retrieve": "retrieve",
            "rewrite_query": "rewrite_query",
            "select_best_context": "select_best_context",
            "grade_documents": "grade_documents",
            "generate": "generate",
            "audit_answer": "audit_answer",
            "end": END,
        },
    )

    workflow.add_edge("retrieve", "controller")
    workflow.add_edge("rewrite_query", "controller")
    workflow.add_edge("select_best_context", "controller")
    workflow.add_edge("grade_documents", "controller")
    workflow.add_edge("generate", "controller")
    workflow.add_edge("audit_answer", "controller")

    return workflow.compile()