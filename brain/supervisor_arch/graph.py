from pathlib import Path
import sys
import time
from typing import Callable, Dict, Any

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
SUPERVISOR_DIR = Path(__file__).resolve().parent
BRAIN_DIR = SUPERVISOR_DIR.parent
FINAL_COMBINED_DIR = BRAIN_DIR / "final_arch"

for path in [str(BRAIN_DIR), str(FINAL_COMBINED_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from langgraph.graph import StateGraph, START, END

from state_shared import GraphState
from retriever_shared import retrieve_and_store
from reranker_shared import rerank_retrieved_docs

# Reuse worker logic from the existing final_combined architecture
from node_retrieval_evaluator import evaluate_retrieval
from node_rewriter import rewrite_query
from node_context_selector import select_best_context
from node_grader import grade_documents
from node_generator import generate
from node_auditor import audit_answer

from supervisor import supervisor_step, choose_next_action, finish_step
from config import RERANK_INPUT_TOP_K, RERANK_OUTPUT_TOP_K


def _run_worker(
    state: GraphState,
    action_name: str,
    worker_fn: Callable[[GraphState], Dict[str, Any]],
):
    start = time.perf_counter()
    updates = worker_fn(state)
    elapsed = time.perf_counter() - start

    history = list(state.get("action_history", []))
    history.append(action_name)

    return {
        **updates,
        "last_action": action_name,
        "step_count": int(state.get("step_count", 0)) + 1,
        "action_history": history,
        "latency_so_far": float(state.get("latency_so_far", 0.0)) + elapsed,
    }


def retrieve_original_node(state: GraphState):
    return _run_worker(state, "retrieve_original", retrieve_and_store)


def retrieve_rewritten_node(state: GraphState):
    return _run_worker(state, "retrieve_rewritten", retrieve_and_store)


def rerank_original_node(state: GraphState):
    return _run_worker(
        state,
        "rerank_original",
        lambda s: rerank_retrieved_docs(
            s,
            input_top_k=RERANK_INPUT_TOP_K,
            output_top_k=RERANK_OUTPUT_TOP_K,
        ),
    )


def rerank_rewritten_node(state: GraphState):
    return _run_worker(
        state,
        "rerank_rewritten",
        lambda s: rerank_retrieved_docs(
            s,
            input_top_k=RERANK_INPUT_TOP_K,
            output_top_k=RERANK_OUTPUT_TOP_K,
        ),
    )


def evaluate_retrieval_node(state: GraphState):
    return _run_worker(state, "evaluate_retrieval", evaluate_retrieval)


def rewrite_query_node(state: GraphState):
    return _run_worker(state, "rewrite_query", rewrite_query)


def select_best_context_node(state: GraphState):
    return _run_worker(state, "select_best_context", select_best_context)


def grade_documents_node(state: GraphState):
    return _run_worker(state, "grade_documents", grade_documents)


def generate_node(state: GraphState):
    return _run_worker(state, "generate", generate)


def audit_answer_node(state: GraphState):
    return _run_worker(state, "audit_answer", audit_answer)


def supervisor_node(state: GraphState):
    return supervisor_step(state)


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("finish", finish_step)

    workflow.add_node("retrieve_original", retrieve_original_node)
    workflow.add_node("rerank_original", rerank_original_node)
    workflow.add_node("evaluate_retrieval", evaluate_retrieval_node)

    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve_rewritten", retrieve_rewritten_node)
    workflow.add_node("rerank_rewritten", rerank_rewritten_node)

    workflow.add_node("select_best_context", select_best_context_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("audit_answer", audit_answer_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        choose_next_action,
        {
            "retrieve_original": "retrieve_original",
            "rerank_original": "rerank_original",
            "evaluate_retrieval": "evaluate_retrieval",
            "rewrite_query": "rewrite_query",
            "retrieve_rewritten": "retrieve_rewritten",
            "rerank_rewritten": "rerank_rewritten",
            "select_best_context": "select_best_context",
            "grade_documents": "grade_documents",
            "generate": "generate",
            "audit_answer": "audit_answer",
            "finish": "finish",
        },
    )

    workflow.add_edge("retrieve_original", "supervisor")
    workflow.add_edge("rerank_original", "supervisor")
    workflow.add_edge("evaluate_retrieval", "supervisor")
    workflow.add_edge("rewrite_query", "supervisor")
    workflow.add_edge("retrieve_rewritten", "supervisor")
    workflow.add_edge("rerank_rewritten", "supervisor")
    workflow.add_edge("select_best_context", "supervisor")
    workflow.add_edge("grade_documents", "supervisor")
    workflow.add_edge("generate", "supervisor")
    workflow.add_edge("audit_answer", "supervisor")

    workflow.add_edge("finish", END)
    
    return workflow.compile()