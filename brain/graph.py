from langgraph.graph import StateGraph, START, END
from state import GraphState

from node_retriever import retrieve_and_split
from node_grader import grade_documents
from node_rewriter import crag_rewrite
from node_generator import generate
from node_auditor import verify_citations

# --- Stub Nodes (For Testing Flow) ---

def safe_refusal(state: GraphState):
    print("[Node Refusal] Safe Refusal triggered...")
    return {"generation": "(Stub) System routed to Safe Refusal Node (Retry Limit Reached)"}

def decide_to_grade(state: GraphState):
    """
    Conditional Edge from Node 1: Skip grading if naive.
    """
    if state.get("ablation_mode", "full") == "naive":
        print("  -> Ablation: 'naive' mode skips Grader/CRAG. Proceed directly to GENERATE.")
        return "generate"
    return "grade_documents"

def decide_to_generate(state: GraphState):
    """
    Conditional Edge from Node 2: Choose routing based on graded_docs and retries.
    """
    graded_docs = state["graded_docs"]
    crag_retries = state.get("crag_retries", 0)
    ablation_mode = state.get("ablation_mode", "full")
    
    if len(graded_docs) > 0:
        print("  -> Decision: Docs are relevant. Proceed to GENERATE.")
        return "generate"
    elif ablation_mode == "no_crag" or crag_retries >= 3:
        print("  -> Decision: Retry limit reached (or no_crag mode). Proceed to SAFE REFUSAL.")
        return "safe_refusal"
    else:
        print("  -> Decision: No relevant docs. Proceed to CRAG REWRITE.")
        return "crag_rewrite"

def decide_to_finalize(state: GraphState):
    """
    Conditional Edge from Node 5: Check if Auditor passed the generation.
    """
    if state.get("citations_pass", True):
        print("  -> Decision: Citations verified. Proceed to END.")
        return "end"
    elif state.get("verify_retries", 0) >= 3:
        print("  -> Decision: Hallucination loop limit reached. Proceed to END.")
        return "end"
    else:
        print("  -> Decision: Hallucination detected. Route back to GENERATE.")
        return "generate"

def decide_to_audit(state: GraphState):
    """
    Conditional Edge from Node 4: Skip auditor if naive or no_auditor.
    """
    ablation_mode = state.get("ablation_mode", "full")
    if ablation_mode in ["naive", "no_auditor"]:
        print("  -> Ablation: Bypassing auditor. Proceed to END.")
        return "end"
    return "verify_citations"

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_and_split", retrieve_and_split)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("crag_rewrite", crag_rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("verify_citations", verify_citations)
    workflow.add_node("safe_refusal", safe_refusal)

    workflow.add_edge(START, "retrieve_and_split")
    workflow.add_conditional_edges(
        "retrieve_and_split",
        decide_to_grade,
        {
            "generate": "generate",
            "grade_documents": "grade_documents"
        }
    )

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "crag_rewrite": "crag_rewrite",
            "safe_refusal": "safe_refusal"
        }
    )

    workflow.add_conditional_edges(
        "generate",
        decide_to_audit,
        {
            "end": END,
            "verify_citations": "verify_citations"
        }
    )
    
    workflow.add_conditional_edges(
        "verify_citations",
        decide_to_finalize,
        {
            "generate": "generate",
            "end": END
        }
    )

    workflow.add_edge("crag_rewrite", "retrieve_and_split") # Loop back to Node 1
    workflow.add_edge("safe_refusal", END)

    return workflow.compile()
