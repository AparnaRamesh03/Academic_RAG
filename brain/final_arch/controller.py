import os
from langchain_core.messages import HumanMessage
from llm_config import build_groq_llm
from state_shared import GraphState

llm = build_groq_llm(temperature=0.0)

ACTIONS = [
    "RETRIEVE",
    "REWRITE_QUERY",
    "SELECT_CONTEXT",
    "GRADE_DOCS",
    "GENERATE",
    "AUDIT",
    "STOP"
]

def controller_node(state: GraphState):
    """
    The learned control node (currently implemented as a prompt-based supervisor for Baseline 2).
    It looks at the current state and decides the next action.
    """
    
    # Update step count
    step_count = state.get("step_count", 0)
    state["step_count"] = step_count + 1
    
    if state.get("done", False) or step_count >= 15:
        state["current_action"] = "STOP"
        state["stop_reason"] = "budget_exhausted" if step_count >= 15 else "done"
        return state
        
    action_history = state.get("action_history", [])
    
    prompt = f"""You are the controller for a modular RAG system. 
Your job is to look at the current state and decide the NEXT SINGLE ACTION to take.

AVAILABLE ACTIONS:
- RETRIEVE: Fetch documents for the search_query.
- REWRITE_QUERY: Rewrite the search_query if retrieved docs are poor.
- SELECT_CONTEXT: Select the best context subset.
- GRADE_DOCS: Filter out irrelevant documents.
- GENERATE: Generate an answer based on graded_docs.
- AUDIT: Check if the generation is grounded and answers the question.
- STOP: Terminate the episode (either successful answer or give up).

CURRENT STATE:
- Step count: {step_count}
- Action history: {action_history}
- Retrieved docs count: {len(state.get("retrieved_docs", []))}
- Graded docs count: {len(state.get("graded_docs", []))}
- Has generation: {bool(state.get("generation", "").strip())}
- Citations pass (audit): {state.get("citations_pass", False)}
- Crag retries: {state.get("crag_retries", 0)}
- Verify retries: {state.get("verify_retries", 0)}

DECISION RULES:
1. If you just started (0 docs), you should RETRIEVE.
2. If you retrieved docs but haven't graded them, you might want to GRADE_DOCS or SELECT_CONTEXT.
3. If you graded docs but haven't generated, GENERATE.
4. If you generated, you must AUDIT.
5. If AUDIT fails, you might REWRITE_QUERY, RETRIEVE again, or GENERATE again.
6. If AUDIT passes, STOP.
7. If you are stuck in a loop or tried too many times, STOP.

OUTPUT EXACTLY ONE OF THE AVAILABLE ACTIONS AND NOTHING ELSE:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().upper()
    
    if decision not in ACTIONS:
        # Fallback heuristic
        if not state.get("retrieved_docs"):
            decision = "RETRIEVE"
        elif state.get("citations_pass", False):
            decision = "STOP"
        else:
            decision = "STOP"
            
    if decision == "STOP":
        state["done"] = True
        state["stop_reason"] = "controller_stopped"
        
    action_history.append(decision)
    state["action_history"] = action_history
    state["current_action"] = decision
    
    return state
