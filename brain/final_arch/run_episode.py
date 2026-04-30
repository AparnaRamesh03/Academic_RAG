import time
from graph import build_graph
from trajectory_logger import log_episode
from reward_fn import compute_reward

app_graph = build_graph()

def run_episode(question: str):
    """
    Run a single episode (benchmark question) through the controller-driven graph.
    """
    print(f"\n--- Starting Episode for Question: '{question}' ---")
    
    initial_state = {
        "original_query": question,
        "search_query": question,
        "retrieved_docs": [],
        "candidate_docs": [],
        "weak_signal_docs": [],
        "graded_docs": [],
        "generation": "",
        "crag_retries": 0,
        "verify_retries": 0,
        "citations_pass": True,
        "auditor_feedback": "",
        "step_count": 0,
        "action_history": [],
        "current_phase": "start",
        "retrieval_rounds": 0,
        "used_rewrite": False,
        "used_grade": False,
        "used_audit": False,
        "top_retrieval_scores": [],
        "num_distinct_sources": 0,
        "question_category": "unknown",
        "question_difficulty": "unknown",
        "latency_so_far": 0.0,
        "done": False,
        "stop_reason": "",
        "current_action": ""
    }

    start_time = time.time()
    final_state = app_graph.invoke(initial_state)
    latency = time.time() - start_time
    
    # Update latency in state for logging and reward
    final_state["latency_so_far"] = latency
    
    reward = compute_reward(final_state)
    
    print("\n--- Episode Finished ---")
    print(f"Action History: {final_state.get('action_history')}")
    print(f"Stop Reason: {final_state.get('stop_reason')}")
    print(f"Reward: {reward}")
    print(f"Latency: {latency:.2f}s")
    
    log_episode(final_state, reward)
    print("Trajectory logged successfully.")
    
if __name__ == "__main__":
    test_question = "What is the role of memory in agentic RAG systems?"
    run_episode(test_question)
