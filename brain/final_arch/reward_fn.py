def compute_reward(state: dict) -> float:
    """
    Compute reward based on:
    reward = answer_quality + grounding_quality - step_penalty - latency_penalty
    """
    reward = 0.0
    
    # Grounding & Answer Quality (Audit Pass/Fail)
    citations_pass = state.get("citations_pass", False)
    generation = state.get("generation", "").strip()
    
    if citations_pass and generation:
        reward += 10.0 # High reward for a supported answer
    elif generation:
        reward -= 5.0 # Penalize unsupported or failed audit answers
    else:
        reward -= 10.0 # Penalize empty answers
        
    # Step Penalty
    step_count = state.get("step_count", 0)
    reward -= step_count * 0.5
    
    # Latency Penalty (Simulated as penalty per action/retry for now)
    # If latency_so_far is tracked, we can subtract it directly
    latency = state.get("latency_so_far", 0.0)
    reward -= latency * 0.1
    
    return reward
