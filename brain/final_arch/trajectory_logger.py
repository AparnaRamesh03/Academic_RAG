import json
import os
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAJECTORY_FILE = LOG_DIR / "trajectories.jsonl"

def log_episode(state: dict, reward: float):
    """
    Logs the final state and reward of an episode for offline RL training.
    """
    record = {
        "question": state.get("original_query"),
        "step_count": state.get("step_count", 0),
        "action_history": state.get("action_history", []),
        "latency": state.get("latency_so_far", 0.0),
        "reward": reward,
        "citations_pass": state.get("citations_pass", False),
        "crag_retries": state.get("crag_retries", 0),
        "verify_retries": state.get("verify_retries", 0),
        "stop_reason": state.get("stop_reason", "unknown")
    }
    
    with open(TRAJECTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
