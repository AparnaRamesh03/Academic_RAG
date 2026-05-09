import sys
from pathlib import Path

# Add brain to path
BRAIN_DIR = Path(__file__).resolve().parents[1]
if str(BRAIN_DIR) not in sys.path:
    sys.path.append(str(BRAIN_DIR))

from rl_arch.policy_runtime import PolicyRuntime

def test_deployment():
    print("Initializing PolicyRuntime...")
    runtime = PolicyRuntime()
    
    if runtime.error:
        print(f"Error: {runtime.error}")
        return
    
    print(f"Successfully loaded model from: {runtime._checkpoint_path()}")
    print(f"Model Architecture: {type(runtime.model).__name__}")
    
    # Test a dummy prediction
    test_state = {
        "retrieved_docs": [],
        "generation": "",
        "auditor_feedback": "",
        "verify_retries": 0,
        "step_count": 0
    }
    
    valid_actions = ["retrieve", "rewrite_query"]
    res = runtime.predict(test_state, valid_actions)
    print(f"Test Prediction: Action='{res['action']}', Confidence={res['confidence']:.4f}")

if __name__ == "__main__":
    test_deployment()
