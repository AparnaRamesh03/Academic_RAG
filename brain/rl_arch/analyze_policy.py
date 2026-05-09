"""
analyze_policy.py
-----------------
Analysis script to compare the learned policy against the rule-based supervisor
using the trajectory JSONL logs.
"""
import json
from pathlib import Path
from collections import defaultdict

CURRENT_DIR = Path(__file__).resolve().parent
TRAJECTORIES_DIR = CURRENT_DIR / "data" / "trajectories"

def analyze_trajectories():
    if not TRAJECTORIES_DIR.exists():
        print(f"Directory not found: {TRAJECTORIES_DIR}")
        return

    total_transitions = 0
    agreements = 0
    disagreements = 0
    fallbacks = 0
    
    rewrite_skipped_attempts = 0
    early_stop_attempts = 0
    
    outcomes_by_source = defaultdict(lambda: defaultdict(int))
    final_outcomes = defaultdict(int)

    for file_path in TRAJECTORIES_DIR.glob("*.jsonl"):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    
                    if data.get("event") == "transition":
                        total_transitions += 1
                        
                        ctrl = data.get("controller", {})
                        rule = ctrl.get("rule_action", "")
                        policy = ctrl.get("policy_action", "")
                        source = ctrl.get("controller_source", "")
                        
                        if ctrl.get("fallback_used", False):
                            fallbacks += 1
                            
                        if rule and policy:
                            if rule == policy:
                                agreements += 1
                            else:
                                disagreements += 1
                                
                        # Detect if policy tried to skip rewrite
                        if rule == "rewrite_query" and policy != "rewrite_query":
                            rewrite_skipped_attempts += 1
                            
                        # Detect if policy tried to early stop
                        if rule == "generate" and policy == "finish":
                            early_stop_attempts += 1

                    elif data.get("event") == "episode_end":
                        outcome = data.get("verification_outcome", "unknown")
                        final_outcomes[outcome] += 1
                        
                        # We don't have exactly which source caused the outcome, 
                        # but we can log the outcome distribution.
                        
                except json.JSONDecodeError:
                    continue

    print("=== Policy vs Rule Analysis ===")
    print(f"Total Transitions Logged: {total_transitions}")
    
    if total_transitions > 0:
        print(f"Fallbacks Triggered: {fallbacks} ({(fallbacks/total_transitions)*100:.2f}%)")
        
    if agreements + disagreements > 0:
        total_decisions = agreements + disagreements
        print(f"Policy/Rule Agreements: {agreements} ({(agreements/total_decisions)*100:.2f}%)")
        print(f"Policy/Rule Disagreements: {disagreements} ({(disagreements/total_decisions)*100:.2f}%)")
    else:
        print("No comparative policy/rule data found (Run benchmark in policy_shadow mode!)")

    print(f"Policy attempted to skip rewrite: {rewrite_skipped_attempts} times")
    print(f"Policy attempted early stop: {early_stop_attempts} times")
    
    print("\n=== Final Verification Outcomes ===")
    for k, v in final_outcomes.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    analyze_trajectories()
