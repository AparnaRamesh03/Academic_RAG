"""
brain/context_marl_ac/evaluate.py
----------------------------------
Deterministic evaluation script for Context-Engineered MARL RAG.
Supports learned policy, random, and fixed 'smoke' test policy modes.
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm

# ── sys.path setup ────────────────────────────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
import context_marl_ac.config as cfg
from context_marl_ac.marl.marl_env import MARLEnv
from context_marl_ac.marl.actors import build_marl_actors
from context_marl_ac.marl.centralized_critic import CentralizedCritic
from context_marl_ac.marl.checkpointing import MARLCheckpointManager
from context_marl_ac.schemas.results_schema import EvalResult
from context_marl_ac.schemas.actions import AGENT_ACTIONS, AGENT_NAMES

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Context-Engineered MARL RAG")
    parser.add_argument("--checkpoint", type=str, default="best_reward.pt", help="Checkpoint filename in checkpoints dir")
    parser.add_argument("--benchmark-path", type=str, 
                        default=str(_BRAIN_ROOT.parent / "evaluation" / "standard_benchmark_v3.json"),
                        help="Path to evaluation benchmark JSON")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--policy-mode", type=str, default="learned", 
                        choices=["learned", "random", "smoke"], 
                        help="Action selection policy: learned (argmax), random, or smoke (fixed path)")
    parser.add_argument("--output-name", type=str, default="context_marl_ac_results.jsonl", help="Output filename")
    return parser.parse_args()

def select_smoke_action(agent: str, state: Any, valid_actions: List[str]) -> str:
    """Fixed deterministic policy for smoke testing."""
    if agent == "rewriter":
        return "keyword_rewrite" if "keyword_rewrite" in valid_actions else "no_rewrite"
    
    if agent == "retriever":
        # Do not retrieve more unless we have nothing
        if not state.retrieved_chunks and "hybrid_rerank" in valid_actions:
            return "hybrid_rerank"
        if "hybrid_rerank" in valid_actions:
            return "hybrid_rerank"
        return valid_actions[0]

    if agent == "grader":
        return "loose_filter" if "loose_filter" in valid_actions else "keep_all"

    if agent == "generator":
        # Force answer generation if evidence exists
        if state.selected_evidence:
            if "generate_with_strict_citations" in valid_actions:
                return "generate_with_strict_citations"
            if "generate_answer" in valid_actions:
                return "generate_answer"
        return "abstain_request_more_evidence" if "abstain_request_more_evidence" in valid_actions else valid_actions[0]

    if agent == "verifier":
        # In smoke mode, verifier always chooses accept if an answer was generated
        if state.generated_answer and "accept" in valid_actions:
            return "accept"
        return valid_actions[0]

    return valid_actions[0]

def evaluate():
    args = parse_args()
    cfg.DRY_RUN = args.dry_run
    
    # 1. Initialize Infrastructure
    env = MARLEnv()
    ckpt_manager = MARLCheckpointManager()
    
    actors = build_marl_actors()
    critic = CentralizedCritic()
    
    # 2. Load Checkpoint
    checkpoint_loaded = False
    if not args.dry_run and args.policy_mode == "learned":
        try:
            print(f"Loading checkpoint: {args.checkpoint}")
            ckpt_manager.load_checkpoint(actors, critic, filename=args.checkpoint)
            checkpoint_loaded = True
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Falling back to policy-mode: smoke if not specified.")
            if args.policy_mode == "learned":
                args.policy_mode = "smoke"
    
    # 3. Load Benchmark
    if args.benchmark_path.endswith(".jsonl"):
        with open(args.benchmark_path, "r") as f:
            benchmark = [json.loads(line) for line in f if line.strip()]
    else:
        with open(args.benchmark_path, "r") as f:
            benchmark = json.load(f)
        
    output_path = cfg.FINAL_EVAL_DIR / args.output_name
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Clear output file if exists
    if os.path.exists(output_path):
        os.remove(output_path)
        
    print(f"Starting evaluation (mode={args.policy_mode}) on {len(benchmark)} questions...")
    
    # 4. Evaluation Loop
    for q_idx, question_dict in enumerate(tqdm(benchmark, desc="Evaluating")):
        state = env.reset(question_dict, index=q_idx + 1)
        trace = []
        
        done = False
        while not done:
            agent_to_act = None
            action_to_take = None
            
            # Find next agent
            for agent_name in AGENT_NAMES:
                mask = env.get_mask(agent_name)
                if sum(mask) > 0:
                    agent_to_act = agent_name
                    valid_actions = [AGENT_ACTIONS[agent_name][i] for i, m in enumerate(mask) if m == 1]
                    
                    if args.policy_mode == "smoke":
                        action_to_take = select_smoke_action(agent_name, state, valid_actions)
                    elif args.policy_mode == "random":
                        action_to_take = random.choice(valid_actions)
                    else:
                        # Learned policy (Argmax)
                        obs = torch.tensor(env.get_obs(agent_name), dtype=torch.float32)
                        mask_t = torch.tensor(mask, dtype=torch.float32)
                        with torch.no_grad():
                            logits = actors[agent_name](obs.unsqueeze(0), mask_t.unsqueeze(0))
                            action_id = torch.argmax(logits, dim=-1).item()
                        action_to_take = AGENT_ACTIONS[agent_name][action_id]
                    break
            
            if not agent_to_act:
                break
                
            # Execute Step
            new_state, reward, done, info = env.step(agent_to_act, action_to_take)
            
            step_data = {
                "step": state.num_steps,
                "agent": agent_to_act,
                "action": action_to_take,
                "status": state.final_status,
                **new_state.to_debug_dict()
            }
            trace.append(step_data)
            state = new_state
            
        # 5. Build Result Row
        final_result = {
            "question":      question_dict.get("question", ""),
            "ground_truth":  question_dict.get("ground_truth", ""),
            "source_file":   question_dict.get("source_file"),
            "category":      question_dict.get("category"),
            "difficulty":    question_dict.get("difficulty"),
            "question_id":   state.question_id,
            "final_answer":  state.generated_answer,
            "selected_evidence": state.selected_evidence,
            "latency_sec":   state.latency_so_far,
            "num_steps":     state.num_steps,
            "num_llm_calls": state.num_llm_calls,
            "token_usage":   state.token_usage,
            "final_status":  state.final_status,
            "verification_pass": (state.final_status == "accepted"),
            "policy_mode":   args.policy_mode,
            "trace":         trace,
            **state.to_debug_dict()
        }
        
        # 6. Save result
        with open(output_path, "a") as f:
            f.write(json.dumps(final_result, ensure_ascii=False) + "\n")
            
    print(f"\nEvaluation Complete. Results saved to {output_path}")

if __name__ == "__main__":
    evaluate()
