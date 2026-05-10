"""
brain/context_marl_ac/evaluate.py
----------------------------------
Deterministic evaluation script for Context-Engineered MARL RAG.
"""

import argparse
import json
import os
import sys
import time
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
    parser.add_argument("--output-name", type=str, default="context_marl_ac_results.jsonl", help="Output filename")
    return parser.parse_args()

def evaluate():
    args = parse_args()
    cfg.DRY_RUN = args.dry_run
    
    # 1. Initialize Infrastructure
    env = MARLEnv()
    ckpt_manager = MARLCheckpointManager()
    
    actors = build_marl_actors()
    critic = CentralizedCritic()
    
    # 2. Load Checkpoint
    if not args.dry_run:
        try:
            print(f"Loading checkpoint: {args.checkpoint}")
            ckpt_manager.load_checkpoint(actors, critic, filename=args.checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Proceeding with random weights.")
    
    # 3. Load Benchmark
    with open(args.benchmark_path, "r") as f:
        benchmark = json.load(f)
        
    output_path = cfg.FINAL_EVAL_DIR / args.output_name
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Clear output file if exists
    if os.path.exists(output_path):
        os.remove(output_path)
        
    print(f"Starting evaluation on {len(benchmark)} questions...")
    
    # 4. Evaluation Loop
    for q_idx, question_dict in enumerate(tqdm(benchmark, desc="Evaluating")):
        state = env.reset(question_dict)
        trace = []
        
        done = False
        while not done:
            agent_to_act = None
            action_to_take = None
            
            # Deterministic selection: find first agent that can act and take argmax action
            for agent_name in AGENT_NAMES:
                mask = env.get_mask(agent_name)
                if sum(mask) > 0:
                    agent_to_act = agent_name
                    
                    obs = torch.tensor(env.get_obs(agent_name), dtype=torch.float32)
                    mask_t = torch.tensor(mask, dtype=torch.float32)
                    
                    with torch.no_grad():
                        logits = actors[agent_name](obs.unsqueeze(0), mask_t.unsqueeze(0))
                        # Deterministic: argmax
                        action_id = torch.argmax(logits, dim=-1).item()
                        
                    action_to_take = AGENT_ACTIONS[agent_name][action_id]
                    break
            
            if not agent_to_act:
                break
                
            # Execute Step
            new_state, reward, done, info = env.step(agent_to_act, action_to_take)
            trace.append({
                "step": state.num_steps,
                "agent": agent_to_act,
                "action": action_to_take,
                "status": state.final_status
            })
            state = new_state
            
        # 5. Build Result Row
        result = EvalResult(
            question=question_dict.get("question", ""),
            ground_truth=question_dict.get("ground_truth", ""),
            source_file=question_dict.get("source_file"),
            category=question_dict.get("category"),
            difficulty=question_dict.get("difficulty"),
            architecture=cfg.ARCHITECTURE_NAME,
            answer=state.generated_answer,
            contexts=[c.get("text", "") for c in state.retrieved_chunks],
            latency_sec=state.latency_so_far,
            status="ok",
            # MARL fields
            question_id=state.question_id,
            question_type=state.query_type,
            retrieved_chunks=state.retrieved_chunks,
            citations=state.citation_candidates,
            verification_result=state.verification_result,
            citation_support_rate=state.citation_support_rate,
            verification_pass=(state.final_status == "accepted"),
            num_steps=state.num_steps,
            num_llm_calls=state.num_llm_calls,
            trace=trace
        )
        
        # 6. Save result
        with open(output_path, "a") as f:
            f.write(result.to_json() + "\n")
            
    print(f"\nEvaluation Complete. Results saved to {output_path}")

if __name__ == "__main__":
    evaluate()
