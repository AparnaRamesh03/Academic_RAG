"""
brain/context_marl_ac/train.py
-------------------------------
Main training script for the Context-Engineered MARL Actor-Critic RAG.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
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
from context_marl_ac.marl.trainer import MARLTrainer
from context_marl_ac.marl.checkpointing import MARLCheckpointManager
from context_marl_ac.logging_utils import MARLLogger
from context_marl_ac.schemas.trajectory import Episode, TrajectoryStep
from context_marl_ac.schemas.actions import AGENT_ACTIONS, AGENT_NAMES

def parse_args():
    parser = argparse.ArgumentParser(description="Train Context-Engineered MARL RAG")
    parser.add_argument("--run-name", type=str, default="marl_run_01", help="Name for this training run")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=cfg.HIDDEN_DIM, help="Hidden dimension for NN")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--benchmark-path", type=str, 
                        default=str(_BRAIN_ROOT.parent / "evaluation" / "standard_benchmark_v3.json"),
                        help="Path to training benchmark JSON")
    parser.add_argument("--checkpoint-every", type=int, default=cfg.CHECKPOINT_EVERY, help="Save model every N episodes")
    return parser.parse_args()

def load_benchmark(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Warning: Benchmark not found at {path}. Using dummy question.")
        return [{"question": "What is the capital of France?", "ground_truth": "Paris", "question_id": "dummy"}]
    with open(path, "r") as f:
        return json.load(f)

def train():
    args = parse_args()
    
    # 1. Update global config with CLI args
    cfg.DRY_RUN = args.dry_run
    
    # 2. Initialize Infrastructure
    env = MARLEnv()
    logger = MARLLogger(run_name=args.run_name)
    ckpt_manager = MARLCheckpointManager()
    
    actors = build_marl_actors()
    critic = CentralizedCritic()
    trainer = MARLTrainer(actors, critic, lr=args.lr)
    
    benchmark = load_benchmark(args.benchmark_path)
    
    print(f"Starting training run: {args.run_name}")
    print(f"Episodes: {args.episodes}, Dry-Run: {args.dry_run}")
    
    best_reward = -float("inf")
    
    # 3. Main Training Loop
    for ep_idx in tqdm(range(1, args.episodes + 1), desc="Training"):
        # Select a random question from benchmark
        question_dict = random.choice(benchmark)
        
        # Reset Env
        state = env.reset(question_dict)
        episode = Episode(
            episode_id=state.question_id,
            question_id=state.question_id,
            question=state.user_query,
            query_type=state.query_type,
            query_complexity=state.query_complexity
        )
        
        done = False
        step_idx = 0
        
        # 4. Episode Rollout
        while not done:
            # Deterministic agent rotation or fixed sequence? 
            # MARL often uses sequential acting for RAG pipelines.
            # Sequence: Rewriter -> Retriever -> Grader -> Generator -> Verifier
            # If Verifier rejects, it can loop back.
            
            # Simple heuristic: pick agent based on action masking or fixed flow
            # For this arch, we iterate through agents until one can act
            agent_to_act = None
            action_to_take = None
            action_id = -1
            action_prob = 0.0
            
            # Determine which agent should act next
            # We check agents in a logical sequence
            for agent_name in AGENT_NAMES:
                mask = env.get_mask(agent_name)
                if sum(mask) > 0:
                    # This agent can act. Sample an action.
                    agent_to_act = agent_name
                    
                    obs = torch.tensor(env.get_obs(agent_name), dtype=torch.float32)
                    mask_t = torch.tensor(mask, dtype=torch.float32)
                    
                    # Forward pass
                    with torch.no_grad():
                        logits = actors[agent_name](obs.unsqueeze(0), mask_t.unsqueeze(0))
                        probs = torch.softmax(logits, dim=-1)
                        
                        # Sample action
                        m = torch.distributions.Categorical(probs)
                        action_tensor = m.sample()
                        action_id = action_tensor.item()
                        action_prob = probs[0, action_id].item()
                        
                    action_to_take = AGENT_ACTIONS[agent_name][action_id]
                    break
            
            if not agent_to_act:
                # No one can act? Terminal.
                break
                
            # Execute Step
            old_state_copy = state.to_dict() # for logging/transition
            prev_global_feats = env.get_global_features()
            
            with torch.no_grad():
                v_s = critic(torch.tensor(prev_global_feats, dtype=torch.float32).unsqueeze(0)).item()
            
            new_state, reward, done, info = env.step(agent_to_act, action_to_take)
            
            # Record transition
            step = TrajectoryStep(
                episode_id=state.question_id,
                step=step_idx,
                agent=agent_to_act,
                observation=env.get_obs(agent_to_act), # Note: this is new obs
                global_features=prev_global_feats,
                obs_names=[], # placeholder
                valid_actions=AGENT_ACTIONS[agent_to_act],
                action_mask=env.get_mask(agent_to_act),
                selected_action=action_to_take,
                action_id=action_id,
                action_probability=action_prob,
                log_probability=0.0, # filled in trainer
                entropy=0.0, # filled in trainer
                critic_value=v_s,
                reward=reward,
                done=done,
                latency_step=0.0 # updated in log_action
            )
            episode.add_step(step)
            
            step_idx += 1
            state = new_state
            
        # 5. Finalize Episode
        episode.total_reward = env.get_global_reward()
        episode.generated_answer = state.generated_answer
        episode.final_status = state.final_status
        episode.num_steps = state.num_steps
        episode.num_llm_calls = state.num_llm_calls
        episode.latency_seconds = state.latency_so_far
        episode.citation_support_rate = state.citation_support_rate
        
        # 6. Periodic Training Update (On-Policy)
        # In this simple version, we update after every episode
        train_metrics = trainer.train_on_episodes([episode])
        
        # 7. Logging
        logger.log_episode(episode)
        if ep_idx % 10 == 0:
            agg_metrics = {
                "mean_reward": episode.total_reward, # simple running average in real ver
                "mean_steps":  episode.num_steps,
                "mean_llm_calls": episode.num_llm_calls,
                **train_metrics
            }
            logger.log_training_metrics(ep_idx, agg_metrics)
            
        # 8. Checkpointing
        if ep_idx % args.checkpoint_every == 0:
            is_best = episode.total_reward > best_reward
            if is_best: best_reward = episode.total_reward
            
            ckpt_manager.save_checkpoint(
                actors, critic, trainer.optimizer, ep_idx,
                metrics={"reward": episode.total_reward},
                config=vars(args),
                is_best=is_best
            )

    print(f"\nTraining Complete. Results saved to {cfg.RESULTS_DIR}")

if __name__ == "__main__":
    train()
