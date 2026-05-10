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
    
    # Support both JSON and JSONL
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
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
    recent_episodes = []
    
    # 3. Main Training Loop
    for ep_idx in tqdm(range(1, args.episodes + 1), desc="Training"):
        idx = random.randint(0, len(benchmark) - 1)
        question_dict = benchmark[idx]
        
        state = env.reset(question_dict, index=idx + 1)
        ep_id = f"episode_{ep_idx:04d}_{state.question_id}"
        
        episode = Episode(
            episode_id=ep_id,
            question_id=state.question_id,
            question=state.user_query,
            query_type=state.query_type,
            query_complexity=state.query_complexity
        )
        
        done = False
        step_idx = 0
        
        # 4. Episode Rollout
        while not done:
            agent_to_act = None
            action_to_take = None
            action_id = -1
            action_prob = 0.0
            log_prob = 0.0
            entropy = 0.0
            valid_action_names = []
            
            for agent_name in AGENT_NAMES:
                mask = env.get_mask(agent_name)
                if sum(mask) > 0:
                    agent_to_act = agent_name
                    valid_action_names = [AGENT_ACTIONS[agent_name][i] for i, m in enumerate(mask) if m == 1]
                    
                    obs = torch.tensor(env.get_obs(agent_name), dtype=torch.float32)
                    mask_t = torch.tensor(mask, dtype=torch.float32)
                    
                    with torch.no_grad():
                        logits = actors[agent_name](obs.unsqueeze(0), mask_t.unsqueeze(0))
                        probs = torch.softmax(logits, dim=-1)
                        
                        m = torch.distributions.Categorical(probs)
                        action_tensor = m.sample()
                        action_id = action_tensor.item()
                        
                        action_prob = probs[0, action_id].item()
                        log_prob = m.log_prob(action_tensor).item()
                        entropy = m.entropy().item()
                        
                    action_to_take = AGENT_ACTIONS[agent_name][action_id]
                    break
            
            if not agent_to_act:
                if state.final_status == "pending":
                    state.final_status = "abstained"
                break
                
            # Execute Step
            prev_global_feats = env.get_global_features()
            prev_obs = env.get_obs(agent_to_act)
            prev_mask = env.get_mask(agent_to_act)
            
            with torch.no_grad():
                v_s = critic(torch.tensor(prev_global_feats, dtype=torch.float32).unsqueeze(0)).item()
            
            new_state, reward, done, info = env.step(agent_to_act, action_to_take)
            
            # Record transition with detailed debug fields
            step = TrajectoryStep(
                episode_id=ep_id,
                step=step_idx,
                agent=agent_to_act,
                observation=prev_obs,
                global_features=prev_global_feats,
                obs_names=[], 
                valid_actions=valid_action_names, 
                action_mask=prev_mask,
                selected_action=action_to_take,
                action_id=action_id,
                action_probability=action_prob,
                log_probability=log_prob,
                entropy=entropy,
                critic_value=v_s,
                reward=reward,
                advantage=0.0,
                done=done,
                latency_step=0.0,
                extra=new_state.to_debug_dict() # ADDED DETAILED LOGGING
            )
            episode.add_step(step)
            state = new_state
            step_idx += 1
            
        # 5. Finalize Episode
        episode.total_reward = env.get_global_reward()
        episode.final_status = state.final_status
        episode.generated_answer = state.generated_answer
        episode.num_steps = state.num_steps
        episode.num_llm_calls = state.num_llm_calls
        episode.latency_seconds = state.latency_so_far
        episode.token_usage = state.token_usage
        
        has_gen = bool(state.generated_answer.strip())
        is_dry_run_ans = "[DRY-RUN" in state.generated_answer
        
        if state.final_status == "accepted":
            episode.verification_pass = True
            if cfg.DRY_RUN and is_dry_run_ans:
                episode.answer_quality = 0.85
                episode.citation_support_rate = 0.95
            else:
                episode.answer_quality = state.verification_result.get("quality_score", 0.0)
                episode.citation_support_rate = state.citation_support_rate
        else:
            episode.verification_pass = False
            if cfg.DRY_RUN and is_dry_run_ans and state.final_status in ["rejected", "timeout"]:
                 episode.answer_quality = 0.85
                 episode.citation_support_rate = 0.0 
            else:
                episode.answer_quality = 0.0
                episode.citation_support_rate = 0.0
            
        episode.has_generated_answer = has_gen
        episode.answer_length = len(state.generated_answer)
        episode.selected_evidence_count = len(state.selected_evidence)
        episode.verifier_decision = state.verification_result.get("decision", "N/A")
        
        logger.log_episode(episode)
        recent_episodes.append(episode)
        
        batch_size = 1 if cfg.DRY_RUN else 4
        if len(recent_episodes) >= batch_size:
            train_metrics = trainer.train_on_episodes(recent_episodes)
            train_metrics.update({
                "mean_reward": sum(e.total_reward for e in recent_episodes) / len(recent_episodes),
                "mean_steps": sum(e.num_steps for e in recent_episodes) / len(recent_episodes),
                "mean_llm_calls": sum(e.num_llm_calls for e in recent_episodes) / len(recent_episodes)
            })
            logger.log_training_metrics(ep_idx, train_metrics)
            recent_episodes = []
            
            if train_metrics["mean_reward"] > best_reward:
                best_reward = train_metrics["mean_reward"]
                ckpt_manager.save_checkpoint(
                    actors=actors, 
                    critic=critic, 
                    optimizer=trainer.optimizer,
                    episode=ep_idx,
                    metrics=train_metrics,
                    config=vars(args),
                    is_best=True,
                    filename="best_reward.pt"
                )

    print(f"Training finished. Logs in {logger.metrics_dir}")

if __name__ == "__main__":
    train()
