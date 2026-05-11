"""
brain/context_marl_ac/train.py
------------------------------
Training script for Supervisor-Guided Fully Free MARL Actor-Critic RAG.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

import context_marl_ac.config as cfg
from context_marl_ac.logging_utils import MARLLogger
from context_marl_ac.marl.actors import build_marl_actors
from context_marl_ac.marl.centralized_critic import CentralizedCritic
from context_marl_ac.marl.checkpointing import MARLCheckpointManager
from context_marl_ac.marl.marl_env import MARLEnv
from context_marl_ac.marl.supervisor_actor import (
    build_supervisor_actor,
    SUPERVISOR_AGENTS,
    SUPERVISOR_ID_TO_AGENT,
)
from context_marl_ac.marl.trainer import MARLTrainer
from context_marl_ac.schemas.actions import AGENT_ACTIONS
from context_marl_ac.schemas.trajectory import Episode, TrajectoryStep


def parse_args():
    parser = argparse.ArgumentParser(description="Train Supervisor-Guided Fully Free MARL RAG")

    parser.add_argument("--run-name", type=str, default="free_marl_run_01")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=cfg.HIDDEN_DIM)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=str(_BRAIN_ROOT.parent / "evaluation" / "standard_benchmark_v3.json"),
    )
    parser.add_argument("--checkpoint-every", type=int, default=cfg.CHECKPOINT_EVERY)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="supervisor_free",
        choices=["supervisor_free"],
    )

    return parser.parse_args()


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Warning: Benchmark not found at {path}. Using dummy question.")
        return [
            {
                "question": "What is the capital of France?",
                "ground_truth": "Paris",
                "question_id": "dummy",
            }
        ]

    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path, "r") as f:
        return json.load(f)


def _distribution_from_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    return probs, dist


def choose_supervisor_agent(
    supervisor,
    global_features: List[float],
    agent_mask: List[int],
) -> Dict[str, Any]:
    global_t = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(agent_mask, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = supervisor(global_t, mask_t)
        probs, dist = _distribution_from_logits(logits)
        agent_tensor = dist.sample()

    agent_id = int(agent_tensor.item())
    agent_name = SUPERVISOR_ID_TO_AGENT[agent_id]

    return {
        "agent_name": agent_name,
        "agent_id": agent_id,
        "prob": float(probs[0, agent_id].item()),
        "log_prob": float(dist.log_prob(agent_tensor).item()),
        "entropy": float(dist.entropy().item()),
    }


def choose_agent_action(
    actors,
    agent_name: str,
    obs: List[float],
    action_mask: List[int],
) -> Dict[str, Any]:
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = actors[agent_name](obs_t, mask_t)
        probs, dist = _distribution_from_logits(logits)
        action_tensor = dist.sample()

    action_id = int(action_tensor.item())
    action_name = AGENT_ACTIONS[agent_name][action_id]

    return {
        "action_name": action_name,
        "action_id": action_id,
        "prob": float(probs[0, action_id].item()),
        "log_prob": float(dist.log_prob(action_tensor).item()),
        "entropy": float(dist.entropy().item()),
    }


def finalize_episode(episode: Episode, state, env: MARLEnv) -> None:
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
            episode.citation_support_rate = state.citation_support_rate
            episode.answer_quality = state.verification_result.get(
                "quality_score",
                state.citation_support_rate if state.citation_support_rate > 0 else 1.0,
            )
    else:
        episode.verification_pass = False

        if cfg.DRY_RUN and is_dry_run_ans and state.final_status in {"rejected", "timeout"}:
            episode.answer_quality = 0.85
            episode.citation_support_rate = 0.0
        else:
            episode.answer_quality = 0.0
            episode.citation_support_rate = 0.0

    episode.has_generated_answer = has_gen
    episode.answer_length = len(state.generated_answer)
    episode.selected_evidence_count = len(state.selected_evidence)
    episode.verifier_decision = state.verification_result.get("decision", "N/A")


def train():
    args = parse_args()
    cfg.DRY_RUN = args.dry_run

    env = MARLEnv()
    logger = MARLLogger(run_name=args.run_name)
    ckpt_manager = MARLCheckpointManager()

    actors = build_marl_actors()
    supervisor = build_supervisor_actor()
    critic = CentralizedCritic()
    trainer = MARLTrainer(actors, critic, supervisor=supervisor, lr=args.lr)

    if args.resume_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        try:
            ckpt_manager.load_checkpoint(
                actors=actors,
                critic=critic,
                optimizer=trainer.optimizer,
                filename=args.resume_checkpoint,
                supervisor=supervisor,
                strict_supervisor=False,
            )
        except Exception as exc:
            print(f"Warning: could not resume checkpoint: {exc}")
            print("Starting from fresh weights.")

    benchmark = load_benchmark(args.benchmark_path)

    print(f"Starting training run: {args.run_name}")
    print(f"Episodes: {args.episodes}, Dry-Run: {args.dry_run}, Routing: {args.routing_mode}")

    best_reward = -float("inf")
    recent_episodes: List[Episode] = []

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
            query_complexity=state.query_complexity,
        )

        done = False
        step_idx = 0

        while not done:
            prev_global_feats = env.get_global_features()
            valid_agents = env.get_valid_agents()
            agent_mask = env.get_agent_mask()

            if not valid_agents or sum(agent_mask) == 0:
                if state.final_status == "pending":
                    state.final_status = "abstained"
                    state.done = True
                break

            supervisor_choice = choose_supervisor_agent(
                supervisor=supervisor,
                global_features=prev_global_feats,
                agent_mask=agent_mask,
            )

            agent_to_act = supervisor_choice["agent_name"]

            # Safety fallback if sampled masked agent somehow appears.
            if agent_to_act not in valid_agents:
                agent_to_act = valid_agents[0]
                supervisor_choice["agent_name"] = agent_to_act
                supervisor_choice["agent_id"] = SUPERVISOR_AGENTS.index(agent_to_act)

            prev_obs = env.get_obs(agent_to_act)
            prev_mask = env.get_mask(agent_to_act)
            valid_action_names = [
                AGENT_ACTIONS[agent_to_act][i]
                for i, m in enumerate(prev_mask)
                if m == 1
            ]

            if not valid_action_names or sum(prev_mask) == 0:
                state.final_status = "error"
                state.done = True
                break

            action_choice = choose_agent_action(
                actors=actors,
                agent_name=agent_to_act,
                obs=prev_obs,
                action_mask=prev_mask,
            )

            action_to_take = action_choice["action_name"]

            with torch.no_grad():
                v_s = critic(
                    torch.tensor(prev_global_feats, dtype=torch.float32).unsqueeze(0)
                ).item()

            new_state, reward, done, info = env.step(agent_to_act, action_to_take)

            step = TrajectoryStep(
                episode_id=ep_id,
                step=step_idx,

                valid_agents=valid_agents,
                agent_mask=agent_mask,
                supervisor_selected_agent=agent_to_act,
                supervisor_agent_id=supervisor_choice["agent_id"],
                supervisor_action_probability=supervisor_choice["prob"],
                supervisor_log_probability=supervisor_choice["log_prob"],
                supervisor_entropy=supervisor_choice["entropy"],

                agent=agent_to_act,
                observation=prev_obs,
                global_features=prev_global_feats,
                obs_names=[],
                valid_actions=valid_action_names,
                action_mask=prev_mask,
                selected_action=action_to_take,
                action_id=action_choice["action_id"],
                action_probability=action_choice["prob"],
                log_probability=action_choice["log_prob"],
                entropy=action_choice["entropy"],

                critic_value=v_s,
                reward=reward,
                advantage=0.0,
                done=done,
                latency_step=0.0,
                extra=new_state.to_debug_dict(),
            )
            episode.add_step(step)

            state = new_state
            step_idx += 1

        finalize_episode(episode, state, env)
        logger.log_episode(episode)
        recent_episodes.append(episode)

        batch_size = 1 if cfg.DRY_RUN else 4
        if len(recent_episodes) >= batch_size:
            train_metrics = trainer.train_on_episodes(recent_episodes)
            train_metrics.update(
                {
                    "mean_reward": sum(e.total_reward for e in recent_episodes) / len(recent_episodes),
                    "mean_steps": sum(e.num_steps for e in recent_episodes) / len(recent_episodes),
                    "mean_llm_calls": sum(e.num_llm_calls for e in recent_episodes) / len(recent_episodes),
                }
            )

            logger.log_training_metrics(ep_idx, train_metrics)

            if train_metrics.get("mean_reward", -float("inf")) > best_reward:
                best_reward = train_metrics["mean_reward"]
                ckpt_manager.save_checkpoint(
                    actors=actors,
                    supervisor=supervisor,
                    critic=critic,
                    optimizer=trainer.optimizer,
                    episode=ep_idx,
                    metrics=train_metrics,
                    config=vars(args),
                    is_best=True,
                    filename="best_reward.pt",
                )

            recent_episodes = []

    if recent_episodes:
        train_metrics = trainer.train_on_episodes(recent_episodes)
        train_metrics.update(
            {
                "mean_reward": sum(e.total_reward for e in recent_episodes) / len(recent_episodes),
                "mean_steps": sum(e.num_steps for e in recent_episodes) / len(recent_episodes),
                "mean_llm_calls": sum(e.num_llm_calls for e in recent_episodes) / len(recent_episodes),
            }
        )

        logger.log_training_metrics(args.episodes, train_metrics)

        if train_metrics.get("mean_reward", -float("inf")) > best_reward:
            ckpt_manager.save_checkpoint(
                actors=actors,
                supervisor=supervisor,
                critic=critic,
                optimizer=trainer.optimizer,
                episode=args.episodes,
                metrics=train_metrics,
                config=vars(args),
                is_best=True,
                filename="best_reward.pt",
            )

    print(f"Training finished. Logs in {logger.metrics_dir}")


if __name__ == "__main__":
    train()