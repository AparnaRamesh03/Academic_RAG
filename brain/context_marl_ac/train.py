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
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

# ── sys.path setup ────────────────────────────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
import context_marl_ac.config as cfg
from context_marl_ac.logging_utils import MARLLogger
from context_marl_ac.marl.actors import build_marl_actors
from context_marl_ac.marl.centralized_critic import CentralizedCritic
from context_marl_ac.marl.checkpointing import MARLCheckpointManager
from context_marl_ac.marl.marl_env import MARLEnv
from context_marl_ac.marl.trainer import MARLTrainer
from context_marl_ac.schemas.actions import AGENT_ACTIONS, AGENT_NAMES
from context_marl_ac.schemas.trajectory import Episode, TrajectoryStep


def parse_args():
    parser = argparse.ArgumentParser(description="Train Context-Engineered MARL RAG")
    parser.add_argument(
        "--run-name",
        type=str,
        default="marl_run_01",
        help="Name for this training run",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to train",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=str(_BRAIN_ROOT.parent / "brain" / "context_marl_ac" / "results" / "checkpoints" / "best_reward.pt"),
        help="Resume training from checkpoint filename in checkpoints directory",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg.LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=cfg.HIDDEN_DIM,
        help="Hidden dimension for NN",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in mock/dry-run mode",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=str(_BRAIN_ROOT.parent / "brain" / "context_marl_ac" / "results" / "benchmark_splits" / "train.jsonl"),
        help="Path to training benchmark JSON or JSONL",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=cfg.CHECKPOINT_EVERY,
        help="Save model every N episodes",
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


def update_from_recent_episodes(
    recent_episodes: List[Episode],
    trainer: MARLTrainer,
    logger: MARLLogger,
    ckpt_manager: MARLCheckpointManager,
    actors,
    critic,
    optimizer,
    epoch: int,
    args,
    best_reward: float,
) -> float:
    """
    Train once on recent episodes, log metrics, and save best checkpoint.
    Returns updated best_reward.
    """
    if not recent_episodes:
        return best_reward

    train_metrics = trainer.train_on_episodes(recent_episodes)
    train_metrics.update(
        {
            "mean_reward": sum(e.total_reward for e in recent_episodes) / len(recent_episodes),
            "mean_steps": sum(e.num_steps for e in recent_episodes) / len(recent_episodes),
            "mean_llm_calls": sum(e.num_llm_calls for e in recent_episodes) / len(recent_episodes),
        }
    )

    logger.log_training_metrics(epoch, train_metrics)

    mean_reward = train_metrics.get("mean_reward", -float("inf"))
    if mean_reward > best_reward:
        best_reward = mean_reward
        ckpt_manager.save_checkpoint(
            actors=actors,
            critic=critic,
            optimizer=optimizer,
            episode=epoch,
            metrics=train_metrics,
            config=vars(args),
            is_best=True,
            filename="best_reward.pt",
        )

    return best_reward


def train():
    args = parse_args()

    # 1. Update global config with CLI args
    cfg.DRY_RUN = args.dry_run

    # 2. Initialize infrastructure
    env = MARLEnv()
    logger = MARLLogger(run_name=args.run_name)
    ckpt_manager = MARLCheckpointManager()

    actors = build_marl_actors()
    critic = CentralizedCritic()
    trainer = MARLTrainer(actors, critic, lr=args.lr)
    if args.resume_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")

        try:
            ckpt_manager.load_checkpoint(
                actors=actors,
                critic=critic,
                optimizer=trainer.optimizer,
                filename=args.resume_checkpoint,
            )
            print("Loaded actors, critic, and optimizer.")
        except TypeError:
            ckpt_manager.load_checkpoint(
                actors=actors,
                critic=critic,
                filename=args.resume_checkpoint,
            )
            print("Loaded actors and critic. Optimizer resume is not supported by checkpoint manager.")
        except Exception as exc:
            print(f"Warning: Could not resume from checkpoint: {exc}")
            print("Starting from fresh weights.")

    benchmark = load_benchmark(args.benchmark_path)

    print(f"Starting training run: {args.run_name}")
    print(f"Episodes: {args.episodes}, Dry-Run: {args.dry_run}")

    best_reward = -float("inf")
    recent_episodes: List[Episode] = []
    batch_size = 1 if cfg.DRY_RUN else 4

    # 3. Main training loop
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

        # 4. Episode rollout
        while not done:
            agent_to_act = None
            action_to_take = None
            action_id = -1
            action_prob = 0.0
            log_prob = 0.0
            entropy = 0.0
            valid_action_names: List[str] = []

            # Agent selection is controlled by action masks.
            for agent_name in AGENT_NAMES:
                mask = env.get_mask(agent_name)
                if sum(mask) > 0:
                    agent_to_act = agent_name
                    valid_action_names = [
                        AGENT_ACTIONS[agent_name][i]
                        for i, m in enumerate(mask)
                        if m == 1
                    ]

                    obs = torch.tensor(env.get_obs(agent_name), dtype=torch.float32)
                    mask_t = torch.tensor(mask, dtype=torch.float32)

                    with torch.no_grad():
                        logits = actors[agent_name](obs.unsqueeze(0), mask_t.unsqueeze(0))
                        probs = torch.softmax(logits, dim=-1)

                        dist = torch.distributions.Categorical(probs)
                        action_tensor = dist.sample()
                        action_id = action_tensor.item()

                        action_prob = probs[0, action_id].item()
                        log_prob = dist.log_prob(action_tensor).item()
                        entropy = dist.entropy().item()

                    action_to_take = AGENT_ACTIONS[agent_name][action_id]
                    break

            if not agent_to_act:
                if state.final_status == "pending":
                    state.final_status = "abstained"
                state.done = True
                done = True
                state.update_latency()
                break

            # Save pre-action state for trajectory learning.
            prev_global_feats = env.get_global_features()
            prev_obs = env.get_obs(agent_to_act)
            prev_mask = env.get_mask(agent_to_act)

            with torch.no_grad():
                v_s = critic(
                    torch.tensor(prev_global_feats, dtype=torch.float32).unsqueeze(0)
                ).item()

            new_state, reward, done, info = env.step(agent_to_act, action_to_take)

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
                extra=new_state.to_debug_dict(),
            )

            episode.add_step(step)
            state = new_state
            step_idx += 1

        # 5. Finalize episode
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

        logger.log_episode(episode)
        recent_episodes.append(episode)

        if len(recent_episodes) >= batch_size:
            best_reward = update_from_recent_episodes(
                recent_episodes=recent_episodes,
                trainer=trainer,
                logger=logger,
                ckpt_manager=ckpt_manager,
                actors=actors,
                critic=critic,
                optimizer=trainer.optimizer,
                epoch=ep_idx,
                args=args,
                best_reward=best_reward,
            )
            recent_episodes = []

    # 6. Final update for leftover episodes
    # If episodes is not divisible by batch_size, the remaining episodes still train once.
    if recent_episodes:
        best_reward = update_from_recent_episodes(
            recent_episodes=recent_episodes,
            trainer=trainer,
            logger=logger,
            ckpt_manager=ckpt_manager,
            actors=actors,
            critic=critic,
            optimizer=trainer.optimizer,
            epoch=args.episodes,
            args=args,
            best_reward=best_reward,
        )

    print(f"Training finished. Logs in {logger.metrics_dir}")


if __name__ == "__main__":
    train()