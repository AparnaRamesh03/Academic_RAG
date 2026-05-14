"""
brain/maddpg/train_maddpg.py
-----------------------------
Training entry point for the stage-conditioned MADDPG-style continuous-control
RAG architecture.

Architecture: maddpg_style_continuous_control
Critic type:  stage_conditioned

Episode flow is identical to the discrete MARL stage gating — `find_active_agent_and_valid_actions`
finds the one agent whose mask is non-empty, the actor outputs a continuous
parameter vector, the mapper picks a discrete action within the valid set, and
`env.step(agent, action, params=params)` executes it.

Run modes (--train-mode):
  dry             — uses cfg.DRY_RUN = True (stub adapters, no LLM/Qdrant)
  live            — real LLM/Qdrant
  offline-replay  — load a saved replay JSONL and run gradient steps with no env

Usage (from brain/):
  python -m maddpg.train_maddpg --train-mode dry --episodes 5 \
      --batch-size 4 --warmup-steps 4 --update-every 1 --min-updates 1 \
      --run-name smoke_stage_conditioned
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ── sys.path: ensure brain/ is importable ────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ── dotenv: load API keys from local-only locations (no Desktop paths) ───────
try:
    from dotenv import load_dotenv
    for _ep in [
        _BRAIN_ROOT / ".env",
        _BRAIN_ROOT.parent / ".env",
    ]:
        if _ep.exists():
            load_dotenv(dotenv_path=_ep)
            break
except ImportError:
    pass

import context_marl_ac.config as cfg

from .continuous_action_mapper import AGENT_ACTION_DIMS, AGENT_DEFAULTS, map_agent_params
from .context_engineering_block import CEB_STATE_DIM, build_ceb_features
from .stage_utils import find_active_agent_and_valid_actions
from .trainer import StageConditionedMADDPGTrainer, TrainerConfig


# ── CSV schema ────────────────────────────────────────────────────────────────

_MAX_RAW_DIM = max(AGENT_ACTION_DIMS.values())
_ALL_PARAM_KEYS: List[str] = list(dict.fromkeys(
    k for defaults in AGENT_DEFAULTS.values() for k in defaults.keys()
))
PARAMS_CSV_FIELDNAMES: List[str] = (
    ["episode", "step", "active_agent", "valid_actions", "discrete_action",
     "reward", "done", "next_active_agent"]
    + [f"raw_{i}" for i in range(_MAX_RAW_DIM)]
    + [f"padded_{i}" for i in range(_MAX_RAW_DIM)]
    + _ALL_PARAM_KEYS
)

EP_CSV_FIELDNAMES: List[str] = [
    "episode", "question_id", "total_reward", "num_steps", "num_llm_calls",
    "final_status", "verification_pass", "citation_support",
    "latency_seconds", "token_usage", "buffer_size", "total_gradient_updates",
    "critic_loss",
    "actor_loss_retriever", "actor_loss_rewriter", "actor_loss_grader",
    "actor_loss_generator", "actor_loss_verifier",
    "trained_so_far",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Train stage-conditioned MADDPG continuous-control RAG")
    p.add_argument("--run-name",        default="maddpg_run_01")
    p.add_argument("--episodes",        type=int, default=200)
    p.add_argument("--train-mode",      default="live",
                   choices=["dry", "live", "offline-replay"])
    p.add_argument("--dry-run",         action="store_true",
                   help="Alias for --train-mode dry (kept for backward compat)")
    p.add_argument("--benchmark-path",  default="")
    p.add_argument("--use-ceb",         action="store_true")
    p.add_argument("--checkpoint-path", default="",
                   help="Resume from existing .pt checkpoint")
    p.add_argument("--checkpoint-every",type=int, default=50)
    p.add_argument("--results-dir",     default="")

    # Hyperparameters (replace module-level constants).
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--warmup-steps",    type=int,   default=50)
    p.add_argument("--update-every",    type=int,   default=4)
    p.add_argument("--min-updates",     type=int,   default=1,
                   help="Fail with nonzero exit if total_gradient_updates < this "
                        "(unless --allow-untrained).")
    p.add_argument("--allow-untrained", action="store_true")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--tau",             type=float, default=0.005)
    p.add_argument("--actor-lr",        type=float, default=1e-3)
    p.add_argument("--critic-lr",       type=float, default=1e-3)
    p.add_argument("--noise-sigma",     type=float, default=0.15)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--error-penalty",   type=float, default=-1.0)
    p.add_argument("--hidden-dim",      type=int,   default=128)

    # Offline-replay support.
    p.add_argument("--replay-in",                default="",
                   help="Load transitions from JSONL before training")
    p.add_argument("--replay-out",               default="",
                   help="Dump collected transitions to JSONL after training")
    p.add_argument("--offline-gradient-steps",   type=int, default=10_000,
                   help="Gradient steps when --train-mode offline-replay")
    p.add_argument("--gradient-steps-per-env-step", type=int, default=1)

    return p.parse_args(argv)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_benchmark(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        print(f"[train_maddpg] Benchmark not found at '{path}'. Using dummy question.")
        return [{
            "question":     "What is the transformer attention mechanism?",
            "ground_truth": "Attention lets the model focus on relevant input.",
            "question_id":  "dummy_q1",
        }]
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(path) as f:
        return json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_features(env: Any, use_ceb: bool) -> np.ndarray:
    if use_ceb:
        return np.array(build_ceb_features(env.state), dtype=np.float32)
    return np.array(env.get_global_features(), dtype=np.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Main training entry ──────────────────────────────────────────────────────

def train(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    train_mode = "dry" if args.dry_run else args.train_mode
    cfg.DRY_RUN = (train_mode == "dry")

    _set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = CEB_STATE_DIM if args.use_ceb else cfg.FEATURE_DIM

    tcfg = TrainerConfig(
        state_dim       = state_dim,
        hidden_dim      = args.hidden_dim,
        actor_lr        = args.actor_lr,
        critic_lr       = args.critic_lr,
        gamma           = args.gamma,
        tau             = args.tau,
        batch_size      = args.batch_size,
        noise_sigma     = args.noise_sigma,
        grad_clip       = args.grad_clip,
        update_every    = args.update_every,
        warmup_steps    = args.warmup_steps,
        error_penalty   = args.error_penalty,
        device          = device,
    )
    trainer = StageConditionedMADDPGTrainer(tcfg)

    # ── Directories ──────────────────────────────────────────────────────────
    base = (
        Path(args.results_dir) if args.results_dir
        else Path(__file__).resolve().parent / "results" / "maddpg"
    )
    ckpt_dir    = base / "checkpoints"
    metrics_dir = base / "metrics"
    traj_dir    = base / "trajectories"
    for d in (ckpt_dir, metrics_dir, traj_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Resume / offline-replay seeding ──────────────────────────────────────
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"[train_maddpg] Resuming from {args.checkpoint_path}")
        trainer.load_checkpoint(Path(args.checkpoint_path))
    if args.replay_in and os.path.exists(args.replay_in):
        n = trainer.load_replay_jsonl(Path(args.replay_in))
        print(f"[train_maddpg] Loaded {n} transitions from {args.replay_in}")

    last_losses: Dict[str, float] = {}
    all_episode_metrics: List[Dict[str, Any]] = []
    best_reward = -float("inf")

    print(
        f"[train_maddpg] run={args.run_name} mode={train_mode} | "
        f"device={device} | CEB={args.use_ceb} | state_dim={state_dim}\n"
        f"  batch={args.batch_size} warmup={args.warmup_steps} "
        f"update_every={args.update_every} min_updates={args.min_updates} "
        f"seed={args.seed}"
    )

    # ── Pure offline-replay training (no env) ────────────────────────────────
    if train_mode == "offline-replay":
        if len(trainer.buffer) < args.batch_size:
            print(
                f"[train_maddpg] offline-replay: buffer has {len(trainer.buffer)} < "
                f"batch_size {args.batch_size}. Aborting."
            )
            return 2
        n_steps = max(1, args.offline_gradient_steps)
        print(f"[train_maddpg] offline-replay: {n_steps} gradient steps over "
              f"buffer of {len(trainer.buffer)} transitions")
        for _ in tqdm(range(n_steps), desc="offline-update"):
            # Bypass should_update() since there are no env steps.
            last_losses = trainer.update()
        _write_aggregate(
            base / f"aggregate_metrics_{args.run_name}.json",
            args, tcfg, trainer, all_episode_metrics, best_reward, last_losses,
        )
        trainer.save_checkpoint(
            ckpt_dir / f"{args.run_name}_offline.pt",
            extra={"run_name": args.run_name, "train_mode": train_mode},
        )
        return 0 if trainer.total_gradient_updates >= args.min_updates or args.allow_untrained else 3

    # ── On-policy / live env training ────────────────────────────────────────
    from context_marl_ac.marl.marl_env import MARLEnv  # imported here so dry-run can preload cfg

    env = MARLEnv()
    benchmark = load_benchmark(
        args.benchmark_path or str(
            Path(__file__).resolve().parent / "results" / "benchmark_splits" / "train.jsonl"
        )
    )

    ep_csv_path  = metrics_dir / f"episode_metrics_{args.run_name}.csv"
    params_path  = metrics_dir / f"action_params_log_{args.run_name}.csv"
    traj_path    = traj_dir    / f"trajectories_{args.run_name}.jsonl"
    agg_path     = base        / f"aggregate_metrics_{args.run_name}.json"

    with (
        open(ep_csv_path, "w", newline="", encoding="utf-8") as ep_csv_f,
        open(params_path, "w", newline="", encoding="utf-8") as params_f,
        open(traj_path,   "w", encoding="utf-8")             as traj_f,
    ):
        params_writer = csv.DictWriter(
            params_f, fieldnames=PARAMS_CSV_FIELDNAMES,
            restval="", extrasaction="ignore",
        )
        params_writer.writeheader()

        ep_csv_writer = csv.DictWriter(
            ep_csv_f, fieldnames=EP_CSV_FIELDNAMES,
            restval="", extrasaction="ignore",
        )
        ep_csv_writer.writeheader()

        for ep_idx in tqdm(range(1, args.episodes + 1), desc=f"MADDPG/{train_mode}"):
            q_idx  = random.randint(0, len(benchmark) - 1)
            q_dict = benchmark[q_idx]
            state  = env.reset(q_dict, index=q_idx + 1)
            qid    = state.question_id

            trainer.reset_noise()
            done = False
            ep_steps = 0
            ep_traj: List[Dict[str, Any]] = []

            while not done:
                active_agent, valid_actions = find_active_agent_and_valid_actions(env)
                if active_agent is None:
                    if state.final_status == "pending":
                        state.final_status = "abstained"
                    state.done = True
                    done = True
                    break

                prev_features = _state_features(env, args.use_ceb)
                raw, params, discrete = trainer.select_action(
                    active_agent, prev_features, valid_actions, explore=True
                )

                try:
                    new_state, reward, done, _info = env.step(
                        active_agent, discrete, params=params
                    )
                    next_features = _state_features(env, args.use_ceb)
                    next_active, next_valid = (
                        (None, []) if done
                        else find_active_agent_and_valid_actions(env)
                    )
                    trainer.push_transition(
                        state_features      = prev_features,
                        active_agent        = active_agent,
                        valid_actions       = valid_actions,
                        raw_action          = raw,
                        mapped_params       = params,
                        discrete_action     = discrete,
                        reward              = reward,
                        next_state_features = next_features,
                        next_active_agent   = next_active,
                        next_valid_actions  = next_valid,
                        done                = done,
                        question_id         = qid,
                        step                = ep_steps + 1,
                        final_status        = new_state.final_status,
                        metrics_snapshot    = {
                            "citation_support_rate":  new_state.citation_support_rate,
                            "num_unsupported_claims": len(new_state.unsupported_claims),
                            "final_status":           new_state.final_status,
                        },
                    )
                except Exception as exc:
                    print(f"  [train-err] ep={ep_idx} agent={active_agent} "
                          f"action={discrete}: {exc}")
                    trainer.push_error_transition(
                        state_features  = prev_features,
                        active_agent    = active_agent,
                        valid_actions   = valid_actions,
                        raw_action      = raw,
                        mapped_params   = params,
                        discrete_action = discrete,
                        question_id     = qid,
                        step            = ep_steps + 1,
                        error_message   = str(exc),
                    )
                    state.final_status = "error"
                    state.done = True
                    done = True
                    new_state = state
                    reward = tcfg.error_penalty

                ep_steps += 1

                _log_params_row(
                    params_writer, ep_idx, ep_steps, active_agent, valid_actions,
                    discrete, reward, done,
                    (next_active if not done else None),
                    raw, params,
                )

                ep_traj.append({
                    "step":             ep_steps,
                    "active_agent":     active_agent,
                    "valid_actions":    list(valid_actions),
                    "discrete_action":  discrete,
                    "raw_action":       raw.tolist(),
                    "mapped_params":    params,
                    "reward":           reward,
                    "done":             done,
                    "next_active_agent": (next_active if not done else None),
                })

                # Gradient updates inside the episode.
                if trainer.should_update():
                    for _ in range(args.gradient_steps_per_env_step):
                        last_losses = trainer.update()

                state = new_state

            ep_reward = env.get_global_reward()
            ep_metrics: Dict[str, Any] = {
                "episode":                ep_idx,
                "question_id":            qid,
                "total_reward":           round(ep_reward, 6),
                "num_steps":              state.num_steps,
                "num_llm_calls":          state.num_llm_calls,
                "final_status":           state.final_status,
                "verification_pass":      int(state.final_status == "accepted"),
                "citation_support":       round(state.citation_support_rate, 4),
                "latency_seconds":        round(state.latency_so_far, 3),
                "token_usage":            state.token_usage,
                "buffer_size":            len(trainer.buffer),
                "total_gradient_updates": trainer.total_gradient_updates,
                "critic_loss":            round(last_losses.get("critic_loss", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_retriever":   round(last_losses.get("actor_loss_retriever", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_rewriter":    round(last_losses.get("actor_loss_rewriter", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_grader":      round(last_losses.get("actor_loss_grader", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_generator":   round(last_losses.get("actor_loss_generator", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_verifier":    round(last_losses.get("actor_loss_verifier", float("nan")), 6)
                                            if last_losses else "",
                "trained_so_far":         trainer.total_gradient_updates > 0,
            }
            all_episode_metrics.append(ep_metrics)
            ep_csv_writer.writerow(ep_metrics)
            ep_csv_f.flush()

            traj_f.write(json.dumps({
                "episode":      ep_idx,
                "question_id":  qid,
                "total_reward": ep_reward,
                "final_status": state.final_status,
                "steps":        ep_traj,
            }, ensure_ascii=False) + "\n")
            traj_f.flush()

            if ep_reward > best_reward:
                best_reward = ep_reward
                trainer.save_checkpoint(
                    ckpt_dir / "best_reward.pt",
                    extra={"run_name": args.run_name, "episode": ep_idx, "metrics": ep_metrics},
                )
            if ep_idx % args.checkpoint_every == 0:
                trainer.save_checkpoint(
                    ckpt_dir / f"ep_{ep_idx:04d}.pt",
                    extra={"run_name": args.run_name, "episode": ep_idx, "metrics": ep_metrics},
                )

        _write_aggregate(
            agg_path, args, tcfg, trainer, all_episode_metrics, best_reward, last_losses,
        )

    if args.replay_out:
        n = trainer.save_replay_jsonl(Path(args.replay_out))
        print(f"[train_maddpg] Dumped {n} transitions -> {args.replay_out}")

    print(
        f"[train_maddpg] Done. best_reward={best_reward:.4f}  "
        f"env_steps={trainer.total_env_steps}  gradient_updates={trainer.total_gradient_updates}\n"
        f"  Results in: {base}"
    )

    if trainer.total_gradient_updates < args.min_updates and not args.allow_untrained:
        print(
            f"  [error] Only {trainer.total_gradient_updates} gradient updates "
            f"performed (min_updates={args.min_updates}). The policy is untrained.\n"
            f"  Pass --allow-untrained to ignore, or run more episodes / lower "
            f"batch_size/warmup_steps."
        )
        return 3
    return 0


# ── Aggregate + params row writers ───────────────────────────────────────────

def _write_aggregate(
    agg_path:           Path,
    args:               argparse.Namespace,
    tcfg:               TrainerConfig,
    trainer:            StageConditionedMADDPGTrainer,
    episode_metrics:    List[Dict[str, Any]],
    best_reward:        float,
    last_losses:        Dict[str, float],
) -> None:
    n = max(len(episode_metrics), 1)
    fail_statuses = {"rejected", "timeout", "error", "generation_failed"}
    agg: Dict[str, Any] = {
        "run_name":               args.run_name,
        "architecture":           trainer.architecture,
        "critic_type":            trainer.critic_type,
        "use_ceb":                args.use_ceb,
        "state_dim":              tcfg.state_dim,
        "episodes":               len(episode_metrics),
        "total_env_steps":        trainer.total_env_steps,
        "total_gradient_updates": trainer.total_gradient_updates,
        "trained":                trainer.total_gradient_updates > 0,
        "best_reward":            best_reward if best_reward != -float("inf") else None,
        "mean_reward":            (sum(m["total_reward"]      for m in episode_metrics) / n)
                                    if episode_metrics else None,
        "verification_pass_rate": (sum(m["verification_pass"] for m in episode_metrics) / n)
                                    if episode_metrics else None,
        "mean_citation_support":  (sum(m["citation_support"]  for m in episode_metrics) / n)
                                    if episode_metrics else None,
        "mean_latency":           (sum(m["latency_seconds"]   for m in episode_metrics) / n)
                                    if episode_metrics else None,
        "failure_rate":           (sum(1 for m in episode_metrics
                                       if m["final_status"] in fail_statuses) / n)
                                    if episode_metrics else None,
        "last_losses":            last_losses,
        "hyperparameters": {
            "batch_size":    args.batch_size,
            "warmup_steps":  args.warmup_steps,
            "update_every":  args.update_every,
            "min_updates":   args.min_updates,
            "gamma":         args.gamma,
            "tau":           args.tau,
            "actor_lr":      args.actor_lr,
            "critic_lr":     args.critic_lr,
            "noise_sigma":   args.noise_sigma,
            "grad_clip":     args.grad_clip,
            "error_penalty": args.error_penalty,
            "seed":          args.seed,
            "hidden_dim":    args.hidden_dim,
        },
    }
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"[train_maddpg] Aggregate metrics -> {agg_path}")


def _log_params_row(
    writer: csv.DictWriter,
    episode: int, step: int,
    active_agent: str, valid_actions: List[str],
    discrete_action: str, reward: float, done: bool,
    next_active_agent: Optional[str],
    raw: np.ndarray, params: Dict[str, Any],
) -> None:
    from .stage_utils import pad_continuous_action
    padded = pad_continuous_action(active_agent, raw)
    row: Dict[str, Any] = {
        "episode":           episode,
        "step":              step,
        "active_agent":      active_agent,
        "valid_actions":     "|".join(valid_actions),
        "discrete_action":   discrete_action,
        "reward":            round(float(reward), 6),
        "done":              int(bool(done)),
        "next_active_agent": next_active_agent or "",
    }
    for i in range(_MAX_RAW_DIM):
        row[f"raw_{i}"]    = round(float(raw[i]), 6) if i < len(raw) else ""
        row[f"padded_{i}"] = round(float(padded[i]), 6) if i < len(padded) else ""
    for k, v in params.items():
        row[k] = v
    writer.writerow(row)


if __name__ == "__main__":
    sys.exit(train())
