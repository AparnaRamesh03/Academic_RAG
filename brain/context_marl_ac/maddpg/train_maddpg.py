"""
brain/context_marl_ac/maddpg/train_maddpg.py
---------------------------------------------
MADDPG training loop for stage-constrained cooperative RAG.

policy_mode = "maddpg_continuous"

Design:
  - Episode flow is identical to discrete MARL (stage-gated via action masking).
  - At each step the MADDPG actor for the active agent outputs a continuous
    action vector in [-1, 1]^action_dim.
  - continuous_action_mapper converts this to:
      1. Numeric RAG params  (top_k, temperature, strictness, …)
      2. A discrete action name  (within the valid masked set)
  - env.step(agent, action) executes the discrete action — existing adapters
    and agents are unchanged.
  - Transitions are stored in a replay buffer; networks are updated off-policy
    with the DDPG actor-critic rule.

Usage:
  # from Academic_RAG/brain/
  python -m context_marl_ac.maddpg.train_maddpg --episodes 200 --dry-run

  # with Context Engineering Block (20-dim state):
  python -m context_marl_ac.maddpg.train_maddpg --episodes 200 --use-ceb --dry-run
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── sys.path: ensure brain/ is importable ────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ── dotenv: load API keys from known locations ────────────────────────────────
try:
    from dotenv import load_dotenv
    for _ep in [
        _BRAIN_ROOT / ".env",
        _BRAIN_ROOT.parent / ".env",
        Path.home() / "Desktop" / "Academic_RAG" / "brain" / ".env",
        Path.home() / "Desktop" / "Multimodal-Academic-RAG" / ".env",
    ]:
        if _ep.exists():
            load_dotenv(dotenv_path=_ep)
            break
except ImportError:
    pass

import context_marl_ac.config as cfg
from context_marl_ac.marl.marl_env import MARLEnv
from context_marl_ac.schemas.actions import AGENT_ACTIONS, AGENT_NAMES

from .maddpg_agent import MADDPGAgentWrapper
from .maddpg_critic import MADDPGCritic
from .replay_buffer import ReplayBuffer, Transition
from .continuous_action_mapper import (
    AGENT_ACTION_DIMS,
    AGENT_DEFAULTS,
    JOINT_ACTION_DIM,
    ORDERED_AGENTS,
    build_joint_action_vector,
    map_agent_params,
    select_discrete_action,
)
from .context_engineering_block import CEB_STATE_DIM, build_ceb_features

# ── Action-params CSV schema (fixed across all agents) ────────────────────────
# All param key names that any agent can produce, plus base + raw columns.
# Rows for agents that don't have a given key leave it blank (restval='').
_MAX_RAW_DIM = max(AGENT_ACTION_DIMS.values())   # 4
_ALL_PARAM_KEYS: List[str] = list(dict.fromkeys(   # dedup, preserve order
    k
    for defaults in AGENT_DEFAULTS.values()
    for k in defaults.keys()
))
PARAMS_CSV_FIELDNAMES: List[str] = (
    ["episode", "step", "agent", "discrete_action", "reward"]
    + [f"raw_{i}" for i in range(_MAX_RAW_DIM)]
    + _ALL_PARAM_KEYS
)

# ── MADDPG hyper-parameters ───────────────────────────────────────────────────
ACTOR_LR        = 1e-3
CRITIC_LR       = 1e-3
GAMMA           = 0.99
TAU             = 0.005      # soft target update
BATCH_SIZE      = 256
BUFFER_CAPACITY = 50_000
HIDDEN_DIM      = 128
NOISE_SIGMA     = 0.15
GRAD_CLIP       = 1.0
UPDATE_EVERY    = 4          # gradient steps every N env steps
WARMUP_STEPS    = 50         # fill buffer before training


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser("Train MADDPG continuous-control RAG")
    p.add_argument("--run-name",        default="maddpg_run_01")
    p.add_argument("--episodes",        type=int, default=200)
    p.add_argument("--dry-run",         action="store_true",
                   help="Use stub adapters (no real LLM / Qdrant calls)")
    p.add_argument("--benchmark-path",  default="",
                   help="Path to JSONL benchmark (train split)")
    p.add_argument("--use-ceb",         action="store_true",
                   help="Use Context Engineering Block (20-dim) instead of base 14-dim")
    p.add_argument("--checkpoint-path", default="",
                   help="Resume from existing .pt checkpoint")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Save periodic checkpoint every N episodes")
    p.add_argument("--results-dir",     default="",
                   help="Override output directory (default: results/maddpg/)")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_benchmark(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        print(f"[train_maddpg] Benchmark not found at '{path}'. Using dummy question.")
        return [{
            "question":    "What is the transformer attention mechanism?",
            "ground_truth": "Attention is a mechanism that allows the model to focus on relevant parts of the input.",
            "question_id": "dummy_q1",
        }]
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(path) as f:
        return json.load(f)


# ── Network construction ──────────────────────────────────────────────────────

def build_maddpg_agents(
    state_dim: int,
    device: str,
) -> Dict[str, MADDPGAgentWrapper]:
    """All agents share obs_dim = state_dim (global features as shared observation)."""
    return {
        name: MADDPGAgentWrapper(
            agent_name  = name,
            obs_dim     = state_dim,
            hidden_dim  = HIDDEN_DIM,
            lr_actor    = ACTOR_LR,
            noise_sigma = NOISE_SIGMA,
            device      = device,
        )
        for name in AGENT_NAMES
    }


# ── State feature helpers ─────────────────────────────────────────────────────

def _state_features(env: MARLEnv, use_ceb: bool) -> np.ndarray:
    if use_ceb:
        return np.array(build_ceb_features(env.state), dtype=np.float32)
    return np.array(env.get_global_features(), dtype=np.float32)


# ── DDPG update ───────────────────────────────────────────────────────────────

def _ddpg_update(
    agents:        Dict[str, MADDPGAgentWrapper],
    critic:        MADDPGCritic,
    target_critic: MADDPGCritic,
    critic_optim:  torch.optim.Optimizer,
    buffer:        ReplayBuffer,
    device:        torch.device,
) -> Dict[str, float]:
    """One MADDPG gradient update. Returns loss metrics dict."""
    batch = buffer.sample(BATCH_SIZE)

    states      = torch.FloatTensor(np.stack([t.state_features      for t in batch])).to(device)
    joint_acts  = torch.FloatTensor(np.stack([t.joint_action        for t in batch])).to(device)
    rewards     = torch.FloatTensor([[t.reward]                     for t in batch]).to(device)
    next_states = torch.FloatTensor(np.stack([t.next_state_features for t in batch])).to(device)
    dones       = torch.FloatTensor([[float(t.done)]                for t in batch]).to(device)

    # ── Critic update ─────────────────────────────────────────────────────────
    with torch.no_grad():
        next_joint_parts = [agents[n].target_action_batch(next_states) for n in ORDERED_AGENTS]
        next_joint = torch.cat(next_joint_parts, dim=-1)
        target_q   = rewards + GAMMA * (1.0 - dones) * target_critic(next_states, next_joint)

    current_q   = critic(states, joint_acts)
    critic_loss = F.mse_loss(current_q, target_q)

    critic_optim.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
    critic_optim.step()

    # ── Actor updates (one per agent, others frozen) ──────────────────────────
    actor_losses: Dict[str, float] = {}
    for active_name in AGENT_NAMES:
        joint_parts = []
        for name in ORDERED_AGENTS:
            if name == active_name:
                a = agents[name].actor_action_batch(states)  # gradient flows here
            else:
                with torch.no_grad():
                    a = agents[name].actor_action_batch(states)
            joint_parts.append(a)
        joint_t = torch.cat(joint_parts, dim=-1)

        actor_loss = -critic(states, joint_t).mean()
        agents[active_name].actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agents[active_name].actor.parameters(), GRAD_CLIP)
        agents[active_name].actor_optim.step()
        actor_losses[active_name] = actor_loss.item()

    # ── Soft target updates ───────────────────────────────────────────────────
    for agent in agents.values():
        agent.soft_update(TAU)
    for tp, p in zip(target_critic.parameters(), critic.parameters()):
        tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)

    return {
        "critic_loss": critic_loss.item(),
        **{f"actor_loss_{n}": v for n, v in actor_losses.items()},
    }


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _save_checkpoint(
    path:          Path,
    agents:        Dict[str, MADDPGAgentWrapper],
    critic:        MADDPGCritic,
    target_critic: MADDPGCritic,
    critic_optim:  torch.optim.Optimizer,
    episode:       int,
    metrics:       Dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "episode":       episode,
        "metrics":       metrics,
        "agents":        {n: a.state_dict() for n, a in agents.items()},
        "critic":        critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "critic_optim":  critic_optim.state_dict(),
    }, path)
    print(f"  [ckpt] saved -> {path}")


def _load_checkpoint(
    path:          str,
    agents:        Dict[str, MADDPGAgentWrapper],
    critic:        MADDPGCritic,
    target_critic: MADDPGCritic,
    critic_optim:  torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    for n, a in agents.items():
        if n in ckpt.get("agents", {}):
            a.load_state_dict(ckpt["agents"][n])
    if "critic"        in ckpt: critic.load_state_dict(ckpt["critic"])
    if "target_critic" in ckpt: target_critic.load_state_dict(ckpt["target_critic"])
    if "critic_optim"  in ckpt: critic_optim.load_state_dict(ckpt["critic_optim"])
    return ckpt.get("episode", 0)


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    args = _parse_args()
    cfg.DRY_RUN = args.dry_run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Directories ───────────────────────────────────────────────────────────
    base = (
        Path(args.results_dir) if args.results_dir
        else Path(__file__).resolve().parent.parent / "results" / "maddpg"
    )
    ckpt_dir    = base / "checkpoints"
    metrics_dir = base / "metrics"
    traj_dir    = base / "trajectories"
    for d in (ckpt_dir, metrics_dir, traj_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Components ────────────────────────────────────────────────────────────
    env       = MARLEnv()
    benchmark = _load_benchmark(
        args.benchmark_path or str(
            Path(__file__).resolve().parent.parent / "results" / "benchmark_splits" / "train.jsonl"
        )
    )

    state_dim = CEB_STATE_DIM if args.use_ceb else cfg.FEATURE_DIM
    agents    = build_maddpg_agents(state_dim, str(device))

    critic        = MADDPGCritic(state_dim, JOINT_ACTION_DIM, HIDDEN_DIM).to(device)
    target_critic = copy.deepcopy(critic)
    for p in target_critic.parameters():
        p.requires_grad_(False)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    buffer   = ReplayBuffer(BUFFER_CAPACITY)
    start_ep = 0

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"[train_maddpg] Resuming from {args.checkpoint_path}")
        start_ep = _load_checkpoint(
            args.checkpoint_path, agents, critic, target_critic, critic_optim
        )

    # ── Output files ──────────────────────────────────────────────────────────
    ep_csv_path   = metrics_dir / f"episode_metrics_{args.run_name}.csv"
    params_path   = metrics_dir / f"action_params_log_{args.run_name}.csv"
    traj_path     = traj_dir    / f"trajectories_{args.run_name}.jsonl"
    agg_path      = base        / f"aggregate_metrics_{args.run_name}.json"

    ep_csv_f  = open(ep_csv_path,  "w", newline="", encoding="utf-8")
    params_f  = open(params_path,  "w", newline="", encoding="utf-8")
    traj_f    = open(traj_path,    "w", encoding="utf-8")

    # params CSV: fixed schema covering all agents; missing columns left blank.
    params_writer = csv.DictWriter(
        params_f, fieldnames=PARAMS_CSV_FIELDNAMES,
        restval="", extrasaction="ignore",
    )
    params_writer.writeheader()

    ep_csv_writer: Optional[csv.DictWriter] = None

    total_steps        = 0
    best_reward        = -float("inf")
    all_episode_metrics: List[Dict] = []

    print(
        f"[train_maddpg] run={args.run_name} | episodes={args.episodes} | "
        f"device={device} | CEB={args.use_ceb} | dry_run={args.dry_run}\n"
        f"  state_dim={state_dim}  joint_action_dim={JOINT_ACTION_DIM}  "
        f"buffer_capacity={BUFFER_CAPACITY}"
    )

    for ep_idx in tqdm(
        range(start_ep + 1, start_ep + args.episodes + 1), desc="MADDPG"
    ):
        q_idx   = random.randint(0, len(benchmark) - 1)
        q_dict  = benchmark[q_idx]
        state   = env.reset(q_dict, index=q_idx + 1)
        qid     = state.question_id

        for agent in agents.values():
            agent.reset_noise()

        done      = False
        ep_steps  = 0
        ep_traj   = []

        # ── Episode rollout ───────────────────────────────────────────────────
        while not done:
            # Find active agent via stage-gated action masking (unchanged logic).
            active_agent  = None
            valid_actions: List[str] = []
            for name in AGENT_NAMES:
                mask = env.get_mask(name)
                if sum(mask) > 0:
                    active_agent  = name
                    valid_actions = [
                        AGENT_ACTIONS[name][i] for i, m in enumerate(mask) if m == 1
                    ]
                    break

            if not active_agent:
                if state.final_status == "pending":
                    state.final_status = "abstained"
                state.done = True
                done = True
                break

            # Pre-step global state features.
            prev_features = _state_features(env, args.use_ceb)

            # MADDPG actor outputs continuous params.
            raw_action = agents[active_agent].select_action(prev_features, explore=True)
            params     = agents[active_agent].map_params(raw_action)

            # Map to discrete action (respects valid_actions from mask).
            discrete_action = select_discrete_action(active_agent, params, valid_actions)

            # Build joint action vector (active agent real, inactive zeroed).
            joint_vec = build_joint_action_vector({active_agent: raw_action})

            # Execute via existing env.step — stage constraints fully preserved.
            # params are injected into state so agents can adapt RAG behaviour.
            new_state, reward, done, info = env.step(active_agent, discrete_action, params=params)

            next_features = _state_features(env, args.use_ceb)
            ep_steps  += 1
            total_steps += 1

            # Store transition.
            buffer.push(Transition(
                state_features      = prev_features,
                agent_raw_actions   = {active_agent: raw_action.copy()},
                mapped_params       = {active_agent: params},
                joint_action        = joint_vec,
                reward              = reward,
                next_state_features = next_features,
                done                = done,
                stage               = active_agent,
                selected_agent      = active_agent,
                action_taken        = discrete_action,
                question_id         = qid,
                step                = ep_steps,
                metrics_snapshot    = {
                    "citation_support_rate":  new_state.citation_support_rate,
                    "num_unsupported_claims": len(new_state.unsupported_claims),
                    "final_status":           new_state.final_status,
                },
            ))

            # Log raw actor output + mapped params (only this agent's param keys
            # are set; all other param columns are left blank via restval='').
            params_row: Dict[str, Any] = {
                "episode":         ep_idx,
                "step":            ep_steps,
                "agent":           active_agent,
                "discrete_action": discrete_action,
                "reward":          round(reward, 6),
                **{f"raw_{i}": round(float(raw_action[i]), 6) for i in range(len(raw_action))},
                **params,
            }
            params_writer.writerow(params_row)

            # Trajectory record.
            ep_traj.append({
                "step":            ep_steps,
                "agent":           active_agent,
                "discrete_action": discrete_action,
                "raw_action":      raw_action.tolist(),
                "mapped_params":   params,
                "reward":          reward,
                "done":            done,
            })

            # DDPG network update (after warmup, every UPDATE_EVERY steps).
            if buffer.is_ready(WARMUP_STEPS) and total_steps % UPDATE_EVERY == 0:
                _ddpg_update(agents, critic, target_critic, critic_optim, buffer, device)

            state = new_state

        # ── Episode summary ────────────────────────────────────────────────────
        ep_reward = env.get_global_reward()
        ep_metrics: Dict[str, Any] = {
            "episode":           ep_idx,
            "question_id":       qid,
            "total_reward":      round(ep_reward, 6),
            "num_steps":         state.num_steps,
            "num_llm_calls":     state.num_llm_calls,
            "final_status":      state.final_status,
            "verification_pass": int(state.final_status == "accepted"),
            "citation_support":  round(state.citation_support_rate, 4),
            "latency_seconds":   round(state.latency_so_far, 3),
            "token_usage":       state.token_usage,
            "buffer_size":       len(buffer),
        }
        all_episode_metrics.append(ep_metrics)

        if ep_csv_writer is None:
            ep_csv_writer = csv.DictWriter(ep_csv_f, fieldnames=list(ep_metrics.keys()))
            ep_csv_writer.writeheader()
        ep_csv_writer.writerow(ep_metrics)
        ep_csv_f.flush()

        # Trajectory JSONL.
        traj_f.write(json.dumps({
            "episode":      ep_idx,
            "question_id":  qid,
            "total_reward": ep_reward,
            "final_status": state.final_status,
            "steps":        ep_traj,
        }, ensure_ascii=False) + "\n")
        traj_f.flush()

        # Checkpoints.
        if ep_reward > best_reward:
            best_reward = ep_reward
            _save_checkpoint(
                ckpt_dir / "best_reward.pt",
                agents, critic, target_critic, critic_optim, ep_idx, ep_metrics,
            )
        if ep_idx % args.checkpoint_every == 0:
            _save_checkpoint(
                ckpt_dir / f"ep_{ep_idx:04d}.pt",
                agents, critic, target_critic, critic_optim, ep_idx, ep_metrics,
            )

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    if all_episode_metrics:
        n   = len(all_episode_metrics)
        agg = {
            "run_name":               args.run_name,
            "episodes":               n,
            "use_ceb":                args.use_ceb,
            "state_dim":              state_dim,
            "joint_action_dim":       JOINT_ACTION_DIM,
            "mean_reward":            sum(m["total_reward"]      for m in all_episode_metrics) / n,
            "mean_steps":             sum(m["num_steps"]         for m in all_episode_metrics) / n,
            "mean_llm_calls":         sum(m["num_llm_calls"]     for m in all_episode_metrics) / n,
            "verification_pass_rate": sum(m["verification_pass"] for m in all_episode_metrics) / n,
            "mean_citation_support":  sum(m["citation_support"]  for m in all_episode_metrics) / n,
            "mean_latency":           sum(m["latency_seconds"]   for m in all_episode_metrics) / n,
            "best_reward":            best_reward,
            "total_env_steps":        total_steps,
        }
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n[train_maddpg] Aggregate metrics -> {agg_path}")

    ep_csv_f.close()
    params_f.close()
    traj_f.close()

    print(
        f"[train_maddpg] Done. best_reward={best_reward:.4f}  "
        f"total_steps={total_steps}\n"
        f"  Results in: {base}"
    )


if __name__ == "__main__":
    train()
