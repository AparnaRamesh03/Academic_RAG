"""
brain/maddpg/trainer.py
-----------------------------
Shared MADDPG-style continuous-control trainer for the stage-gated RAG env.

This is the single source of truth for:
  - building agents + stage-conditioned critic
  - one DDPG-style gradient update (critic + per-agent actor)
  - replay buffer push for a single env step
  - error-path terminal transition push
  - checkpoint save / load

`train_maddpg.py`, `live_maddpg_runner.py`, and `evaluate_maddpg.py` MUST go
through this module — do not duplicate the update logic.

Hyperparameters are passed in via the `TrainerConfig` dataclass; there are no
module-level constants gating gradient updates.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from context_marl_ac.schemas.actions import AGENT_NAMES

from .continuous_action_mapper import (
    AGENT_ACTION_DIMS,
    AGENT_DEFAULTS,
    map_agent_params,
    select_discrete_action,
)
from .maddpg_agent import MADDPGAgentWrapper
from .maddpg_critic import StageConditionedCritic
from .noise import OUNoise
from .replay_buffer import ReplayBuffer, Transition
from .stage_utils import (
    AGENT_TO_ID,
    MAX_CONTINUOUS_ACTION_DIM,
    MAX_DISCRETE_ACTIONS,
    NUM_AGENTS,
    ORDERED_AGENTS,
    agent_one_hot,
    discrete_action_one_hot,
    find_active_agent_and_valid_actions,
    pad_continuous_action,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    state_dim:     int
    hidden_dim:    int   = 128
    actor_lr:      float = 1e-3
    critic_lr:     float = 1e-3
    gamma:         float = 0.99
    tau:           float = 0.005
    batch_size:    int   = 64
    buffer_capacity: int = 50_000
    noise_sigma:   float = 0.15
    grad_clip:     float = 1.0
    update_every:  int   = 4
    warmup_steps:  int   = 50
    error_penalty: float = -1.0
    device:        str   = "cpu"


# ── Trainer ───────────────────────────────────────────────────────────────────

class StageConditionedMADDPGTrainer:
    """
    Single-file implementation of the MADDPG-style stage-conditioned trainer.

    Composition:
      - `agents`: dict[str, MADDPGAgentWrapper]  — one deterministic actor per role
      - `critic`: StageConditionedCritic         — Q(s, a_active, agent, discrete)
      - `target_critic`: deep-copied critic
      - `buffer`: ReplayBuffer of Transition

    All gradient updates flow through `.update()`. Episode rollout helpers
    (`rollout_step`, `push_error_transition`) are provided so train scripts
    only have to call them, not reimplement the per-step logic.
    """

    critic_type: str = StageConditionedCritic.critic_type
    architecture: str = "maddpg_style_continuous_control"

    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.agents: Dict[str, MADDPGAgentWrapper] = {
            name: MADDPGAgentWrapper(
                agent_name  = name,
                obs_dim     = cfg.state_dim,
                hidden_dim  = cfg.hidden_dim,
                lr_actor    = cfg.actor_lr,
                noise_sigma = cfg.noise_sigma,
                device      = cfg.device,
            )
            for name in AGENT_NAMES
        }

        self.critic = StageConditionedCritic(
            state_dim                 = cfg.state_dim,
            num_agents                = NUM_AGENTS,
            max_discrete_actions      = MAX_DISCRETE_ACTIONS,
            max_continuous_action_dim = MAX_CONTINUOUS_ACTION_DIM,
            hidden_dim                = cfg.hidden_dim,
        ).to(self.device)

        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr
        )

        self.buffer = ReplayBuffer(cfg.buffer_capacity)

        self.total_env_steps:        int = 0
        self.total_gradient_updates: int = 0

    # ── Rollout helpers ───────────────────────────────────────────────────────

    def reset_noise(self) -> None:
        for a in self.agents.values():
            a.reset_noise()

    def select_action(
        self,
        active_agent: str,
        state_features: np.ndarray,
        valid_actions: List[str],
        explore: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        """
        Returns (raw_action, mapped_params, discrete_action) for the active agent.
        """
        raw = self.agents[active_agent].select_action(state_features, explore=explore)
        params = map_agent_params(active_agent, raw)
        discrete = select_discrete_action(active_agent, params, valid_actions)
        return raw, params, discrete

    def push_transition(
        self,
        *,
        state_features:      np.ndarray,
        active_agent:        str,
        valid_actions:       List[str],
        raw_action:          np.ndarray,
        mapped_params:       Dict[str, Any],
        discrete_action:     str,
        reward:              float,
        next_state_features: np.ndarray,
        next_active_agent:   Optional[str],
        next_valid_actions:  List[str],
        done:                bool,
        question_id:         str,
        step:                int,
        final_status:        str,
        metrics_snapshot:    Optional[Dict[str, Any]] = None,
    ) -> None:
        t = Transition(
            state_features          = np.asarray(state_features, dtype=np.float32),
            active_agent            = active_agent,
            active_agent_id         = AGENT_TO_ID[active_agent],
            valid_actions           = list(valid_actions),
            raw_action              = np.asarray(raw_action, dtype=np.float32),
            action_padded           = pad_continuous_action(active_agent, raw_action),
            mapped_params           = dict(mapped_params),
            discrete_action         = discrete_action,
            discrete_action_one_hot = discrete_action_one_hot(active_agent, discrete_action),
            reward                  = float(reward),
            next_state_features     = np.asarray(next_state_features, dtype=np.float32),
            next_active_agent       = next_active_agent,
            next_active_agent_id    = AGENT_TO_ID[next_active_agent] if next_active_agent else None,
            next_valid_actions      = list(next_valid_actions),
            done                    = bool(done),
            question_id             = str(question_id),
            step                    = int(step),
            final_status            = str(final_status),
            metrics_snapshot        = dict(metrics_snapshot or {}),
        )
        self.buffer.push(t)
        self.total_env_steps += 1

    def push_error_transition(
        self,
        *,
        state_features:    np.ndarray,
        active_agent:      Optional[str],
        valid_actions:     List[str],
        raw_action:        Optional[np.ndarray],
        mapped_params:     Optional[Dict[str, Any]],
        discrete_action:   Optional[str],
        question_id:       str,
        step:              int,
        error_message:     str,
    ) -> None:
        """
        Push a terminal NEGATIVE transition when env.step raises. This keeps the
        learning signal alive even when an LLM call or verifier crashes.
        """
        agent = active_agent or ORDERED_AGENTS[0]
        raw = raw_action if raw_action is not None else np.zeros(
            AGENT_ACTION_DIMS[agent], dtype=np.float32
        )
        discrete = discrete_action or (valid_actions[0] if valid_actions else "")
        self.push_transition(
            state_features      = state_features,
            active_agent        = agent,
            valid_actions       = valid_actions,
            raw_action          = raw,
            mapped_params       = mapped_params or dict(AGENT_DEFAULTS.get(agent, {})),
            discrete_action     = discrete,
            reward              = self.cfg.error_penalty,
            next_state_features = state_features,
            next_active_agent   = None,
            next_valid_actions  = [],
            done                = True,
            question_id         = question_id,
            step                = step,
            final_status        = "error",
            metrics_snapshot    = {"error_message": str(error_message)[:500]},
        )

    # ── Gradient update ───────────────────────────────────────────────────────

    def should_update(self) -> bool:
        cfg = self.cfg
        return (
            len(self.buffer) >= cfg.warmup_steps
            and len(self.buffer) >= cfg.batch_size
            and self.total_env_steps % cfg.update_every == 0
        )

    def update(self) -> Dict[str, float]:
        """One critic step + one actor step per agent that has samples in batch."""
        cfg = self.cfg
        batch = self.buffer.sample(cfg.batch_size)
        device = self.device

        states     = torch.from_numpy(np.stack([t.state_features      for t in batch])).float().to(device)
        agent_ohs  = torch.from_numpy(np.stack([agent_one_hot(t.active_agent) for t in batch])).to(device)
        disc_ohs   = torch.from_numpy(np.stack([t.discrete_action_one_hot for t in batch])).to(device)
        act_pads   = torch.from_numpy(np.stack([t.action_padded      for t in batch])).to(device)
        rewards    = torch.tensor([[t.reward] for t in batch], dtype=torch.float32, device=device)
        next_states= torch.from_numpy(np.stack([t.next_state_features for t in batch])).float().to(device)
        dones      = torch.tensor([[float(t.done)] for t in batch], dtype=torch.float32, device=device)

        # ── Build next-step inputs for the target critic, transition-by-transition.
        next_agent_ohs_np  = np.zeros((len(batch), NUM_AGENTS), dtype=np.float32)
        next_disc_ohs_np   = np.zeros((len(batch), MAX_DISCRETE_ACTIONS), dtype=np.float32)
        next_act_pads_np   = np.zeros((len(batch), MAX_CONTINUOUS_ACTION_DIM), dtype=np.float32)

        for i, t in enumerate(batch):
            if t.done or t.next_active_agent is None or not t.next_valid_actions:
                continue  # leave zeros
            na = t.next_active_agent
            with torch.no_grad():
                ns = torch.from_numpy(t.next_state_features).float().unsqueeze(0).to(device)
                raw_next = self.agents[na].target_actor(ns).squeeze(0).cpu().numpy()
            params_next = map_agent_params(na, raw_next)
            try:
                disc_next = select_discrete_action(na, params_next, t.next_valid_actions)
            except Exception:
                disc_next = t.next_valid_actions[0]
            next_agent_ohs_np[i] = agent_one_hot(na)
            next_disc_ohs_np[i]  = discrete_action_one_hot(na, disc_next)
            next_act_pads_np[i]  = pad_continuous_action(na, raw_next)

        next_agent_ohs = torch.from_numpy(next_agent_ohs_np).to(device)
        next_disc_ohs  = torch.from_numpy(next_disc_ohs_np).to(device)
        next_act_pads  = torch.from_numpy(next_act_pads_np).to(device)

        # ── Critic loss ───────────────────────────────────────────────────────
        with torch.no_grad():
            target_q = rewards + cfg.gamma * (1.0 - dones) * self.target_critic(
                next_states, next_agent_ohs, next_disc_ohs, next_act_pads
            )

        current_q = self.critic(states, agent_ohs, disc_ohs, act_pads)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_optim.step()

        # ── Actor updates: one per agent, only on batch rows where it was active ─
        actor_losses: Dict[str, float] = {}
        for agent_name in ORDERED_AGENTS:
            idxs = [i for i, t in enumerate(batch) if t.active_agent == agent_name]
            if not idxs:
                continue
            sel_states = states[idxs]
            sel_disc   = disc_ohs[idxs]  # treat discrete context as fixed (non-differentiable)
            sel_agent  = agent_ohs[idxs]

            raw_actions = self.agents[agent_name].actor(sel_states)  # gradient flows
            # Pad raw_actions on the fly so gradient flows into the critic's
            # active-continuous-action slot.
            agent_dim = AGENT_ACTION_DIMS[agent_name]
            if agent_dim < MAX_CONTINUOUS_ACTION_DIM:
                pad = torch.zeros(
                    (raw_actions.size(0), MAX_CONTINUOUS_ACTION_DIM - agent_dim),
                    device=device, dtype=raw_actions.dtype,
                )
                act_padded = torch.cat([raw_actions, pad], dim=-1)
            else:
                act_padded = raw_actions[:, :MAX_CONTINUOUS_ACTION_DIM]

            actor_loss = -self.critic(sel_states, sel_agent, sel_disc, act_padded).mean()
            self.agents[agent_name].actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agents[agent_name].actor.parameters(), cfg.grad_clip
            )
            self.agents[agent_name].actor_optim.step()
            actor_losses[agent_name] = float(actor_loss.item())

        # ── Soft target updates ───────────────────────────────────────────────
        for a in self.agents.values():
            a.soft_update(cfg.tau)
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(cfg.tau * p.data + (1.0 - cfg.tau) * tp.data)

        self.total_gradient_updates += 1
        return {
            "critic_loss": float(critic_loss.item()),
            **{f"actor_loss_{n}": v for n, v in actor_losses.items()},
        }

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "critic_type":            self.critic_type,
            "architecture":           self.architecture,
            "state_dim":              self.cfg.state_dim,
            "hidden_dim":             self.cfg.hidden_dim,
            "total_env_steps":        self.total_env_steps,
            "total_gradient_updates": self.total_gradient_updates,
            "agents":                 {n: a.state_dict() for n, a in self.agents.items()},
            "critic":                 self.critic.state_dict(),
            "target_critic":          self.target_critic.state_dict(),
            "critic_optim":           self.critic_optim.state_dict(),
            "extra":                  dict(extra or {}),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu")
        if ckpt.get("critic_type") not in (None, self.critic_type):
            print(
                f"  [warn] Checkpoint critic_type={ckpt.get('critic_type')} != "
                f"{self.critic_type}; loading actors only."
            )
        for n, a in self.agents.items():
            if n in ckpt.get("agents", {}):
                try:
                    a.load_state_dict(ckpt["agents"][n])
                except RuntimeError as e:
                    print(f"  [warn] Could not load actor '{n}': {e}")
        if ckpt.get("critic_type") in (None, self.critic_type):
            if "critic" in ckpt:
                try:
                    self.critic.load_state_dict(ckpt["critic"])
                except RuntimeError as e:
                    print(f"  [warn] Could not load critic: {e}")
            if "target_critic" in ckpt:
                try:
                    self.target_critic.load_state_dict(ckpt["target_critic"])
                except RuntimeError:
                    pass
            if "critic_optim" in ckpt:
                try:
                    self.critic_optim.load_state_dict(ckpt["critic_optim"])
                except (ValueError, RuntimeError):
                    pass
        self.total_env_steps = int(ckpt.get("total_env_steps", self.total_env_steps))
        self.total_gradient_updates = int(ckpt.get("total_gradient_updates", self.total_gradient_updates))
        return ckpt

    # ── Offline-replay JSONL serialization ───────────────────────────────────

    def save_replay_jsonl(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for d in self.buffer.to_list():
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
        return n

    def load_replay_jsonl(self, path: Path) -> int:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        n = self.buffer.extend_from_dicts(rows)
        self.total_env_steps += n
        return n
