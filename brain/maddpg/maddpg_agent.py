"""
brain/maddpg/maddpg_agent.py
---------------------------------------------
Per-agent MADDPG wrapper: actor + target actor + optimizer + OUNoise.
"""
import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.optim as optim

from .maddpg_actor import MADDPGActor
from .noise import OUNoise
from .continuous_action_mapper import AGENT_ACTION_DIMS, AGENT_DEFAULTS, map_agent_params


class MADDPGAgentWrapper:
    def __init__(
        self,
        agent_name: str,
        obs_dim: int,
        hidden_dim: int = 128,
        lr_actor: float = 1e-3,
        noise_sigma: float = 0.15,
        device: str = "cpu",
    ):
        self.name       = agent_name
        self.action_dim = AGENT_ACTION_DIMS[agent_name]
        self.device     = torch.device(device)

        self.actor = MADDPGActor(obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad_(False)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.noise = OUNoise(self.action_dim, sigma=noise_sigma)

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """Return continuous action in [-1, 1]^action_dim."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0).cpu().numpy()
        self.actor.train()
        if explore:
            action = np.clip(action + self.noise.sample(), -1.0, 1.0)
        return action.astype(np.float32)

    def target_action_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Batch target-actor inference; no gradient."""
        obs_batch = obs_batch.to(self.device)
        with torch.no_grad():
            return self.target_actor(obs_batch)

    def actor_action_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Batch actor inference; gradient flows for the active agent."""
        return self.actor(obs_batch.to(self.device))

    # ── Param mapping ──────────────────────────────────────────────────────────

    def map_params(self, raw: np.ndarray) -> Dict[str, Any]:
        """Map raw [-1,1] action to RAG execution params with safe fallback."""
        try:
            return map_agent_params(self.name, raw)
        except Exception:
            return dict(AGENT_DEFAULTS.get(self.name, {}))

    # ── Noise / soft update ────────────────────────────────────────────────────

    def reset_noise(self):
        self.noise.reset()

    def soft_update(self, tau: float = 0.005):
        for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    # ── Serialisation ──────────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        return {
            "actor":        self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "actor_optim":  self.actor_optim.state_dict(),
        }

    def load_state_dict(self, d: Dict):
        self.actor.load_state_dict(d["actor"])
        self.target_actor.load_state_dict(d["target_actor"])
        self.actor_optim.load_state_dict(d["actor_optim"])
