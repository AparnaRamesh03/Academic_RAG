"""
brain/maddpg/maddpg_actor.py
---------------------------------------------
MADDPG per-agent deterministic actor network.
Input:  observation vector  (obs_dim,)
Output: continuous action in [-1, 1]^action_dim  via tanh
"""
import torch
import torch.nn as nn


class MADDPGActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),   # outputs in [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, obs_dim)  →  (batch, action_dim) in [-1, 1]"""
        return self.net(obs)
