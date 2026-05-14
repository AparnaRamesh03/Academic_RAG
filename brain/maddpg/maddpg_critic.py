"""
brain/maddpg/maddpg_critic.py
----------------------------------------------
MADDPG centralized critic.
Input:  global state features  (state_dim,)
        joint continuous action (joint_action_dim,)
Output: Q(s, a)  scalar
"""
import torch
import torch.nn as nn


class MADDPGCritic(nn.Module):
    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + joint_action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        """
        state:         (batch, state_dim)
        joint_actions: (batch, joint_action_dim)
        returns:       (batch, 1)
        """
        return self.net(torch.cat([state, joint_actions], dim=-1))
