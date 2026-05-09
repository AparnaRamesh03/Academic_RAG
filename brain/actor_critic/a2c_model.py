"""
a2c_model.py
------------
Offline Actor-Critic Neural Network Architecture.

Note: This is designed for offline actor-critic training from existing
trajectory logs, not true on-policy A2C. It learns from fixed historical
data rather than collecting fresh rollouts.
"""
from typing import Tuple

import torch
import torch.nn as nn


class ActorCriticNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 5,
    ):
        super().__init__()

        # Actor backbone
        self.actor_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Critic backbone
        self.critic_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head: maps actor representation to action logits
        self.actor_head = nn.Linear(hidden_dim, output_dim)

        # Critic head: maps critic representation to state value V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        
        Returns:
            action_logits: Tensor of shape (batch_size, output_dim)
            state_value: Tensor of shape (batch_size, 1)
        """
        actor_features = self.actor_net(x)
        critic_features = self.critic_net(x)
        
        action_logits = self.actor_head(actor_features)
        state_value = self.critic_head(critic_features)
        
        return action_logits, state_value
