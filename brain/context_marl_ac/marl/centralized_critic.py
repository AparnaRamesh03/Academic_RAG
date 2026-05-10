"""
brain/context_marl_ac/marl/centralized_critic.py
------------------------------------------------
Centralized critic that estimates V(s) based on global state features.
"""

import torch
import torch.nn as nn
from context_marl_ac.config import HIDDEN_DIM, FEATURE_DIM

class CentralizedCritic(nn.Module):
    """
    Estimates state value V(s) from global feature vector.
    Used for advantage estimation across all agents.
    """
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar state value V(s).
        """
        return self.net(features)
