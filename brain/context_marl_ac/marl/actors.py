"""
brain/context_marl_ac/marl/actors.py
-----------------------------------
Decentralized actor policies for the 5 agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from context_marl_ac.config import HIDDEN_DIM
from context_marl_ac.schemas.observations import OBS_DIM
from context_marl_ac.schemas.actions import AGENT_ACTIONS

class ActorNet(nn.Module):
    """
    MLP-based actor policy for a single agent.
    Input: local observation vector.
    Output: action logits.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns action logits. If mask is provided, sets illegal actions to -inf.
        """
        logits = self.net(obs)
        if mask is not None:
            # Mask illegal actions with a large negative value
            logits = logits.masked_fill(mask == 0, -1e9)
        return logits

def build_marl_actors() -> nn.ModuleDict:
    """
    Builds a dictionary of actor networks, one for each agent.
    """
    actors = nn.ModuleDict()
    for agent_name, actions in AGENT_ACTIONS.items():
        obs_dim = OBS_DIM[agent_name]
        action_dim = len(actions)
        actors[agent_name] = ActorNet(obs_dim, action_dim)
    return actors
