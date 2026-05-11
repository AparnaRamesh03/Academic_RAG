"""
brain/context_marl_ac/marl/supervisor_actor.py
----------------------------------------------
Supervisor actor for fully free MARL routing.

The supervisor chooses which agent should act next:
    retriever / rewriter / grader / generator / verifier

The chosen agent's own actor then chooses the concrete action.
"""

from typing import List, Dict

import torch
import torch.nn as nn

from context_marl_ac.config import FEATURE_DIM, HIDDEN_DIM
from context_marl_ac.schemas.actions import AGENT_NAMES


SUPERVISOR_AGENTS: List[str] = list(AGENT_NAMES)

SUPERVISOR_AGENT_TO_ID: Dict[str, int] = {
    name: idx for idx, name in enumerate(SUPERVISOR_AGENTS)
}

SUPERVISOR_ID_TO_AGENT: Dict[int, str] = {
    idx: name for name, idx in SUPERVISOR_AGENT_TO_ID.items()
}


class SupervisorActor(nn.Module):
    """
    MLP policy that chooses the next agent from the global ContextState features.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_agents: int = len(SUPERVISOR_AGENTS),
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents),
        )

    def forward(self, global_features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns logits over agents.

        Args:
            global_features: Tensor of shape [batch, FEATURE_DIM]
            mask: optional binary mask of shape [batch, num_agents]
        """
        logits = self.net(global_features)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        return logits


def build_supervisor_actor() -> SupervisorActor:
    return SupervisorActor()