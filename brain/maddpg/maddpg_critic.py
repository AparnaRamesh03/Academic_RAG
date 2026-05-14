"""
brain/maddpg/maddpg_critic.py
----------------------------------------------
MADDPG-style centralised critics.

The default critic is the **StageConditionedCritic**: because the RAG
environment is stage-gated (exactly one agent acts at each step), we condition
the critic on (state, active_agent, discrete_action_context, padded_continuous_action)
instead of training Q(state, joint_action) where joint_action is mostly zeros.

    Q(
        state_features              (state_dim,)
        active_agent_one_hot        (NUM_AGENTS,)
        discrete_action_one_hot     (MAX_DISCRETE_ACTIONS,)
        active_continuous_padded    (MAX_CONTINUOUS_ACTION_DIM,)
    ) -> scalar

The legacy joint-action critic is kept under MADDPGCritic for backward-compat
checkpoint loading only — new training MUST use StageConditionedCritic.
"""
import torch
import torch.nn as nn


class StageConditionedCritic(nn.Module):
    """
    Centralised Q-network that is conditioned on which agent is currently
    active. Suitable for stage-gated multi-agent environments where only one
    agent's action takes effect per step.
    """

    critic_type: str = "stage_conditioned"

    def __init__(
        self,
        state_dim: int,
        num_agents: int,
        max_discrete_actions: int,
        max_continuous_action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.max_discrete_actions = max_discrete_actions
        self.max_continuous_action_dim = max_continuous_action_dim
        self.hidden_dim = hidden_dim

        in_dim = state_dim + num_agents + max_discrete_actions + max_continuous_action_dim
        self.input_dim = in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state:                    torch.Tensor,
        active_agent_one_hot:     torch.Tensor,
        discrete_action_one_hot:  torch.Tensor,
        active_continuous_padded: torch.Tensor,
    ) -> torch.Tensor:
        """
        state                     : (batch, state_dim)
        active_agent_one_hot      : (batch, num_agents)
        discrete_action_one_hot   : (batch, max_discrete_actions)
        active_continuous_padded  : (batch, max_continuous_action_dim)
        returns                   : (batch, 1)
        """
        x = torch.cat(
            [state, active_agent_one_hot, discrete_action_one_hot, active_continuous_padded],
            dim=-1,
        )
        return self.net(x)


class MADDPGCritic(nn.Module):
    """
    Legacy joint-action critic. Retained ONLY so that old checkpoints can be
    loaded without crashing. New training paths must use StageConditionedCritic.
    """

    critic_type: str = "joint_action_legacy"

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
        return self.net(torch.cat([state, joint_actions], dim=-1))
