"""
brain/context_marl_ac/maddpg/
MADDPG continuous-control extension for the stage-constrained cooperative MARL RAG.

policy_mode: 'discrete_marl'       — existing A2C discrete actors (unchanged)
             'maddpg_continuous'   — MADDPG actors output continuous params that
                                     select and parameterise discrete stage actions.
"""
from .maddpg_actor import MADDPGActor
from .maddpg_critic import MADDPGCritic
from .maddpg_agent import MADDPGAgentWrapper
from .replay_buffer import ReplayBuffer, Transition
from .noise import OUNoise, GaussianNoise
from .continuous_action_mapper import (
    map_agent_params,
    select_discrete_action,
    build_joint_action_vector,
    AGENT_ACTION_DIMS,
    JOINT_ACTION_DIM,
    ORDERED_AGENTS,
    AGENT_DEFAULTS,
)
from .context_engineering_block import build_ceb_features, CEB_STATE_DIM

__all__ = [
    "MADDPGActor",
    "MADDPGCritic",
    "MADDPGAgentWrapper",
    "ReplayBuffer",
    "Transition",
    "OUNoise",
    "GaussianNoise",
    "map_agent_params",
    "select_discrete_action",
    "build_joint_action_vector",
    "AGENT_ACTION_DIMS",
    "JOINT_ACTION_DIM",
    "ORDERED_AGENTS",
    "AGENT_DEFAULTS",
    "build_ceb_features",
    "CEB_STATE_DIM",
]
