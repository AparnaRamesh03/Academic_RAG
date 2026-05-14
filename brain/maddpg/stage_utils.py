"""
brain/maddpg/stage_utils.py
----------------------------
Shared helpers for the stage-conditioned MADDPG-style architecture.

This module centralises:
  - canonical agent ordering / IDs
  - one-hot encodings for active agent and discrete action
  - padding for the active-agent continuous action so the critic always sees
    a fixed-width vector regardless of which agent is acting
  - stage-gated detection of "who is the active agent right now"

All staged-MADDPG modules (trainer, train script, live runner, evaluator) MUST
use these helpers — do not duplicate the active-agent loop in callers.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from context_marl_ac.schemas.actions import AGENT_ACTIONS, AGENT_NAMES

from .continuous_action_mapper import AGENT_ACTION_DIMS


ORDERED_AGENTS: List[str] = list(AGENT_NAMES)                    # canonical order
AGENT_TO_ID: dict[str, int] = {n: i for i, n in enumerate(ORDERED_AGENTS)}
NUM_AGENTS: int = len(ORDERED_AGENTS)

MAX_CONTINUOUS_ACTION_DIM: int = max(AGENT_ACTION_DIMS.values())          # 4
MAX_DISCRETE_ACTIONS: int = max(len(v) for v in AGENT_ACTIONS.values())   # 5


# ── One-hot encodings ────────────────────────────────────────────────────────

def agent_one_hot(agent_name: Optional[str]) -> np.ndarray:
    """Length-NUM_AGENTS one-hot. All zeros if agent_name is None/unknown."""
    vec = np.zeros(NUM_AGENTS, dtype=np.float32)
    if agent_name is None:
        return vec
    idx = AGENT_TO_ID.get(agent_name)
    if idx is None:
        return vec
    vec[idx] = 1.0
    return vec


def discrete_action_one_hot(
    agent_name: Optional[str],
    action_name: Optional[str],
) -> np.ndarray:
    """
    Length-MAX_DISCRETE_ACTIONS one-hot. The index is the action's position
    within the agent's own action list (AGENT_ACTIONS[agent_name]). All-zero
    if either argument is None or unknown.

    Using a single shared MAX_DISCRETE_ACTIONS width lets the critic consume
    a fixed-shape discrete context tensor for any agent.
    """
    vec = np.zeros(MAX_DISCRETE_ACTIONS, dtype=np.float32)
    if agent_name is None or action_name is None:
        return vec
    actions = AGENT_ACTIONS.get(agent_name, [])
    if action_name not in actions:
        return vec
    vec[actions.index(action_name)] = 1.0
    return vec


def pad_continuous_action(
    agent_name: Optional[str],
    raw_action: Optional[np.ndarray],
) -> np.ndarray:
    """
    Pad/clip the active agent's continuous action to MAX_CONTINUOUS_ACTION_DIM.

    - all-zero if either argument is None
    - clipped to [-1, 1] (sanity, in case the actor's tanh was bypassed)
    - truncated if longer than MAX (should not happen if dims are consistent)
    """
    vec = np.zeros(MAX_CONTINUOUS_ACTION_DIM, dtype=np.float32)
    if agent_name is None or raw_action is None:
        return vec
    arr = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
    dim = min(len(arr), MAX_CONTINUOUS_ACTION_DIM)
    vec[:dim] = arr[:dim]
    return vec


# ── Stage-gated active-agent detection ───────────────────────────────────────

def find_active_agent_and_valid_actions(env: Any) -> Tuple[Optional[str], List[str]]:
    """
    Return the first agent whose mask has at least one valid action, plus the
    list of those action names.

    Returns (None, []) if no agent has a valid action (episode is effectively
    over — the caller should mark it done/abstained).
    """
    for name in ORDERED_AGENTS:
        try:
            mask = env.get_mask(name)
        except Exception:
            mask = []
        if mask and sum(mask) > 0:
            valid = [AGENT_ACTIONS[name][i] for i, m in enumerate(mask) if m == 1]
            return name, valid
    return None, []
