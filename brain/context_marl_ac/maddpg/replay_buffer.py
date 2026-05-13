"""
brain/context_marl_ac/maddpg/replay_buffer.py
----------------------------------------------
Off-policy replay buffer for MADDPG.
Stores SARS transitions with per-episode metadata.
"""
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class Transition:
    state_features:      np.ndarray            # global state  (state_dim,)
    agent_raw_actions:   Dict[str, np.ndarray]  # raw [-1,1] outputs per active agent
    mapped_params:       Dict[str, Dict]        # human-readable RAG params
    joint_action:        np.ndarray             # (joint_action_dim,) concatenated
    reward:              float
    next_state_features: np.ndarray
    done:                bool
    stage:               str                   # which stage ran (= selected_agent)
    selected_agent:      str
    action_taken:        str                   # discrete action name actually executed
    question_id:         str
    step:                int
    metrics_snapshot:    Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self._buf.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        k = min(batch_size, len(self._buf))
        return random.sample(list(self._buf), k)

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, min_size: int = 256) -> bool:
        return len(self._buf) >= min_size
