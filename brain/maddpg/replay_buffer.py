"""
brain/maddpg/replay_buffer.py
----------------------------------------------
Off-policy replay buffer for stage-conditioned MADDPG-style training.

Each Transition captures one env.step. Because the environment is stage-gated,
only one agent acts per step — the buffer therefore stores the active agent
identity, that agent's continuous action (padded for fixed-width critic input),
the discrete action it executed, and the *next* active agent so the target
critic knows which actor to roll forward.

JSON-serialisable: every numpy array is stored as a list when calling
to_dict()/from_dict(), so trajectories can be saved as JSONL for offline-replay
training (see trainer.py `--train-mode offline-replay`).
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Transition:
    # ── State ────────────────────────────────────────────────────────────────
    state_features:           np.ndarray              # (state_dim,)

    # ── Active agent and its action this step ────────────────────────────────
    active_agent:             str
    active_agent_id:          int
    valid_actions:            List[str]

    raw_action:               np.ndarray              # raw [-1, 1] vector, len = agent's action_dim
    action_padded:            np.ndarray              # (MAX_CONTINUOUS_ACTION_DIM,) padded

    mapped_params:            Dict[str, Any]
    discrete_action:          str
    discrete_action_one_hot:  np.ndarray              # (MAX_DISCRETE_ACTIONS,)

    # ── Reward / next state / done ───────────────────────────────────────────
    reward:                   float

    next_state_features:      np.ndarray              # (state_dim,)
    next_active_agent:        Optional[str]
    next_active_agent_id:     Optional[int]
    next_valid_actions:       List[str]

    done:                     bool

    # ── Metadata ─────────────────────────────────────────────────────────────
    question_id:              str
    step:                     int
    final_status:             str
    metrics_snapshot:         Dict[str, Any]          = field(default_factory=dict)

    # ── Legacy fields (kept so old code paths don't crash; not used by trainer) ─
    joint_action:             Optional[np.ndarray]    = None
    stage:                    Optional[str]           = None
    selected_agent:           Optional[str]           = None
    action_taken:             Optional[str]           = None

    # ── Serialization (JSON-safe) ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transition":
        def _arr(x, default_shape=None):
            if x is None:
                if default_shape is None:
                    return None
                return np.zeros(default_shape, dtype=np.float32)
            return np.asarray(x, dtype=np.float32)

        return cls(
            state_features          = _arr(d["state_features"]),
            active_agent            = d["active_agent"],
            active_agent_id         = int(d["active_agent_id"]),
            valid_actions           = list(d.get("valid_actions", [])),
            raw_action              = _arr(d["raw_action"]),
            action_padded           = _arr(d["action_padded"]),
            mapped_params           = dict(d.get("mapped_params", {})),
            discrete_action         = d["discrete_action"],
            discrete_action_one_hot = _arr(d["discrete_action_one_hot"]),
            reward                  = float(d["reward"]),
            next_state_features     = _arr(d["next_state_features"]),
            next_active_agent       = d.get("next_active_agent"),
            next_active_agent_id    = (int(d["next_active_agent_id"])
                                       if d.get("next_active_agent_id") is not None else None),
            next_valid_actions      = list(d.get("next_valid_actions", [])),
            done                    = bool(d["done"]),
            question_id             = str(d.get("question_id", "")),
            step                    = int(d.get("step", 0)),
            final_status            = str(d.get("final_status", "")),
            metrics_snapshot        = dict(d.get("metrics_snapshot", {})),
            joint_action            = _arr(d.get("joint_action"), default_shape=None)
                                       if d.get("joint_action") is not None else None,
            stage                   = d.get("stage"),
            selected_agent          = d.get("selected_agent"),
            action_taken            = d.get("action_taken"),
        )


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        k = min(batch_size, len(self._buf))
        return random.sample(list(self._buf), k)

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, min_size: int = 256) -> bool:
        return len(self._buf) >= min_size

    # ── JSONL persistence for offline-replay training ────────────────────────

    def to_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._buf]

    def extend_from_dicts(self, rows: List[Dict[str, Any]]) -> int:
        n = 0
        for row in rows:
            try:
                self._buf.append(Transition.from_dict(row))
                n += 1
            except Exception:
                continue
        return n
