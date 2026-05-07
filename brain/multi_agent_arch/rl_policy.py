"""
rl_policy.py
------------
Contextual bandit policy for supervisor routing in the multi-agent RAG system.

Components
----------
EpisodeBuffer
    Accumulates (state_key_str, action) transitions during a single query
    episode so they can receive a retrospective reward update at episode end.

RLPolicy
    Q-table keyed on discrete state tuples.  Action selection is ε-greedy:
    with probability ε the caller should fall back to rule-based routing
    (returning None signals this); otherwise the highest-Q valid action is
    returned.

    Learning uses a Monte Carlo / TD(0) update:
        Q[s, a] ← Q[s, a] + α × (reward − Q[s, a])

    The Q-table is persisted to disk after every episode update via an atomic
    write (tmp → rename) so the policy survives server restarts and improves
    continuously over time.
"""
from __future__ import annotations

import json
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import decay constants so the policy stays in sync with config
try:
    from config import RL_EPSILON_DECAY as _DEFAULT_DECAY, RL_EPSILON_MIN as _DEFAULT_MIN
except ImportError:
    _DEFAULT_DECAY = 0.995
    _DEFAULT_MIN   = 0.05

# ── Valid actions the bandit may choose from ──────────────────────────────────

VALID_ACTIONS: List[str] = [
    "retriever_agent",
    "rewrite_agent",
    "evidence_agent",
    "answer_agent",
    "verification_agent",
    "finish",
]


# ── Episode buffer ────────────────────────────────────────────────────────────


class EpisodeBuffer:
    """
    Records (state_key_str, action) transitions for one episode.

    Rather than holding a module-level buffer (which would be unsafe for
    concurrent requests), transitions are serialised into GraphState as a list
    of [state_key_str, action] pairs and re-hydrated here for the update step.
    """

    def __init__(self) -> None:
        self.transitions: List[Tuple[str, str]] = []

    def record(self, state_key: tuple, action: str) -> None:
        self.transitions.append((str(state_key), action))

    def clear(self) -> None:
        self.transitions.clear()

    @classmethod
    def from_state_list(cls, raw: List) -> "EpisodeBuffer":
        """Re-hydrate a buffer from the list stored in GraphState."""
        buf = cls()
        for item in raw or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                buf.transitions.append((str(item[0]), str(item[1])))
        return buf


# ── Policy ────────────────────────────────────────────────────────────────────


class RLPolicy:
    """
    Contextual bandit with a persistent Q-table for supervisor routing.

    Parameters
    ----------
    policy_path : Path
        File path for JSON persistence.
    epsilon : float
        Exploration rate.  When a random draw is below epsilon, `act()` returns
        None so the caller falls back to rule-based routing.
    alpha : float
        TD learning rate.
    valid_actions : list[str] | None
        Allowable action labels.  Defaults to VALID_ACTIONS.
    """

    def __init__(
        self,
        policy_path: Path,
        epsilon: float = 0.15,
        alpha: float = 0.05,
        valid_actions: Optional[List[str]] = None,
        epsilon_decay: float = _DEFAULT_DECAY,
        epsilon_min: float = _DEFAULT_MIN,
    ) -> None:
        self.policy_path = policy_path
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.valid_actions = valid_actions or VALID_ACTIONS
        self._lock = threading.Lock()

        # Q-table: {state_key_str: {action: float}}
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.episode_count: int = 0

        self._load()

    # ── Action selection ──────────────────────────────────────────────────────

    def act(
        self,
        state_key: tuple,
        valid_actions: Optional[List[str]] = None,
        explore: bool = True,
    ) -> Optional[str]:
        """
        Return the RL-chosen action, or None to signal rule-based fallback.

        Returns None when:
        - explore=True and random draw < epsilon  (intentional exploration)
        - The state has never been seen yet        (no data to exploit)
        - No known Q-value among the valid actions (no learned preference)
        """
        actions = valid_actions or self.valid_actions

        # ε-greedy exploration → defer to caller's rule-based logic.
        if explore and random.random() < self.epsilon:
            return None

        key = str(state_key)
        with self._lock:
            q_values = self.q_table.get(key, {})

        if not q_values:
            return None  # Unseen state — no preference yet.

        # Only consider valid actions that the Q-table has an entry for.
        valid_q = {a: q_values[a] for a in actions if a in q_values}
        if not valid_q:
            return None

        return max(valid_q, key=valid_q.__getitem__)

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(self, buffer: EpisodeBuffer, reward: float) -> None:
        """
        Monte Carlo update: propagate terminal reward to every transition.
        Decays epsilon by epsilon_decay (clamped at epsilon_min).
        Saves the updated Q-table to disk atomically.
        """
        if not buffer.transitions:
            return

        with self._lock:
            for state_key_str, action in buffer.transitions:
                old_q = self.q_table[state_key_str].get(action, 0.0)
                self.q_table[state_key_str][action] = (
                    old_q + self.alpha * (reward - old_q)
                )
            self.episode_count += 1
            # Decay exploration rate after every episode.
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

        self._save()
        print(
            f"[RL] Policy updated: episode={self.episode_count}, "
            f"reward={reward:.4f}, transitions={len(buffer.transitions)}, "
            f"known_states={len(self.q_table)}, epsilon={self.epsilon:.4f}"
        )

    def update_from_state_transitions(
        self, raw_transitions: List, reward: float
    ) -> None:
        """
        Convenience wrapper: re-hydrate a buffer from the raw list stored in
        GraphState and call update().
        """
        buf = EpisodeBuffer.from_state_list(raw_transitions)
        self.update(buf, reward)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.policy_path.exists():
            print(f"[RL] No existing policy at {self.policy_path}. Starting fresh.")
            return
        try:
            with self.policy_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state_key, actions in data.get("q_table", {}).items():
                self.q_table[state_key] = defaultdict(float, actions)
            self.episode_count = int(data.get("episode_count", 0))
            # Restore decayed epsilon if persisted; fall back to current value.
            if "epsilon" in data:
                self.epsilon = float(data["epsilon"])
            print(
                f"[RL] Loaded policy from {self.policy_path} "
                f"({self.episode_count} episodes, "
                f"{len(self.q_table)} known states, "
                f"epsilon={self.epsilon:.4f})"
            )
        except Exception as exc:
            print(f"[RL] Warning: could not load policy — {exc}. Starting fresh.")

    def _save(self) -> None:
        try:
            self.policy_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
                "q_table": {k: dict(v) for k, v in self.q_table.items()},
            }
            tmp = self.policy_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self.policy_path)
        except Exception as exc:
            print(f"[RL] Warning: could not save policy — {exc}")

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return a human-readable summary of the current policy."""
        with self._lock:
            n_states = len(self.q_table)
            top_actions: Dict[str, str] = {}
            for state_key, actions in self.q_table.items():
                if actions:
                    top_actions[state_key] = max(actions, key=actions.__getitem__)

        return {
            "episode_count": self.episode_count,
            "known_states": n_states,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "alpha": self.alpha,
            "preferred_action_per_state": top_actions,
        }
