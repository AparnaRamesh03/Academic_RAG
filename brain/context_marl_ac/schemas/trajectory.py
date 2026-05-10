"""
brain/context_marl_ac/schemas/trajectory.py
---------------------------------------------
Dataclasses for recording MARL episode trajectories.

A trajectory is the complete record of one episode:
  - metadata (episode id, question, query type, etc.)
  - a sequence of TrajectoryStep objects — one per (agent, action) pair

Trajectories are serialised to JSONL in:
    results/trajectories/train_trajectories.jsonl

Each line is one TrajectoryStep serialised to a flat dict.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# TrajectoryStep — one (agent, action) transition inside an episode
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryStep:
    """
    Records everything needed to reconstruct and train from one step.

    Fields
    ------
    episode_id          : unique episode identifier (uuid4 string)
    step                : 0-indexed step counter within the episode
    agent               : agent name ("retriever" | "rewriter" | ...)
    observation         : local observation vector (List[float])
    obs_names           : feature names for the observation vector
    valid_actions       : list of valid action names at this step
    action_mask         : binary mask aligned to agent action list
    selected_action     : chosen action name (str)
    action_id           : integer id of selected_action
    action_probability  : softmax probability of selected action
    log_probability     : log of action_probability
    entropy             : policy entropy at this step
    critic_value        : V(s) estimated by centralized critic
    reward              : step + terminal reward
    advantage           : computed advantage A_t (filled after episode)
    done                : True if this was the final step
    latency_step        : wall-clock seconds for this step
    extra               : optional dict for any additional debug info
    """
    episode_id:         str
    step:               int
    agent:              str
    observation:        List[float]
    global_features:    List[float]
    obs_names:          List[str]
    valid_actions:      List[str]
    action_mask:        List[int]
    selected_action:    str
    action_id:          int
    action_probability: float
    log_probability:    float
    entropy:            float
    critic_value:       float
    reward:             float
    advantage:          float = 0.0   # filled in by trainer after episode
    done:               bool  = False
    latency_step:       float = 0.0
    extra:              Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flat JSON-serializable dict for JSONL logging."""
        return {
            "episode_id":         self.episode_id,
            "step":               self.step,
            "agent":              self.agent,
            "observation":        self.observation,
            "global_features":    self.global_features,
            "obs_names":          self.obs_names,
            "valid_actions":      self.valid_actions,
            "action_mask":        self.action_mask,
            "selected_action":    self.selected_action,
            "action_id":          self.action_id,
            "action_probability": round(self.action_probability, 6),
            "log_probability":    round(self.log_probability, 6),
            "entropy":            round(self.entropy, 6),
            "critic_value":       round(self.critic_value, 6),
            "reward":             round(self.reward, 6),
            "advantage":          round(self.advantage, 6),
            "done":               self.done,
            "latency_step":       round(self.latency_step, 4),
            **self.extra,
        }

    def to_json(self) -> str:
        """Serialise to a single JSONL line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Episode — the full trajectory for one benchmark question
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    """
    Complete record of one training or evaluation episode.

    Holds:
    - episode-level metadata (id, question, query type, etc.)
    - a list of TrajectoryStep objects in order
    - episode-level outcome metrics (filled at end of episode)
    """
    episode_id:     str
    question_id:    str
    question:       str
    query_type:     str         = "factual"
    query_complexity: str       = "medium"

    # Steps accumulate as the episode runs
    steps: List[TrajectoryStep] = field(default_factory=list)

    # Episode outcome (filled at end)
    total_reward:          float = 0.0
    answer_quality:        float = 0.0
    citation_support_rate: float = 0.0
    verification_pass:     bool  = False
    final_status:          str   = "pending"
    generated_answer:      str   = ""
    num_steps:             int   = 0
    num_llm_calls:         int   = 0
    latency_seconds:       float = 0.0
    token_usage:           int   = 0

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def compute_advantages(self, gamma: float = 0.99) -> None:
        """
        Back-fill advantage estimates for all steps using Monte Carlo returns.
        Called by the trainer after the full episode reward sequence is known.

        A_t = G_t - V(s_t)
        G_t = R_t + γ * R_{t+1} + γ² * R_{t+2} + ...
        """
        g = 0.0
        for step in reversed(self.steps):
            g = step.reward + gamma * g
            step.advantage = round(g - step.critic_value, 6)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Episode-level summary for episode_metrics.csv."""
        return {
            "episode_id":          self.episode_id,
            "question_id":         self.question_id,
            "query_type":          self.query_type,
            "total_reward":        round(self.total_reward, 4),
            "answer_quality":      round(self.answer_quality, 4),
            "citation_support_rate": round(self.citation_support_rate, 4),
            "verification_pass":   self.verification_pass,
            "final_status":        self.final_status,
            "num_steps":           self.num_steps,
            "num_llm_calls":       self.num_llm_calls,
            "latency_seconds":     round(self.latency_seconds, 4),
            "token_usage":         self.token_usage,
        }

    def steps_to_jsonl(self) -> str:
        """Return all steps as newline-separated JSON lines."""
        return "\n".join(s.to_json() for s in self.steps)
