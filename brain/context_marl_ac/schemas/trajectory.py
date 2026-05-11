"""
brain/context_marl_ac/schemas/trajectory.py
-------------------------------------------
Dataclasses for recording fully free MARL trajectories.

Each step stores:
1. Supervisor decision: which agent was chosen.
2. Selected agent decision: which action was chosen.
3. Critic value, reward, advantage, and debug state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TrajectoryStep:
    episode_id: str
    step: int

    # Supervisor-level decision
    valid_agents: List[str] = field(default_factory=list)
    agent_mask: List[int] = field(default_factory=list)
    supervisor_selected_agent: str = ""
    supervisor_agent_id: int = -1
    supervisor_action_probability: float = 0.0
    supervisor_log_probability: float = 0.0
    supervisor_entropy: float = 0.0

    # Selected agent/action decision
    agent: str = ""
    observation: List[float] = field(default_factory=list)
    global_features: List[float] = field(default_factory=list)
    obs_names: List[str] = field(default_factory=list)
    valid_actions: List[str] = field(default_factory=list)
    action_mask: List[int] = field(default_factory=list)
    selected_action: str = ""
    action_id: int = -1
    action_probability: float = 0.0
    log_probability: float = 0.0
    entropy: float = 0.0

    # Value/reward
    critic_value: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    done: bool = False
    latency_step: float = 0.0

    # Debug payload
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,

            "valid_agents": self.valid_agents,
            "agent_mask": self.agent_mask,
            "supervisor_selected_agent": self.supervisor_selected_agent,
            "supervisor_agent_id": self.supervisor_agent_id,
            "supervisor_action_probability": round(self.supervisor_action_probability, 6),
            "supervisor_log_probability": round(self.supervisor_log_probability, 6),
            "supervisor_entropy": round(self.supervisor_entropy, 6),

            "agent": self.agent,
            "observation": self.observation,
            "global_features": self.global_features,
            "obs_names": self.obs_names,
            "valid_actions": self.valid_actions,
            "action_mask": self.action_mask,
            "selected_action": self.selected_action,
            "action_id": self.action_id,
            "action_probability": round(self.action_probability, 6),
            "log_probability": round(self.log_probability, 6),
            "entropy": round(self.entropy, 6),

            "critic_value": round(self.critic_value, 6),
            "reward": round(self.reward, 6),
            "advantage": round(self.advantage, 6),
            "done": self.done,
            "latency_step": round(self.latency_step, 4),

            **self.extra,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Episode:
    episode_id: str
    question_id: str
    question: str
    query_type: str = "factual"
    query_complexity: str = "medium"

    steps: List[TrajectoryStep] = field(default_factory=list)

    total_reward: float = 0.0
    answer_quality: float = 0.0
    citation_support_rate: float = 0.0
    verification_pass: bool = False
    final_status: str = "pending"
    generated_answer: str = ""
    num_steps: int = 0
    num_llm_calls: int = 0
    latency_seconds: float = 0.0
    token_usage: int = 0

    has_generated_answer: bool = False
    answer_length: int = 0
    selected_evidence_count: int = 0
    verifier_decision: str = "N/A"

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def compute_advantages(self, gamma: float = 0.99) -> None:
        g = 0.0
        for step in reversed(self.steps):
            g = step.reward + gamma * g
            step.advantage = round(g - step.critic_value, 6)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "question_id": self.question_id,
            "query_type": self.query_type,
            "total_reward": round(self.total_reward, 4),
            "answer_quality": round(self.answer_quality, 4),
            "citation_support_rate": round(self.citation_support_rate, 4),
            "verification_pass": self.verification_pass,
            "final_status": self.final_status,
            "num_steps": self.num_steps,
            "num_llm_calls": self.num_llm_calls,
            "latency_seconds": round(self.latency_seconds, 4),
            "token_usage": self.token_usage,
            "has_gen": self.has_generated_answer,
            "ans_len": self.answer_length,
            "ev_count": self.selected_evidence_count,
            "ver_dec": self.verifier_decision,
        }

    def steps_to_jsonl(self) -> str:
        return "\n".join(step.to_json() for step in self.steps)