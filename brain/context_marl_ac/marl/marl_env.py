"""
brain/context_marl_ac/marl/marl_env.py
--------------------------------------
Multi-Agent RL environment for Supervisor-Guided Fully Free MARL RAG.
"""

from typing import Dict, Any, Tuple, Optional, List

from context_marl_ac.config import MAX_STEPS_PER_EPISODE, MAX_LLM_CALLS_PER_EPISODE
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.context_engineering.context_builder import initialize_context

from context_marl_ac.agents.retriever_agent import RetrieverAgent
from context_marl_ac.agents.rewriter_agent import RewriterAgent
from context_marl_ac.agents.grader_agent import GraderAgent
from context_marl_ac.agents.generator_agent import GeneratorAgent
from context_marl_ac.agents.verifier_agent import VerifierAgent

from context_marl_ac.schemas.observations import get_observation
from context_marl_ac.schemas.actions import AGENT_NAMES
from context_marl_ac.marl.action_masking import (
    get_action_mask,
    get_valid_actions,
    get_valid_agents,
    get_agent_mask,
)
from context_marl_ac.marl.reward import calculate_reward


class MARLEnv:
    """
    Fully free MARL environment.

    Supervisor chooses a valid agent.
    Selected agent chooses a valid action.
    Environment executes the action and calculates shared reward.
    """

    def __init__(self):
        self.agents = {
            "retriever": RetrieverAgent(),
            "rewriter": RewriterAgent(),
            "grader": GraderAgent(),
            "generator": GeneratorAgent(),
            "verifier": VerifierAgent(),
        }
        self.state: Optional[ContextState] = None
        self.total_reward: float = 0.0
        self.gold_answer: str = ""
        self.gold_chunks: List[str] = []

    def reset(self, question_dict: Dict[str, Any], index: int = 1) -> ContextState:
        self.state = initialize_context(question_dict, index=index)
        self.total_reward = 0.0
        self.gold_answer = question_dict.get("ground_truth", "")

        gold_chunks = question_dict.get("source_file", [])
        if isinstance(gold_chunks, str):
            gold_chunks = [gold_chunks]
        self.gold_chunks = gold_chunks

        return self.state

    def step(self, agent_name: str, action_name: str) -> Tuple[ContextState, float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Env must be reset() before step()")

        if self.state.done:
            return self.state, 0.0, True, {"msg": "Episode already finished"}

        valid_agents = get_valid_agents(self.state)
        if agent_name not in valid_agents:
            self.state.record_action(agent_name, f"INVALID_AGENT::{action_name}")
            self.state.final_status = "error"
            self.state.done = True
            reward, reward_components = calculate_reward(
                self.state,
                action_name,
                True,
                gold_answer=self.gold_answer,
                gold_chunks=self.gold_chunks,
            )
            self.total_reward += reward
            return self.state, reward, True, {
                "agent": agent_name,
                "action": action_name,
                "status": self.state.final_status,
                "invalid": True,
                "reward_components": reward_components,
            }

        valid_actions = get_valid_actions(agent_name, self.state)
        if action_name not in valid_actions:
            self.state.record_action(agent_name, f"INVALID_ACTION::{action_name}")
            self.state.final_status = "error"
            self.state.done = True
            reward, reward_components = calculate_reward(
                self.state,
                action_name,
                True,
                gold_answer=self.gold_answer,
                gold_chunks=self.gold_chunks,
            )
            self.total_reward += reward
            return self.state, reward, True, {
                "agent": agent_name,
                "action": action_name,
                "status": self.state.final_status,
                "invalid": True,
                "reward_components": reward_components,
            }

        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_name}")

        self.state = agent.act(self.state, action_name)

        if self.state.num_steps >= MAX_STEPS_PER_EPISODE:
            self.state.done = True
            if self.state.final_status == "pending":
                self.state.final_status = "timeout"

        if self.state.num_llm_calls >= MAX_LLM_CALLS_PER_EPISODE:
            self.state.done = True
            if self.state.final_status == "pending":
                self.state.final_status = "timeout"

        reward, reward_components = calculate_reward(
            self.state,
            action_name,
            self.state.done,
            gold_answer=self.gold_answer,
            gold_chunks=self.gold_chunks,
        )

        self.total_reward += reward

        info = {
            "agent": agent_name,
            "action": action_name,
            "status": self.state.final_status,
            "reward_components": reward_components,
        }

        return self.state, reward, self.state.done, info

    def get_obs(self, agent_name: str) -> List[float]:
        if not self.state:
            return []
        return get_observation(agent_name, self.state)

    def get_mask(self, agent_name: str) -> List[int]:
        if not self.state:
            return []
        return get_action_mask(agent_name, self.state)

    def get_valid_actions(self, agent_name: str) -> List[str]:
        if not self.state:
            return []
        return get_valid_actions(agent_name, self.state)

    def get_valid_agents(self) -> List[str]:
        if not self.state:
            return []
        return get_valid_agents(self.state)

    def get_agent_mask(self) -> List[int]:
        if not self.state:
            return [0 for _ in AGENT_NAMES]
        return get_agent_mask(self.state)

    def get_global_features(self) -> List[float]:
        if not self.state:
            return []
        from context_marl_ac.context_engineering.feature_encoder import encode_features
        return encode_features(self.state)

    def get_global_reward(self) -> float:
        return self.total_reward