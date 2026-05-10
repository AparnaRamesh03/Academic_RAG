"""
brain/context_marl_ac/marl/marl_env.py
--------------------------------------
The Multi-Agent Reinforcement Learning Environment for Context-Engineered RAG.
"""

import time
from typing import Dict, Any, Tuple, Optional, List

from context_marl_ac.config import MAX_STEPS_PER_EPISODE, MAX_LLM_CALLS_PER_EPISODE
from context_marl_ac.schemas.context_state import ContextState
from context_marl_ac.context_engineering.context_builder import initialize_context

# Import Agents
from context_marl_ac.agents.retriever_agent import RetrieverAgent
from context_marl_ac.agents.rewriter_agent import RewriterAgent
from context_marl_ac.agents.grader_agent import GraderAgent
from context_marl_ac.agents.generator_agent import GeneratorAgent
from context_marl_ac.agents.verifier_agent import VerifierAgent
from context_marl_ac.schemas.observations import get_observation
from context_marl_ac.marl.action_masking import get_action_mask
from context_marl_ac.marl.reward import calculate_reward

class MARLEnv:
    """
    Orchestrates the episode flow for the 5 MARL agents.
    """
    def __init__(self):
        self.agents = {
            "retriever": RetrieverAgent(),
            "rewriter":  RewriterAgent(),
            "grader":    GraderAgent(),
            "generator": GeneratorAgent(),
            "verifier":  VerifierAgent()
        }
        self.state: Optional[ContextState] = None
        self.total_reward: float = 0.0

    def reset(self, question_dict: Dict[str, Any]) -> ContextState:
        """
        Starts a new episode for a given question.
        """
        self.state = initialize_context(question_dict)
        self.total_reward = 0.0
        self.gold_answer = question_dict.get("ground_truth", "")
        self.gold_chunks = question_dict.get("source_file", []) # simplified
        return self.state

    def step(self, agent_name: str, action_name: str) -> Tuple[ContextState, float, bool, Dict[str, Any]]:
        """
        Performs one action by one agent and returns (new_state, reward, done, info).
        """
        if self.state is None:
            raise RuntimeError("Env must be reset() before step()")
        if self.state.done:
            return self.state, 0.0, True, {"msg": "Episode already finished"}

        # 1. Execute agent action
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        # The agent mutates self.state in-place
        self.state = agent.act(self.state, action_name)
        
        # 2. Check for environment-level termination
        if self.state.num_steps >= MAX_STEPS_PER_EPISODE:
            self.state.done = True
            if self.state.final_status == "pending":
                self.state.final_status = "timeout"
                
        if self.state.num_llm_calls >= MAX_LLM_CALLS_PER_EPISODE:
            self.state.done = True
            if self.state.final_status == "pending":
                self.state.final_status = "timeout"

        # 3. Calculate Reward
        reward, reward_components = calculate_reward(
            self.state, 
            action_name, 
            self.state.done,
            gold_answer=self.gold_answer,
            gold_chunks=self.gold_chunks if isinstance(self.gold_chunks, list) else [self.gold_chunks]
        )

        self.total_reward += reward
        
        info = {
            "agent": agent_name,
            "action": action_name,
            "status": self.state.final_status,
            "reward_components": reward_components
        }
        
        return self.state, reward, self.state.done, info

    def get_obs(self, agent_name: str) -> List[float]:
        """Returns the local observation for a specific agent."""
        if not self.state: return []
        return get_observation(agent_name, self.state)

    def get_mask(self, agent_name: str) -> List[int]:
        """Returns the valid action mask for a specific agent."""
        if not self.state: return []
        return get_action_mask(agent_name, self.state)

    def get_global_features(self) -> List[float]:
        """Returns the global 14-dim feature vector for the critic."""
        if not self.state: return []
        from context_marl_ac.context_engineering.feature_encoder import encode_features
        return encode_features(self.state)

    def get_global_reward(self) -> float:
        """
        Returns the total accumulated reward for the current episode.
        """
        return self.total_reward
