"""
brain/context_marl_ac/agents/base_agent.py
-----------------------------------------
Base class for all MARL agents.
"""

from abc import ABC, abstractmethod
from typing import Any, List
from context_marl_ac.schemas.context_state import ContextState

class BaseAgent(ABC):
    """
    Common interface for all agents in the MARL system.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def act(self, state: ContextState, action_name: str) -> ContextState:
        """
        Perform an action and mutate the state.
        """
        pass
        
    def log_action(self, state: ContextState, action_name: str):
        """
        Standard logging for action execution.
        """
        state.record_action(self.name, action_name)
        state.num_llm_calls += 1 # Default assumption, overridden if no LLM call
        state.update_latency()
