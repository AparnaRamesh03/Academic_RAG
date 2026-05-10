"""
brain/context_marl_ac/marl/replay_buffer.py
-------------------------------------------
Lightweight storage for episode trajectories.
"""

from typing import List, Dict, Any
from context_marl_ac.schemas.trajectory import Episode

class ReplayBuffer:
    """
    Stores completed episodes for training.
    """
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: List[Episode] = []

    def add_episode(self, episode: Episode):
        """
        Add a completed episode to the buffer.
        """
        self.buffer.append(episode)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def get_all(self) -> List[Episode]:
        """
        Return all episodes and clear the buffer.
        """
        episodes = list(self.buffer)
        self.buffer = []
        return episodes

    def __len__(self):
        return len(self.buffer)
