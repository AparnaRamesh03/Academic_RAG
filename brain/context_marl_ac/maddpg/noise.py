"""
brain/context_marl_ac/maddpg/noise.py
--------------------------------------
Exploration noise processes for MADDPG continuous action space.
"""
import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()


class GaussianNoise:
    """Simple i.i.d. Gaussian noise for exploration."""

    def __init__(self, size: int, sigma: float = 0.1):
        self.size = size
        self.sigma = sigma

    def reset(self):
        pass

    def sample(self) -> np.ndarray:
        return np.random.randn(self.size) * self.sigma
