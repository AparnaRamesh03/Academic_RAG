from typing import Tuple

import torch
import torch.nn as nn


class ControllerPolicyNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 5,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)