"""
brain/context_marl_ac/marl/checkpointing.py
-------------------------------------------
Checkpointing for Supervisor-Guided Fully Free MARL.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from context_marl_ac.config import CHECKPOINTS_DIR


class MARLCheckpointManager:
    def __init__(self, checkpoints_dir: Path = CHECKPOINTS_DIR):
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def save_checkpoint(
        self,
        actors: nn.ModuleDict,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        episode: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        is_best: bool = False,
        filename: Optional[str] = None,
        supervisor: Optional[nn.Module] = None,
    ):
        if filename is None:
            filename = f"episode_{episode:04d}.pt"

        checkpoint_path = self.checkpoints_dir / filename

        state = {
            "episode": episode,
            "actors_state_dict": actors.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        }

        if supervisor is not None:
            state["supervisor_state_dict"] = supervisor.state_dict()

        torch.save(state, checkpoint_path)

        if is_best:
            best_path = self.checkpoints_dir / "best_reward.pt"
            torch.save(state, best_path)

        config_path = self.checkpoints_dir / "run_config.json"
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

    def load_checkpoint(
        self,
        actors: nn.ModuleDict,
        critic: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: str = "best_reward.pt",
        supervisor: Optional[nn.Module] = None,
        strict_supervisor: bool = False,
    ) -> Dict[str, Any]:
        checkpoint_path = self.checkpoints_dir / filename

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu")

        if "actors_state_dict" in state:
            actors.load_state_dict(state["actors_state_dict"])
        else:
            raise KeyError("Checkpoint missing actors_state_dict")

        if "critic_state_dict" in state:
            critic.load_state_dict(state["critic_state_dict"])
        else:
            raise KeyError("Checkpoint missing critic_state_dict")

        if supervisor is not None:
            if "supervisor_state_dict" in state:
                supervisor.load_state_dict(state["supervisor_state_dict"])
            elif strict_supervisor:
                raise KeyError("Checkpoint missing supervisor_state_dict")
            else:
                print("[checkpointing] No supervisor_state_dict found. Supervisor will start fresh.")

        if optimizer is not None and "optimizer_state_dict" in state:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            except ValueError as exc:
                print(f"[checkpointing] Optimizer state not loaded: {exc}")

        return {
            "episode": state.get("episode", 0),
            "metrics": state.get("metrics", {}),
            "config": state.get("config", {}),
        }