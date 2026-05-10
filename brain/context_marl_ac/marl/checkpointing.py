"""
brain/context_marl_ac/marl/checkpointing.py
-------------------------------------------
Logic for saving and loading MARL models and training state.
"""

import json
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

from context_marl_ac.config import CHECKPOINTS_DIR

class MARLCheckpointManager:
    """
    Handles saving and loading of actor/critic networks and optimizer state.
    """
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
        filename: Optional[str] = None
    ):
        """
        Saves a checkpoint to disk.
        """
        if filename is None:
            filename = f"episode_{episode:04d}.pt"
            
        checkpoint_path = self.checkpoints_dir / filename
        
        state = {
            "episode": episode,
            "actors_state_dict": actors.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config
        }
        
        torch.save(state, checkpoint_path)
        
        # If best, also save as best_reward.pt
        if is_best:
            best_path = self.checkpoints_dir / "best_reward.pt"
            torch.save(state, best_path)
            
        # Save run config separately for easy inspection
        config_path = self.checkpoints_dir / "run_config.json"
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

    def load_checkpoint(
        self,
        actors: nn.ModuleDict,
        critic: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: str = "best_reward.pt"
    ) -> Dict[str, Any]:
        """
        Loads a checkpoint from disk.
        """
        checkpoint_path = self.checkpoints_dir / filename
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        state = torch.load(checkpoint_path, map_location="cpu")
        
        actors.load_state_dict(state["actors_state_dict"])
        critic.load_state_dict(state["critic_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            
        return {
            "episode": state.get("episode", 0),
            "metrics": state.get("metrics", {}),
            "config":  state.get("config", {})
        }
