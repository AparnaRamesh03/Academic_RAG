"""
brain/context_marl_ac/logging_utils.py
--------------------------------------
Logging utilities for MARL training and evaluation.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from context_marl_ac.config import (
    METRICS_DIR, TRAJECTORIES_DIR, RESULTS_DIR
)
from context_marl_ac.schemas.trajectory import Episode, TrajectoryStep

class MARLLogger:
    """
    Handles CSV and JSONL logging for the MARL system.
    """
    def __init__(self, run_name: str = "latest"):
        self.run_name = run_name
        self.metrics_dir = METRICS_DIR
        self.trajectories_dir = TRAJECTORIES_DIR
        
        # Ensure directories exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.trajectories_dir, exist_ok=True)
        
        # File paths
        self.episode_csv = self.metrics_dir / f"episode_metrics_{run_name}.csv"
        self.training_csv = self.metrics_dir / f"training_metrics_{run_name}.csv"
        self.trajectory_jsonl = self.trajectories_dir / f"train_trajectories_{run_name}.jsonl"
        
        self._init_csvs()

    def _init_csvs(self):
        """Initialize CSV headers if files don't exist."""
        if not os.path.exists(self.episode_csv):
            with open(self.episode_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode_id", "question_id", "query_type", "total_reward",
                    "answer_quality", "citation_support_rate", "verification_pass",
                    "final_status", "num_steps", "num_llm_calls", "latency_seconds"
                ])
                
        if not os.path.exists(self.training_csv):
            with open(self.training_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "mean_reward", "mean_steps", "mean_llm_calls",
                    "loss", "actor_loss", "critic_loss", "entropy"
                ])

    def log_episode(self, episode: Episode):
        """Log episode summary to CSV and full trajectory to JSONL."""
        # 1. CSV Summary
        summary = episode.to_summary_dict()
        with open(self.episode_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writerow(summary)
            
        # 2. JSONL Trajectory
        with open(self.trajectory_jsonl, "a") as f:
            f.write(episode.steps_to_jsonl() + "\n")

    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log aggregate training metrics to CSV."""
        row = {
            "epoch": epoch,
            "mean_reward":    metrics.get("mean_reward", 0.0),
            "mean_steps":     metrics.get("mean_steps", 0.0),
            "mean_llm_calls": metrics.get("mean_llm_calls", 0.0),
            "loss":           metrics.get("loss", 0.0),
            "actor_loss":     metrics.get("actor_loss", 0.0),
            "critic_loss":    metrics.get("critic_loss", 0.0),
            "entropy":        metrics.get("entropy", 0.0)
        }
        with open(self.training_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    def log_eval_result(self, result_dict: Dict[str, Any], output_file: Path):
        """Append one evaluation result row to a JSONL file."""
        os.makedirs(output_file.parent, exist_ok=True)
        with open(output_file, "a") as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
