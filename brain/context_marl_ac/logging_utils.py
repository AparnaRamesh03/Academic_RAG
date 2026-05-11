"""
brain/context_marl_ac/logging_utils.py
--------------------------------------
Logging utilities for MARL training and evaluation.

Updated for Supervisor-Guided Fully Free MARL:
- logs supervisor_loss
- logs agent_actor_loss
- keeps old fields for compatibility
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Any

from context_marl_ac.config import METRICS_DIR, TRAJECTORIES_DIR
from context_marl_ac.schemas.trajectory import Episode


class MARLLogger:
    """
    Handles CSV and JSONL logging for the MARL system.
    """

    EPISODE_FIELDS = [
        "episode_id",
        "question_id",
        "query_type",
        "total_reward",
        "answer_quality",
        "citation_support_rate",
        "verification_pass",
        "final_status",
        "num_steps",
        "num_llm_calls",
        "latency_seconds",
        "token_usage",
        "has_gen",
        "ans_len",
        "ev_count",
        "ver_dec",
    ]

    TRAINING_FIELDS = [
        "epoch",
        "mean_reward",
        "mean_steps",
        "mean_llm_calls",
        "loss",
        "actor_loss",
        "supervisor_loss",
        "agent_actor_loss",
        "critic_loss",
        "entropy",
        "entropy_loss",
    ]

    def __init__(self, run_name: str = "latest"):
        self.run_name = run_name
        self.metrics_dir = METRICS_DIR
        self.trajectories_dir = TRAJECTORIES_DIR

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.trajectories_dir, exist_ok=True)

        self.episode_csv = self.metrics_dir / f"episode_metrics_{run_name}.csv"
        self.training_csv = self.metrics_dir / f"training_metrics_{run_name}.csv"
        self.trajectory_jsonl = self.trajectories_dir / f"train_trajectories_{run_name}.jsonl"

        # Fresh run: delete existing files for specific runs.
        if run_name != "latest":
            for path in [self.episode_csv, self.training_csv, self.trajectory_jsonl]:
                if os.path.exists(path):
                    os.remove(path)

        self._init_csvs()

    def _init_csvs(self) -> None:
        """
        Initialize CSV headers if files do not exist.
        """
        if not os.path.exists(self.episode_csv):
            with open(self.episode_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.EPISODE_FIELDS)
                writer.writeheader()

        if not os.path.exists(self.training_csv):
            with open(self.training_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.TRAINING_FIELDS)
                writer.writeheader()

    def log_episode(self, episode: Episode) -> None:
        """
        Log episode summary to CSV and full trajectory to JSONL.
        """
        summary = episode.to_summary_dict()

        row = {
            field: summary.get(field, "")
            for field in self.EPISODE_FIELDS
        }

        with open(self.episode_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.EPISODE_FIELDS)
            writer.writerow(row)

        with open(self.trajectory_jsonl, "a") as f:
            text = episode.steps_to_jsonl()
            if text.strip():
                f.write(text + "\n")

    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log aggregate training metrics to CSV.
        """
        row = {
            "epoch": epoch,
            "mean_reward": metrics.get("mean_reward", 0.0),
            "mean_steps": metrics.get("mean_steps", 0.0),
            "mean_llm_calls": metrics.get("mean_llm_calls", 0.0),
            "loss": metrics.get("loss", 0.0),
            "actor_loss": metrics.get("actor_loss", 0.0),
            "supervisor_loss": metrics.get("supervisor_loss", 0.0),
            "agent_actor_loss": metrics.get("agent_actor_loss", 0.0),
            "critic_loss": metrics.get("critic_loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "entropy_loss": metrics.get("entropy_loss", 0.0),
        }

        with open(self.training_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.TRAINING_FIELDS)
            writer.writerow(row)

    def log_eval_result(self, result_dict: Dict[str, Any], output_file: Path) -> None:
        """
        Append one evaluation result row to a JSONL file.
        """
        os.makedirs(output_file.parent, exist_ok=True)

        with open(output_file, "a") as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")