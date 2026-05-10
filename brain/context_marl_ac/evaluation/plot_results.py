"""
brain/context_marl_ac/evaluation/plot_results.py
------------------------------------------------
Generates training progress plots from logged CSV metrics.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── sys.path setup ────────────────────────────────────────────────────────────
_MARL_ROOT = Path(__file__).resolve().parents[1]
_BRAIN_ROOT = _MARL_ROOT.parent

# ── Imports ───────────────────────────────────────────────────────────────────
from context_marl_ac.config import METRICS_DIR, PLOTS_DIR

def plot_training_progress(run_name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Load Data
    episode_csv = METRICS_DIR / f"episode_metrics_{run_name}.csv"
    training_csv = METRICS_DIR / f"training_metrics_{run_name}.csv"
    
    if not os.path.exists(episode_csv) or not os.path.exists(training_csv):
        print(f"Error: CSV metrics for {run_name} not found in {METRICS_DIR}")
        return

    ep_df = pd.read_csv(episode_csv)
    tr_df = pd.read_csv(training_csv)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # ──────── Plot 1: Mean Reward vs Episode ─────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(tr_df["epoch"], tr_df["mean_reward"], label="Mean Reward", color="blue", linewidth=2)
    plt.fill_between(tr_df["epoch"], tr_df["mean_reward"] - 0.1, tr_df["mean_reward"] + 0.1, alpha=0.2)
    plt.title(f"MARL Training Progress: Mean Reward ({run_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(PLOTS_DIR / f"reward_curve_{run_name}.png")
    plt.close()

    # ──────── Plot 2: Loss Curves ────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(tr_df["epoch"], tr_df["loss"], label="Total Loss", color="black", linestyle="--")
    plt.plot(tr_df["epoch"], tr_df["actor_loss"], label="Actor Loss", color="red")
    plt.plot(tr_df["epoch"], tr_df["critic_loss"], label="Critic Loss", color="green")
    plt.title(f"Training Loss: {run_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(PLOTS_DIR / f"loss_curves_{run_name}.png")
    plt.close()

    # ──────── Plot 3: Efficiency (Steps & LLM Calls) ──────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(tr_df["epoch"], tr_df["mean_steps"], label="Avg Steps", color="purple")
    plt.plot(tr_df["epoch"], tr_df["mean_llm_calls"], label="Avg LLM Calls", color="orange")
    plt.title(f"Efficiency Metrics: {run_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(PLOTS_DIR / f"efficiency_metrics_{run_name}.png")
    plt.close()

    # ──────── Plot 4: Final Status Distribution ──────────────────────────────
    plt.figure(figsize=(8, 8))
    status_counts = ep_df["final_status"].value_counts()
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=sns.color_palette("viridis"))
    plt.title(f"Episode Outcome Distribution: {run_name}")
    plt.savefig(PLOTS_DIR / f"status_distribution_{run_name}.png")
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    plot_training_progress(args.run_name)
