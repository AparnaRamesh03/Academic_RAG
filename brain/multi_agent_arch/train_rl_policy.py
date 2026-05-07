"""
train_rl_policy.py
------------------
Offline RL warm-up script for the multi_agent_arch contextual bandit policy.

Runs queries DIRECTLY through the compiled LangGraph (no FastAPI server needed).
Only Qdrant must be running.

Usage
-----
    # Basic: 3 rounds over the built-in stress-test set
    python train_rl_policy.py

    # More rounds
    python train_rl_policy.py --rounds 10

    # Use the full benchmark dataset
    python train_rl_policy.py --dataset ../../evaluation/gold_standard_benchmark_v2_1.json --rounds 3

    # Dry-run (RL disabled, just see what the graph does)
    python train_rl_policy.py --no-rl

    # Print the policy stats and exit (no training)
    python train_rl_policy.py --stats

    # Reset the policy and start fresh
    python train_rl_policy.py --reset

How training works
------------------
Each query is a full episode:
  1. Graph runs: supervisor → agents → finish
  2. _rl_finish_node computes the terminal reward from the episode outcome
  3. Monte Carlo update propagates the reward to every (state_key, action)
     transition recorded in rl_transitions during that episode
  4. Updated Q-table is saved atomically to results/rl_policy.json

After each round the script prints per-round stats (mean reward, citations_pass
rate, mean step_count, reward distribution) so you can track convergence.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── Path setup ────────────────────────────────────────────────────────────────

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent
ROOT_DIR = BRAIN_DIR.parent

for p in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if p not in sys.path:
        sys.path.append(p)

# Load environment variables (.env in brain/)
try:
    from dotenv import load_dotenv
    env_path = BRAIN_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[ENV] Loaded: {env_path}")
    else:
        print(f"[ENV] Warning: no .env found at {env_path}")
except ImportError:
    print("[ENV] python-dotenv not installed — relying on system environment variables.")

# ── Imports (after env is loaded so API keys are set) ─────────────────────────

import os
os.environ["RL_TRAINING_MODE"] = "1"

from config import RL_ENABLED, RL_POLICY_PATH
from graph import build_graph
from supervisor import _policy
from rl_reward import compute_episode_reward

# ── Built-in training questions ───────────────────────────────────────────────
# These cover all major query types the system handles.

BUILTIN_QUESTIONS: List[Dict[str, str]] = [
    # direct_fact
    {"query": "What is Naive RAG according to the survey?",
     "category": "direct_fact"},
    {"query": "What does BERT stand for?",
     "category": "direct_fact"},
    {"query": "What BLEU score did the Transformer achieve on the WMT 2014 English-to-German translation task?",
     "category": "direct_fact"},
    {"query": "What core mechanism does TabNet use to choose which features to reason from at each decision step?",
     "category": "direct_fact"},
    {"query": "How deep is the deepest residual network reported in the ResNet paper?",
     "category": "direct_fact"},
    # definition_explanation
    {"query": "What is the degradation problem described in the ResNet paper?",
     "category": "definition_explanation"},
    {"query": "What is self-attention in the Transformer paper?",
     "category": "definition_explanation"},
    {"query": "Why does the RAG survey claim retrieval helps reduce hallucination in LLMs?",
     "category": "definition_explanation"},
    {"query": "Which two pre-training tasks are used in BERT?",
     "category": "definition_explanation"},
    # intra_paper_comparison
    {"query": "How does Advanced RAG differ from Naive RAG according to the survey?",
     "category": "intra_paper_comparison"},
    {"query": "How does the Transformer differ from recurrent and convolutional sequence models?",
     "category": "intra_paper_comparison"},
    {"query": "How does BERT compare feature-based transfer approaches with fine-tuning approaches?",
     "category": "intra_paper_comparison"},
    {"query": "How do plain networks differ from residual networks in the ResNet paper?",
     "category": "intra_paper_comparison"},
    # cross_paper_comparison (harder)
    {"query": "How do the Transformer and ResNet papers each argue that architectural design can overcome training bottlenecks in deep learning?",
     "category": "cross_paper_comparison"},
    {"query": "Compare what BERT and the Transformer paper each claim about architectural improvements over earlier sequence models.",
     "category": "cross_paper_comparison"},
    # figure_grounded
    {"query": "What does Figure 3 show about the differences between Naive RAG, Advanced RAG, and Modular RAG?",
     "category": "figure_grounded"},
    {"query": "According to Figure 1 in the Transformer paper, what are the major components of the encoder and decoder?",
     "category": "figure_grounded"},
    # underspecified / adversarial
    {"query": "Which architecture solves the efficiency problem best and why?",
     "category": "adversarial_superlative"},
    {"query": "How do the papers describe solving the problem where deeper networks become harder to optimize as depth increases?",
     "category": "paraphrase_hard"},
]


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(graph, query: str) -> Dict[str, Any]:
    """Run one full query through the compiled graph and return the final state."""
    initial_state = {
        "original_query": query,
        "search_query": query,
        "step_count": 0,
        "action_history": [],
        "crag_retries": 0,
        "verify_retries": 0,
        "citations_pass": False,
        "done": False,
        "latency_so_far": 0.0,
        "rl_transitions": [],
    }
    result = graph.invoke(initial_state)
    return result


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _round_summary(episode_results: List[Dict]) -> Dict:
    rewards = [r["reward"] for r in episode_results]
    steps   = [r["step_count"] for r in episode_results]
    passes  = [r["citations_pass"] for r in episode_results]

    def _bucket(r):
        if r >= 0.8:  return "great (≥0.8)"
        if r >= 0.5:  return "ok (0.5–0.8)"
        if r >= 0.0:  return "weak (0–0.5)"
        return "bad (<0)"

    buckets: Dict[str, int] = {}
    for r in rewards:
        b = _bucket(r)
        buckets[b] = buckets.get(b, 0) + 1

    return {
        "episodes": len(episode_results),
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "min_reward":  round(min(rewards), 4) if rewards else 0.0,
        "max_reward":  round(max(rewards), 4) if rewards else 0.0,
        "citations_pass_rate": round(sum(passes) / len(passes), 3) if passes else 0.0,
        "mean_steps":  round(sum(steps)   / len(steps),  2) if steps else 0.0,
        "reward_distribution": buckets,
    }


def _print_summary(label: str, summary: Dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Episodes        : {summary['episodes']}")
    print(f"  Mean reward     : {summary['mean_reward']}")
    print(f"  Reward range    : [{summary['min_reward']}, {summary['max_reward']}]")
    print(f"  citations_pass  : {summary['citations_pass_rate']*100:.1f}%")
    print(f"  Mean steps      : {summary['mean_steps']}")
    print(f"  Distribution    : {summary['reward_distribution']}")
    if _policy:
        print(f"  Known RL states : {len(_policy.q_table)}")
        print(f"  Total episodes  : {_policy.episode_count}")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the multi_agent_arch RL bandit policy offline."
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of training rounds (each round = one pass over all questions). Default: 3",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to a JSON benchmark file (list of {question, category, ...}). "
             "If omitted, uses the built-in stress-test question set.",
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True,
        help="Shuffle questions each round (default: True).",
    )
    parser.add_argument(
        "--no-shuffle", dest="shuffle", action="store_false",
        help="Disable shuffling.",
    )
    parser.add_argument(
        "--no-rl", action="store_true",
        help="Disable RL updates (dry-run — useful to baseline before training).",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print current policy stats and exit without training.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the saved policy file and start fresh.",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between episodes (rate-limit Groq). Default: 1.0",
    )
    args = parser.parse_args()

    # ── --stats ───────────────────────────────────────────────────────────────
    if args.stats:
        if _policy is None:
            print("[RL] RL is disabled (RL_ENABLED=False in config.py).")
        else:
            stats = _policy.stats()
            print(json.dumps(stats, indent=2))
        return

    # ── --reset ───────────────────────────────────────────────────────────────
    if args.reset:
        if RL_POLICY_PATH.exists():
            RL_POLICY_PATH.unlink()
            print(f"[RL] Deleted policy file: {RL_POLICY_PATH}")
        else:
            print(f"[RL] No policy file to delete at: {RL_POLICY_PATH}")
        if _policy:
            from collections import defaultdict
            _policy.q_table = defaultdict(lambda: defaultdict(float))
            _policy.episode_count = 0
            print("[RL] In-memory policy reset.")
        return

    # ── --no-rl ───────────────────────────────────────────────────────────────
    if args.no_rl:
        # Patch config to disable RL for this run.
        import config as cfg
        cfg.RL_ENABLED = False
        print("[RL] RL disabled for this run (dry-run mode).")

    # ── Load questions ────────────────────────────────────────────────────────
    if args.dataset:
        dataset_path = Path(args.dataset)
        with dataset_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        questions = [
            {"query": item["question"], "category": item.get("category", "unknown")}
            for item in raw
            if "question" in item
        ]
        print(f"[Dataset] Loaded {len(questions)} questions from {dataset_path.name}")
    else:
        questions = BUILTIN_QUESTIONS
        print(f"[Dataset] Using {len(questions)} built-in training questions.")

    # ── Build graph ───────────────────────────────────────────────────────────
    print("\n[Graph] Compiling LangGraph...")
    graph = build_graph()
    print("[Graph] Ready.\n")

    all_results: List[Dict] = []
    total_start = time.perf_counter()

    for round_idx in range(1, args.rounds + 1):
        round_questions = list(questions)
        if args.shuffle:
            random.shuffle(round_questions)

        print(f"\n{'─'*60}")
        print(f"  ROUND {round_idx}/{args.rounds}  ({len(round_questions)} episodes)")
        print(f"{'─'*60}")

        round_results: List[Dict] = []

        for ep_idx, item in enumerate(round_questions, start=1):
            query = item["query"]
            category = item.get("category", "unknown")
            print(f"\n[R{round_idx} E{ep_idx:02d}/{len(round_questions):02d}] [{category}]")
            print(f"  Query: {query[:90]}{'...' if len(query) > 90 else ''}")

            ep_start = time.perf_counter()
            try:
                final_state = run_episode(graph, query)
                ep_latency = time.perf_counter() - ep_start

                # Reward was computed inside _rl_finish_node; recompute here for logging.
                reward = compute_episode_reward(final_state)

                ep_result = {
                    "round": round_idx,
                    "query": query,
                    "category": category,
                    "reward": reward,
                    "stop_reason": final_state.get("stop_reason", ""),
                    "citations_pass": bool(final_state.get("citations_pass", False)),
                    "step_count": int(final_state.get("step_count", 0)),
                    "crag_retries": int(final_state.get("crag_retries", 0)),
                    "verify_retries": int(final_state.get("verify_retries", 0)),
                    "latency_sec": round(ep_latency, 2),
                    "rl_transitions": len(final_state.get("rl_transitions", [])),
                }

                print(
                    f"  → stop={ep_result['stop_reason']}  "
                    f"citations={ep_result['citations_pass']}  "
                    f"reward={reward:.3f}  "
                    f"steps={ep_result['step_count']}  "
                    f"latency={ep_latency:.1f}s"
                )

            except Exception as exc:
                ep_latency = time.perf_counter() - ep_start
                print(f"  ✗ Episode failed: {exc}")
                ep_result = {
                    "round": round_idx,
                    "query": query,
                    "category": category,
                    "reward": -0.3,
                    "stop_reason": "error",
                    "citations_pass": False,
                    "step_count": 0,
                    "crag_retries": 0,
                    "verify_retries": 0,
                    "latency_sec": round(ep_latency, 2),
                    "rl_transitions": 0,
                    "error": str(exc),
                }

            round_results.append(ep_result)
            all_results.append(ep_result)

            # Rate-limit between Groq API calls.
            if ep_idx < len(round_questions) and args.delay > 0:
                time.sleep(args.delay)

        # Print round summary.
        summary = _round_summary(round_results)
        _print_summary(f"Round {round_idx} Summary", summary)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    final_summary = _round_summary(all_results)
    _print_summary(
        f"TRAINING COMPLETE  ({args.rounds} rounds, {len(all_results)} episodes, "
        f"{total_elapsed/60:.1f} min)",
        final_summary,
    )

    # Save training log.
    log_path = CURRENT_DIR / "results" / "rl_training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rounds": args.rounds,
                "total_episodes": len(all_results),
                "total_elapsed_sec": round(total_elapsed, 2),
                "final_summary": final_summary,
                "episodes": all_results,
            },
            f,
            indent=2,
        )
    print(f"[✓] Training log saved to: {log_path}")

    if _policy:
        print(f"[✓] Policy saved to: {RL_POLICY_PATH}")
        print(f"    Total episodes learned: {_policy.episode_count}")
        print(f"    Known states in Q-table: {len(_policy.q_table)}")


if __name__ == "__main__":
    main()
