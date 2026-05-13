"""
brain/context_marl_ac/maddpg/evaluate_maddpg.py
------------------------------------------------
Evaluation and comparison script for MADDPG continuous-control RAG.

Three runnable modes (--mode):
  maddpg          — MADDPG with Context Engineering Block (20-dim state)
  maddpg_no_ceb   — MADDPG without CEB (14-dim base state)
  discrete_marl   — Fixed smoke policy on the existing discrete MARL system
  compare_all     — Run all three and produce comparison_summary.csv

Outputs (written to results/maddpg/ by default):
  eval_{mode}.jsonl            — per-question result rows
  aggregate_metrics.json       — per-mode aggregate stats
  comparison_summary.csv       — side-by-side numeric comparison

Usage:
  # from Academic_RAG/brain/
  python -m context_marl_ac.maddpg.evaluate_maddpg --mode compare_all --dry-run

  # evaluate trained checkpoint on real benchmark:
  python -m context_marl_ac.maddpg.evaluate_maddpg \\
      --mode maddpg --checkpoint results/maddpg/checkpoints/best_reward.pt \\
      --benchmark-path results/benchmark_splits/test.jsonl
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ── sys.path ──────────────────────────────────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ── dotenv ────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    for _ep in [
        _BRAIN_ROOT / ".env",
        _BRAIN_ROOT.parent / ".env",
        Path.home() / "Desktop" / "Academic_RAG" / "brain" / ".env",
        Path.home() / "Desktop" / "Multimodal-Academic-RAG" / ".env",
    ]:
        if _ep.exists():
            load_dotenv(dotenv_path=_ep)
            break
except ImportError:
    pass

import context_marl_ac.config as cfg
from context_marl_ac.marl.marl_env import MARLEnv
from context_marl_ac.schemas.actions import AGENT_ACTIONS, AGENT_NAMES

from .maddpg_agent import MADDPGAgentWrapper
from .maddpg_critic import MADDPGCritic
from .continuous_action_mapper import (
    JOINT_ACTION_DIM,
    select_discrete_action,
)
from .context_engineering_block import CEB_STATE_DIM, build_ceb_features
from .train_maddpg import build_maddpg_agents, HIDDEN_DIM

# Default smoke-policy actions used for discrete_marl baseline.
_SMOKE: Dict[str, str] = {
    "retriever": "hybrid_rerank",
    "grader":    "medium_filter",
    "generator": "generate_with_strict_citations",
    "verifier":  "verify_answer",
    "rewriter":  "keyword_rewrite",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser("Evaluate / compare MADDPG RAG")
    p.add_argument("--mode", default="maddpg",
                   choices=["maddpg", "maddpg_no_ceb", "discrete_marl", "compare_all"],
                   help="Evaluation mode")
    p.add_argument("--checkpoint", default="",
                   help="Path to MADDPG .pt checkpoint (used for maddpg / maddpg_no_ceb)")
    p.add_argument("--benchmark-path", default="",
                   help="Path to evaluation JSONL (test split)")
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--output-dir",    default="",
                   help="Override results directory")
    p.add_argument("--n-questions",   type=int, default=0,
                   help="Limit to first N questions (0 = all)")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_benchmark(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        print(f"[evaluate_maddpg] Benchmark not found at '{path}'. Using dummy.")
        return [{
            "question":    "What is attention in transformers?",
            "ground_truth": "...",
            "question_id": "dummy_q1",
        }]
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(path) as f:
        return json.load(f)


# ── Checkpoint loading ────────────────────────────────────────────────────────

def _load_maddpg(
    ckpt_path: str,
    state_dim: int,
    device:    str,
) -> Tuple[Dict[str, MADDPGAgentWrapper], MADDPGCritic]:
    agents = build_maddpg_agents(state_dim, device)
    critic = MADDPGCritic(state_dim, JOINT_ACTION_DIM, HIDDEN_DIM)

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for n, a in agents.items():
            if n in ckpt.get("agents", {}):
                a.load_state_dict(ckpt["agents"][n])
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
        print(f"[evaluate_maddpg] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[evaluate_maddpg] No checkpoint at '{ckpt_path}'. Using random weights.")
    return agents, critic


# ── Result builder ────────────────────────────────────────────────────────────

def _build_result(
    state:      Any,
    q_dict:     Dict[str, Any],
    trace:      List[Dict],
    policy_mode: str,
) -> Dict[str, Any]:
    return {
        "question_id":            state.question_id,
        "question":               q_dict.get("question", ""),
        "ground_truth":           q_dict.get("ground_truth", ""),
        "category":               q_dict.get("category"),
        "difficulty":             q_dict.get("difficulty"),
        "policy_mode":            policy_mode,
        "final_status":           state.final_status,
        "final_answer":           state.generated_answer,
        "verification_pass":      int(state.final_status == "accepted"),
        "citation_support":       state.citation_support_rate,
        "num_unsupported_claims": len(state.unsupported_claims),
        "num_steps":              state.num_steps,
        "num_llm_calls":          state.num_llm_calls,
        "latency_seconds":        state.latency_so_far,
        "token_usage":            state.token_usage,
        "selected_evidence_count": len(state.selected_evidence),
        "verifier_decision":      (state.verification_result or {}).get("decision", "N/A"),
        "trace":                  trace,
    }


# ── MADDPG episode runner ─────────────────────────────────────────────────────

def _run_maddpg_episode(
    env:       MARLEnv,
    q_dict:    Dict[str, Any],
    q_idx:     int,
    agents:    Dict[str, MADDPGAgentWrapper],
    use_ceb:   bool,
    policy_label: str,
) -> Dict[str, Any]:
    state = env.reset(q_dict, index=q_idx + 1)
    trace: List[Dict] = []
    done  = False

    while not done:
        active_agent  = None
        valid_actions: List[str] = []
        for name in AGENT_NAMES:
            mask = env.get_mask(name)
            if sum(mask) > 0:
                active_agent  = name
                valid_actions = [AGENT_ACTIONS[name][i] for i, m in enumerate(mask) if m == 1]
                break

        if not active_agent:
            if state.final_status == "pending":
                state.final_status = "abstained"
            state.done = True
            done = True
            break

        # Build observation.
        if use_ceb:
            obs = np.array(build_ceb_features(env.state), dtype=np.float32)
        else:
            obs = np.array(env.get_global_features(), dtype=np.float32)

        # Greedy action (no exploration noise).
        raw_action = agents[active_agent].select_action(obs, explore=False)
        params     = agents[active_agent].map_params(raw_action)
        discrete   = select_discrete_action(active_agent, params, valid_actions)

        try:
            new_state, reward, done, _ = env.step(active_agent, discrete, params=params)
        except Exception as exc:
            state.final_status = "error"
            state.done = True
            done = True
            trace.append({"agent": active_agent, "error": str(exc)})
            break

        trace.append({
            "step":            new_state.num_steps,
            "agent":           active_agent,
            "discrete_action": discrete,
            "raw_action":      raw_action.tolist(),
            "mapped_params":   params,
            "reward":          reward,
            "done":            done,
        })
        state = new_state

    return _build_result(state, q_dict, trace, policy_label)


# ── Discrete MARL (smoke) episode runner ──────────────────────────────────────

def _run_discrete_episode(
    env:    MARLEnv,
    q_dict: Dict[str, Any],
    q_idx:  int,
) -> Dict[str, Any]:
    state = env.reset(q_dict, index=q_idx + 1)
    trace: List[Dict] = []
    done  = False

    while not done:
        active_agent  = None
        valid_actions: List[str] = []
        for name in AGENT_NAMES:
            mask = env.get_mask(name)
            if sum(mask) > 0:
                active_agent  = name
                valid_actions = [AGENT_ACTIONS[name][i] for i, m in enumerate(mask) if m == 1]
                break

        if not active_agent:
            if state.final_status == "pending":
                state.final_status = "abstained"
            state.done = True
            done = True
            break

        preferred = _SMOKE.get(active_agent, valid_actions[0])
        discrete  = preferred if preferred in valid_actions else valid_actions[0]

        try:
            new_state, reward, done, _ = env.step(active_agent, discrete)
        except Exception as exc:
            state.final_status = "error"
            state.done = True
            done = True
            trace.append({"agent": active_agent, "action": discrete, "error": str(exc)})
            break

        trace.append({
            "step":   new_state.num_steps,
            "agent":  active_agent,
            "action": discrete,
            "reward": reward,
            "done":   done,
        })
        state = new_state

    return _build_result(state, q_dict, trace, "discrete_marl")


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def _aggregate(results: List[Dict]) -> Dict[str, Any]:
    n = len(results) or 1
    fail_statuses = {"rejected", "timeout", "error", "generation_failed"}
    return {
        "n_questions":             n,
        "verification_pass_rate":  sum(r["verification_pass"]      for r in results) / n,
        "mean_citation_support":   sum(r["citation_support"]        for r in results) / n,
        "mean_unsupported_claims": sum(r["num_unsupported_claims"]  for r in results) / n,
        "mean_steps":              sum(r["num_steps"]               for r in results) / n,
        "mean_llm_calls":          sum(r["num_llm_calls"]           for r in results) / n,
        "mean_latency":            sum(r["latency_seconds"]         for r in results) / n,
        "mean_token_usage":        sum(r["token_usage"]             for r in results) / n,
        "mean_evidence_count":     sum(r["selected_evidence_count"] for r in results) / n,
        "failure_rate":            sum(
            1 for r in results if r["final_status"] in fail_statuses
        ) / n,
    }


# ── JSONL writer ──────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, results: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Main evaluation entry point ───────────────────────────────────────────────

def evaluate():
    args = _parse_args()
    cfg.DRY_RUN = args.dry_run
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parent.parent / "results" / "maddpg"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = _load_benchmark(
        args.benchmark_path or str(
            Path(__file__).resolve().parent.parent / "results" / "benchmark_splits" / "test.jsonl"
        )
    )
    if args.n_questions > 0:
        benchmark = benchmark[:args.n_questions]

    env = MARLEnv()

    modes = (
        ["maddpg", "maddpg_no_ceb", "discrete_marl"]
        if args.mode == "compare_all"
        else [args.mode]
    )

    all_results:    Dict[str, List[Dict]] = {}
    loaded_agents:  Dict[int, Tuple] = {}  # keyed by state_dim to avoid double-loading

    for mode in modes:
        print(f"\n{'-'*60}\n  Mode: {mode}\n{'-'*60}")
        results: List[Dict] = []

        if mode in ("maddpg", "maddpg_no_ceb"):
            use_ceb   = (mode == "maddpg")
            state_dim = CEB_STATE_DIM if use_ceb else cfg.FEATURE_DIM

            if state_dim not in loaded_agents:
                loaded_agents[state_dim] = _load_maddpg(args.checkpoint, state_dim, device)
            agents, _ = loaded_agents[state_dim]

            for q_idx, q_dict in enumerate(tqdm(benchmark, desc=mode)):
                r = _run_maddpg_episode(env, q_dict, q_idx, agents, use_ceb, mode)
                results.append(r)

        elif mode == "discrete_marl":
            for q_idx, q_dict in enumerate(tqdm(benchmark, desc=mode)):
                r = _run_discrete_episode(env, q_dict, q_idx)
                results.append(r)

        all_results[mode] = results
        _write_jsonl(out_dir / f"eval_{mode}.jsonl", results)
        print(f"  -> {len(results)} questions evaluated. Results: eval_{mode}.jsonl")

    # ── Aggregate JSON ─────────────────────────────────────────────────────────
    agg_all: Dict[str, Dict] = {m: _aggregate(r) for m, r in all_results.items()}
    agg_path = out_dir / "aggregate_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(agg_all, f, indent=2)
    print(f"\nAggregate metrics -> {agg_path}")

    # ── Comparison CSV (only when multiple modes ran) ─────────────────────────
    if len(all_results) > 1:
        comp_path = out_dir / "comparison_summary.csv"
        metric_keys = list(next(iter(agg_all.values())).keys())
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["policy"] + metric_keys)
            w.writeheader()
            for mode, agg in agg_all.items():
                w.writerow({"policy": mode, **agg})
        print(f"Comparison CSV -> {comp_path}")

    print("\n-- Summary " + "-"*57)
    for mode, agg in agg_all.items():
        print(
            f"  {mode:20s}  pass={agg['verification_pass_rate']:.1%}  "
            f"cit={agg['mean_citation_support']:.2f}  "
            f"steps={agg['mean_steps']:.1f}  "
            f"latency={agg['mean_latency']:.1f}s"
        )


if __name__ == "__main__":
    evaluate()
