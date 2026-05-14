"""
brain/maddpg/evaluate_maddpg.py
------------------------------------------------
Evaluation script for the stage-conditioned MADDPG-style RAG architecture.

Modes:
  maddpg_ceb       — MADDPG with Context Engineering Block (20-dim state)
  maddpg_no_ceb    — MADDPG without CEB (14-dim base state)
  discrete_smoke_policy — fixed (un-trained) action policy on the discrete MARL env;
                         enabled only via --compare-baseline.

Behaviour:
  - Missing checkpoint is a HARD error by default.
  - --allow-random-policy is required for smoke runs that should accept random weights.
  - Every result file embeds checkpoint metadata (path, exists, trained flag, etc.).

Usage (from brain/):
  python -m maddpg.evaluate_maddpg \\
      --mode maddpg_ceb \\
      --checkpoint results/maddpg/checkpoints/best_reward.pt \\
      --benchmark-path results/benchmark_splits/test.jsonl \\
      --n-questions 9
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ── sys.path ──────────────────────────────────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ── dotenv (local paths only) ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    for _ep in [_BRAIN_ROOT / ".env", _BRAIN_ROOT.parent / ".env"]:
        if _ep.exists():
            load_dotenv(dotenv_path=_ep)
            break
except ImportError:
    pass

import context_marl_ac.config as cfg
from context_marl_ac.schemas.actions import AGENT_ACTIONS

from .context_engineering_block import CEB_STATE_DIM, build_ceb_features
from .stage_utils import find_active_agent_and_valid_actions
from .trainer import StageConditionedMADDPGTrainer, TrainerConfig

# Fixed (un-trained) action choices used for the discrete_smoke_policy baseline.
_SMOKE_POLICY: Dict[str, str] = {
    "retriever": "hybrid_rerank",
    "grader":    "medium_filter",
    "generator": "generate_with_strict_citations",
    "verifier":  "verify_answer",
    "rewriter":  "keyword_rewrite",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate stage-conditioned MADDPG RAG")
    p.add_argument("--mode", default="maddpg_ceb",
                   choices=["maddpg_ceb", "maddpg_no_ceb"],
                   help="Evaluation mode")
    p.add_argument("--checkpoint", default="",
                   help="Path to MADDPG .pt checkpoint")
    p.add_argument("--benchmark-path", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-dir", default="")
    p.add_argument("--n-questions", type=int, default=0,
                   help="Limit to first N questions (0 = all)")
    p.add_argument("--allow-random-policy", action="store_true",
                   help="Allow evaluation without a checkpoint (smoke tests only).")
    p.add_argument("--compare-baseline", action="store_true",
                   help="Also run the discrete_smoke_policy baseline.")
    p.add_argument("--hidden-dim", type=int, default=128)
    return p.parse_args()


# ── Benchmark loader ──────────────────────────────────────────────────────────

def _load_benchmark(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        print(f"[evaluate_maddpg] Benchmark not found at '{path}'. Using dummy.")
        return [{
            "question":     "What is attention in transformers?",
            "ground_truth": "...",
            "question_id":  "dummy_q1",
        }]
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(path) as f:
        return json.load(f)


# ── Feature builder ───────────────────────────────────────────────────────────

def _state_features(env: Any, use_ceb: bool) -> np.ndarray:
    if use_ceb:
        return np.array(build_ceb_features(env.state), dtype=np.float32)
    return np.array(env.get_global_features(), dtype=np.float32)


# ── Result builder ────────────────────────────────────────────────────────────

def _build_result(state: Any, q_dict: Dict[str, Any],
                  trace: List[Dict], policy_mode: str) -> Dict[str, Any]:
    return {
        "question_id":             state.question_id,
        "question":                q_dict.get("question", ""),
        "ground_truth":            q_dict.get("ground_truth", ""),
        "category":                q_dict.get("category"),
        "difficulty":              q_dict.get("difficulty"),
        "policy_mode":             policy_mode,
        "final_status":            state.final_status,
        "final_answer":            state.generated_answer,
        "verification_pass":       int(state.final_status == "accepted"),
        "citation_support":        state.citation_support_rate,
        "num_unsupported_claims":  len(state.unsupported_claims),
        "num_steps":               state.num_steps,
        "num_llm_calls":           state.num_llm_calls,
        "latency_seconds":         state.latency_so_far,
        "token_usage":             state.token_usage,
        "selected_evidence_count": len(state.selected_evidence),
        "verifier_decision":       (state.verification_result or {}).get("decision", "N/A"),
        "trace":                   trace,
    }


# ── MADDPG episode runner ─────────────────────────────────────────────────────

def _run_maddpg_episode(env: Any, q_dict: Dict[str, Any], q_idx: int,
                         trainer: StageConditionedMADDPGTrainer,
                         use_ceb: bool, policy_label: str) -> Dict[str, Any]:
    state = env.reset(q_dict, index=q_idx + 1)
    trace: List[Dict] = []
    done = False

    while not done:
        active_agent, valid_actions = find_active_agent_and_valid_actions(env)
        if active_agent is None:
            if state.final_status == "pending":
                state.final_status = "abstained"
            state.done = True
            done = True
            break

        obs = _state_features(env, use_ceb)
        raw, params, discrete = trainer.select_action(
            active_agent, obs, valid_actions, explore=False
        )

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
            "active_agent":    active_agent,
            "discrete_action": discrete,
            "mapped_params":   params,
            "reward":          reward,
            "done":            done,
        })
        state = new_state

    return _build_result(state, q_dict, trace, policy_label)


# ── Discrete smoke-policy episode runner ─────────────────────────────────────

def _run_smoke_episode(env: Any, q_dict: Dict[str, Any], q_idx: int) -> Dict[str, Any]:
    state = env.reset(q_dict, index=q_idx + 1)
    trace: List[Dict] = []
    done = False
    while not done:
        active_agent, valid_actions = find_active_agent_and_valid_actions(env)
        if active_agent is None:
            if state.final_status == "pending":
                state.final_status = "abstained"
            state.done = True
            done = True
            break
        preferred = _SMOKE_POLICY.get(active_agent, valid_actions[0])
        discrete = preferred if preferred in valid_actions else valid_actions[0]
        try:
            new_state, reward, done, _ = env.step(active_agent, discrete)
        except Exception as exc:
            state.final_status = "error"
            state.done = True
            done = True
            trace.append({"agent": active_agent, "action": discrete, "error": str(exc)})
            break
        trace.append({"step": new_state.num_steps, "agent": active_agent,
                       "action": discrete, "reward": reward, "done": done})
        state = new_state
    return _build_result(state, q_dict, trace, "discrete_smoke_policy")


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
        "failure_rate":            sum(1 for r in results if r["final_status"] in fail_statuses) / n,
    }


def _write_jsonl(path: Path, results: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(argv: Optional[List[str]] = None) -> int:
    args = _parse_args()
    cfg.DRY_RUN = args.dry_run
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parent / "results" / "maddpg"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = _load_benchmark(
        args.benchmark_path or str(
            Path(__file__).resolve().parent / "results" / "benchmark_splits" / "test.jsonl"
        )
    )
    if args.n_questions > 0:
        benchmark = benchmark[:args.n_questions]

    # ── Checkpoint policy ────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    ckpt_exists = bool(ckpt_path) and os.path.exists(ckpt_path)
    if not ckpt_exists and not args.allow_random_policy:
        print(
            f"[evaluate_maddpg] ERROR: checkpoint not found: '{ckpt_path}'. "
            f"Pass --allow-random-policy explicitly for smoke tests."
        )
        return 2

    use_ceb = (args.mode == "maddpg_ceb")
    state_dim = CEB_STATE_DIM if use_ceb else cfg.FEATURE_DIM
    tcfg = TrainerConfig(state_dim=state_dim, hidden_dim=args.hidden_dim, device=device)
    trainer = StageConditionedMADDPGTrainer(tcfg)

    if ckpt_exists:
        trainer.load_checkpoint(Path(ckpt_path))

    from context_marl_ac.marl.marl_env import MARLEnv  # late import for dry-run safety
    env = MARLEnv()

    print(f"\n{'-'*60}\n  Mode: {args.mode}  (CEB={use_ceb}, state_dim={state_dim})\n"
          f"  Checkpoint: {ckpt_path or '(none)'}  exists={ckpt_exists}  "
          f"trained_updates={trainer.total_gradient_updates}\n{'-'*60}")

    all_results: Dict[str, List[Dict]] = {}
    results: List[Dict] = []
    for q_idx, q_dict in enumerate(tqdm(benchmark, desc=args.mode)):
        results.append(_run_maddpg_episode(env, q_dict, q_idx, trainer, use_ceb, args.mode))
    all_results[args.mode] = results
    _write_jsonl(out_dir / f"eval_{args.mode}.jsonl", results)
    print(f"  -> {len(results)} questions evaluated.")

    if args.compare_baseline:
        bres: List[Dict] = []
        for q_idx, q_dict in enumerate(tqdm(benchmark, desc="discrete_smoke_policy")):
            bres.append(_run_smoke_episode(env, q_dict, q_idx))
        all_results["discrete_smoke_policy"] = bres
        _write_jsonl(out_dir / "eval_discrete_smoke_policy.jsonl", bres)

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg_all: Dict[str, Dict[str, Any]] = {m: _aggregate(rs) for m, rs in all_results.items()}

    # Always embed checkpoint metadata.
    metadata = {
        "architecture":           trainer.architecture,
        "critic_type":            trainer.critic_type,
        "checkpoint_path":        ckpt_path or None,
        "checkpoint_exists":      ckpt_exists,
        "trained":                trainer.total_gradient_updates > 0,
        "total_gradient_updates": trainer.total_gradient_updates,
        "state_dim":              state_dim,
        "use_ceb":                use_ceb,
        "allow_random_policy":    bool(args.allow_random_policy),
    }
    agg_payload: Dict[str, Any] = {"metadata": metadata, "modes": agg_all}

    agg_path = out_dir / "aggregate_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(agg_payload, f, indent=2)
    print(f"\nAggregate metrics -> {agg_path}")

    if len(all_results) > 1:
        comp_path = out_dir / "comparison_summary.csv"
        metric_keys = list(next(iter(agg_all.values())).keys())
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["policy"] + metric_keys)
            w.writeheader()
            for mode, agg in agg_all.items():
                w.writerow({"policy": mode, **agg})
        print(f"Comparison CSV -> {comp_path}")

    print("\n-- Summary " + "-" * 57)
    for mode, agg in agg_all.items():
        print(f"  {mode:24s}  pass={agg['verification_pass_rate']:.1%}  "
              f"cit={agg['mean_citation_support']:.2f}  "
              f"steps={agg['mean_steps']:.1f}  latency={agg['mean_latency']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(evaluate())
