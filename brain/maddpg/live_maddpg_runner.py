"""
brain/maddpg/live_maddpg_runner.py
-----------------------------------
Live training + evaluation pipeline for the stage-conditioned MADDPG-style RAG.

Runs both no-CEB and CEB variants end-to-end with real LLM/Qdrant calls,
then compares against the discrete MARL baseline (loaded from existing
final_eval/ files when present).

This file is a thin orchestration layer over `trainer.StageConditionedMADDPGTrainer`.
All gradient-update logic lives in trainer.py — do not duplicate it here.

Usage (from brain/):
  python -m maddpg.live_maddpg_runner --episodes 30 --n-eval 9
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ── sys.path ──────────────────────────────────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ── dotenv (local paths only — never print API keys) ─────────────────────────
try:
    from dotenv import load_dotenv
    _loaded = False
    for _ep in [_BRAIN_ROOT / ".env", _BRAIN_ROOT.parent / ".env"]:
        if _ep.exists():
            load_dotenv(dotenv_path=_ep)
            _loaded = True
            break
    if not _loaded:
        print("[env] No local .env found. API keys must be set in environment.")
except ImportError:
    pass

import context_marl_ac.config as cfg

from .context_engineering_block import CEB_STATE_DIM, build_ceb_features
from .stage_utils import find_active_agent_and_valid_actions
from .trainer import StageConditionedMADDPGTrainer, TrainerConfig
from .train_maddpg import (
    EP_CSV_FIELDNAMES,
    PARAMS_CSV_FIELDNAMES,
    _log_params_row,
)


# ── NLP metrics (no external deps) ───────────────────────────────────────────

def _tok(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def token_f1(pred: str, gold: str) -> float:
    p_toks, g_toks = set(_tok(pred)), set(_tok(gold))
    if not p_toks or not g_toks:
        return 0.0
    tp = len(p_toks & g_toks)
    prec = tp / len(p_toks)
    rec  = tp / len(g_toks)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def rouge_l(pred: str, gold: str) -> float:
    p, g = _tok(pred), _tok(gold)
    m, n = len(p), len(g)
    if not m or not n:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if p[i-1] == g[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m
    rec  = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def _src_pdfs(chunks: List[Dict]) -> Set[str]:
    out: Set[str] = set()
    for c in chunks:
        sf = c.get("metadata", {}).get("source_file") or c.get("source_file") or ""
        if sf:
            out.add(sf.split("_p")[0].strip())
    return out

def src_precision_recall(retrieved: List[Dict], expected: List[str]) -> Tuple[float, float]:
    ret_pdfs = _src_pdfs(retrieved)
    exp_pdfs = {s.split("_p")[0].strip() for s in expected if s}
    if not exp_pdfs:
        return 0.0, 0.0
    tp = len(ret_pdfs & exp_pdfs)
    prec = tp / len(ret_pdfs) if ret_pdfs else 0.0
    rec  = tp / len(exp_pdfs)
    return prec, rec


# ── Benchmark loading ─────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def _load_benchmark(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")
    return _load_jsonl(p)


# ── State features ────────────────────────────────────────────────────────────

def _state_features(env: Any, use_ceb: bool) -> np.ndarray:
    if use_ceb:
        return np.array(build_ceb_features(env.state), dtype=np.float32)
    return np.array(env.get_global_features(), dtype=np.float32)


# ── Training loop (delegates to trainer.update()) ────────────────────────────

def train_variant(
    *,
    variant:          str,
    use_ceb:          bool,
    benchmark:        List[Dict],
    episodes:         int,
    ckpt_dir:         Path,
    metrics_dir:      Path,
    traj_dir:         Path,
    device:           str,
    checkpoint_every: int,
    hparams:          argparse.Namespace,
) -> Tuple[Path, int, Dict[str, Any]]:
    """
    Train one variant. Returns (best_reward_ckpt, total_gradient_updates, meta).
    """
    print(f"\n{'='*60}\n  TRAINING: maddpg_{variant}   episodes={episodes}   CEB={use_ceb}\n{'='*60}")

    state_dim = CEB_STATE_DIM if use_ceb else cfg.FEATURE_DIM
    tcfg = TrainerConfig(
        state_dim     = state_dim,
        hidden_dim    = hparams.hidden_dim,
        actor_lr      = hparams.actor_lr,
        critic_lr     = hparams.critic_lr,
        gamma         = hparams.gamma,
        tau           = hparams.tau,
        batch_size    = hparams.batch_size,
        noise_sigma   = hparams.noise_sigma,
        grad_clip     = hparams.grad_clip,
        update_every  = hparams.update_every,
        warmup_steps  = hparams.warmup_steps,
        error_penalty = hparams.error_penalty,
        device        = device,
    )
    trainer = StageConditionedMADDPGTrainer(tcfg)

    from context_marl_ac.marl.marl_env import MARLEnv  # late import for dry-run safety
    env = MARLEnv()

    run_name    = f"maddpg_{variant}_live"
    ep_path     = metrics_dir / f"ep_metrics_{run_name}.csv"
    params_path = metrics_dir / f"action_params_{run_name}.csv"
    traj_path   = traj_dir    / f"trajectories_{run_name}.jsonl"
    best_ckpt   = ckpt_dir    / f"best_{run_name}.pt"
    best_reward = -float("inf")
    step_errors = 0

    with (
        open(ep_path,     "w", newline="", encoding="utf-8") as ep_f,
        open(params_path, "w", newline="", encoding="utf-8") as prm_f,
        open(traj_path,   "w", encoding="utf-8")             as trj_f,
    ):
        prm_writer = csv.DictWriter(prm_f, fieldnames=PARAMS_CSV_FIELDNAMES,
                                    restval="", extrasaction="ignore")
        prm_writer.writeheader()
        ep_writer = csv.DictWriter(ep_f, fieldnames=EP_CSV_FIELDNAMES,
                                   restval="", extrasaction="ignore")
        ep_writer.writeheader()

        last_losses: Dict[str, float] = {}
        for ep_idx in tqdm(range(1, episodes + 1), desc=f"maddpg_{variant}"):
            q_idx  = random.randint(0, len(benchmark) - 1)
            q_dict = benchmark[q_idx]
            state  = env.reset(q_dict, index=q_idx + 1)
            trainer.reset_noise()
            done = False
            ep_steps = 0
            ep_traj: List[Dict[str, Any]] = []

            while not done:
                active_agent, valid_actions = find_active_agent_and_valid_actions(env)
                if active_agent is None:
                    if state.final_status == "pending":
                        state.final_status = "abstained"
                    state.done = True
                    done = True
                    break

                prev_features = _state_features(env, use_ceb)
                raw, params, discrete = trainer.select_action(
                    active_agent, prev_features, valid_actions, explore=True
                )

                try:
                    new_state, reward, done, _info = env.step(
                        active_agent, discrete, params=params
                    )
                    next_features = _state_features(env, use_ceb)
                    next_active, next_valid = (
                        (None, []) if done
                        else find_active_agent_and_valid_actions(env)
                    )
                    trainer.push_transition(
                        state_features      = prev_features,
                        active_agent        = active_agent,
                        valid_actions       = valid_actions,
                        raw_action          = raw,
                        mapped_params       = params,
                        discrete_action     = discrete,
                        reward              = reward,
                        next_state_features = next_features,
                        next_active_agent   = next_active,
                        next_valid_actions  = next_valid,
                        done                = done,
                        question_id         = state.question_id,
                        step                = ep_steps + 1,
                        final_status        = new_state.final_status,
                        metrics_snapshot    = {
                            "citation_support_rate":  new_state.citation_support_rate,
                            "num_unsupported_claims": len(new_state.unsupported_claims),
                            "final_status":           new_state.final_status,
                        },
                    )
                except Exception as exc:
                    step_errors += 1
                    print(f"  [train-err] ep={ep_idx} agent={active_agent} action={discrete}: {exc}")
                    trainer.push_error_transition(
                        state_features  = prev_features,
                        active_agent    = active_agent,
                        valid_actions   = valid_actions,
                        raw_action      = raw,
                        mapped_params   = params,
                        discrete_action = discrete,
                        question_id     = state.question_id,
                        step            = ep_steps + 1,
                        error_message   = str(exc),
                    )
                    state.final_status = "error"
                    state.done = True
                    done = True
                    new_state = state
                    reward = tcfg.error_penalty

                ep_steps += 1
                _log_params_row(
                    prm_writer, ep_idx, ep_steps, active_agent, valid_actions,
                    discrete, reward, done,
                    (None if done else find_active_agent_and_valid_actions(env)[0]),
                    raw, params,
                )

                ep_traj.append({
                    "step":             ep_steps,
                    "active_agent":     active_agent,
                    "valid_actions":    list(valid_actions),
                    "discrete_action":  discrete,
                    "raw_action":       raw.tolist(),
                    "mapped_params":    params,
                    "reward":           reward,
                    "done":             done,
                })

                if trainer.should_update():
                    last_losses = trainer.update()

                state = new_state

            ep_reward = env.get_global_reward()
            ep_metrics = {
                "episode":                ep_idx,
                "question_id":            state.question_id,
                "total_reward":           round(ep_reward, 6),
                "num_steps":              state.num_steps,
                "num_llm_calls":          state.num_llm_calls,
                "final_status":           state.final_status,
                "verification_pass":      int(state.final_status == "accepted"),
                "citation_support":       round(state.citation_support_rate, 4),
                "latency_seconds":        round(state.latency_so_far, 3),
                "token_usage":            state.token_usage,
                "buffer_size":            len(trainer.buffer),
                "total_gradient_updates": trainer.total_gradient_updates,
                "critic_loss":            round(last_losses.get("critic_loss", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_retriever":   round(last_losses.get("actor_loss_retriever", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_rewriter":    round(last_losses.get("actor_loss_rewriter", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_grader":      round(last_losses.get("actor_loss_grader", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_generator":   round(last_losses.get("actor_loss_generator", float("nan")), 6)
                                            if last_losses else "",
                "actor_loss_verifier":    round(last_losses.get("actor_loss_verifier", float("nan")), 6)
                                            if last_losses else "",
                "trained_so_far":         trainer.total_gradient_updates > 0,
            }
            ep_writer.writerow(ep_metrics)
            ep_f.flush()
            trj_f.write(json.dumps({
                "episode":      ep_idx,
                "question_id":  state.question_id,
                "trajectory":   ep_traj,
                "final_status": state.final_status,
                "total_reward": ep_reward,
            }) + "\n")

            if ep_reward > best_reward:
                best_reward = ep_reward
                trainer.save_checkpoint(
                    best_ckpt,
                    extra={"variant": variant, "episode": ep_idx, "metrics": ep_metrics},
                )
            if ep_idx % checkpoint_every == 0:
                trainer.save_checkpoint(
                    ckpt_dir / f"{run_name}_ep{ep_idx:04d}.pt",
                    extra={"variant": variant, "episode": ep_idx, "metrics": ep_metrics},
                )

    meta = {
        "variant":                variant,
        "best_reward":            best_reward if best_reward != -float("inf") else None,
        "total_env_steps":        trainer.total_env_steps,
        "total_gradient_updates": trainer.total_gradient_updates,
        "step_errors":            step_errors,
        "critic_type":            trainer.critic_type,
        "architecture":           trainer.architecture,
        "state_dim":              state_dim,
        "use_ceb":                use_ceb,
        "hyperparameters":        vars(hparams),
    }
    print(
        f"  [done] best_reward={best_reward:.4f}  "
        f"gradient_updates={trainer.total_gradient_updates}  step_errors={step_errors}  "
        f"checkpoint -> {best_ckpt}"
    )
    return best_ckpt, trainer.total_gradient_updates, meta


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_variant(
    *,
    variant:    str,
    use_ceb:    bool,
    ckpt_path:  Path,
    benchmark:  List[Dict],
    n_eval:     int,
    out_dir:    Path,
    device:     str,
    hparams:    argparse.Namespace,
) -> List[Dict]:
    print(f"\n{'='*60}\n  EVALUATION: maddpg_{variant}   n={n_eval}   CEB={use_ceb}\n{'='*60}")

    state_dim = CEB_STATE_DIM if use_ceb else cfg.FEATURE_DIM
    tcfg = TrainerConfig(state_dim=state_dim, device=device, hidden_dim=hparams.hidden_dim)
    trainer = StageConditionedMADDPGTrainer(tcfg)

    if ckpt_path.exists():
        trainer.load_checkpoint(ckpt_path)
        print(f"  [ckpt] Loaded {ckpt_path}  (updates={trainer.total_gradient_updates})")
    else:
        print(f"  [warn] Checkpoint not found: {ckpt_path}. Using random weights.")

    from context_marl_ac.marl.marl_env import MARLEnv
    env = MARLEnv()

    eval_qs = benchmark[:n_eval]
    results: List[Dict] = []

    for q_idx, q_dict in enumerate(tqdm(eval_qs, desc=f"eval maddpg_{variant}")):
        state = env.reset(q_dict, index=q_idx + 1)
        done = False
        trace: List[Dict] = []

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
                "step":             new_state.num_steps,
                "active_agent":     active_agent,
                "discrete_action":  discrete,
                "mapped_params":    params,
                "reward":           reward,
                "done":             done,
            })
            state = new_state

        gold = q_dict.get("ground_truth", "")
        pred = state.generated_answer or ""
        exp_s = q_dict.get("source_file", [])
        if isinstance(exp_s, str):
            exp_s = [exp_s]

        src_p, src_r = src_precision_recall(state.retrieved_chunks, exp_s)
        tf1 = token_f1(pred, gold)
        rl  = rouge_l(pred, gold)

        results.append({
            "question_id":        state.question_id,
            "question":           q_dict.get("question", ""),
            "ground_truth":       gold,
            "category":           q_dict.get("category"),
            "difficulty":         q_dict.get("difficulty"),
            "policy":             f"maddpg_{variant}",
            "data_source":        "live_llm",
            "final_status":       state.final_status,
            "final_answer":       pred,
            "token_f1":           round(tf1, 4),
            "rouge_l":            round(rl, 4),
            "correctness":        round(tf1, 4),
            "faithfulness":       round(state.citation_support_rate, 4),
            "citation_support":   round(state.citation_support_rate, 4),
            "source_precision":   round(src_p, 4),
            "source_recall":      round(src_r, 4),
            "verification_pass":  int(state.final_status == "accepted"),
            "unsupported_claims": len(state.unsupported_claims),
            "latency_seconds":    state.latency_so_far,
            "num_llm_calls":      state.num_llm_calls,
            "token_usage":        state.token_usage,
            "num_steps":          state.num_steps,
            "trace":              trace,
        })

    out_path = out_dir / f"maddpg_{variant}_live.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(results)} results -> {out_path}")
    return results


# ── Load real discrete_marl baseline ──────────────────────────────────────────

def _load_discrete_baseline(eval_dir: Path, test_qids: List[str]) -> List[Dict]:
    candidates = sorted(eval_dir.glob("learned_eval_*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(eval_dir.glob("*.jsonl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)

    by_qid: Dict[str, Dict] = {}
    for fpath in candidates:
        try:
            for row in _load_jsonl(fpath):
                qid = row.get("question_id")
                if qid and qid not in by_qid:
                    by_qid[qid] = row
        except Exception:
            continue

    dc_path = eval_dir.parent / "defense_comparison" / "discrete_marl_real.jsonl"
    if dc_path.exists():
        for row in _load_jsonl(dc_path):
            qid = row.get("question_id")
            if qid and qid not in by_qid:
                by_qid[qid] = row

    results: List[Dict] = []
    for qid in test_qids:
        row = by_qid.get(qid)
        if not row:
            continue
        gold = row.get("ground_truth", "")
        pred = row.get("final_answer", "")
        exp_s = row.get("source_file", [])
        if isinstance(exp_s, str):
            exp_s = [exp_s]
        retrieved = row.get("retrieved_chunks", [])
        src_p, src_r = src_precision_recall(retrieved, exp_s)
        tf1 = token_f1(pred, gold)
        rl  = rouge_l(pred, gold)
        results.append({
            "question_id":       qid,
            "question":          row.get("question", ""),
            "ground_truth":      gold,
            "category":          row.get("category"),
            "difficulty":        row.get("difficulty"),
            "policy":            "discrete_marl",
            "data_source":       "real_llm",
            "final_status":      row.get("final_status", ""),
            "final_answer":      pred,
            "token_f1":          round(tf1, 4),
            "rouge_l":           round(rl, 4),
            "correctness":       round(tf1, 4),
            "faithfulness":      round(float(row.get("citation_support_rate", 0)), 4),
            "citation_support":  round(float(row.get("citation_support_rate", 0)), 4),
            "source_precision":  round(src_p, 4),
            "source_recall":     round(src_r, 4),
            "verification_pass": int(row.get("final_status") == "accepted"),
            "unsupported_claims": len(row.get("unsupported_claims", [])),
            "latency_seconds":   float(row.get("latency_seconds", 0)),
            "num_llm_calls":     int(row.get("num_llm_calls", 0)),
            "token_usage":       int(row.get("token_usage", 0)),
            "num_steps":         int(row.get("num_steps", 0)),
        })
    print(f"  [baseline] Loaded {len(results)} real discrete_marl results.")
    return results


# ── Aggregate + comparison table ──────────────────────────────────────────────

_METRIC_COLS = [
    "token_f1", "rouge_l", "correctness", "faithfulness",
    "citation_support", "source_precision", "source_recall",
    "verification_pass", "unsupported_claims",
    "latency_seconds", "num_llm_calls", "token_usage", "num_steps",
]

def _aggregate(results: List[Dict]) -> Dict[str, Any]:
    n = len(results) or 1
    fail = {"rejected", "error", "timeout", "generation_failed", "abstained"}
    agg: Dict[str, Any] = {"n_questions": n, "data_source": results[0].get("data_source", "?") if results else "?"}
    for col in _METRIC_COLS:
        vals = [float(r.get(col, 0)) for r in results]
        agg[f"mean_{col}"] = round(sum(vals) / n, 4)
    agg["failure_rate"] = round(sum(1 for r in results if r["final_status"] in fail) / n, 4)
    return agg


def _print_table(agg_all: Dict[str, Dict]) -> None:
    policies = list(agg_all.keys())
    col_w = 22
    sep = "-" * (20 + col_w * len(policies))
    print(f"\n{'='*80}\n  LIVE DEFENSE COMPARISON TABLE\n{'='*80}")
    header = f"{'Metric':<20}" + "".join(f"{p:>{col_w}}" for p in policies)
    print(header); print(sep)
    rows = [
        ("n questions",      "n_questions"),
        ("data source",      "data_source"),
        ("Token F1",         "mean_token_f1"),
        ("ROUGE-L",          "mean_rouge_l"),
        ("Citation Support", "mean_citation_support"),
        ("Verif. Pass",      "mean_verification_pass"),
        ("Latency (s)",      "mean_latency_seconds"),
        ("Failure Rate",     "failure_rate"),
    ]
    for label, key in rows:
        vals = [agg_all[p].get(key, "N/A") for p in policies]
        row = f"{label:<20}" + "".join(
            f"{str(v):>{col_w}}" if isinstance(v, str) else f"{v:>{col_w}.4f}"
            for v in vals
        )
        print(row)
    print(sep)


# ── Interpretation writer (rewritten to be honest about training status) ─────

def _write_interpretation(
    agg_all:        Dict[str, Dict],
    out_path:       Path,
    metas:          Dict[str, Dict[str, Any]],
    hparams:        argparse.Namespace,
) -> None:
    pols = list(agg_all.keys())
    lines: List[str] = [
        "# Live MADDPG-style (stage-conditioned) vs Discrete MARL",
        "",
        f"**Evaluation date:** {time.strftime('%Y-%m-%d')}",
        f"**Architecture:** maddpg_style_continuous_control",
        f"**Critic type:** stage_conditioned",
        f"**Data source:** Live LLM inference (no stubs).",
        "",
        "## Training status",
        "",
        "| Variant | Episodes | Env steps | Gradient updates | Trained? |",
        "|---|---:|---:|---:|---|",
    ]
    for variant, meta in metas.items():
        n_updates = meta.get("total_gradient_updates", 0)
        trained = "YES" if n_updates and n_updates > 0 else ("SKIPPED" if meta.get("skip") else "NO (random policy)")
        lines.append(
            f"| maddpg_{variant} | {hparams.episodes} | "
            f"{meta.get('total_env_steps', '?')} | {n_updates} | {trained} |"
        )
    lines += [
        "",
        f"**Hyperparameters:** batch_size={hparams.batch_size}, warmup_steps={hparams.warmup_steps}, "
        f"update_every={hparams.update_every}, gamma={hparams.gamma}, tau={hparams.tau}, "
        f"seed={hparams.seed}.",
        "",
    ]

    any_untrained = any(
        (m.get("total_gradient_updates") or 0) == 0 and not m.get("skip")
        for m in metas.values()
    )
    if any_untrained:
        lines += [
            "> **WARNING:** One or more variants performed zero gradient updates. Their metrics "
            "below reflect a *random* (untrained) policy, not a learned one. Re-run with more "
            "episodes, smaller warmup_steps, or smaller batch_size before drawing conclusions.",
            "",
        ]

    lines += [
        "## Per-policy aggregate metrics",
        "",
        "| Policy | Token F1 | ROUGE-L | Citation | Verif. Pass | Latency (s) | Failure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pol in pols:
        a = agg_all[pol]
        lines.append(
            f"| {pol} | {a.get('mean_token_f1', 0):.4f} | "
            f"{a.get('mean_rouge_l', 0):.4f} | {a.get('mean_citation_support', 0):.4f} | "
            f"{a.get('mean_verification_pass', 0):.4f} | "
            f"{a.get('mean_latency_seconds', 0):.2f} | {a.get('failure_rate', 0):.4f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- The MADDPG-style architecture here is *stage-conditioned*: the centralised critic "
        "is Q(state, active_agent, discrete_action_context, padded_continuous_action), not "
        "Q(state, joint_action). This matches the stage-gated nature of the RAG environment "
        "(only one agent acts per step).",
        "- Discrete action selection is *non-differentiable*: actors learn continuous RAG "
        "parameters; the discrete action is selected from the continuous params after each "
        "actor forward pass and treated as fixed execution context by the critic.",
        "- This is not a claim of production-readiness. It is a demonstration that the staged "
        "MADDPG-style training loop performs real gradient updates against a stage-conditioned "
        "critic and produces a deterministic continuous-control policy over the same masked "
        "action space as the discrete baseline.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [interp] Saved -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Live stage-conditioned MADDPG training + evaluation")
    p.add_argument("--episodes",         type=int,  default=20)
    p.add_argument("--n-eval",           type=int,  default=9)
    p.add_argument("--benchmark-train",  default="")
    p.add_argument("--benchmark-eval",   default="")
    p.add_argument("--checkpoint-every", type=int,  default=10)
    p.add_argument("--output-dir",       default="")
    p.add_argument("--skip-training",    action="store_true")
    p.add_argument("--eval-only-variant", default="both",
                   choices=["no_ceb", "ceb", "both"])

    # Hyperparameters (forwarded to TrainerConfig)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--warmup-steps", type=int,   default=50)
    p.add_argument("--update-every", type=int,   default=4)
    p.add_argument("--min-updates",  type=int,   default=1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--tau",          type=float, default=0.005)
    p.add_argument("--actor-lr",     type=float, default=1e-3)
    p.add_argument("--critic-lr",    type=float, default=1e-3)
    p.add_argument("--noise-sigma",  type=float, default=0.15)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--error-penalty",type=float, default=-1.0)
    p.add_argument("--hidden-dim",   type=int,   default=128)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg.DRY_RUN = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parent / "results" / "defense_comparison_live"
    )
    ckpt_dir    = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    traj_dir    = out_dir / "trajectories"
    for d in (ckpt_dir, metrics_dir, traj_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_bm_path = args.benchmark_train or str(
        Path(__file__).resolve().parent / "results" / "benchmark_splits" / "train.jsonl"
    )
    eval_bm_path = args.benchmark_eval or str(
        Path(__file__).resolve().parent / "results" / "benchmark_splits" / "test.jsonl"
    )
    train_bm = _load_benchmark(train_bm_path)
    eval_bm = _load_benchmark(eval_bm_path)
    if args.n_eval > 0:
        eval_bm = eval_bm[:args.n_eval]
    eval_qids = [q.get("question_id", f"Q{i+1:03d}") for i, q in enumerate(eval_bm)]

    print(f"\n[live_runner] episodes={args.episodes}  n_eval={len(eval_bm)}  device={device}")
    print(f"  train benchmark: {train_bm_path}  ({len(train_bm)} questions)")
    print(f"  eval  benchmark: {eval_bm_path}  ({len(eval_bm)} questions)")
    print(f"  output dir:      {out_dir}")

    all_results: Dict[str, List[Dict]] = {}
    metas: Dict[str, Dict[str, Any]] = {}

    ckpt_no_ceb = ckpt_dir / "best_maddpg_no_ceb_live.pt"
    ckpt_ceb    = ckpt_dir / "best_maddpg_ceb_live.pt"

    variants = ["no_ceb", "ceb"] if args.eval_only_variant == "both" else [args.eval_only_variant]

    # ── Training ──
    if not args.skip_training:
        for variant in variants:
            use_ceb = (variant == "ceb")
            ckpt_path, _n_updates, meta = train_variant(
                variant          = variant,
                use_ceb          = use_ceb,
                benchmark        = train_bm,
                episodes         = args.episodes,
                ckpt_dir         = ckpt_dir,
                metrics_dir      = metrics_dir,
                traj_dir         = traj_dir,
                device           = device,
                checkpoint_every = args.checkpoint_every,
                hparams          = args,
            )
            metas[variant] = meta
            if variant == "no_ceb": ckpt_no_ceb = ckpt_path
            else:                   ckpt_ceb    = ckpt_path
    else:
        print("[live_runner] --skip-training: using existing checkpoints.")
        for variant in variants:
            metas[variant] = {"skip": True, "total_gradient_updates": 0}

    meta_path = out_dir / "training_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "episodes":              args.episodes if not args.skip_training else None,
            "skip_training":         bool(args.skip_training),
            "metas":                 metas,
            "hyperparameters":       vars(args),
            "architecture":          "maddpg_style_continuous_control",
            "critic_type":           "stage_conditioned",
        }, f, indent=2, default=str)
    print(f"[meta] Saved -> {meta_path}")

    # ── Evaluation ──
    for variant in variants:
        use_ceb   = (variant == "ceb")
        ckpt_path = ckpt_ceb if use_ceb else ckpt_no_ceb
        results   = evaluate_variant(
            variant   = variant,
            use_ceb   = use_ceb,
            ckpt_path = ckpt_path,
            benchmark = eval_bm,
            n_eval    = len(eval_bm),
            out_dir   = out_dir,
            device    = device,
            hparams   = args,
        )
        all_results[f"maddpg_{variant}"] = results

    # ── Baseline ──
    eval_dir = Path(__file__).resolve().parent / "results" / "final_eval"
    baseline = _load_discrete_baseline(eval_dir, eval_qids)
    if baseline:
        all_results["discrete_marl"] = baseline
        with open(out_dir / "discrete_marl_baseline.jsonl", "w", encoding="utf-8") as f:
            for r in baseline:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Aggregate ──
    agg_all = {pol: _aggregate(rs) for pol, rs in all_results.items()}
    with open(out_dir / "aggregate_metrics.json", "w") as f:
        json.dump(agg_all, f, indent=2)

    # Save flat episode_metrics.csv across policies.
    flat: List[Dict] = []
    for pol, rs in all_results.items():
        for r in rs:
            row = {k: v for k, v in r.items() if k != "trace"}
            row["policy"] = pol
            flat.append(row)
    if flat:
        keys = list(flat[0].keys())
        with open(out_dir / "episode_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in flat:
                w.writerow(row)

    # Save comparison CSV.
    metric_keys = list(next(iter(agg_all.values())).keys()) if agg_all else []
    if agg_all:
        with open(out_dir / "comparison_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["policy"] + metric_keys)
            w.writeheader()
            for pol, agg in agg_all.items():
                w.writerow({"policy": pol, **agg})

    _print_table(agg_all)
    _write_interpretation(agg_all, out_dir / "results_interpretation.md",
                           metas=metas, hparams=args)
    print(f"\nAll outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
