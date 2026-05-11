"""
brain/context_marl_ac/evaluate.py
----------------------------------
Deterministic evaluation script for Context-Engineered MARL RAG.
Supports learned policy, random policy, and fixed smoke-test policy modes.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

# ── sys.path setup ────────────────────────────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
import context_marl_ac.config as cfg
from context_marl_ac.marl.marl_env import MARLEnv
from context_marl_ac.marl.actors import build_marl_actors
from context_marl_ac.marl.centralized_critic import CentralizedCritic
from context_marl_ac.marl.checkpointing import MARLCheckpointManager
from context_marl_ac.schemas.actions import AGENT_ACTIONS, AGENT_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Context-Engineered MARL RAG")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_reward.pt",
        help="Checkpoint filename in checkpoints dir",
    )

    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=str(_BRAIN_ROOT.parent / "brain" / "context_marl_ac" / "results" / "benchmark_splits" / "test.jsonl"),
        help="Path to evaluation benchmark JSON or JSONL",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in mock/dry-run mode",
    )

    parser.add_argument(
        "--policy-mode",
        type=str,
        default="learned",
        choices=["learned", "random", "smoke"],
        help="Action selection policy: learned, random, or smoke",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="context_marl_ac_results.jsonl",
        help="Output filename",
    )

    return parser.parse_args()


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    """
    Load JSON or JSONL benchmark file.
    """
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path, "r") as f:
        return json.load(f)


def select_smoke_action(agent: str, state: Any, valid_actions: List[str]) -> str:
    """
    Fixed deterministic policy for smoke testing.

    Smoke path:
        retriever -> grader -> generator -> verifier.verify_answer
    """
    if not valid_actions:
        raise ValueError(f"No valid actions available for agent={agent}")

    if agent == "retriever":
        if "hybrid_rerank" in valid_actions:
            return "hybrid_rerank"
        if "hybrid_retrieve" in valid_actions:
            return "hybrid_retrieve"
        return valid_actions[0]

    if agent == "grader":
        if "loose_filter" in valid_actions:
            return "loose_filter"
        if "keep_all" in valid_actions:
            return "keep_all"
        return valid_actions[0]

    if agent == "generator":
        if state.selected_evidence:
            if "generate_with_strict_citations" in valid_actions:
                return "generate_with_strict_citations"
            if "generate_answer" in valid_actions:
                return "generate_answer"
            if "generate_short_answer" in valid_actions:
                return "generate_short_answer"

        if "abstain_request_more_evidence" in valid_actions:
            return "abstain_request_more_evidence"

        return valid_actions[0]

    if agent == "verifier":
        if state.generated_answer and "verify_answer" in valid_actions:
            return "verify_answer"
        return valid_actions[0]

    if agent == "rewriter":
        # Rewriter should only appear in recovery mode now.
        if "keyword_rewrite" in valid_actions:
            return "keyword_rewrite"
        if "no_rewrite" in valid_actions:
            return "no_rewrite"
        return valid_actions[0]

    return valid_actions[0]


def select_action(
    args,
    agent_name: str,
    state: Any,
    env: MARLEnv,
    actors,
) -> str:
    """
    Select one valid action according to policy mode.
    """
    mask = env.get_mask(agent_name)
    valid_actions = [
        AGENT_ACTIONS[agent_name][i]
        for i, m in enumerate(mask)
        if m == 1
    ]

    if not valid_actions:
        raise ValueError(f"No valid actions for agent={agent_name}")

    if args.policy_mode == "smoke":
        return select_smoke_action(agent_name, state, valid_actions)

    if args.policy_mode == "random":
        return random.choice(valid_actions)

    # learned mode: argmax over masked actor logits
    obs = torch.tensor(env.get_obs(agent_name), dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)

    with torch.no_grad():
        logits = actors[agent_name](obs.unsqueeze(0), mask_t.unsqueeze(0))
        action_id = torch.argmax(logits, dim=-1).item()

    return AGENT_ACTIONS[agent_name][action_id]


def build_final_result(
    state: Any,
    question_dict: Dict[str, Any],
    trace: List[Dict[str, Any]],
    policy_mode: str,
) -> Dict[str, Any]:
    """
    Build one JSONL result row from final ContextState.
    """
    return {
        "question_id": state.question_id,
        "question": question_dict.get("question", ""),
        "ground_truth": question_dict.get("ground_truth", ""),
        "category": question_dict.get("category"),
        "difficulty": question_dict.get("difficulty"),
        "source_file": question_dict.get("source_file"),
        "architecture": cfg.ARCHITECTURE_NAME,

        "final_status": state.final_status,
        "final_answer": state.generated_answer,
        "answer": state.generated_answer,
        "generated_answer_length": len(state.generated_answer) if state.generated_answer else 0,

        "selected_evidence_count": len(state.selected_evidence),
        "selected_evidence": state.selected_evidence,
        "retrieved_chunks": state.retrieved_chunks,
        "citations": state.citation_candidates,

        "citation_support_rate": state.citation_support_rate,
        "verification_pass": state.final_status == "accepted",
        "verifier_decision": state.verification_result.get("decision", "N/A"),
        "verifier_reason": state.verification_result.get("reason", ""),
        "verification_result": state.verification_result,
        "unsupported_claims": state.unsupported_claims,

        "latency_seconds": state.latency_so_far,
        "latency_sec": state.latency_so_far,
        "num_steps": state.num_steps,
        "num_llm_calls": state.num_llm_calls,
        "token_usage": state.token_usage,

        "policy_mode": policy_mode,
        "trace": trace,

        **state.to_debug_dict(),
    }


def evaluate():
    args = parse_args()
    cfg.DRY_RUN = args.dry_run

    # 1. Initialize infrastructure
    env = MARLEnv()
    ckpt_manager = MARLCheckpointManager()

    actors = build_marl_actors()
    critic = CentralizedCritic()

    # 2. Load checkpoint when using learned policy
    if not args.dry_run and args.policy_mode == "learned":
        try:
            print(f"Loading checkpoint: {args.checkpoint}")
            ckpt_manager.load_checkpoint(
                actors=actors,
                critic=critic,
                filename=args.checkpoint,
            )
        except Exception as exc:
            print(f"Error loading checkpoint: {exc}. Falling back to policy-mode: smoke.")
            args.policy_mode = "smoke"

    # 3. Load benchmark
    benchmark = load_benchmark(args.benchmark_path)

    output_path = cfg.FINAL_EVAL_DIR / args.output_name
    os.makedirs(output_path.parent, exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Starting evaluation (mode={args.policy_mode}) on {len(benchmark)} questions...")

    # 4. Evaluation loop
    for q_idx, question_dict in enumerate(tqdm(benchmark, desc="Evaluating")):
        state = env.reset(question_dict, index=q_idx + 1)
        trace: List[Dict[str, Any]] = []

        done = False

        while not done:
            agent_to_act = None
            action_to_take = None

            # Pick the first agent with a valid action.
            # The mask controls the stage flow.
            for agent_name in AGENT_NAMES:
                mask = env.get_mask(agent_name)
                if sum(mask) > 0:
                    agent_to_act = agent_name
                    try:
                        action_to_take = select_action(
                            args=args,
                            agent_name=agent_name,
                            state=state,
                            env=env,
                            actors=actors,
                        )
                    except Exception as exc:
                        state.final_status = "error"
                        state.done = True
                        state.update_latency()
                        state.verification_result = {
                            "decision": "FAIL",
                            "reason": f"Action selection failed: {type(exc).__name__}: {exc}",
                            "verified_claims": [],
                        }

                        trace.append({
                            "step": state.num_steps,
                            "agent": agent_name,
                            "action": None,
                            "status": state.final_status,
                            "error": f"{type(exc).__name__}: {exc}",
                            **state.to_debug_dict(),
                        })

                        done = True
                    break

            if done:
                break

            # No valid agent means the episode got stuck.
            if not agent_to_act:
                if state.final_status == "pending":
                    state.final_status = "abstained"
                state.done = True
                state.update_latency()

                trace.append({
                    "step": state.num_steps,
                    "agent": None,
                    "action": None,
                    "status": state.final_status,
                    "error": "No valid agent/action available.",
                    **state.to_debug_dict(),
                })

                done = True
                break

            # Execute the selected step.
            try:
                new_state, reward, done, info = env.step(agent_to_act, action_to_take)

            except Exception as exc:
                state.final_status = "error"
                state.done = True
                state.update_latency()
                state.verification_result = {
                    "decision": "FAIL",
                    "reason": f"Evaluation step failed: {type(exc).__name__}: {exc}",
                    "verified_claims": [],
                }

                trace.append({
                    "step": state.num_steps,
                    "agent": agent_to_act,
                    "action": action_to_take,
                    "status": state.final_status,
                    "error": f"{type(exc).__name__}: {exc}",
                    **state.to_debug_dict(),
                })

                done = True
                break

            # Record successful step.
            step_data = {
                "step": new_state.num_steps,
                "agent": agent_to_act,
                "action": action_to_take,
                "reward": reward,
                "status": new_state.final_status,
                "done": done,
                **new_state.to_debug_dict(),
            }

            trace.append(step_data)
            state = new_state

        # 5. Write result row
        final_result = build_final_result(
            state=state,
            question_dict=question_dict,
            trace=trace,
            policy_mode=args.policy_mode,
        )

        with open(output_path, "a") as f:
            f.write(json.dumps(final_result, ensure_ascii=False) + "\n")

    print(f"\nEvaluation Complete. Results saved to {output_path}")


if __name__ == "__main__":
    evaluate()