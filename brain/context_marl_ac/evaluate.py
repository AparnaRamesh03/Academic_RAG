"""
brain/context_marl_ac/evaluate.py
---------------------------------
Evaluation for Supervisor-Guided Fully Free MARL RAG.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

import context_marl_ac.config as cfg
from context_marl_ac.marl.actors import build_marl_actors
from context_marl_ac.marl.centralized_critic import CentralizedCritic
from context_marl_ac.marl.checkpointing import MARLCheckpointManager
from context_marl_ac.marl.marl_env import MARLEnv
from context_marl_ac.marl.supervisor_actor import (
    build_supervisor_actor,
    SUPERVISOR_AGENTS,
    SUPERVISOR_ID_TO_AGENT,
)
from context_marl_ac.schemas.actions import AGENT_ACTIONS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Supervisor-Guided Fully Free MARL RAG")

    parser.add_argument("--checkpoint", type=str, default="best_reward.pt")
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=str(_BRAIN_ROOT.parent / "evaluation" / "standard_benchmark_v3.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="learned",
        choices=["learned", "random", "smoke"],
    )
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="supervisor_free",
        choices=["supervisor_free"],
    )
    parser.add_argument("--output-name", type=str, default="context_marl_ac_results.jsonl")

    return parser.parse_args()


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path, "r") as f:
        return json.load(f)


def select_smoke_agent(valid_agents: List[str], state: Any) -> str:
    """
    Deterministic debug policy. This is not used for learned results.
    """
    if state.generated_answer and "verifier" in valid_agents:
        return "verifier"

    if not state.retrieved_chunks:
        if "retriever" in valid_agents:
            return "retriever"
        if "rewriter" in valid_agents:
            return "rewriter"

    if state.retrieved_chunks and "grader" in valid_agents:
        return "grader"

    if state.selected_evidence and "generator" in valid_agents:
        return "generator"

    return valid_agents[0]


def select_smoke_action(agent: str, state: Any, valid_actions: List[str]) -> str:
    if agent == "retriever":
        if state.retrieved_chunks and "retrieve_more" in valid_actions:
            return "retrieve_more"
        if "hybrid_rerank" in valid_actions:
            return "hybrid_rerank"
        return valid_actions[0]

    if agent == "rewriter":
        if "keyword_rewrite" in valid_actions:
            return "keyword_rewrite"
        return valid_actions[0]

    if agent == "grader":
        if "loose_filter" in valid_actions:
            return "loose_filter"
        if "keep_all" in valid_actions:
            return "keep_all"
        return valid_actions[0]

    if agent == "generator":
        if "generate_with_strict_citations" in valid_actions:
            return "generate_with_strict_citations"
        if "generate_answer" in valid_actions:
            return "generate_answer"
        return valid_actions[0]

    if agent == "verifier":
        if state.generated_answer and not state.verification_result and "verify_answer" in valid_actions:
            return "verify_answer"
        if "request_more_retrieval" in valid_actions:
            return "request_more_retrieval"
        return valid_actions[0]

    return valid_actions[0]


def choose_learned_supervisor(supervisor, global_features, agent_mask):
    global_t = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(agent_mask, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = supervisor(global_t, mask_t)
        probs = torch.softmax(logits, dim=-1)
        agent_id = int(torch.argmax(probs, dim=-1).item())

    return SUPERVISOR_ID_TO_AGENT[agent_id], agent_id, float(probs[0, agent_id].item())


def choose_learned_action(actors, agent_name, obs, action_mask):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = actors[agent_name](obs_t, mask_t)
        probs = torch.softmax(logits, dim=-1)
        action_id = int(torch.argmax(probs, dim=-1).item())

    return AGENT_ACTIONS[agent_name][action_id], action_id, float(probs[0, action_id].item())


def evaluate():
    args = parse_args()
    cfg.DRY_RUN = args.dry_run

    env = MARLEnv()
    ckpt_manager = MARLCheckpointManager()

    actors = build_marl_actors()
    supervisor = build_supervisor_actor()
    critic = CentralizedCritic()

    if not args.dry_run and args.policy_mode == "learned":
        try:
            print(f"Loading checkpoint: {args.checkpoint}")
            ckpt_manager.load_checkpoint(
                actors=actors,
                supervisor=supervisor,
                critic=critic,
                filename=args.checkpoint,
                strict_supervisor=False,
            )
        except Exception as exc:
            print(f"Error loading checkpoint: {exc}. Falling back to smoke mode.")
            args.policy_mode = "smoke"

    benchmark = load_benchmark(args.benchmark_path)

    output_path = cfg.FINAL_EVAL_DIR / args.output_name
    os.makedirs(output_path.parent, exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    print(
        f"Starting evaluation "
        f"(mode={args.policy_mode}, routing={args.routing_mode}) "
        f"on {len(benchmark)} questions..."
    )

    for q_idx, question_dict in enumerate(tqdm(benchmark, desc="Evaluating")):
        state = env.reset(question_dict, index=q_idx + 1)
        trace = []
        done = False

        while not done:
            global_features = env.get_global_features()
            valid_agents = env.get_valid_agents()
            agent_mask = env.get_agent_mask()

            if not valid_agents or sum(agent_mask) == 0:
                if state.final_status == "pending":
                    state.final_status = "abstained"
                    state.done = True
                break

            if args.policy_mode == "smoke":
                agent_to_act = select_smoke_agent(valid_agents, state)
                supervisor_agent_id = SUPERVISOR_AGENTS.index(agent_to_act)
                supervisor_prob = 1.0

            elif args.policy_mode == "random":
                agent_to_act = random.choice(valid_agents)
                supervisor_agent_id = SUPERVISOR_AGENTS.index(agent_to_act)
                supervisor_prob = 1.0 / max(len(valid_agents), 1)

            else:
                agent_to_act, supervisor_agent_id, supervisor_prob = choose_learned_supervisor(
                    supervisor,
                    global_features,
                    agent_mask,
                )

                if agent_to_act not in valid_agents:
                    agent_to_act = valid_agents[0]
                    supervisor_agent_id = SUPERVISOR_AGENTS.index(agent_to_act)

            action_mask = env.get_mask(agent_to_act)
            valid_actions = [
                AGENT_ACTIONS[agent_to_act][i]
                for i, m in enumerate(action_mask)
                if m == 1
            ]

            if not valid_actions or sum(action_mask) == 0:
                state.final_status = "error"
                state.done = True
                break

            if args.policy_mode == "smoke":
                action_to_take = select_smoke_action(agent_to_act, state, valid_actions)
                action_id = AGENT_ACTIONS[agent_to_act].index(action_to_take)
                action_prob = 1.0

            elif args.policy_mode == "random":
                action_to_take = random.choice(valid_actions)
                action_id = AGENT_ACTIONS[agent_to_act].index(action_to_take)
                action_prob = 1.0 / max(len(valid_actions), 1)

            else:
                obs = env.get_obs(agent_to_act)
                action_to_take, action_id, action_prob = choose_learned_action(
                    actors,
                    agent_to_act,
                    obs,
                    action_mask,
                )

                if action_to_take not in valid_actions:
                    action_to_take = valid_actions[0]
                    action_id = AGENT_ACTIONS[agent_to_act].index(action_to_take)

            new_state, reward, done, info = env.step(agent_to_act, action_to_take)

            trace.append(
                {
                    "step": state.num_steps,
                    "valid_agents": valid_agents,
                    "agent_mask": agent_mask,
                    "supervisor_selected_agent": agent_to_act,
                    "supervisor_agent_id": supervisor_agent_id,
                    "supervisor_probability": supervisor_prob,
                    "agent": agent_to_act,
                    "valid_actions": valid_actions,
                    "action_mask": action_mask,
                    "action": action_to_take,
                    "action_id": action_id,
                    "action_probability": action_prob,
                    "reward": reward,
                    "status": new_state.final_status,
                    "done": done,
                    **new_state.to_debug_dict(),
                }
            )

            state = new_state

        final_result = {
            "question_id": state.question_id,
            "question": question_dict.get("question", ""),
            "ground_truth": question_dict.get("ground_truth", ""),
            "category": question_dict.get("category"),
            "difficulty": question_dict.get("difficulty"),
            "source_file": question_dict.get("source_file"),
            "architecture": cfg.ARCHITECTURE_NAME,
            "routing_mode": args.routing_mode,
            "policy_mode": args.policy_mode,
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
            "trace": trace,
            **state.to_debug_dict(),
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(final_result, ensure_ascii=False) + "\n")

    print(f"\nEvaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    evaluate()