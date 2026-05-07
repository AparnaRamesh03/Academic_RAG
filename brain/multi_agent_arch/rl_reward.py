"""
rl_reward.py
------------
Terminal reward function for the multi-agent RAG contextual bandit.

The reward is computed once at the end of each episode (inside finish_step)
and then propagated backwards to every (state_key, action) transition that
was recorded during the episode.

Reward range: [-0.3, 1.0]

Design principles
-----------------
- Highest reward for a grounded answer reached with zero retries (efficient).
- Graduated penalties for extra rewrite / verification rounds.
- Negative rewards for hard failure modes (loops, max-step exhaustion).
- Small efficiency adjustment that penalises unnecessarily long episodes.
"""
from __future__ import annotations

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent

for path in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from state_shared import GraphState


def compute_episode_reward(final_state: GraphState) -> float:
    """
    Return a scalar reward in [-0.3, 1.0] that reflects the quality and
    efficiency of the completed episode.
    """
    stop_reason = str(final_state.get("stop_reason", "") or "")
    citations_pass = bool(final_state.get("citations_pass", False))
    verification_outcome = str(final_state.get("verification_outcome", "") or "")
    crag_retries = int(final_state.get("crag_retries", 0) or 0)
    verify_retries = int(final_state.get("verify_retries", 0) or 0)
    step_count = int(final_state.get("step_count", 0) or 0)

    # ── Base reward ───────────────────────────────────────────────────────────
    if stop_reason == "grounded_answer_ready":
        if verify_retries == 0 and crag_retries == 0:
            # Perfect: first-pass grounded answer.
            base = 1.0
        elif verify_retries == 0:
            # Needed a query rewrite but verified cleanly.
            base = 0.85
        else:
            # Required verification retries — still acceptable.
            base = max(0.60, 0.85 - 0.10 * verify_retries)

    elif citations_pass and verification_outcome == "grounded_incomplete":
        # Honest partial answer accepted by the verifier.
        base = 0.50

    elif citations_pass:
        # Citations passed but full grounding not confirmed.
        base = 0.30

    elif stop_reason == "audit_retry_limit_reached":
        base = -0.10

    elif stop_reason in {"agent_loop_detected", "max_steps_reached"}:
        base = -0.20

    else:
        # no_retrieval_progress, supervisor_stopped, etc.
        base = 0.0

    # ── Efficiency adjustment ─────────────────────────────────────────────────
    # A short episode that still achieves a good outcome is preferable.
    # Penalise long episodes to encourage the policy to find direct paths.
    efficiency = max(-0.15, 0.05 - 0.01 * step_count)

    reward = round(max(-0.30, min(1.0, base + efficiency)), 4)

    print(
        f"[RL] Episode reward: {reward:.4f}  "
        f"(stop={stop_reason}, citations_pass={citations_pass}, "
        f"crag_retries={crag_retries}, verify_retries={verify_retries}, "
        f"steps={step_count})"
    )
    return reward
