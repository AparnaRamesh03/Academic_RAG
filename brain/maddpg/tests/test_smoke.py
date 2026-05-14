"""
brain/maddpg/tests/test_smoke.py
---------------------------------
Smoke checks for the stage-conditioned MADDPG-style architecture.

Run from brain/:
    python -m maddpg.tests.test_smoke

Each check prints PASS/FAIL and the overall script exits non-zero if any
check fails. Designed to be runnable without pytest.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch

# ── sys.path: ensure brain/ is importable ────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

from maddpg.maddpg_critic import StageConditionedCritic
from maddpg.replay_buffer import ReplayBuffer, Transition
from maddpg.stage_utils import (
    AGENT_TO_ID,
    MAX_CONTINUOUS_ACTION_DIM,
    MAX_DISCRETE_ACTIONS,
    NUM_AGENTS,
    agent_one_hot,
    discrete_action_one_hot,
    pad_continuous_action,
)


_FAILED: List[Tuple[str, str]] = []


def _check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        _FAILED.append((name, traceback.format_exc()))


# ── 1. Critic shape test ──────────────────────────────────────────────────────

def test_critic_shape() -> None:
    state_dim = 20
    critic = StageConditionedCritic(
        state_dim                 = state_dim,
        num_agents                = NUM_AGENTS,
        max_discrete_actions      = MAX_DISCRETE_ACTIONS,
        max_continuous_action_dim = MAX_CONTINUOUS_ACTION_DIM,
        hidden_dim                = 64,
    )
    batch = 7
    s   = torch.randn(batch, state_dim)
    a   = torch.zeros(batch, NUM_AGENTS); a[:, 0] = 1.0
    d   = torch.zeros(batch, MAX_DISCRETE_ACTIONS); d[:, 0] = 1.0
    act = torch.randn(batch, MAX_CONTINUOUS_ACTION_DIM).tanh()
    out = critic(s, a, d, act)
    assert out.shape == (batch, 1), f"expected (batch,1) got {tuple(out.shape)}"


# ── 2. Replay transition test ────────────────────────────────────────────────

def test_replay_transition_fields() -> None:
    state = np.random.randn(14).astype(np.float32)
    raw = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    t = Transition(
        state_features          = state,
        active_agent            = "retriever",
        active_agent_id         = AGENT_TO_ID["retriever"],
        valid_actions           = ["dense_retrieve", "hybrid_rerank"],
        raw_action              = raw,
        action_padded           = pad_continuous_action("retriever", raw),
        mapped_params           = {"top_k": 10},
        discrete_action         = "hybrid_rerank",
        discrete_action_one_hot = discrete_action_one_hot("retriever", "hybrid_rerank"),
        reward                  = 0.5,
        next_state_features     = state,
        next_active_agent       = "grader",
        next_active_agent_id    = AGENT_TO_ID["grader"],
        next_valid_actions      = ["medium_filter"],
        done                    = False,
        question_id             = "q1",
        step                    = 1,
        final_status            = "pending",
    )
    buf = ReplayBuffer(capacity=10)
    buf.push(t)
    buf.push(t)
    sample = buf.sample(2)
    assert len(sample) == 2
    assert sample[0].active_agent == "retriever"
    assert sample[0].next_active_agent == "grader"
    assert sample[0].action_padded.shape == (MAX_CONTINUOUS_ACTION_DIM,)
    assert sample[0].discrete_action_one_hot.shape == (MAX_DISCRETE_ACTIONS,)

    # Round-trip via JSON.
    d = t.to_dict()
    t2 = Transition.from_dict(d)
    assert t2.active_agent == "retriever"
    assert np.allclose(t2.state_features, t.state_features)


# ── 3. Dry-run training update test ──────────────────────────────────────────

def test_dry_run_training() -> None:
    import context_marl_ac.config as cfg
    cfg.DRY_RUN = True

    from maddpg.train_maddpg import train as train_fn
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rc = train_fn([
            "--train-mode", "dry",
            "--episodes",   "3",
            "--batch-size", "2",
            "--warmup-steps", "2",
            "--update-every", "1",
            "--min-updates",  "1",
            "--run-name",     "smoke_stage_conditioned",
            "--results-dir",  tmp,
            "--seed",         "0",
            "--hidden-dim",   "32",
        ])
        assert rc == 0, f"train returned nonzero: {rc}"

        import json
        agg_path = Path(tmp) / "aggregate_metrics_smoke_stage_conditioned.json"
        assert agg_path.exists(), f"aggregate metrics missing: {agg_path}"
        agg = json.loads(agg_path.read_text())
        assert agg["trained"] is True, f"trained=False; updates={agg['total_gradient_updates']}"
        assert agg["total_gradient_updates"] >= 1
        assert agg["critic_type"] == "stage_conditioned"


# ── 4. Exception transition test ─────────────────────────────────────────────

def test_error_transition_push() -> None:
    from maddpg.trainer import StageConditionedMADDPGTrainer, TrainerConfig
    tcfg = TrainerConfig(state_dim=14, hidden_dim=16, device="cpu")
    trainer = StageConditionedMADDPGTrainer(tcfg)
    state = np.random.randn(14).astype(np.float32)
    trainer.push_error_transition(
        state_features  = state,
        active_agent    = "retriever",
        valid_actions   = ["dense_retrieve"],
        raw_action      = None,
        mapped_params   = None,
        discrete_action = None,
        question_id     = "q_err",
        step            = 0,
        error_message   = "stub failure",
    )
    assert len(trainer.buffer) == 1
    sampled = trainer.buffer.sample(1)[0]
    assert sampled.done is True
    assert sampled.reward == tcfg.error_penalty
    assert sampled.final_status == "error"
    assert sampled.next_active_agent is None


# ── 5. Evaluation checkpoint test ────────────────────────────────────────────

def test_evaluation_requires_checkpoint() -> None:
    import context_marl_ac.config as cfg
    cfg.DRY_RUN = True
    from maddpg.evaluate_maddpg import evaluate
    # No checkpoint, no --allow-random-policy → must return nonzero.
    rc = evaluate([
        "--mode", "maddpg_no_ceb",
        "--dry-run",
        "--n-questions", "1",
    ])
    assert rc != 0, "evaluate should refuse to run without a checkpoint"


# ── 6. Param wiring test ─────────────────────────────────────────────────────

def test_param_mapper_ranges() -> None:
    from maddpg.continuous_action_mapper import map_agent_params
    raw = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p = map_agent_params("retriever", raw)
    assert 5 <= p["top_k"] <= 30
    assert 0.0 <= p["source_diversity"] <= 1.0
    p = map_agent_params("generator", raw)
    assert 128 <= p["max_tokens"] <= 1024
    p = map_agent_params("verifier", raw)
    assert "support_threshold" in p
    # Removed params must not be present anymore.
    assert "confidence_threshold" not in map_agent_params("verifier", raw)


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("  Stage-conditioned MADDPG smoke checks")
    print("=" * 60)

    _check("critic shape (StageConditionedCritic forward)", test_critic_shape)
    _check("replay buffer Transition has new schema",       test_replay_transition_fields)
    _check("error/exception transition is pushed",          test_error_transition_push)
    _check("evaluate refuses missing checkpoint",           test_evaluation_requires_checkpoint)
    _check("param mapper produces valid ranges",            test_param_mapper_ranges)
    _check("dry-run training performs >=1 gradient update", test_dry_run_training)

    print()
    if _FAILED:
        print(f"  {len(_FAILED)} check(s) FAILED.")
        for name, tb in _FAILED:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    print("  All smoke checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
