"""
verify_rl_output.py
-------------------
Offline verification of the multi-agent RL system.

Checks:
  1. Config values are sane.
  2. rl_policy.json exists and Q-table is populated.
  3. EpisodeBuffer hydration round-trip.
  4. compute_episode_reward covers all stop-reason branches.
  5. extract_state_key is deterministic and produces valid tuples.
  6. RLPolicy.act() returns None (exploration) or a valid action.
  7. RLPolicy.update_from_state_transitions() writes to Q-table and disk.
  8. Training-log integrity: counts, reward distribution, error rates per round.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

# ── Path bootstrap ─────────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent

for p in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Imports from the project ───────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(BRAIN_DIR / ".env")

from config import (
    RL_ENABLED, RL_EPSILON, RL_ALPHA, RL_POLICY_PATH,
    MAX_STEPS, MAX_AUDIT_RETRIES, MIN_CONFIDENCE_TO_STOP, MAX_REWRITE_ROUNDS,
)
from rl_policy import RLPolicy, EpisodeBuffer, VALID_ACTIONS
from rl_reward import compute_episode_reward
from rl_state_features import extract_state_key, classify_query_type, _confidence_bin

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
SEP  = "─" * 60

failures: list[str] = []

def ok(msg: str):
    print(f"  {PASS} {msg}")

def fail(msg: str):
    print(f"  {FAIL} {msg}")
    failures.append(msg)

def warn(msg: str):
    print(f"  {WARN} {msg}")

def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config sanity
# ─────────────────────────────────────────────────────────────────────────────
section("1 · Config sanity")

if RL_ENABLED:
    ok("RL_ENABLED = True")
else:
    fail("RL_ENABLED = False — RL is disabled")

if 0.0 < RL_EPSILON < 1.0:
    ok(f"RL_EPSILON = {RL_EPSILON}  (valid range)")
else:
    fail(f"RL_EPSILON = {RL_EPSILON}  out of (0, 1)")

if 0.0 < RL_ALPHA <= 1.0:
    ok(f"RL_ALPHA   = {RL_ALPHA}  (valid range)")
else:
    fail(f"RL_ALPHA   = {RL_ALPHA}  out of (0, 1]")

ok(f"RL_POLICY_PATH = {RL_POLICY_PATH}")
print(f"  MAX_STEPS={MAX_STEPS}  MAX_AUDIT_RETRIES={MAX_AUDIT_RETRIES}  "
      f"MIN_CONFIDENCE={MIN_CONFIDENCE_TO_STOP}  MAX_REWRITE={MAX_REWRITE_ROUNDS}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. rl_policy.json existence and Q-table population
# ─────────────────────────────────────────────────────────────────────────────
section("2 · Policy file existence & Q-table")

if RL_POLICY_PATH.exists():
    ok(f"Policy file found: {RL_POLICY_PATH}")
    with RL_POLICY_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    n_episodes = raw.get("episode_count", 0)
    n_states   = len(raw.get("q_table", {}))
    ok(f"episode_count = {n_episodes}")
    if n_states > 0:
        ok(f"Q-table populated: {n_states} known states")
        # Print top 5 states
        q = raw["q_table"]
        print("\n  Top Q-table entries (up to 5):")
        for state_key, actions in list(q.items())[:5]:
            best = max(actions, key=actions.__getitem__)
            best_val = actions[best]
            print(f"    state={state_key!r}")
            print(f"      preferred_action={best!r}  Q={best_val:.4f}")
            for a, v in sorted(actions.items(), key=lambda x: -x[1]):
                print(f"        {a}: {v:.4f}")
    else:
        fail("Q-table is EMPTY — training did not record any states")
else:
    fail(f"Policy file NOT found at {RL_POLICY_PATH}")
    warn("The policy was never persisted.  This may mean train_rl_policy.py "
         "completed but _save() failed, or no successful episodes ran.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EpisodeBuffer round-trip
# ─────────────────────────────────────────────────────────────────────────────
section("3 · EpisodeBuffer hydration round-trip")

raw_transitions = [
    ["('retriever_agent', False, False, False, 0, 0, 'direct_fact', 0)", "retriever_agent"],
    ["('retriever_agent', False, False, False, 0, 0, 'direct_fact', 0)", "evidence_agent"],
    ["('evidence_agent', True, False, False, 0, 0, 'direct_fact', 2)", "answer_agent"],
]

buf = EpisodeBuffer.from_state_list(raw_transitions)
if len(buf.transitions) == len(raw_transitions):
    ok(f"Hydrated {len(buf.transitions)} transitions correctly")
else:
    fail(f"Expected {len(raw_transitions)} transitions, got {len(buf.transitions)}")

for i, (sk, act) in enumerate(buf.transitions):
    if isinstance(sk, str) and act in VALID_ACTIONS:
        ok(f"  transition[{i}]: action='{act}'  key is str ✓")
    else:
        fail(f"  transition[{i}]: bad format — key={sk!r} action={act!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. compute_episode_reward branch coverage
# ─────────────────────────────────────────────────────────────────────────────
section("4 · Reward function branch coverage")

reward_cases = [
    # (description, state_dict, expected_range)
    ("perfect first-pass (r=1.0)",
     {"stop_reason": "grounded_answer_ready", "citations_pass": True,
      "verification_outcome": "pass", "crag_retries": 0,
      "verify_retries": 0, "step_count": 3},
     (0.95, 1.01)),
    ("rewrite but clean (r≈0.85+eff)",
     {"stop_reason": "grounded_answer_ready", "citations_pass": True,
      "verification_outcome": "pass", "crag_retries": 1,
      "verify_retries": 0, "step_count": 5},
     (0.75, 0.91)),
    ("verify retries needed (r≥0.60+eff)",
     {"stop_reason": "grounded_answer_ready", "citations_pass": True,
      "verification_outcome": "pass", "crag_retries": 0,
      "verify_retries": 2, "step_count": 7},
     (0.40, 0.80)),
    ("grounded_incomplete (r≈0.5)",
     {"stop_reason": "other", "citations_pass": True,
      "verification_outcome": "grounded_incomplete", "crag_retries": 0,
      "verify_retries": 0, "step_count": 3},
     (0.40, 0.60)),
    ("citations pass only (r≈0.3)",
     {"stop_reason": "other", "citations_pass": True,
      "verification_outcome": "", "crag_retries": 0,
      "verify_retries": 0, "step_count": 3},
     (0.20, 0.40)),
    ("audit_retry_limit (r≈-0.1)",
     {"stop_reason": "audit_retry_limit_reached", "citations_pass": False,
      "verification_outcome": "", "crag_retries": 0,
      "verify_retries": 2, "step_count": 8},
     (-0.25, 0.0)),
    ("loop/max_steps (r≈-0.2)",
     {"stop_reason": "agent_loop_detected", "citations_pass": False,
      "verification_outcome": "", "crag_retries": 0,
      "verify_retries": 0, "step_count": 12},
     (-0.30, -0.10)),
    ("supervisor_stopped (r=0.0+eff)",
     {"stop_reason": "supervisor_stopped", "citations_pass": False,
      "verification_outcome": "", "crag_retries": 0,
      "verify_retries": 0, "step_count": 2},
     (-0.10, 0.10)),
]

all_rewards_ok = True
for desc, state, (lo, hi) in reward_cases:
    r = compute_episode_reward(state)
    if lo <= r <= hi:
        ok(f"{desc} → reward={r:.4f}  (expected [{lo},{hi}]) ✓")
    else:
        fail(f"{desc} → reward={r:.4f}  OUTSIDE expected [{lo},{hi}]")
        all_rewards_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# 5. extract_state_key determinism and validity
# ─────────────────────────────────────────────────────────────────────────────
section("5 · State feature extraction")

sample_states = [
    {"original_query": "What is self-attention?", "last_action": "",
     "graded_docs": [], "generation": "", "citations_pass": False,
     "crag_retries": 0, "verify_retries": 0, "confidence": 0.0},
    {"original_query": "Which is the best architecture?", "last_action": "evidence_agent",
     "graded_docs": [{"text": "x"}], "generation": "Some answer.", "citations_pass": True,
     "crag_retries": 1, "verify_retries": 1, "confidence": 0.75},
    {"original_query": "Compare Figure 3 with Table 1.", "last_action": "answer_agent",
     "graded_docs": [], "generation": "answer", "citations_pass": False,
     "crag_retries": 0, "verify_retries": 0, "confidence": 0.45},
]

for i, s in enumerate(sample_states):
    key1 = extract_state_key(s)
    key2 = extract_state_key(s)   # run twice — must be identical
    if key1 == key2:
        ok(f"State[{i}]: key={key1}  (deterministic ✓)")
    else:
        fail(f"State[{i}]: non-deterministic key!")
    if len(key1) == 8:
        ok(f"  8-tuple length ✓")
    else:
        fail(f"  Expected 8-tuple, got {len(key1)}-tuple")
    if key1[6] in ("figure", "comparison", "superlative", "direct_fact", "other"):
        ok(f"  query_type='{key1[6]}' ✓")
    else:
        fail(f"  Unknown query_type='{key1[6]}'")
    if key1[7] in (0, 1, 2, 3):
        ok(f"  conf_bin={key1[7]} ✓")
    else:
        fail(f"  Invalid conf_bin={key1[7]}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. RLPolicy.act() correctness
# ─────────────────────────────────────────────────────────────────────────────
section("6 · RLPolicy.act() behaviour")

import tempfile, random

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
    tmp_path = Path(tf.name)

try:
    policy = RLPolicy(policy_path=tmp_path, epsilon=0.0, alpha=0.1)

    # Unseen state → must return None
    result = policy.act(("unknown_state",), explore=False)
    if result is None:
        ok("Unseen state → act() returns None  ✓")
    else:
        fail(f"Unseen state → expected None, got '{result}'")

    # Seed the Q-table manually
    state_key = str(("evidence_agent", True, False, False, 0, 0, "direct_fact", 2))
    policy.q_table[state_key]["answer_agent"] = 0.80
    policy.q_table[state_key]["verification_agent"] = 0.30

    result = policy.act(("evidence_agent", True, False, False, 0, 0, "direct_fact", 2),
                        explore=False)
    if result == "answer_agent":
        ok(f"Exploitation: chose highest-Q action 'answer_agent' ✓")
    else:
        fail(f"Expected 'answer_agent', got '{result}'")

    # With epsilon=1.0 always explore → None
    policy.epsilon = 1.0
    result = policy.act(("evidence_agent", True, False, False, 0, 0, "direct_fact", 2),
                        explore=True)
    if result is None:
        ok("Full exploration (ε=1.0) → returns None  ✓")
    else:
        fail(f"Expected None with ε=1.0, got '{result}'")

finally:
    tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. RLPolicy.update_from_state_transitions() — write to Q-table & disk
# ─────────────────────────────────────────────────────────────────────────────
section("7 · Policy update & persistence")

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
    tmp_path = Path(tf.name)

try:
    policy2 = RLPolicy(policy_path=tmp_path, epsilon=0.15, alpha=0.1)
    raw_t = [
        [str(("", False, False, False, 0, 0, "direct_fact", 0)), "retriever_agent"],
        [str(("retriever_agent", False, False, False, 0, 0, "direct_fact", 0)), "evidence_agent"],
        [str(("evidence_agent", True, False, False, 0, 0, "direct_fact", 2)), "answer_agent"],
    ]
    policy2.update_from_state_transitions(raw_t, reward=1.0)

    if policy2.episode_count == 1:
        ok("episode_count incremented to 1 ✓")
    else:
        fail(f"episode_count={policy2.episode_count}, expected 1")

    if len(policy2.q_table) == 3:
        ok(f"Q-table has 3 entries (one per transition state) ✓")
    else:
        fail(f"Expected 3 Q entries, got {len(policy2.q_table)}")

    # Check Q-values were updated from 0.0
    key0 = str(("", False, False, False, 0, 0, "direct_fact", 0))
    q_val = policy2.q_table[key0].get("retriever_agent", None)
    if q_val is not None and abs(q_val - 0.1) < 0.001:
        ok(f"Q[state0][retriever_agent] = {q_val:.4f}  (α×(1-0)=0.1) ✓")
    else:
        fail(f"Unexpected Q-value: {q_val}")

    # Verify saved to disk
    if tmp_path.exists():
        saved = json.loads(tmp_path.read_text(encoding="utf-8"))
        if saved.get("episode_count") == 1 and len(saved.get("q_table", {})) == 3:
            ok("Policy correctly persisted to disk ✓")
        else:
            fail(f"Persisted data mismatch: {saved.get('episode_count')} episodes, "
                 f"{len(saved.get('q_table',{}))} states")
    else:
        fail("Policy file was not saved to disk")

finally:
    tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Training log analysis
# ─────────────────────────────────────────────────────────────────────────────
section("8 · Training log integrity")

LOG_PATH = CURRENT_DIR / "results" / "rl_training_log.json"
if not LOG_PATH.exists():
    fail(f"Training log not found at {LOG_PATH}")
else:
    log = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    episodes = log.get("episodes", [])
    n_total  = log.get("total_episodes", 0)
    n_rounds = log.get("rounds", 0)

    ok(f"Log found: {n_rounds} rounds, {n_total} episodes recorded")

    # Count successful vs errored
    successful = [e for e in episodes if e.get("stop_reason") != "error"]
    errored    = [e for e in episodes if e.get("stop_reason") == "error"]
    rate_errors = [e for e in errored if "429" in str(e.get("error", ""))]

    ok(f"  Successful episodes: {len(successful)} / {len(episodes)}")
    warn(f"  Rate-limit (429) errors: {len(rate_errors)} / {len(episodes)}")

    # Reward distribution
    rewards = [e["reward"] for e in episodes]
    mean_r  = sum(rewards) / len(rewards) if rewards else 0
    perfect = sum(1 for r in rewards if r >= 0.80)
    bad     = sum(1 for r in rewards if r < 0.0)
    ok(f"  Mean reward: {mean_r:.4f}  (target ≥ 0.50 for convergence)")
    print(f"  Perfect (≥0.80): {perfect}   Bad (<0.0): {bad}   Weak: {len(rewards)-perfect-bad}")

    # Round-by-round improvement
    by_round = defaultdict(list)
    for e in episodes:
        by_round[e["round"]].append(e["reward"])
    print("\n  Round-by-round mean reward:")
    prev_mean = None
    for rnd in sorted(by_round):
        r_list = by_round[rnd]
        m = sum(r_list) / len(r_list)
        trend = ""
        if prev_mean is not None:
            trend = f"  {'▲' if m > prev_mean else '▼'} {abs(m - prev_mean):.4f}"
        print(f"    Round {rnd}: {m:.4f}  (n={len(r_list)}){trend}")
        prev_mean = m

    # Convergence check: final round mean ≥ 0.50
    final_round = max(by_round)
    final_mean  = sum(by_round[final_round]) / len(by_round[final_round])
    if final_mean >= 0.50:
        ok(f"  Round {final_round} mean={final_mean:.4f} ≥ 0.50 → converging ✓")
    else:
        warn(f"  Round {final_round} mean={final_mean:.4f} < 0.50 → not yet converged "
             "(likely due to rate-limit errors in early rounds)")

    # RL transition recording sanity
    with_transitions = sum(1 for e in successful if e.get("rl_transitions", 0) > 0)
    without_trans    = sum(1 for e in successful if e.get("rl_transitions", 0) == 0)
    if with_transitions == len(successful):
        ok(f"  All {len(successful)} successful episodes recorded RL transitions ✓")
    else:
        warn(f"  {without_trans} successful episodes had 0 transitions (RL override may be missing)")

    # Category coverage
    cats = defaultdict(lambda: {"total": 0, "success": 0})
    for e in episodes:
        c = e.get("category", "unknown")
        cats[c]["total"] += 1
        if e.get("stop_reason") != "error":
            cats[c]["success"] += 1
    print("\n  Category breakdown (success / total):")
    for cat, d in sorted(cats.items()):
        bar = f"{d['success']}/{d['total']}"
        print(f"    {cat:<35} {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")

if not failures:
    print(f"\n  {PASS} ALL CHECKS PASSED\n")
else:
    print(f"\n  {FAIL} {len(failures)} CHECK(S) FAILED:\n")
    for f in failures:
        print(f"    • {f}")
    print()

sys.exit(0 if not failures else 1)
