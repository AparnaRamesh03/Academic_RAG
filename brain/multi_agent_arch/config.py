from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from qdrant_config import SIMPLE_TOP_K

ARCHITECTURE_NAME = "multi_agent_arch"

TOP_K = SIMPLE_TOP_K
RETRIEVAL_EVAL_TOP_K = 5
WEAK_SIGNAL_TOP_K = 3
GRADE_TOP_K = 6

RERANK_INPUT_TOP_K = 20
RERANK_OUTPUT_TOP_K = 10

MAX_REWRITE_ROUNDS = 1
MAX_AUDIT_RETRIES = 1

MAX_STEPS = 14
MIN_CONFIDENCE_TO_STOP = 0.70  # Recalibrated: reachable via rerank-score interpolation (~0.55–0.82 range)

# ── Reinforcement Learning (bandit routing) ──────────────────────────────────
RL_ENABLED      = True   # Set to False to use pure rule-based routing.
RL_EPSILON      = 0.15   # Initial exploration rate.
RL_EPSILON_DECAY = 0.995 # Multiply epsilon by this after every episode update.
RL_EPSILON_MIN  = 0.05   # Epsilon floor — always keep some exploration.
RL_ALPHA        = 0.15   # TD learning rate (higher = faster warm-up convergence).
RL_POLICY_PATH  = Path(__file__).resolve().parent / "results" / "rl_policy.json"