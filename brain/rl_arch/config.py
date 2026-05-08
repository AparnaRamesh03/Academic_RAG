from __future__ import annotations

# Phase 4 starts with a small controller action space.
# Keep this minimal for feasibility.
ENABLE_GRADE_DOCS_ACTION = False

# Hard safety limits for controller behavior.
MAX_RL_CONTROLLER_STEPS = 12
MAX_REWRITE_ACTIONS = 2
MAX_VERIFY_ACTIONS = 2

# Ordered action list used by the RL controller.
BASE_ACTIONS = [
    "retrieve",
    "rewrite_query",
    "answer",
    "verify",
    "stop",
]

OPTIONAL_ACTIONS = [
    "grade_docs",
]

STATE_FEATURE_VERSION = "v1"

# Clipping values keep the encoded state numerically stable.
MAX_DOC_COUNT_CLIP = 20
MAX_LATENCY_CLIP = 120.0
MAX_TEXT_LENGTH_CLIP = 4000

# -----------------------------
# RL reward configuration
# -----------------------------

REWARD_GROUNDED_PASS = 10.0
REWARD_GROUNDED_INCOMPLETE = 6.0
REWARD_CITATIONS_PASS_BONUS = 2.0

PENALTY_UNSUPPORTED = -10.0
PENALTY_NEEDS_REVISION = -4.0
PENALTY_EMPTY_ANSWER = -6.0

PENALTY_PER_STEP = -0.5
PENALTY_REWRITE_ACTION = -0.25
PENALTY_VERIFY_ACTION = -0.25

BONUS_EARLY_STOP_GOOD = 1.5
PENALTY_BAD_STOP = -5.0

PENALTY_MAX_STEP_TERMINATION = -3.0
PENALTY_AUDIT_RETRY_LIMIT = -2.5

# Small bonus when evidence quality improves after an action.
REWARD_EVIDENCE_IMPROVEMENT = 1.0

# Threshold used for judging whether evidence quality improved meaningfully.
EVIDENCE_IMPROVEMENT_DELTA = 0.05

from pathlib import Path

RL_DATA_DIR = Path(__file__).resolve().parent / "data"
RL_TRAJECTORY_DIR = RL_DATA_DIR / "trajectories"
RL_TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

ENABLE_TRAJECTORY_LOGGING = True
RL_TRAJECTORY_FILE_PREFIX = "phase4_traj"