"""
brain/context_marl_ac/config.py
---------------------------------
Central configuration for the Context-Engineered MARL Actor-Critic RAG.

All hyperparameters, path constants, and flags live here.
Other modules import from this file — do not scatter magic numbers.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # brain/context_marl_ac/
BRAIN_ROOT = _THIS_DIR.parent                        # brain/
REPO_ROOT  = BRAIN_ROOT.parent                       # Academic_RAG/

RESULTS_DIR        = _THIS_DIR / "results"
METRICS_DIR        = RESULTS_DIR / "metrics"
TRAJECTORIES_DIR   = RESULTS_DIR / "trajectories"
CHECKPOINTS_DIR    = RESULTS_DIR / "checkpoints"
PLOTS_DIR          = RESULTS_DIR / "plots"
FINAL_EVAL_DIR     = RESULTS_DIR / "final_eval"

# ---------------------------------------------------------------------------
# Architecture identifier
# ---------------------------------------------------------------------------
ARCHITECTURE_NAME = "context_marl_ac"

# ---------------------------------------------------------------------------
# Dry-run / mock mode
# Adapters check this flag and return stub data instead of calling
# external services (Qdrant, Groq).  Set via --dry-run CLI flag.
# ---------------------------------------------------------------------------
DRY_RUN: bool = False

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
DEFAULT_TOP_K          = 8
RERANK_INPUT_TOP_K     = 20
RERANK_OUTPUT_TOP_K    = 10

# ---------------------------------------------------------------------------
# Episode limits
# ---------------------------------------------------------------------------
MAX_STEPS_PER_EPISODE      = 12
MAX_LLM_CALLS_PER_EPISODE  = 15
MAX_RETRIEVAL_RETRIES      = 2
MAX_REWRITES               = 2
MAX_VERIFICATION_RETRIES   = 2

# ---------------------------------------------------------------------------
# Neural model dimensions
# ---------------------------------------------------------------------------
FEATURE_DIM = 14      # output dim of feature_encoder.py
HIDDEN_DIM  = 64      # actor / critic hidden layer size

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE  = 3e-4
GAMMA          = 0.99
ENTROPY_COEF   = 0.01
VALUE_COEF     = 0.5
GRAD_CLIP_NORM = 1.0
CHECKPOINT_EVERY = 50   # save checkpoint every N episodes

# ---------------------------------------------------------------------------
# Cooperative reward weights  (must sum to 1.0 for the positive terms)
# ---------------------------------------------------------------------------
W_ANSWER_QUALITY    = 0.30
W_CITATION_SUPPORT  = 0.25
W_VERIFICATION_PASS = 0.20
W_RETRIEVAL_F1      = 0.15
W_LATENCY_COST      = 0.05   # subtracted; applied once at terminal, capped at 1.0 normalized
W_STEP_COST         = 0.02   # subtracted per step

# Penalty magnitudes (applied as negative rewards)
PENALTY_HALLUCINATION        = -0.30
PENALTY_UNSUPPORTED_CLAIM    = -0.10
PENALTY_REPEATED_ACTION      = -0.05
PENALTY_INVALID_ACTION       = -0.10
PENALTY_NO_ANSWER            = -0.50
PENALTY_MAX_STEPS            = -0.20
