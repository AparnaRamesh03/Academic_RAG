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

MAX_RETRIEVAL_RETRIES      = 3
MAX_REWRITES               = 2

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
W_LATENCY_COST      = 0.05   # subtracted
W_STEP_COST         = 0.05   # subtracted

# Penalty magnitudes (applied as negative rewards)
PENALTY_HALLUCINATION        = -0.30
PENALTY_UNSUPPORTED_CLAIM    = -0.10
PENALTY_REPEATED_ACTION      = -0.05
PENALTY_INVALID_ACTION       = -0.10
PENALTY_NO_ANSWER            = -0.50
PENALTY_MAX_STEPS            = -0.20

# ---------------------------------------------------------------------------
# Fully free supervisor MARL routing
# ---------------------------------------------------------------------------
ROUTING_MODE = "supervisor_free"

# Supervisor is allowed to choose the next valid agent.
# Masks block impossible actions, not the whole RAG flow.
SUPERVISOR_ENTROPY_COEF = 0.01

# Keep fully free routing bounded.
MAX_SUPERVISOR_REPEATS = 2
MAX_NO_PROGRESS_STEPS = 3

# Recommended for fully free routing to reduce loops.


# Penalize bad supervisor routing habits.
PENALTY_REPEATED_AGENT = -0.04
PENALTY_REPEATED_SAME_ACTION = -0.08
PENALTY_REWRITE_REPEAT_BEFORE_RETRIEVAL = -0.08

# Penalize skipping useful context processing on harder questions.
PENALTY_SKIP_GRADER_COMPLEX = -0.08
BONUS_GRADER_USED_COMPLEX = 0.03

# Penalize weak generation choices on complex questions.
PENALTY_SHORT_ANSWER_COMPLEX = -0.08
PENALTY_TOO_SHORT_COMPLEX_ANSWER = -0.05
PENALTY_TOO_SHORT_COMPLEX_ANSWER_TERMINAL = -0.05

# ---------------------------------------------------------------------------
# Fully free MARL curriculum masks
# ---------------------------------------------------------------------------

# Prevent the supervisor from learning bad early shortcuts on complex questions.
# This still keeps the architecture supervisor-free, but blocks weak generation
# before evidence grading on harder questions.
REQUIRE_GRADER_FOR_COMPLEX_GENERATION = True


# ---------------------------------------------------------------------------
# Groq TPM / rate-limit handling
# ---------------------------------------------------------------------------

# Retry Groq 429 errors instead of crashing training.
GROQ_RETRY_MAX_ATTEMPTS = 6
GROQ_RETRY_BASE_SLEEP_SECONDS = 2.0
GROQ_RETRY_MAX_SLEEP_SECONDS = 30.0

# Small pacing delay before every Groq-backed LLM call.
# Increase to 2.0 or 3.0 if TPM errors continue.
GROQ_MIN_SECONDS_BETWEEN_CALLS = 2.5

# Reduce prompt size sent to generator/verifier.
LLM_MAX_EVIDENCE_DOCS = 4
LLM_MAX_EVIDENCE_CHARS_PER_DOC = 900

# Reduce prompt size sent to LLM grader.
LLM_MAX_GRADER_CHUNKS = 16
LLM_MAX_GRADER_CHARS_PER_CHUNK = 800

# Limit source-diversity retries:
# one normal retrieval + one diversity retry.
MAX_SOURCE_DIVERSITY_RETRIEVALS = 2

# ---------------------------------------------------------------------------
# Constrained action masking
# ---------------------------------------------------------------------------

# First action must be retrieval. This ensures evidence grounding.
REQUIRE_INITIAL_RETRIEVAL = True


# Retrieval/recovery limits.
MAX_RETRIEVAL_RETRIES = 3
MAX_VERIFICATION_RETRIES = 1
MAX_STEPS_PER_EPISODE = 14
MAX_LLM_CALLS_PER_EPISODE = 8


# Bound non-LLM grader actions so keep_all/rerank_only do not pass too much noise.
MAX_KEEP_ALL_DOCS = 4
MAX_RERANK_ONLY_DOCS = 5

# Lightweight verifier used by Context MARL.
LIGHTWEIGHT_VERIFIER_MODEL = "llama-3.1-8b-instant"

# Use local/non-LLM grading during training.
FAST_TRAINING_GRADER = True

RETRIEVAL_TOP_K = 5

# If your retriever_adapter supports these, use them there too.
RERANK_INPUT_TOP_K = 10
RERANK_KEEP_TOP_K = 5
