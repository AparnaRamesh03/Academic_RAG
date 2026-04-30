from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from qdrant_config import SIMPLE_TOP_K

ARCHITECTURE_NAME = "final_combined"

TOP_K = SIMPLE_TOP_K
RETRIEVAL_EVAL_TOP_K = 5
WEAK_SIGNAL_TOP_K = 3
GRADE_TOP_K = 6

# NEW: reranker settings
RERANK_INPUT_TOP_K = 20
RERANK_OUTPUT_TOP_K = 10

MAX_REWRITE_ROUNDS = 1
MAX_AUDIT_RETRIES = 1