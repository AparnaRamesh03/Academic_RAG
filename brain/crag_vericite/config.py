from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from qdrant_config import SIMPLE_TOP_K

ARCHITECTURE_NAME = "crag_vericite"

TOP_K = SIMPLE_TOP_K
RETRIEVAL_EVAL_TOP_K = 5
WEAK_SIGNAL_TOP_K = 3

MAX_REWRITE_ROUNDS = 1
MAX_AUDIT_RETRIES = 1