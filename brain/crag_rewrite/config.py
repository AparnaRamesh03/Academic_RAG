from pathlib import Path
import sys

# Allow this folder to import shared files from ../
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from qdrant_config import SIMPLE_TOP_K

ARCHITECTURE_NAME = "crag_rewrite"

# Final number of docs passed to generation
TOP_K = 3

# How many top docs to inspect when judging retrieval quality
RETRIEVAL_EVAL_TOP_K = 5

# How many weak-signal docs to pass into the rewrite prompt
WEAK_SIGNAL_TOP_K = 3

# Keep CRAG simple and evaluation-friendly:
# only one rewrite attempt
MAX_REWRITE_ROUNDS = 1