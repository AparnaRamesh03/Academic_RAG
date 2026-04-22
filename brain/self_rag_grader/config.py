from pathlib import Path
import sys

# Allow this folder to import shared files from ../
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ARCHITECTURE_NAME = "self_rag_grader"

# Number of top retrieved docs to send into the grader.
# Keep this smaller than full retrieval to reduce cost and latency.
GRADE_TOP_K = 6