from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from qdrant_config import SIMPLE_TOP_K

ARCHITECTURE_NAME = "agentic_scholar"

TOP_K = SIMPLE_TOP_K
RETRIEVAL_EVAL_TOP_K = 5
WEAK_SIGNAL_TOP_K = 3
GRADE_TOP_K = 6

MAX_REWRITE_ROUNDS = 1
MAX_AUDIT_RETRIES = 1

# ── Scholarly Fallback Configuration ────────────────────────────────────────
# Max results to request from each trusted scholarly API.
# These are free unauthenticated tiers — keep limits conservative.
SCHOLAR_FALLBACK_MAX_RESULTS = 5   # Semantic Scholar
ARXIV_FALLBACK_MAX_RESULTS = 3     # arXiv

# Trusted academic sources. Only these APIs are ever queried.
TRUSTED_SCHOLARLY_APIS = [
    "api.semanticscholar.org",
    "export.arxiv.org",
]
