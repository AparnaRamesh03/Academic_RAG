from pathlib import Path
import sys

# Allow this folder to import shared files from ../
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from qdrant_config import SIMPLE_TOP_K

ARCHITECTURE_NAME = "simple_hybrid_rag"
TOP_K = SIMPLE_TOP_K