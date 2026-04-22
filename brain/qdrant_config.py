import os
from dotenv import load_dotenv

load_dotenv()

# Shared Qdrant / retrieval configuration for all brain architectures.

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "academic_papers")

DENSE_EMBED_MODEL = "BAAI/bge-m3"
SPARSE_EMBED_MODEL = "Qdrant/bm25"

# Retrieval breadth
DENSE_PREFETCH_LIMIT = 20
SPARSE_PREFETCH_LIMIT = 20
FINAL_FUSION_LIMIT = 20

# Simple baseline top-k passed to generation
SIMPLE_TOP_K = 8