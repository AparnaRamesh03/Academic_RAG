import os
from qdrant_client import QdrantClient
from fastembed import SparseTextEmbedding
from qdrant_client import models
from state import GraphState
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "academic_papers"
DENSE_EMBED_MODEL = "nomic-embed-text"

client = QdrantClient(url=QDRANT_URL)
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

def retrieve_and_split(state: GraphState):
    """
    Node 1: Hybrid Retrieval & Splitter
    Queries Qdrant using RRF and splits into Candidates and Weak Signals.
    """
    query = state["search_query"]
    print(f"\n[Node 1] Retrieving context for query: '{query}'")
    
    # 1. Dense and Sparse Embeddings
    from ollama import embeddings
    dense_vec = embeddings(model=DENSE_EMBED_MODEL, prompt=query)["embedding"]
    sparse_vec = list(sparse_model.embed([query]))[0]
    
    # 2. Qdrant Hybrid Search (RRF)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=10),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(), 
                    values=sparse_vec.values.tolist()
                ),
                using="sparse",
                limit=10
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=15, # Get Top 5 chunks total
        with_payload=True,
    )
    
    candidate_docs = []
    weak_signal_docs = []
    
    # 3. Splitting Logic
    print("[Node 1] Evaluating RRF Scores:")
    for idx, point in enumerate(results.points):
        doc = {"text": point.payload.get("text", ""), "metadata": point.payload, "score": point.score}
        print(f"  -> Doc {idx+1} Score: {point.score:.5f}")
        
        if point.score > 0.4:
            candidate_docs.append(doc)
        else:
            weak_signal_docs.append(doc)
            
    print(f"[Node 1] Split: {len(candidate_docs)} Candidates | {len(weak_signal_docs)} Weak Signals")
    
    ablation_mode = state.get("ablation_mode", "full")
    if ablation_mode == "naive":
        print(f"[Node 1] Ablation: 'naive' mode active. Fast-tracking {len(candidate_docs) + len(weak_signal_docs)} directly to Generator.")
        return {
            "candidate_docs": candidate_docs,
            "weak_signal_docs": weak_signal_docs,
            "graded_docs": candidate_docs + weak_signal_docs
        }
        
    return {
        "candidate_docs": candidate_docs,
        "weak_signal_docs": weak_signal_docs
    }
