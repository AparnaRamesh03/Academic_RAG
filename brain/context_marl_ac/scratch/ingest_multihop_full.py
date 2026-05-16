import os
import sys
import json
import random
from datasets import load_dataset
from pathlib import Path

# Setup paths to import ingestion modules
PROJECT_ROOT = Path("/Users/spartan/Documents/Academic_RAG")
INGESTION_DIR = PROJECT_ROOT / "ingestion"
if str(INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(INGESTION_DIR))

# Ensure brain is in path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chunker
import indexer
from brain.qdrant_config import COLLECTION_NAME, QDRANT_URL

NUM_QUESTIONS = 100
DENSE_VECTOR_SIZE = 1024 # BGE-M3

def sanitize_filename(name: str) -> str:
    keepcharacters = (' ','.','_','-')
    sanitized = "".join(c for c in name if c.isalnum() or c in keepcharacters).rstrip()
    return sanitized.replace(" ", "_")[:50] + ".md"

def main():
    print(f"Loading MultiHopRAG dataset...")
    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    
    # Map corpus by url
    corpus_by_url = {doc["url"]: doc for doc in corpus}
    
    print(f"Total corpus documents: {len(corpus)}")
    
    # Pick 100 random questions
    random.seed(42)
    selected_indices = random.sample(range(len(ds)), min(NUM_QUESTIONS, len(ds)))
    selected_questions = [ds[i] for i in selected_indices]
    
    # We will ingest the ENTIRE corpus (609 docs) to ensure "checking the database" means the whole thing
    docs_to_ingest = list(corpus_by_url.keys())
    print(f"Total documents to ingest (full corpus): {len(docs_to_ingest)}")
    
    # 1. Ensure collection exists with CORRECT dimensions for BGE-M3
    print(f"Checking collection '{COLLECTION_NAME}'...")
    import qdrant_client
    from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, Modifier, SparseIndexParams
    
    qdrant = qdrant_client.QdrantClient(url=QDRANT_URL)
    
    try:
        if qdrant.collection_exists(COLLECTION_NAME):
            info = qdrant.get_collection(COLLECTION_NAME)
            size = info.config.params.vectors["dense"].size
            if size != DENSE_VECTOR_SIZE:
                print(f"[*] Dimension mismatch (found {size}, expected {DENSE_VECTOR_SIZE}). Recreating...")
                qdrant.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config={"dense": VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE)},
                    sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF, index=SparseIndexParams(on_disk=True))},
                )
            else:
                print(f"[✓] Collection '{COLLECTION_NAME}' already exists with correct dimensions.")
        else:
            print(f"[*] Creating collection '{COLLECTION_NAME}'...")
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"dense": VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF, index=SparseIndexParams(on_disk=True))},
            )
    except Exception as e:
        print(f"[!] Warning: collection setup failed: {e}")
        
    # 2. Index chunks file by file
    print("Chunking and Indexing...")
    for idx, url in enumerate(docs_to_ingest, 1):
        doc = corpus_by_url[url]
        title = doc.get("title", "Unknown Title")
        text = doc.get("body", "")
        filename = sanitize_filename(title)
        
        # Build markdown text
        markdown_text = f"# {title}\n\n{text}"
        
        # Chunk it
        chunks = chunker.chunk_markdown(markdown_text, source_file=filename)
        
        if (idx % 50 == 0) or (idx == len(docs_to_ingest)):
            print(f"  -> Progress: {idx}/{len(docs_to_ingest)} documents processed.")
            
        if chunks:
            # The indexer will handle embeddings and upsert
            indexer.index_chunks(chunks)
            
    print("[✓] Indexing complete.")
    
    # 3. Build benchmark file (100 questions)
    benchmark_file = PROJECT_ROOT / "brain" / "context_marl_ac" / "results" / "benchmark_splits" / "test_multihop_rag_large.jsonl"
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving benchmark file to {benchmark_file}")
    with open(benchmark_file, "w", encoding="utf-8") as f:
        for idx, q in enumerate(selected_questions, start=1):
            source_files = []
            for ev in q.get("evidence_list", []):
                doc = corpus_by_url.get(ev["url"])
                if doc:
                    title = doc.get("title", "Unknown")
                    source_files.append(sanitize_filename(title))
            
            # Remove duplicates
            source_files = list(set(source_files))
            
            row = {
                "question_id": f"MH-L-{idx:03d}",
                "question": q["query"],
                "ground_truth": q["answer"],
                "source_file": source_files,
                "category": q.get("question_type", "multi_chunk_synthesis"),
                "difficulty": "hard"
            }
            f.write(json.dumps(row) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
