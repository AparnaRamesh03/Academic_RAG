import json
import random
from datasets import load_dataset
from pathlib import Path

PROJECT_ROOT = Path("/Users/spartan/Documents/Academic_RAG")
NUM_QUESTIONS = 200

def sanitize_filename(name: str) -> str:
    keepcharacters = (' ','.','_','-')
    sanitized = "".join(c for c in name if c.isalnum() or c in keepcharacters).rstrip()
    return sanitized.replace(" ", "_")[:50] + ".md"

def main():
    print("Loading MultiHopRAG dataset for benchmark generation...")
    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    
    corpus_by_url = {doc["url"]: doc for doc in corpus}
    
    random.seed(42)
    selected_indices = random.sample(range(len(ds)), min(NUM_QUESTIONS, len(ds)))
    selected_questions = [ds[i] for i in selected_indices]
    
    benchmark_file = PROJECT_ROOT / "brain" / "context_marl_ac" / "results" / "benchmark_splits" / "test_multihop_rag_200.jsonl"
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating 200-question benchmark file: {benchmark_file}")
    with open(benchmark_file, "w", encoding="utf-8") as f:
        for idx, q in enumerate(selected_questions, start=1):
            source_files = []
            for ev in q.get("evidence_list", []):
                doc = corpus_by_url.get(ev["url"])
                if doc:
                    title = doc.get("title", "Unknown")
                    source_files.append(sanitize_filename(title))
            
            source_files = list(set(source_files))
            
            row = {
                "question_id": f"MH-200-{idx:03d}",
                "question": q["query"],
                "ground_truth": q["answer"],
                "source_file": source_files,
                "category": q.get("question_type", "multi_chunk_synthesis"),
                "difficulty": "hard"
            }
            f.write(json.dumps(row) + "\n")
            
    print("Done! Benchmark file ready.")

if __name__ == "__main__":
    main()
