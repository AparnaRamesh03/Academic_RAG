
import os
import sys
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# Import MARL components
from context_marl_ac.adapters.llm_adapter import generate_answer

def sanitize_filename(name: str) -> str:
    keepcharacters = (' ','.','_','-')
    sanitized = "".join(c for c in name if c.isalnum() or c in keepcharacters).rstrip()
    return sanitized.replace(" ", "_")[:50] + ".md"

def clean_answer(answer: str) -> str:
    cleaned = answer.strip()
    prefixes_to_strip = [
        "The answer is ", "According to the article, ", "Based on the evidence, ",
        "The name of the company is ", "The character is ", "The individual is ",
        "Yes, ", "No, "
    ]
    for prefix in prefixes_to_strip:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    
    if cleaned.lower().startswith("yes,"):
        cleaned = "Yes"
    if cleaned.lower().startswith("no,"):
        cleaned = "No"
    
    return cleaned.strip().rstrip(".")

def evaluate_ground_truth():
    benchmark_path = _BRAIN_ROOT / "context_marl_ac" / "results" / "benchmark_splits" / "test_multihop_15.jsonl"
    output_path = _BRAIN_ROOT / "context_marl_ac" / "results" / "final_eval" / "test_15_ground_truth_results.jsonl"
    
    print(f"Loading benchmark from {benchmark_path}...")
    questions = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
            
    print("Loading MultiHopRAG corpus...")
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    
    # Map corpus by sanitized title
    corpus_by_filename = {}
    for doc in corpus:
        filename = sanitize_filename(doc["title"])
        corpus_by_filename[filename] = doc["body"]
        
    results = []
    print(f"Evaluating {len(questions)} questions using ground-truth chunks...")
    
    for q in tqdm(questions):
        question_text = q["question"]
        gold_files = q.get("source_file", [])
        
        # Prepare "evidence_pack" as expected by generate_answer
        gold_evidence = []
        for filename in gold_files:
            if filename in corpus_by_filename:
                gold_evidence.append({
                    "text": corpus_by_filename[filename],
                    "metadata": {"source_file": filename}
                })
        
        try:
            # Call adapter directly
            raw_answer, tokens = generate_answer(question_text, gold_evidence)
            generated_answer = clean_answer(raw_answer)
        except Exception as e:
            generated_answer = f"Error: {e}"
            
        res_row = {
            "question_id": q["question_id"],
            "question": question_text,
            "ground_truth": q["ground_truth"],
            "answer": generated_answer,
            "category": q["category"],
            "source_file": gold_files,
            "architecture": "ground_truth_reasoning",
            "status": "ok"
        }
        results.append(res_row)
        
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
            
    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    evaluate_ground_truth()
