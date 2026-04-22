import json
import time
from pathlib import Path

import requests

BENCHMARK_PATH = Path("./gold_standard_dev_24.json")
OUTPUT_PATH = Path("./results_simple_dev_24.json")
API_URL = "http://localhost:8000/ask"
ARCHITECTURE_NAME = "simple_hybrid_rag"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    benchmark = load_json(BENCHMARK_PATH)
    results = []

    total = len(benchmark)
    print(f"Loaded {total} benchmark questions.")

    for i, item in enumerate(benchmark, start=1):
        question = item["question"]

        print(f"[{i}/{total}] {question}")

        start = time.perf_counter()
        try:
            response = requests.post(
                API_URL,
                json={"query": question},
                timeout=300,
            )
            latency = time.perf_counter() - start

            if response.status_code == 200:
                payload = response.json()
                results.append(
                    {
                        "question": item["question"],
                        "ground_truth": item["ground_truth"],
                        "source_file": item.get("source_file"),
                        "category": item.get("category"),
                        "difficulty": item.get("difficulty"),
                        "architecture": ARCHITECTURE_NAME,
                        "answer": payload.get("answer", ""),
                        "contexts": payload.get("context_used", []),
                        "citations": payload.get("citations", []),
                        "latency_sec": round(latency, 4),
                        "status": "ok",
                        "error": None,
                    }
                )
            else:
                results.append(
                    {
                        "question": item["question"],
                        "ground_truth": item["ground_truth"],
                        "source_file": item.get("source_file"),
                        "category": item.get("category"),
                        "difficulty": item.get("difficulty"),
                        "architecture": ARCHITECTURE_NAME,
                        "answer": "",
                        "contexts": [],
                        "citations": [],
                        "latency_sec": round(latency, 4),
                        "status": "error",
                        "error": {
                            "status_code": response.status_code,
                            "detail": response.text,
                        },
                    }
                )

        except Exception as e:
            latency = time.perf_counter() - start
            results.append(
                {
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "source_file": item.get("source_file"),
                    "category": item.get("category"),
                    "difficulty": item.get("difficulty"),
                    "architecture": ARCHITECTURE_NAME,
                    "answer": "",
                    "contexts": [],
                    "citations": [],
                    "latency_sec": round(latency, 4),
                    "status": "error",
                    "error": str(e),
                }
            )

    save_json(results, OUTPUT_PATH)
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()