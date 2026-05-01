"""
generate_results.py — General-purpose benchmark result collector.

Queries the currently running backend for every question in the gold standard
benchmark and saves the raw results as a JSON file for later metric computation.

Usage (run ONE architecture backend first, then):
    python generate_results.py --arch simple_hybrid_rag
    python generate_results.py --arch final_arch
    python generate_results.py --arch agentic_scholar

Options:
    --arch      Architecture name tag written into results (required)
    --benchmark Path to benchmark JSON (default: gold_standard_dev_24.json)
    --port      Backend port (default: 8000)
    --output    Output JSON path (default: results_<arch>.json)
    --timeout   Per-request timeout in seconds (default: 300)
"""

import argparse
import json
import time
from pathlib import Path

import requests


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect benchmark results from a running RAG backend."
    )
    parser.add_argument(
        "--arch",
        required=True,
        help="Architecture name (e.g. simple_hybrid_rag, final_arch, agentic_scholar)",
    )
    parser.add_argument(
        "--benchmark",
        default="gold_standard_dev_24.json",
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port the backend is listening on (default: 8000).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results_<arch>.json).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds (default: 300).",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    output_path = Path(args.output) if args.output else Path(f"results_{args.arch}.json")
    api_url = f"http://localhost:{args.port}/ask"

    print(f"Architecture : {args.arch}")
    print(f"Benchmark    : {benchmark_path}")
    print(f"API endpoint : {api_url}")
    print(f"Output       : {output_path}")
    print()

    benchmark = load_json(benchmark_path)
    total = len(benchmark)
    print(f"Loaded {total} benchmark questions.")

    # Verify backend is alive before starting
    try:
        health_resp = requests.get(
            f"http://localhost:{args.port}/health", timeout=10
        )
        if health_resp.status_code == 200:
            info = health_resp.json()
            running_arch = info.get("architecture", "unknown")
            print(f"Backend healthy. Running architecture: '{running_arch}'")
            if running_arch != args.arch:
                print(
                    f"  [WARNING] --arch flag says '{args.arch}' but backend "
                    f"reports '{running_arch}'. Results will be tagged as '{args.arch}'."
                )
        else:
            print(f"[WARNING] Health check returned HTTP {health_resp.status_code}.")
    except Exception as e:
        print(f"[WARNING] Could not reach backend health endpoint: {e}")
        print("  Proceeding anyway — make sure the backend is running.")

    print()

    # Resumable logic: load existing if available
    results = []
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {output_path.name}. Resuming...")
        except Exception as e:
            print(f"Could not load existing {output_path.name}, starting fresh. ({e})")
            results = []

    answered_qids = {r.get("original_question_id") for r in results if r.get("original_question_id") is not None}

    print()
    for i, item in enumerate(benchmark, start=1):
        q_id = item.get("original_question_id", i)
        question = item["question"]
        print(f"[{i:02d}/{total}] {question[:80]}")

        if q_id in answered_qids:
            print(f"       SKIPPED (already answered)")
            continue

        start = time.perf_counter()
        try:
            response = requests.post(
                api_url,
                json={"query": question},
                timeout=args.timeout,
            )
            latency = time.perf_counter() - start

            if response.status_code == 200:
                payload = response.json()
                results.append({
                    "original_question_id": q_id,
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "source_file": item.get("source_file"),
                    "category": item.get("category"),
                    "difficulty": item.get("difficulty"),
                    "architecture": args.arch,
                    "answer": payload.get("answer", ""),
                    "contexts": payload.get("context_used", []),
                    "citations": payload.get("citations", []),
                    "latency_sec": round(latency, 4),
                    "status": "ok",
                    "error": None,
                })
                print(f"       OK ({latency:.1f}s)")
            else:
                print(f"       ERROR HTTP {response.status_code} - Skipping save so it can be retried later.")
                
            # Sleep for 12 seconds to respect Gemini's 15 Requests-Per-Minute Free Tier Limit
            # Advanced architectures make 3-5 LLM calls per question, so pacing is required.
            time.sleep(12)

        except Exception as exc:
            print(f"       TIMEOUT/ERROR: {exc}")
            # Sleep for 12 seconds even on error to avoid rate limiting
            time.sleep(12)
            continue

        # Save incrementally after each successful question
        save_json(results, output_path)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print()
    print(f"Done. {ok_count}/{total} questions answered successfully.")
    print(f"Results saved to → {output_path}")


if __name__ == "__main__":
    main()
