import argparse
import time
from pathlib import Path
from typing import Any, Dict

import requests

from result_utils import (
    load_json,
    save_json,
    normalize_error,
    build_success_result,
    build_error_result,
    validate_backend_architecture,
)

API_BASE_URL = "http://localhost:8000"
ASK_DEBUG_URL = f"{API_BASE_URL}/ask_debug"

ARCHITECTURE_NAME = "final_combined"
REQUEST_TIMEOUT_SEC = 300


def parse_backend_error_detail(response: requests.Response) -> Any:
    try:
        payload = response.json()
        if isinstance(payload, dict) and "detail" in payload:
            return payload["detail"]
        return payload
    except Exception:
        return response.text


def build_debug_success_result(
    *,
    benchmark_item: Dict[str, Any],
    payload: Dict[str, Any],
    latency_sec: float,
) -> Dict[str, Any]:
    base = build_success_result(
        benchmark_item=benchmark_item,
        architecture=ARCHITECTURE_NAME,
        answer=payload.get("answer", ""),
        contexts=payload.get("context_used", []),
        citations=payload.get("citations", []),
        latency_sec=latency_sec,
    )

    base.update(
        {
            "original_query": payload.get("original_query", benchmark_item["question"]),
            "final_search_query": payload.get("final_search_query", benchmark_item["question"]),
            "retrieved_docs": payload.get("retrieved_docs", []),
            "candidate_docs": payload.get("candidate_docs", []),
            "weak_signal_docs": payload.get("weak_signal_docs", []),
            "graded_docs": payload.get("graded_docs", []),
            "crag_retries": payload.get("crag_retries", 0),
            "verify_retries": payload.get("verify_retries", 0),
            "citations_pass": payload.get("citations_pass", False),
            "auditor_feedback": payload.get("auditor_feedback", ""),
            "claim_verification": payload.get("claim_verification", []),
        }
    )
    return base


def build_debug_error_result(
    *,
    benchmark_item: Dict[str, Any],
    latency_sec: float,
    error: Dict[str, Any],
) -> Dict[str, Any]:
    base = build_error_result(
        benchmark_item=benchmark_item,
        architecture=ARCHITECTURE_NAME,
        latency_sec=latency_sec,
        error=error,
    )
    base.update(
        {
            "original_query": benchmark_item["question"],
            "final_search_query": benchmark_item["question"],
            "retrieved_docs": [],
            "candidate_docs": [],
            "weak_signal_docs": [],
            "graded_docs": [],
            "crag_retries": 0,
            "verify_retries": 0,
            "citations_pass": False,
            "auditor_feedback": "",
            "claim_verification": [],
        }
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Run debug benchmark against final_combined backend.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="./gold_standard_dev_24.json",
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/final_combined/dev24_debug_results.json",
        help="Path to save debug results JSON",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    output_path = Path(args.output)

    validate_backend_architecture(
        api_base_url=API_BASE_URL,
        expected_architecture=ARCHITECTURE_NAME,
    )

    benchmark = load_json(str(benchmark_path))
    results = []

    total = len(benchmark)
    print(f"Loaded {total} benchmark questions from: {benchmark_path}")

    for i, item in enumerate(benchmark, start=1):
        question = item["question"]
        print(f"[{i}/{total}] {question}")

        start = time.perf_counter()

        try:
            response = requests.post(
                ASK_DEBUG_URL,
                json={"query": question},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            latency = time.perf_counter() - start

            if response.status_code == 200:
                payload: Dict[str, Any] = response.json()
                results.append(
                    build_debug_success_result(
                        benchmark_item=item,
                        payload=payload,
                        latency_sec=latency,
                    )
                )
            else:
                detail = parse_backend_error_detail(response)
                results.append(
                    build_debug_error_result(
                        benchmark_item=item,
                        latency_sec=latency,
                        error=normalize_error(
                            error_type="http_error",
                            message=f"Backend returned HTTP {response.status_code}.",
                            status_code=response.status_code,
                            detail=detail,
                        ),
                    )
                )

        except Exception as e:
            latency = time.perf_counter() - start
            results.append(
                build_debug_error_result(
                    benchmark_item=item,
                    latency_sec=latency,
                    error=normalize_error(
                        error_type="exception",
                        message=str(e),
                    ),
                )
            )

    save_json(results, str(output_path))
    print(f"[✓] Saved debug results to: {output_path}")


if __name__ == "__main__":
    main()