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

# ---- Config -----------------------------------------------------------------

BENCHMARK_PATH = Path("./gold_standard_dev_24.json")
OUTPUT_PATH = Path("./results/supervisor_arch/dev24_results.json")

API_BASE_URL = "http://localhost:8000"
ASK_URL = f"{API_BASE_URL}/ask"

ARCHITECTURE_NAME = "supervisor_arch"
REQUEST_TIMEOUT_SEC = 300

# -----------------------------------------------------------------------------


def parse_backend_error_detail(response: requests.Response) -> Any:
    try:
        payload = response.json()
        if isinstance(payload, dict) and "detail" in payload:
            return payload["detail"]
        return payload
    except Exception:
        return response.text


def main() -> None:
    # 1. Confirm correct backend
    health_payload = validate_backend_architecture(
        api_base_url=API_BASE_URL,
        expected_architecture=ARCHITECTURE_NAME,
    )
    print(f"[✓] Backend health check passed: {health_payload}")

    # 2. Load benchmark
    benchmark = load_json(str(BENCHMARK_PATH))
    results = []

    total = len(benchmark)
    print(f"Loaded {total} benchmark questions from: {BENCHMARK_PATH}")

    # 3. Run benchmark
    for i, item in enumerate(benchmark, start=1):
        question = item["question"]
        print(f"[{i}/{total}] {question}")

        start = time.perf_counter()

        try:
            response = requests.post(
                ASK_URL,
                json={"query": question},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            latency = time.perf_counter() - start

            if response.status_code == 200:
                payload: Dict[str, Any] = response.json()

                results.append(
                    build_success_result(
                        benchmark_item=item,
                        architecture=ARCHITECTURE_NAME,
                        answer=payload.get("answer", ""),
                        contexts=payload.get("context_used", []),
                        citations=payload.get("citations", []),
                        latency_sec=latency,
                    )
                )
            else:
                detail = parse_backend_error_detail(response)
                results.append(
                    build_error_result(
                        benchmark_item=item,
                        architecture=ARCHITECTURE_NAME,
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
                build_error_result(
                    benchmark_item=item,
                    architecture=ARCHITECTURE_NAME,
                    latency_sec=latency,
                    error=normalize_error(
                        error_type="exception",
                        message=str(e),
                    ),
                )
            )

    # 4. Save
    save_json(results, str(OUTPUT_PATH))
    print(f"[✓] Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()