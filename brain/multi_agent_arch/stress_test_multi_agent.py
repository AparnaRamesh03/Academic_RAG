import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

API_BASE_URL = "http://localhost:8000"
ASK_DEBUG_URL = f"{API_BASE_URL}/ask_debug"
OUTPUT_PATH = Path("./results/multi_agent_arch/stress_test_results.json")
REQUEST_TIMEOUT_SEC = 300

TEST_CASES: List[Dict[str, str]] = [
    {
        "id": "cross_paper_comparison",
        "query": "How do the Transformer and ResNet papers each argue that architectural design can overcome training bottlenecks in deep learning?",
    },
    {
        "id": "single_paper_direct_fact",
        "query": "What is Naive RAG according to the survey?",
    },
    {
        "id": "figure_grounded",
        "query": "What does Figure 3 show about the differences between Naive RAG, Advanced RAG, and Modular RAG?",
    },
    {
        "id": "paraphrase_hard",
        "query": "How do the papers describe solving the problem where deeper networks become harder to optimize as depth increases?",
    },
    {
        "id": "noisy_or_underspecified",
        "query": "Which architecture solves the efficiency problem best and why?",
    },
    {
        "id": "cross_paper_missing_side",
        "query": "Compare what BERT and the Transformer paper each claim about architectural improvements over earlier sequence models.",
    },
]


def parse_backend_error_detail(response: requests.Response) -> Any:
    try:
        payload = response.json()
        if isinstance(payload, dict) and "detail" in payload:
            return payload["detail"]
        return payload
    except Exception:
        return response.text


def summarize_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    action_history = payload.get("action_history", []) or []
    agent_status = payload.get("agent_status", {}) or {}
    verification_outcome = payload.get("verification_outcome", "")
    auditor_feedback = payload.get("auditor_feedback", "")
    answer = payload.get("answer", "") or ""

    issues = []

    if not answer.strip():
        issues.append("empty_answer")

    if not payload.get("graded_docs"):
        issues.append("no_graded_docs")

    if len(action_history) >= 6 and action_history[-6:-3] == action_history[-3:]:
        issues.append("repeating_action_loop")

    if payload.get("stop_reason") in {"supervisor_stopped", "max_steps_reached", "agent_loop_detected"}:
        issues.append(f"stop_reason:{payload.get('stop_reason')}")

    if verification_outcome in {"needs_revision", "empty_or_ungrounded"}:
        issues.append(f"verification:{verification_outcome}")

    if payload.get("citations_pass") is False and answer.strip():
        issues.append("answered_but_not_verified")

    if "lack" in auditor_feedback.lower() or "missing" in auditor_feedback.lower():
        issues.append("coverage_gap_flagged")

    if not payload.get("agent_trace"):
        issues.append("no_agent_trace")

    if not agent_status:
        issues.append("no_agent_status")

    return {
        "answer_preview": answer[:220],
        "step_count": payload.get("step_count", 0),
        "action_history": action_history,
        "stop_reason": payload.get("stop_reason", ""),
        "citations_pass": payload.get("citations_pass", False),
        "verification_outcome": verification_outcome,
        "auditor_feedback": auditor_feedback,
        "issues": issues,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i, case in enumerate(TEST_CASES, start=1):
        query = case["query"]
        print(f"\n[{i}/{len(TEST_CASES)}] {case['id']}")
        print(f"Query: {query}")

        start = time.perf_counter()
        try:
            response = requests.post(
                ASK_DEBUG_URL,
                json={"query": query},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            latency = time.perf_counter() - start

            if response.status_code == 200:
                payload = response.json()
                summary = summarize_result(payload)

                print(f"  -> stop_reason: {summary['stop_reason']}")
                print(f"  -> citations_pass: {summary['citations_pass']}")
                print(f"  -> verification_outcome: {summary['verification_outcome']}")
                print(f"  -> action_history: {summary['action_history']}")
                print(f"  -> issues: {summary['issues']}")

                results.append(
                    {
                        "id": case["id"],
                        "query": query,
                        "latency_sec": latency,
                        "summary": summary,
                        "raw": payload,
                    }
                )
            else:
                detail = parse_backend_error_detail(response)
                print(f"  -> HTTP {response.status_code}")
                results.append(
                    {
                        "id": case["id"],
                        "query": query,
                        "latency_sec": latency,
                        "error": {
                            "status_code": response.status_code,
                            "detail": detail,
                        },
                    }
                )

        except Exception as e:
            latency = time.perf_counter() - start
            print(f"  -> exception: {e}")
            results.append(
                {
                    "id": case["id"],
                    "query": query,
                    "latency_sec": latency,
                    "error": {
                        "exception": str(e),
                    },
                }
            )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Saved stress test results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()