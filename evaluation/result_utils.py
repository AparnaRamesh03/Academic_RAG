import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def save_json(data: Any, filepath: str) -> None:
    """
    Save Python data as pretty JSON.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """
    Load JSON data from disk.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_error(
    *,
    error_type: str,
    message: str,
    status_code: Optional[int] = None,
    detail: Any = None,
) -> Dict[str, Any]:
    """
    Standardize all saved errors into one object shape.

    Example outputs:
    {
      "type": "http_error",
      "message": "Backend returned HTTP 500.",
      "status_code": 500,
      "detail": "..."
    }

    {
      "type": "exception",
      "message": "Connection refused",
      "status_code": None,
      "detail": None
    }
    """
    return {
        "type": error_type,
        "message": message,
        "status_code": status_code,
        "detail": detail,
    }


def build_success_result(
    *,
    benchmark_item: Dict[str, Any],
    architecture: str,
    answer: str,
    contexts: List[str],
    citations: List[Dict[str, Any]],
    latency_sec: float,
) -> Dict[str, Any]:
    """
    Build one canonical successful result row.
    """
    return {
        "question": benchmark_item["question"],
        "ground_truth": benchmark_item["ground_truth"],
        "source_file": benchmark_item.get("source_file"),
        "category": benchmark_item.get("category"),
        "difficulty": benchmark_item.get("difficulty"),
        "architecture": architecture,
        "answer": answer,
        "contexts": contexts,
        "citations": citations,
        "latency_sec": round(latency_sec, 4),
        "status": "ok",
        "error": None,
    }


def build_error_result(
    *,
    benchmark_item: Dict[str, Any],
    architecture: str,
    latency_sec: float,
    error: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build one canonical failed result row.
    """
    return {
        "question": benchmark_item["question"],
        "ground_truth": benchmark_item["ground_truth"],
        "source_file": benchmark_item.get("source_file"),
        "category": benchmark_item.get("category"),
        "difficulty": benchmark_item.get("difficulty"),
        "architecture": architecture,
        "answer": "",
        "contexts": [],
        "citations": [],
        "latency_sec": round(latency_sec, 4),
        "status": "error",
        "error": error,
    }


def validate_backend_architecture(
    *,
    api_base_url: str,
    expected_architecture: str,
    timeout_sec: float = 20.0,
) -> Dict[str, Any]:
    """
    Call /health and confirm the backend architecture matches the runner.

    Returns the parsed health payload if valid.
    Raises RuntimeError if anything is wrong.
    """
    health_url = f"{api_base_url.rstrip('/')}/health"

    try:
        response = requests.get(health_url, timeout=timeout_sec)
    except Exception as e:
        raise RuntimeError(f"Could not reach backend health endpoint: {e}")

    if response.status_code != 200:
        raise RuntimeError(
            f"Health check failed with HTTP {response.status_code}: {response.text}"
        )

    try:
        payload = response.json()
    except Exception as e:
        raise RuntimeError(f"Health endpoint did not return valid JSON: {e}")

    actual_architecture = str(payload.get("architecture", "")).strip()

    if actual_architecture != expected_architecture:
        raise RuntimeError(
            f"Backend architecture mismatch. "
            f"Expected '{expected_architecture}', got '{actual_architecture}'."
        )

    return payload