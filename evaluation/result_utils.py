"""
result_utils.py
---------------
Shared utilities for all generate_*_results.py benchmark scripts.

Provides:
  - load_json / save_json
  - normalize_error
  - build_success_result
  - build_error_result
  - validate_backend_architecture
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Error normalization
# ---------------------------------------------------------------------------

def normalize_error(
    *,
    error_type: str,
    message: str,
    status_code: Optional[int] = None,
    detail: Any = None,
) -> Dict[str, Any]:
    """Build a standardised error dict for failed benchmark items."""
    err: Dict[str, Any] = {
        "error_type": error_type,
        "message": message,
    }
    if status_code is not None:
        err["status_code"] = status_code
    if detail is not None:
        err["detail"] = detail
    return err


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------

def build_success_result(
    *,
    benchmark_item: Dict[str, Any],
    architecture: str,
    answer: str,
    contexts: List[str],
    citations: List[Dict[str, Any]],
    latency_sec: float,
) -> Dict[str, Any]:
    """Build a standardised success result dict from a benchmark item + response."""
    return {
        # ── benchmark metadata ──────────────────────────────────────────────
        "question": benchmark_item.get("question", ""),
        "ground_truth": benchmark_item.get("ground_truth", ""),
        "source_file": benchmark_item.get("source_file", []),
        "category": benchmark_item.get("category", ""),
        "difficulty": benchmark_item.get("difficulty", ""),
        "original_question_id": benchmark_item.get("original_question_id"),
        # ── response ────────────────────────────────────────────────────────
        "architecture": architecture,
        "answer": answer,
        "context_used": contexts,
        "citations": citations,
        # ── run metadata ────────────────────────────────────────────────────
        "latency_sec": latency_sec,
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
    """Build a standardised error result dict for a failed benchmark item."""
    return {
        # ── benchmark metadata ──────────────────────────────────────────────
        "question": benchmark_item.get("question", ""),
        "ground_truth": benchmark_item.get("ground_truth", ""),
        "source_file": benchmark_item.get("source_file", []),
        "category": benchmark_item.get("category", ""),
        "difficulty": benchmark_item.get("difficulty", ""),
        "original_question_id": benchmark_item.get("original_question_id"),
        # ── response ────────────────────────────────────────────────────────
        "architecture": architecture,
        "answer": "",
        "context_used": [],
        "citations": [],
        # ── run metadata ────────────────────────────────────────────────────
        "latency_sec": latency_sec,
        "status": "error",
        "error": error,
    }


# ---------------------------------------------------------------------------
# Backend health / architecture validation
# ---------------------------------------------------------------------------

def validate_backend_architecture(
    *,
    api_base_url: str,
    expected_architecture: str,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Hit GET /health on the backend and assert the architecture matches.

    Returns the parsed health payload on success.
    Raises RuntimeError if the server is unreachable or the architecture
    does not match.
    """
    health_url = f"{api_base_url}/health"
    try:
        resp = requests.get(health_url, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach backend at {api_base_url}. "
            "Is the server running?"
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Backend health check failed: {exc}") from exc

    payload = resp.json()
    actual = payload.get("architecture", "")
    if actual != expected_architecture:
        raise RuntimeError(
            f"Architecture mismatch: expected '{expected_architecture}', "
            f"but backend reports '{actual}'. "
            "Make sure you started the correct backend."
        )

    return payload
