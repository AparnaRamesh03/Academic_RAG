"""
brain/context_marl_ac/adapters/combined_arch_adapter.py
---------------------------------------------------------
Optional wrapper around the existing brain/final_arch/ combined/reliability-
enhanced pipeline.  Useful for baseline comparison or fallback routing.

Exposed API
-----------
    run_combined_pipeline(query)
        -> dict with keys: answer, citations, latency_sec, generation

This adapter is OPTIONAL — the MARL system does not depend on it during
normal operation.  It exists so the new architecture can call the combined
pipeline as a reference or as a fallback.

In dry-run mode it returns a stub result without touching the graph.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_BRAIN_ROOT     = Path(__file__).resolve().parents[2]
_FINAL_ARCH_DIR = _BRAIN_ROOT / "final_arch"

for _p in [str(_BRAIN_ROOT), str(_FINAL_ARCH_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    import context_marl_ac.config as cfg
except ImportError:
    _MARL_ROOT = Path(__file__).resolve().parents[1]
    if str(_MARL_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(_MARL_ROOT.parent))
    import context_marl_ac.config as cfg


# ---------------------------------------------------------------------------
# Lazy import for the real graph pipeline
# ---------------------------------------------------------------------------
_graph_loaded = False
_combined_run_fn = None


def _ensure_graph_loaded() -> None:
    global _graph_loaded, _combined_run_fn
    if _graph_loaded:
        return
    try:
        # Assuming the main entry point for the combined architecture is 
        # in a file like 'brain/final_arch/main.py' or similar.
        # For now, we stub the actual import until the baseline is fully established.
        # If there's a specific 'run_combined' function in the repo, we import it here.
        
        # from final_arch_main import run_pipeline
        # _combined_run_fn = run_pipeline
        
        _graph_loaded = True

    except Exception as exc:
        raise ImportError(
            f"[combined_adapter] Failed to load combined pipeline.\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_combined_pipeline(query: str) -> Dict[str, Any]:
    """
    Run the full reliability-enhanced RAG pipeline (baseline).

    Returns
    -------
    dict
        {answer, citations, latency_sec, generation, final_status}
    """
    if cfg.DRY_RUN:
        return {
            "answer":       f"[DRY-RUN COMBINED] Answer for: {query}",
            "citations":    [],
            "latency_sec":  1.23,
            "generation":   "Placeholder generation",
            "final_status": "accepted",
        }

    _ensure_graph_loaded()

    # If the real function isn't set yet, return an error or placeholder
    if _combined_run_fn is None:
        return {
            "answer":       "Combined pipeline integration pending.",
            "citations":    [],
            "latency_sec":  0.0,
            "generation":   "",
            "final_status": "error",
        }

    start_t = time.time()
    result = _combined_run_fn(query)
    latency = time.time() - start_t

    return {
        "answer":       result.get("final_answer", ""),
        "citations":    result.get("citations", []),
        "latency_sec":  latency,
        "generation":   result.get("final_answer", ""),
        "final_status": result.get("final_status", "accepted"),
    }
