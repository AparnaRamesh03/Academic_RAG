"""
brain/context_marl_ac/adapters/citation_adapter.py
----------------------------------------------------
Wraps brain/citation_utils.py and brain/final_arch/claim_verifier.py
to expose citation-related functions needed by GeneratorAgent and VerifierAgent.

Exposed API
-----------
    build_citations(chunks)
        -> List[dict]  (serializable CitationItem dicts)

    compute_citation_support(answer, citations, evidence_pack)
        -> float  (0.0 – 1.0, fraction of claims that have citation support)

    detect_unsupported_claims(answer, evidence_pack)
        -> List[str]  (list of unsupported claim strings)

In dry-run mode all functions return deterministic stub values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_BRAIN_ROOT = Path(__file__).resolve().parents[2]
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
# Lazy imports
# ---------------------------------------------------------------------------
_citation_utils_loaded = False
_build_citations_fn    = None
_verify_claims_fn      = None


def _ensure_citation_loaded() -> None:
    global _citation_utils_loaded, _build_citations_fn, _verify_claims_fn
    if _citation_utils_loaded:
        return
    try:
        from citation_utils import build_citations_from_docs
        from claim_verifier import verify_claims

        _build_citations_fn = build_citations_from_docs
        _verify_claims_fn   = verify_claims
        _citation_utils_loaded = True

    except Exception as exc:
        raise ImportError(
            f"[citation_adapter] Failed to load citation utilities.\n"
            f"Make sure brain/ and brain/final_arch/ are on sys.path.\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Evidence pack → legacy docs conversion (shared helper)
# ---------------------------------------------------------------------------

def _evidence_pack_to_docs(evidence_pack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for item in evidence_pack:
        docs.append({
            "text": item.get("text", ""),
            "metadata": {
                "source_file":    item.get("source", "Unknown Source"),
                "page_number":    item.get("page", "Unknown Page"),
                "section_header": item.get("section", "Unknown Section"),
                "content_type":   "text",
            },
            "score": item.get("retrieval_score", 0.0),
        })
    return docs


def _is_evidence_pack(items: List[Dict[str, Any]]) -> bool:
    if not items:
        return False
    return "chunk_id" in items[0] or "retrieval_score" in items[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build serializable citation dicts from a list of retrieved chunks or
    evidence_pack items.

    Parameters
    ----------
    chunks : list
        Either raw retriever chunks {text, metadata, score}
        or evidence_pack items {chunk_id, source, page, section, text, ...}.

    Returns
    -------
    List[dict] — each dict has:
        source_file, page_number, section_header, excerpt, content_type
    """
    if cfg.DRY_RUN:
        return [
            {
                "source_file":    "AttentionIsAllYouNeed.pdf",
                "page_number":    3,
                "section_header": "Model Architecture",
                "excerpt":        "Transformer models introduced by Vaswani et al...",
                "content_type":   "text",
            }
        ]

    _ensure_citation_loaded()

    # Normalise to legacy format if needed
    docs = _evidence_pack_to_docs(chunks) if _is_evidence_pack(chunks) else chunks

    citation_objects = _build_citations_fn(docs)

    # Convert Pydantic models to plain dicts for JSON-serializability
    return [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in citation_objects]


def compute_citation_support(
    answer: str,
    citations: List[Dict[str, Any]],
    evidence_pack: List[Dict[str, Any]],
) -> float:
    """
    Estimate what fraction of the answer is supported by the provided citations.

    Uses the claim verifier to split the answer into claims and check each
    claim against the evidence pack.

    Returns
    -------
    float in [0.0, 1.0] — citation support rate.
    """
    if cfg.DRY_RUN:
        if not answer.strip() or not evidence_pack:
            return 0.0
        return 0.95

    if not answer.strip() or not evidence_pack:
        return 0.0

    _ensure_citation_loaded()

    docs = _evidence_pack_to_docs(evidence_pack) if _is_evidence_pack(evidence_pack) else evidence_pack

    # Reuse claim verifier — gold answer NOT used here, only evidence vs answer
    try:
        result = _verify_claims_fn("", answer, docs)
        claims = result.get("claims", [])
        if not claims:
            return 0.0
        supported = sum(1 for c in claims if c.get("supported", False))
        return round(supported / len(claims), 4)
    except Exception:
        return 0.0


def detect_unsupported_claims(
    answer: str,
    evidence_pack: List[Dict[str, Any]],
) -> List[str]:
    """
    Identify which claims in the answer are NOT supported by the evidence pack.

    Parameters
    ----------
    answer        : str — generated answer.
    evidence_pack : list — evidence_pack items or raw chunk dicts.

    Returns
    -------
    List[str] — text of each unsupported claim.
    """
    if cfg.DRY_RUN:
        return []

    if not answer.strip() or not evidence_pack:
        return [answer] if answer.strip() else []

    _ensure_citation_loaded()

    docs = _evidence_pack_to_docs(evidence_pack) if _is_evidence_pack(evidence_pack) else evidence_pack

    try:
        result = _verify_claims_fn("", answer, docs)
        claims = result.get("claims", [])
        return [
            c.get("claim_text", "")
            for c in claims
            if not c.get("supported", True)
        ]
    except Exception:
        return []
