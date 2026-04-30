from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL

llm = build_groq_llm(temperature=0.0)


def _build_context_blocks(docs: List[dict]) -> str:
    """
    Build verifier context blocks with explicit doc ids so the model can map
    each claim to supporting document ids.
    """
    blocks = []

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown Source")
        section = metadata.get("section_header", "Unknown Section")
        page = metadata.get("page_number", "Unknown Page")
        content_type = metadata.get("content_type", "text")

        block = (
            f"DOCUMENT [{idx}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Content Type: {content_type}\n"
            f"Text:\n{doc.get('text', '')}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def _doc_lookup(docs: List[dict]) -> Dict[int, Dict[str, Any]]:
    """
    1-indexed lookup for document metadata used after verification.
    """
    lookup: Dict[int, Dict[str, Any]] = {}
    for idx, doc in enumerate(docs, start=1):
        meta = doc.get("metadata", {}) or {}
        lookup[idx] = {
            "source_file": meta.get("source_file", "Unknown Source"),
            "page_number": meta.get("page_number", "Unknown Page"),
            "section_header": meta.get("section_header", "Unknown Section"),
            "content_type": meta.get("content_type", "text"),
            "excerpt": str(doc.get("text", ""))[:300],
        }
    return lookup


def split_answer_into_claims(answer: str) -> List[str]:
    """
    Split answer into claim-like units.
    Keeps this simple and deterministic.
    """
    text = answer.strip()
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    claims = []

    for piece in pieces:
        clean = " ".join(piece.strip().split())
        if not clean:
            continue
        if len(clean) < 8:
            continue
        claims.append(clean)

    return claims


def _extract_json_block(text: str) -> Dict[str, Any]:
    """
    Try to safely extract JSON from an LLM response.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    raise ValueError("Could not parse claim verification JSON response.")


def _normalize_doc_ids(raw_ids: Any, max_doc_id: int) -> List[int]:
    """
    Normalize supporting_doc_ids into a clean sorted unique list of valid ints.
    """
    if not isinstance(raw_ids, list):
        return []

    cleaned = []
    for value in raw_ids:
        try:
            doc_id = int(value)
        except Exception:
            continue

        if 1 <= doc_id <= max_doc_id and doc_id not in cleaned:
            cleaned.append(doc_id)

    return cleaned


def _build_support_records(doc_ids: List[int], doc_index: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert doc ids into richer source records for debug output.
    """
    support_records = []

    for doc_id in doc_ids:
        if doc_id not in doc_index:
            continue

        meta = doc_index[doc_id]
        support_records.append(
            {
                "doc_id": doc_id,
                "source_file": meta["source_file"],
                "page_number": meta["page_number"],
                "section_header": meta["section_header"],
                "content_type": meta["content_type"],
                "excerpt": meta["excerpt"],
            }
        )

    return support_records


def verify_claims(
    query: str,
    answer: str,
    docs: List[dict],
) -> Dict[str, Any]:
    """
    Verify the generated answer claim by claim against retrieved documents.

    Returns:
    {
        "decision": "PASS" | "FAIL",
        "overall_feedback": str,
        "claims": [
            {
                "claim_id": int,
                "claim_text": str,
                "supported": bool,
                "feedback": str,
                "supporting_doc_ids": [1, 3],
                "supporting_sources": [
                    {
                        "doc_id": 1,
                        "source_file": "...",
                        "page_number": 4,
                        "section_header": "...",
                        "content_type": "text",
                        "excerpt": "..."
                    }
                ]
            }
        ]
    }
    """
    claims = split_answer_into_claims(answer)

    if not claims:
        return {
            "decision": "FAIL",
            "overall_feedback": "The answer is empty or too fragmentary to verify.",
            "claims": [],
        }

    context = _build_context_blocks(docs)
    doc_index = _doc_lookup(docs)

    numbered_claims = "\n".join(
        f"{idx}. {claim}" for idx, claim in enumerate(claims, start=1)
    )

    prompt = f"""You are verifying an academic QA answer claim by claim using retrieved evidence.

User Question:
{query}

Retrieved Documents:
---
{context}
---

Generated Answer:
{answer}

Claims to verify:
{numbered_claims}

Task:
For each claim:
- mark supported = true only if the claim is clearly grounded in the retrieved documents
- mark supported = false if the claim is unsupported, too broad, too strong, or not clearly grounded
- feedback must be short
- if supported, feedback can be "supported"
- if unsupported, feedback should briefly say what is wrong
- supporting_doc_ids must list the DOCUMENT numbers that directly support the claim
- if a claim is unsupported, use an empty list for supporting_doc_ids
- do not include document ids that are only loosely related

Decision rule:
- PASS only if ALL claims are supported and the answer is sufficiently specific to the user question
- FAIL if any claim is unsupported or if the answer is too broad/incomplete for the question

Return ONLY valid JSON in exactly this schema:
{{
  "decision": "PASS or FAIL",
  "overall_feedback": "short sentence",
  "claims": [
    {{
      "claim_id": 1,
      "claim_text": "exact claim text",
      "supported": true,
      "feedback": "supported",
      "supporting_doc_ids": [1]
    }}
  ]
}}
"""

    print("\n[Claim Verifier] Verifying answer claim by claim...")
    print(f"  -> Groq model: {GROQ_MODEL}")
    print(f"  -> Claims to verify: {len(claims)}")

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = _extract_json_block(response.content)

    decision = str(parsed.get("decision", "FAIL")).strip().upper()
    overall_feedback = str(
        parsed.get(
            "overall_feedback",
            "The answer contains unsupported or insufficiently grounded claims.",
        )
    ).strip()

    raw_claims = parsed.get("claims", [])
    cleaned_claims = []

    for idx, claim in enumerate(raw_claims, start=1):
        claim_id = int(claim.get("claim_id", idx))
        claim_text = str(
            claim.get("claim_text", claims[min(idx - 1, len(claims) - 1)])
        ).strip()
        supported = bool(claim.get("supported", False))
        feedback = str(claim.get("feedback", "unsupported")).strip()

        supporting_doc_ids = _normalize_doc_ids(
            claim.get("supporting_doc_ids", []),
            max_doc_id=len(docs),
        )

        # Unsupported claims should not keep support ids even if the model outputs some.
        if not supported:
            supporting_doc_ids = []

        supporting_sources = _build_support_records(supporting_doc_ids, doc_index)

        cleaned_claims.append(
            {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "supported": supported,
                "feedback": feedback,
                "supporting_doc_ids": supporting_doc_ids,
                "supporting_sources": supporting_sources,
            }
        )

    if len(cleaned_claims) != len(claims):
        cleaned_claims = []
        for idx, claim_text in enumerate(claims, start=1):
            cleaned_claims.append(
                {
                    "claim_id": idx,
                    "claim_text": claim_text,
                    "supported": False,
                    "feedback": "Claim verification output was incomplete.",
                    "supporting_doc_ids": [],
                    "supporting_sources": [],
                }
            )
        decision = "FAIL"
        overall_feedback = "Claim verification output was incomplete."

    return {
        "decision": "PASS" if decision == "PASS" else "FAIL",
        "overall_feedback": overall_feedback,
        "claims": cleaned_claims,
    }