"""
brain/context_marl_ac/context_engineering/citation_context_builder.py
---------------------------------------------------------------------
Prepares formatted context for the generator and manages citation mapping.
"""

from typing import List, Dict, Any

def build_citation_context(evidence_pack: List[Dict[str, Any]]) -> str:
    """
    Creates a formatted string of documents for the LLM prompt.
    """
    if not evidence_pack:
        return "No evidence provided."
        
    blocks = []
    for item in evidence_pack:
        cit_id = item.get("citation_id", "[?]")
        source = item.get("source", "Unknown")
        section = item.get("section", "Unknown")
        page = item.get("page", "Unknown")
        text = item.get("text", "")
        
        block = (
            f"DOCUMENT {cit_id}\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )
        blocks.append(block)
        
    return "\n\n---\n\n".join(blocks)

def reassign_citation_ids(evidence_pack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures citation IDs are sequential [1], [2], etc. after filtering/compression.
    """
    for idx, item in enumerate(evidence_pack):
        item["citation_id"] = f"[{idx + 1}]"
    return evidence_pack
