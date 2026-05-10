"""
brain/context_marl_ac/context_engineering/evidence_pack_builder.py
------------------------------------------------------------------
Converts raw retrieved chunks into structured evidence records.
"""

from typing import List, Dict, Any

def build_evidence_pack(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts raw chunks into structured evidence records.
    
    Input: List[{text, metadata, score}]
    Output: List[{chunk_id, source, page, section, text, retrieval_score, grade, citation_id}]
    """
    evidence_pack = []
    
    for idx, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        
        item = {
            "chunk_id": str(idx),
            "source": str(metadata.get("source_file", "Unknown Source")),
            "page": str(metadata.get("page_number", "Unknown Page")),
            "section": str(metadata.get("section_header", "Unknown Section")),
            "text": str(chunk.get("text", "")),
            "retrieval_score": float(chunk.get("score", 0.0)),
            "grade": "ungraded",
            "citation_id": f"[{idx + 1}]"
        }
        evidence_pack.append(item)
        
    return evidence_pack
