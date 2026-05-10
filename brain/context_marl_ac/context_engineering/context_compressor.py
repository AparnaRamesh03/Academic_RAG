"""
brain/context_marl_ac/context_engineering/context_compressor.py
---------------------------------------------------------------
Handles deduplication and budget enforcement for context.
"""

import hashlib
from typing import List, Dict, Any

def compress_context(
    evidence_pack: List[Dict[str, Any]], 
    max_chars: int = 8000,
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Deduplicates, filters by score, and enforces character budget.
    """
    if not evidence_pack:
        return []
        
    # 1. Filter by score
    filtered = [item for item in evidence_pack if item.get("retrieval_score", 0.0) >= min_score]
    
    # 2. Deduplicate by text hash
    unique_items = []
    seen_hashes = set()
    
    for item in filtered:
        text = item.get("text", "").strip()
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        
        if text_hash not in seen_hashes:
            unique_items.append(item)
            seen_hashes.add(text_hash)
            
    # 3. Enforce budget (priority by retrieval score)
    # Already sorted by retrieval if they come from hybrid search, but let's be safe
    sorted_items = sorted(unique_items, key=lambda x: x.get("retrieval_score", 0.0), reverse=True)
    
    compressed = []
    current_chars = 0
    
    for item in sorted_items:
        text_len = len(item.get("text", ""))
        if current_chars + text_len <= max_chars:
            compressed.append(item)
            current_chars += text_len
        elif not compressed: # ensure at least one if possible
             compressed.append(item)
             break
        else:
            break
            
    return compressed
