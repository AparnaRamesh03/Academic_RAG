"""
brain/context_marl_ac/context_engineering/query_analyzer.py
-----------------------------------------------------------
Analyzes the user query to classify its type and complexity.
Used by ContextBuilder to populate initial ContextState fields.
"""

import re
from typing import Dict, Any, Tuple

def analyze_query(query: str) -> Dict[str, Any]:
    """
    Classify query type and estimate complexity.
    """
    query_lower = query.lower()
    
    # 1. Determine Query Type
    query_type = "factual"  # default
    
    type_rules = {
        "conceptual": [r"\bexplain\b", r"\bdescribe\b", r"\bhow does\b", r"\bwhy\b", r"\bmechanism\b", r"\btheory\b"],
        "comparison": [r"\bcompare\b", r"\bdifference\b", r"\bvs\b", r"\bversus\b", r"\bcontrast\b", r"\balternative\b"],
        "section_specific": [r"\bsection\b", r"\bchapter\b", r"\bfigure\b", r"\btable\b", r"\bconclusion\b"],
        "multi_hop": [r"\band\b", r"\bconnect\b", r"\bboth\b", r"\brelation\b"],
        "definition": [r"\bdefine\b", r"\bdefinition\b", r"\bwhat is a\b", r"\bwhat does\b"],
        "summarization": [r"\bsummarize\b", r"\boverview\b", r"\bsummary\b", r"\bmain points\b"]
    }
    
    for qtype, patterns in type_rules.items():
        if any(re.search(p, query_lower) for p in patterns):
            query_type = qtype
            break
            
    # 2. Estimate Complexity
    # Heuristics: length, specific keywords, number of sources likely needed
    word_count = len(query.split())
    
    complexity = "low"
    if word_count > 15 or query_type in ["comparison", "multi_hop"]:
        complexity = "high"
    elif word_count > 8 or query_type in ["conceptual", "summarization"]:
        complexity = "medium"
        
    # 3. Specific flags
    requires_multiple_sources = query_type in ["comparison", "multi_hop", "summarization"]
    requires_strict_citation = query_type in ["factual", "definition", "section_specific"]
    
    return {
        "query_type": query_type,
        "query_complexity": complexity,
        "requires_multiple_sources": requires_multiple_sources,
        "requires_strict_citation": requires_strict_citation
    }
