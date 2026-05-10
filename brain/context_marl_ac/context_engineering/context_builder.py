"""
brain/context_marl_ac/context_engineering/context_builder.py
-----------------------------------------------------------
Orchestrator for the context engineering phase.
"""

from typing import Dict, Any, List
import sys
from pathlib import Path

# Add brain root to sys.path for relative imports if needed
_BRAIN_ROOT = Path(__file__).resolve().parents[2]
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

try:
    from context_marl_ac.schemas.context_state import ContextState
    from context_marl_ac.context_engineering.query_analyzer import analyze_query
    from context_marl_ac.context_engineering.evidence_pack_builder import build_evidence_pack
except ImportError:
    # Fallback for internal structure within context_engineering
    from schemas.context_state import ContextState
    from context_engineering.query_analyzer import analyze_query
    from context_engineering.evidence_pack_builder import build_evidence_pack

def initialize_context(question_dict: Dict[str, Any], index: int = 1) -> ContextState:
    """
    Creates initial ContextState and performs query analysis.
    """
    state = ContextState.from_question(question_dict, index=index)
    
    # 1. Analyze query
    analysis = analyze_query(state.user_query)
    
    state.query_type = analysis["query_type"]
    state.query_complexity = analysis["query_complexity"]
    state.requires_multiple_sources = analysis["requires_multiple_sources"]
    state.requires_strict_citation = analysis["requires_strict_citation"]
    
    return state

def update_evidence(state: ContextState, raw_chunks: List[Dict[str, Any]]):
    """
    Updates the state with new retrieved evidence.
    """
    state.retrieved_chunks = raw_chunks
    state.retrieval_scores = [c.get("score", 0.0) for c in raw_chunks]
    
    # Initially, evidence pack matches raw chunks
    state.selected_evidence = build_evidence_pack(raw_chunks)
    
    return state
