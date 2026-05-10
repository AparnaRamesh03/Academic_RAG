"""
brain/context_marl_ac/context_engineering/feature_encoder.py
-------------------------------------------------------------
Encodes ContextState into a fixed-length numerical feature vector.
FEATURE_DIM = 14
"""

from typing import List, Dict, Any

def encode_features(state: Any) -> List[float]:
    """
    Converts ContextState to List[float] of length 14.
    Expects state to be a ContextState object or equivalent dict.
    """
    # Helper to get attributes from dataclass or dict
    def get_val(obj, key, default=0):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    # Mapping of types and complexities to IDs
    query_types = ["factual", "conceptual", "comparison", "section_specific", "multi_hop", "definition", "summarization"]
    complexities = ["low", "medium", "high"]
    
    q_type = get_val(state, "query_type", "factual")
    q_comp = get_val(state, "query_complexity", "medium")
    
    type_id = query_types.index(q_type) if q_type in query_types else 0
    comp_id = complexities.index(q_comp) if q_comp in complexities else 1
    
    # Retrieval stats
    retrieved = get_val(state, "retrieved_chunks", [])
    scores = get_val(state, "retrieval_scores", [])
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    # Grading stats
    graded = get_val(state, "graded_chunks", [])
    rel_ratio = len(graded) / len(retrieved) if retrieved else 0.0
    
    # Selection stats
    selected = get_val(state, "selected_evidence", [])
    
    # Verification stats
    ver_result = get_val(state, "verification_result", {})
    ver_failed = 1.0 if ver_result.get("decision") == "FAIL" else 0.0
    
    # Progress stats
    retry_count = get_val(state, "retry_count", 0)
    num_steps = get_val(state, "num_steps", 0)
    llm_calls = get_val(state, "num_llm_calls", 0)
    latency = get_val(state, "latency_so_far", 0.0)
    
    # Action history
    prev_actions = get_val(state, "previous_actions", [])
    last_action_id = 0.0
    if prev_actions:
        # Just a simple heuristic for last action "intensity"
        last_action_id = len(prev_actions) / 10.0

    features = [
        float(type_id) / 6.0,                  # 1: query_type_id (0-6)
        float(comp_id) / 2.0,                  # 2: query_complexity_id (0-2)
        float(avg_score),                      # 3: retrieval_confidence
        min(float(len(retrieved)) / 20.0, 1.0),# 4: num_retrieved_chunks (capped at 20)
        float(rel_ratio),                      # 5: graded_relevance_ratio
        min(float(len(selected)) / 10.0, 1.0), # 6: selected_evidence_count (capped at 10)
        float(get_val(state, "citation_support_rate", 0.0)), # 7: citation_support_rate
        min(float(len(get_val(state, "unsupported_claims", []))) / 5.0, 1.0), # 8: unsupported_claim_count
        float(ver_failed),                     # 9: verification_failed flag
        min(float(retry_count) / 5.0, 1.0),    # 10: retry_count
        min(float(latency) / 60.0, 1.0),       # 11: latency_so_far (capped at 60s)
        min(float(num_steps) / 10.0, 1.0),     # 12: num_steps
        min(float(llm_calls) / 15.0, 1.0),     # 13: num_llm_calls
        min(last_action_id, 1.0)               # 14: previous_action_intensity
    ]
    
    return features
