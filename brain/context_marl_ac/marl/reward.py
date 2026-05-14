"""
brain/context_marl_ac/marl/reward.py
------------------------------------
Cooperative reward function for the MARL system.
"""

import re
from typing import Dict, Any, Tuple
from context_marl_ac.config import (
    W_ANSWER_QUALITY, W_CITATION_SUPPORT, W_VERIFICATION_PASS, W_RETRIEVAL_F1,
    W_LATENCY_COST, W_STEP_COST,
    PENALTY_HALLUCINATION, PENALTY_UNSUPPORTED_CLAIM, PENALTY_REPEATED_ACTION,
    PENALTY_INVALID_ACTION, PENALTY_NO_ANSWER, PENALTY_MAX_STEPS
)
from context_marl_ac.schemas.context_state import ContextState


def _token_f1(pred: str, gold: str) -> float:
    """Token-set F1 between predicted and gold answer. Used as a quality proxy."""
    p = set(re.findall(r"\b\w+\b", (pred or "").lower()))
    g = set(re.findall(r"\b\w+\b", (gold or "").lower()))
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec = tp / len(p)
    rec  = tp / len(g)
    return 2 * prec * rec / (prec + rec)

def calculate_reward(
    state: ContextState, 
    action_name: str, 
    is_terminal: bool,
    gold_answer: str = "",
    gold_chunks: list = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculates the shared cooperative reward for the current step.
    
    Returns:
        (total_reward, reward_components_dict)
    """
    components = {}
    reward = 0.0
    
    # 1. Per-step cost (small fixed cost to discourage purely random exploration)
    reward -= W_STEP_COST
    components["step_cost"] = -W_STEP_COST
    
    # 2. Penalty for repeated actions (encourages variety/efficiency)
    if state.num_steps > 1 and state.previous_actions and state.last_action_for(state.previous_actions[-1]["agent"]) == action_name:
         # Penalty if same agent does same action twice in a row (unless it's retrieval)
         if state.previous_actions[-1]["agent"] != "retriever":
             reward += PENALTY_REPEATED_ACTION
             components["penalty_repeated"] = PENALTY_REPEATED_ACTION

    # 3. Terminal Rewards (Positive & Negative)
    if is_terminal:
        # A. Answer Quality — token F1 against gold (not a constant!)
        q_score = 0.0
        if state.generated_answer:
            if "DRY-RUN" in state.generated_answer:
                q_score = 0.85
            elif gold_answer:
                q_score = _token_f1(state.generated_answer, gold_answer)

        reward += W_ANSWER_QUALITY * q_score
        components["answer_quality"] = float(W_ANSWER_QUALITY * q_score)

        # A2. Latency cost — charged once at episode end, normalized to [0, 1]
        # by 120 s reference (typical Groq budget). Caps prevent latency dominating
        # over quality terms.
        latency_norm = min(state.latency_so_far / 120.0, 1.0)
        latency_cost = W_LATENCY_COST * latency_norm
        reward -= latency_cost
        components["latency_cost"] = -latency_cost
        
        # B. Citation Support & Source Accuracy
        reward += W_CITATION_SUPPORT * state.citation_support_rate
        components["citation_support"] = W_CITATION_SUPPORT * state.citation_support_rate
        
        if state.citation_candidates and state.expected_sources:
            cit_sources = {c.get("source_file") for c in state.citation_candidates if c.get("source_file")}
            exp_sources = set(state.expected_sources)
            correct_cit = len(cit_sources.intersection(exp_sources))
            cit_acc = correct_cit / len(cit_sources) if cit_sources else 0.0
            components["citation_source_accuracy"] = cit_acc
            reward += 0.1 * cit_acc
        
        # C. Verification Pass
        if state.final_status == "accepted":
            reward += W_VERIFICATION_PASS
            components["verification_pass"] = W_VERIFICATION_PASS
        elif state.final_status == "rejected":
            reward += PENALTY_HALLUCINATION
            components["penalty_hallucination"] = PENALTY_HALLUCINATION
            
        # D. Source-level Retrieval Metrics
        if state.expected_sources and state.retrieved_chunks:
            ret_sources = {c.get("metadata", {}).get("source_file") for c in state.retrieved_chunks if c.get("metadata", {}).get("source_file")}
            exp_sources = set(state.expected_sources)
            intersection = ret_sources.intersection(exp_sources)
            
            hit = 1.0 if intersection else 0.0
            precision = len(intersection) / len(ret_sources) if ret_sources else 0.0
            recall = len(intersection) / len(exp_sources) if exp_sources else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            components["source_hit_at_k"] = hit
            components["source_precision_at_k"] = precision
            components["source_recall_at_k"] = recall
            components["source_f1_at_k"] = f1
            
            reward += W_RETRIEVAL_F1 * f1
            components["retrieval_f1"] = W_RETRIEVAL_F1 * f1
        elif gold_chunks and state.retrieved_chunks:
            retrieved_texts = {c.get("text", "").strip() for c in state.retrieved_chunks}
            gold_texts = {str(gc).strip() for gc in gold_chunks}
            intersection = len(retrieved_texts.intersection(gold_texts))
            recall = intersection / len(gold_texts) if gold_texts else 0.0
            precision = intersection / len(retrieved_texts) if retrieved_texts else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            reward += W_RETRIEVAL_F1 * f1
            components["retrieval_f1"] = W_RETRIEVAL_F1 * f1

        # E. Critical Penalties
        if not (state.generated_answer or "").strip():
            reward += PENALTY_NO_ANSWER
            components["penalty_no_answer"] = PENALTY_NO_ANSWER
        if len(state.unsupported_claims) > 0:
            p = PENALTY_UNSUPPORTED_CLAIM * len(state.unsupported_claims)
            reward += p
            components["penalty_unsupported"] = p
        if state.final_status == "timeout":
            reward += PENALTY_MAX_STEPS
            components["penalty_timeout"] = PENALTY_MAX_STEPS

    return round(reward, 4), components
