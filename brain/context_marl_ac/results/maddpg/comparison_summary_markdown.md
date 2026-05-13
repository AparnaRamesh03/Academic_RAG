# Defense-Ready Evaluation: Policy Comparison Summary

**Benchmark:** 9 questions (Q001–Q009), identical across all systems  
**Data sources:** `discrete_marl` = real LLM (Groq); MADDPG = dry-run stubs pending real training  
**Generated:** 2026-05-13

---

## Metric Definitions

| Symbol | Metric | Notes |
|--------|--------|-------|
| TF1 | Token F1 | Set-based precision/recall over lowercased tokens |
| RL | ROUGE-L | Longest Common Subsequence F1 |
| Corr | Correctness | Identical to Token F1 (lexical proxy) |
| Faith | Faithfulness | Fraction of claims supported by retrieved evidence |
| Cit | Citation Support Rate | Citations present and matching evidence |
| Src-P | Source Precision | Retrieved PDFs that are relevant / all retrieved PDFs |
| Src-R | Source Recall | Relevant PDFs retrieved / all relevant PDFs |
| VP | Verification Pass Rate | Fraction of episodes where verifier accepted |
| UC | Avg Unsupported Claims | Claims in final answer not backed by evidence |
| Lat | Avg Latency (s) | Wall-clock time per episode |
| LLM | Avg LLM Calls | Groq API calls per episode |
| Steps | Avg Workflow Steps | Stages executed per episode |
| Tok | Avg Token Usage | Prompt + completion tokens per episode |
| Fail | Failure Rate | Episodes ending in rejected/error/timeout |

---

## Comparison Table

| Metric | discrete_marl | maddpg_no_ce | maddpg_with_ce |
|--------|:-------------:|:------------:|:--------------:|
| **n questions** | 9 | 9 | 9 |
| **data source** | real LLM | dry-run stub | dry-run stub |
| **Token F1** | **0.2373** | 0.0476 | 0.0476 |
| **ROUGE-L** | **0.1912** | 0.0350 | 0.0350 |
| **Correctness** | **0.2373** | 0.0476 | 0.0476 |
| **Faithfulness** | 0.7778 | 1.0000* | 1.0000* |
| **Citation Support** | 0.7778 | 1.0000* | 1.0000* |
| **Source Precision** | **0.1111** | 0.0370 | 0.0370 |
| **Source Recall** | 0.1111 | 0.1111 | 0.1111 |
| **Verif. Pass Rate** | 0.6667 | 1.0000* | 1.0000* |
| **Unsupported Claims** | 0.5556 | 0.0000* | 0.0000* |
| **Avg Latency (s)** | 37.61 | 0.0011 | **0.0010** |
| **Avg LLM Calls** | **2.11** | 3.00 | 3.00 |
| **Avg Workflow Steps** | 4.0 | 4.0 | 4.0 |
| **Avg Token Usage** | 3413.78 | 0.00 | 0.00 |
| **Failure Rate** | 0.3333 | **0.0000** | **0.0000** |

> \* Stub artifact: dry-run verifier always passes; faithfulness/citation/pass-rate are 1.0 by construction, not earned.  
> Bold = best observed value for that metric.

---

## Notes on Data Integrity

- `discrete_marl` numbers reflect **real LLM inference** against the actual Qdrant index and Groq API.  
  Token F1 and ROUGE-L are computed against human-written ground-truth answers.
- `maddpg_no_ce` and `maddpg_with_ce` use a **fixed stub answer** during dry-run:  
  `"[DRY-RUN] Answer about Transformer models introduced by Vaswani et al..."`.  
  All NLP quality metrics (Token F1, ROUGE-L, Correctness) reflect similarity to ground-truth, not actual generation quality.
- Both MADDPG variants are numerically identical because the stub environment ignores actor parameter outputs for answer generation.
- Latency for MADDPG stubs (~1 ms) measures only the actor forward pass + env step; it excludes all real API calls.
- After real training with live inference, MADDPG metrics will be directly comparable to `discrete_marl`.
