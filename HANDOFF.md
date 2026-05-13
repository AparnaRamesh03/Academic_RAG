# MADDPG RAG Project — Agent Handoff Summary
**Date:** 2026-05-13  
**Working directory:** `c:\Users\juand\Documents\Academic_RAG`  
**Branch:** `RL_update`  
**User email:** juan.liangliao@sjsu.edu

---

## 1. Original Goal

Build and evaluate a MADDPG (Multi-Agent DDPG) continuous-control extension on top of an existing discrete MARL RAG system for an academic thesis/defense. The deliverable was:

- Train two MADDPG variants (no-CEB and with-CEB) for 20 episodes using **real LLM calls** (Groq API)
- Evaluate all three policies (maddpg_no_ceb, maddpg_ceb, discrete_marl baseline) on 9 test questions
- Produce a comparison table and defense-ready interpretation
- Understand why the CEB variant failed

---

## 2. What Has Been Completed

### 2a. MADDPG Module (fully built)
All files under `brain/context_marl_ac/maddpg/`:
- `noise.py` — OUNoise (sigma=0.15) and GaussianNoise
- `continuous_action_mapper.py` — maps actor output [-1,1]^d → RAG params (top_k, temperature, etc.) + discrete action name; JOINT_ACTION_DIM=16
- `maddpg_actor.py` — deterministic MLP actor (tanh output)
- `maddpg_critic.py` — centralized Q(state, joint_action)
- `maddpg_agent.py` — per-agent wrapper with soft target updates
- `replay_buffer.py` — off-policy transition buffer, capacity=50,000
- `context_engineering_block.py` — 20-dim CEB state (14-dim base + 6 extra features)
- `train_maddpg.py` — DDPG training loop
- `evaluate_maddpg.py` — 3-mode comparison evaluator
- `live_maddpg_runner.py` — self-contained live train+eval pipeline (the main script used)

### 2b. Live Training Run (completed)
Both variants trained for 20 episodes each with real Groq LLM calls:

| Variant | Episodes | Accept rate | Avg reward | Buffer size |
|---|---|---|---|---|
| maddpg_no_ceb | 20 | 65% (13/20) | −2.15 | 133 |
| maddpg_ceb | 20 | 55% (11/20) | −3.34 | 164 |

Checkpoints saved to `brain/context_marl_ac/results/defense_comparison_live/checkpoints/`:
- `best_maddpg_no_ceb_live.pt`, `maddpg_no_ceb_live_ep0010.pt`, `maddpg_no_ceb_live_ep0020.pt`
- `best_maddpg_ceb_live.pt`, `maddpg_ceb_live_ep0010.pt`, `maddpg_ceb_live_ep0020.pt`

### 2c. Live Evaluation Run (completed)
All three policies evaluated on 9 test questions. Results in `brain/context_marl_ac/results/defense_comparison_live/`:

| Metric | maddpg_no_ceb | maddpg_ceb | discrete_marl |
|---|---|---|---|
| Token F1 | 0.1853 | 0.0000 | **0.3243** |
| ROUGE-L | 0.1244 | 0.0000 | **0.1937** |
| Faithfulness | 0.1852 | 0.0000 | **0.6919** |
| Verification Pass | 22.2% | 0.0% | **66.7%** |
| Failure Rate | 77.8% | **100%** | 33.3% |
| Avg Latency (s) | 43.9 | 15.5 | 53.8 |
| Avg LLM Calls | 4.00 | 0.00 | 2.44 |

### 2d. Root Cause Analysis (completed)
The CEB failure has been fully diagnosed — see Section 6.

### 2e. Output Files Generated
All in `brain/context_marl_ac/results/defense_comparison_live/`:
- `comparison_summary.csv` — 3-row aggregate comparison
- `episode_metrics.csv` — per-question results for all 3 policies (27 rows)
- `aggregate_metrics.json` — full JSON aggregate stats
- `results_interpretation.md` — 5-section defense write-up
- `maddpg_no_ceb_live.jsonl`, `maddpg_ceb_live.jsonl` — per-question eval trajectories
- `discrete_marl_baseline.jsonl` — baseline results
- `metrics/ep_metrics_*.csv` — per-episode training metrics
- `metrics/action_params_*.csv` — per-step actor outputs during training

---

## 3. Decisions, Assumptions, Preferences, Constraints

### API / Security
- **Groq API key** must ONLY be loaded via `load_dotenv(dotenv_path=...)` from `brain/.env`
- **Never** use `export $(grep -v '^#' .env | xargs)` or expose GROQ_API_KEY in bash commands — user explicitly rejected this
- Model used: `llama-3.3-70b-versatile` (via `GROQ_MODEL` env var)

### Architecture
- MADDPG is **additive** — existing discrete MARL code (`train.py`, `evaluate.py`) is completely untouched
- Stage-gated action masking is **fully preserved** — MADDPG only selects within the valid masked action set
- All agents share one global state observation (not local per-agent obs)
- State dim: 14 (no-CEB) or 20 (CEB); joint action dim: 16

### Hyperparameters (current, from `train_maddpg.py`)
```python
ACTOR_LR        = 1e-3
CRITIC_LR       = 1e-3
GAMMA           = 0.99
TAU             = 0.005
BATCH_SIZE      = 256
BUFFER_CAPACITY = 50_000
HIDDEN_DIM      = 128
NOISE_SIGMA     = 0.15
GRAD_CLIP       = 1.0
UPDATE_EVERY    = 4
WARMUP_STEPS    = 500   # ← CRITICAL: this is the bug (see Section 6)
```

### Benchmark splits
- `brain/context_marl_ac/results/benchmark_splits/train.jsonl` — 38 questions (used for training)
- `brain/context_marl_ac/results/benchmark_splits/val.jsonl` — 4 questions (unused so far)
- `brain/context_marl_ac/results/benchmark_splits/test.jsonl` — 18 questions (first 9 used for eval)
- Source papers: Transformer, BERT, ResNet, TabNet, RAG Survey, Norwegian English teaching

---

## 4. Important Technical Details

### How to Run (from `brain/` directory)
```bash
# Full live training + eval (both variants):
python -m context_marl_ac.maddpg.live_maddpg_runner --episodes 200 --n-eval 9

# Skip training, eval only:
python -m context_marl_ac.maddpg.live_maddpg_runner --skip-training --n-eval 9

# Dry-run (no API calls):
python -m context_marl_ac.maddpg.train_maddpg --episodes 200 --dry-run
python -m context_marl_ac.maddpg.train_maddpg --episodes 200 --use-ceb --dry-run
```

### Agent Action Dimensions
| Agent | Dim | Params controlled |
|---|---|---|
| retriever | 4 | dense_sparse_weight, top_k [5–30], rerank_threshold, source_diversity |
| rewriter | 3 | rewrite_strength, query_expansion_weight, source_focus_weight |
| grader | 3 | relevance_threshold, evidence_keep_ratio, strictness_score |
| generator | 4 | temperature, citation_strictness, max_tokens [128–1024], answer_detail_level |
| verifier | 2 | support_threshold, confidence_threshold |

### CEB Extra State Features (dims 15–20)
source_diversity, evidence_coverage, step_fraction, llm_call_fraction, query_length_norm, requires_multiple_sources

### Rate Limiting
Groq hits rate limits on nearly every grading LLM call. The `_call_with_retry` in `brain/context_marl_ac/adapters/llm_adapter.py` retries up to 10 times with ~2–3s waits. Eval of 9 questions takes ~30–90 min due to this.

---

## 5. Current Status

| Item | Status |
|---|---|
| MADDPG module code | ✅ Complete |
| 20-episode live training (both variants) | ✅ Complete |
| 9-question live evaluation (all 3 policies) | ✅ Complete |
| Results files and comparison table | ✅ Complete |
| Root cause analysis of CEB failure | ✅ Complete |
| **Actual policy learning (gradient updates)** | ❌ Never happened — see Section 6 |
| Re-training with fixed warmup / more episodes | ❌ Not done |
| CEB vs. no-CEB meaningful comparison | ❌ Blocked until re-training |

---

## 6. Problems, Blockers, Root Cause of CEB Failure

### Critical Bug: WARMUP_STEPS Never Reached
```python
# train_maddpg.py
WARMUP_STEPS = 500

if buffer.is_ready(WARMUP_STEPS) and total_steps % UPDATE_EVERY == 0:
    _ddpg_update(agents, ...)   # ← was NEVER called
```

- 20 training episodes × ~6.5 steps/episode ≈ **133–164 total transitions**
- `buffer.is_ready(500)` was always `False`
- `_ddpg_update()` was **never executed** during either variant's training
- Both networks evaluated with **random initialized weights** (zero gradient updates)

### Why CEB Fails 100% in Eval (Proximate)
- Training used OU noise (`explore=True`) → random weights + noise = some lucky episodes (55% accept)
- Eval used greedy mode (`explore=False`) → pure random network output, no noise
- CEB's random 20-dim actor deterministically produces params that crash `env.step` on every question
- Crash trace: retriever step completes fully (`num_steps=1`, `num_llm_calls=0`, ~15s latency from Qdrant call), then `calculate_reward()` inside `env.step` throws an unhandled exception
- Exception is caught in the eval loop, marking status as `"error"`

### Why no-CEB Gets 22% Pass (Also Random)
- Random 14-dim actor happens to produce params that don't crash reward calculation
- 22% pass rate is pure chance from the random initialization, not a trained policy

### Implication for Defense
**Neither variant was actually trained.** The comparison table does NOT show trained MADDPG vs. baseline. It shows random-initialized MADDPG vs. a properly evaluated discrete MARL baseline. This must be fixed before the results are defensible.

---

## 7. Recommended Next Steps

### Step 1 — Fix WARMUP_STEPS (required)
In `brain/context_marl_ac/maddpg/train_maddpg.py`, change:
```python
WARMUP_STEPS = 500   # current — never reached in 20 episodes
```
to either:
```python
WARMUP_STEPS = 50    # option A: ~1/3 of what 20 episodes generates
# OR
WARMUP_STEPS = 100   # option B: crossed after ~15 episodes
```
This is a one-line fix. Do it before any re-training.

### Step 2 — Re-train with Enough Episodes
Minimum to cross warmup + get meaningful learning:
```bash
# Minimum viable (crosses warmup after ~50 episodes):
python -m context_marl_ac.maddpg.live_maddpg_runner --episodes 100 --n-eval 9

# Recommended (meaningful convergence):
python -m context_marl_ac.maddpg.live_maddpg_runner --episodes 300 --n-eval 9
```
Note: 100 live episodes ≈ 2–3 hours due to Groq rate limits. 300 episodes ≈ 6–9 hours.

### Step 3 — Fix the CEB eval crash
Before re-running, identify and fix the exception in `calculate_reward()` that triggers when the retriever runs with CEB's greedy params. Read `brain/context_marl_ac/marl/reward.py` to find what crashes. Add a try/except or guard on whatever computation fails with edge-case retriever output.

### Step 4 — Re-run Evaluation
```bash
python -m context_marl_ac.maddpg.live_maddpg_runner --skip-training --n-eval 18
```
Use all 18 test questions (not just 9) for statistical power.

### Step 5 — Re-generate comparison table and interpretation
The `live_maddpg_runner.py` generates all output files automatically at the end of the run. After a successful re-run, the files in `results/defense_comparison_live/` will be updated.

---

## 8. Formatting / Style Notes

- User is working toward a **thesis defense** — all analysis should be framed with that context
- Keep responses **concise and technical** — no verbose summaries at the end of responses
- Use markdown tables for results comparisons
- File references should use markdown link syntax: `[filename.py](relative/path.py)`
- **No emojis**
- When discussing results, always distinguish between "trained policy" and "random policy" — this distinction is critical given the warmup bug
- The user checks progress frequently during long-running tasks — use background task output files (not TaskOutput which buffers stale snapshots) to verify true progress
- **Never** expose API keys in bash commands; always use `load_dotenv()`

---

## 9. Key File Locations

| File | Purpose |
|---|---|
| `brain/context_marl_ac/maddpg/live_maddpg_runner.py` | Main train+eval pipeline |
| `brain/context_marl_ac/maddpg/train_maddpg.py` | DDPG training loop + hyperparams |
| `brain/context_marl_ac/maddpg/continuous_action_mapper.py` | RAG param mapping |
| `brain/context_marl_ac/maddpg/context_engineering_block.py` | CEB 20-dim state |
| `brain/context_marl_ac/adapters/llm_adapter.py` | Groq API wrapper with retry |
| `brain/context_marl_ac/marl/marl_env.py` | Environment step/reset |
| `brain/context_marl_ac/marl/reward.py` | Reward calculation (likely crash source) |
| `brain/context_marl_ac/results/defense_comparison_live/comparison_summary.csv` | Current (flawed) results |
| `brain/context_marl_ac/results/benchmark_splits/train.jsonl` | 38 training questions |
| `brain/context_marl_ac/results/benchmark_splits/test.jsonl` | 18 test questions |
| `brain/.env` | Groq API key (load via dotenv only) |
