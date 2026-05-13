# MADDPG Continuous-Control Extension

Extends the existing stage-constrained cooperative MARL RAG system with
MADDPG-style continuous action parameterisation, without replacing or
breaking any existing discrete MARL code.

---

## 1. Existing Discrete MARL Setup

The existing system (`brain/context_marl_ac/`) trains five cooperative agents
with A2C (on-policy) and a centralized critic:

| Agent     | Discrete actions | Role |
|-----------|-----------------|------|
| retriever | dense_retrieve, sparse_retrieve, hybrid_retrieve, hybrid_rerank, retrieve_more | Chooses retrieval strategy |
| rewriter  | no_rewrite, simple_rewrite, keyword_rewrite, expanded_rewrite, multi_query_rewrite | Reformulates query (recovery only) |
| grader    | keep_all, loose_filter, medium_filter, strict_filter, rerank_only | Filters retrieved evidence |
| generator | generate_answer, generate_with_strict_citations, generate_short_answer, abstain, regenerate | Produces answer |
| verifier  | verify_answer, request_regeneration, request_more_retrieval, request_rewrite | Accepts or triggers recovery |

**Stage flow (unchanged):**
```
START → retriever(hybrid_rerank)
      → grader(any)
      → generator(any except abstain/regenerate when evidence exists)
      → verifier(verify_answer)
         ↓ on FAIL + retry_count < 2
      → [request_regeneration | request_more_retrieval | request_rewrite]
         → recovery path → verifier again
```

**Training:** A2C, actor networks output logit distributions over discrete
actions, masked by stage-gated rules. Centralized critic estimates V(s) from
14-dim global feature vector. On-policy Monte Carlo advantage estimation.

---

## 2. What MADDPG Adds

MADDPG replaces the **discrete action selection** with **continuous
parameterisation** while keeping the stage-gated flow identical.

Key additions:

| | Discrete MARL | MADDPG extension |
|--|--|--|
| Actor output | Softmax logits → categorical sample | Tanh → continuous [-1,1]^d |
| Critic input | V(s) from 14-dim features | Q(s, joint_a) from state + 16-dim joint action |
| Update rule | A2C on-policy | DDPG off-policy with replay buffer |
| Stage flow | Unchanged | Unchanged (masking fully preserved) |
| Exploration | Entropy bonus | Ornstein-Uhlenbeck noise |
| Discrete action | Sampled from masked distribution | Derived from continuous params via mapper |

### How Continuous Params Map to Discrete Actions

Each MADDPG actor outputs a per-agent continuous vector. The
`continuous_action_mapper.py` converts each to:

1. **Numeric RAG parameters** (logged for analysis)
2. **A discrete action name** (always within the valid masked set)

| Agent | Continuous dims | Key mapping rule |
|-------|----------------|-----------------|
| retriever | 4 | `rerank_threshold≥0.5` → hybrid_rerank; else dense/sparse/hybrid by `dense_sparse_weight` |
| rewriter  | 3 | `rewrite_strength≥0.7` → multi_query; `≥0.4` → keyword; else simple |
| grader    | 3 | `strictness_score≥0.7` → strict; `≥0.4` → medium; else loose/keep_all |
| generator | 4 | `citation_strictness≥0.65` → strict_citations; `detail≥0.4` → generate_answer |
| verifier  | 2 | verify_answer always first; recovery: sup≥0.55+conf≥0.55→regenerate, else retrieval/rewrite |

Mapped numeric params (`top_k`, `temperature`, `max_tokens`, etc.) are logged
to `action_params_log.csv` and available for future adapter-level injection.

---

## 3. Context Engineering Block (CEB)

Located in `maddpg/context_engineering_block.py`.

Builds a **20-dim** state representation by extending the existing 14-dim
global features with 6 additional context-engineering features:

| Dim | Feature | Description |
|-----|---------|-------------|
| 15 | source_diversity | unique source files / total retrieved chunks |
| 16 | evidence_coverage | selected_evidence / retrieved_chunks |
| 17 | step_fraction | num_steps / MAX_STEPS |
| 18 | llm_call_fraction | num_llm_calls / MAX_LLM_CALLS |
| 19 | query_length_norm | len(query) / 300 |
| 20 | requires_multiple_sources | binary flag |

When `--use-ceb` is passed, all actors and the critic use 20-dim input
instead of 14-dim.

---

## 4. How Stage Constraints Are Preserved

The MADDPG training loop calls the **exact same** stage-gating logic as the
discrete system:

```python
# Existing action_masking.get_action_mask() — unchanged
for name in AGENT_NAMES:
    mask = env.get_mask(name)          # ← existing code, untouched
    if sum(mask) > 0:
        active_agent  = name
        valid_actions = [...]           # only currently legal actions
        break

# MADDPG adds: continuous params → pick from valid_actions only
raw    = agents[active_agent].select_action(obs)
params = agents[active_agent].map_params(raw)
action = select_discrete_action(active_agent, params, valid_actions)

# Existing env.step — unchanged
env.step(active_agent, action)
```

The continuous actor never bypasses masking. `select_discrete_action` always
returns a string from `valid_actions`, falling back to `valid_actions[0]`.

---

## 5. File Structure

```
brain/context_marl_ac/maddpg/
├── __init__.py
├── noise.py                    OUNoise + GaussianNoise
├── continuous_action_mapper.py [-1,1]^d → RAG params + discrete action
├── maddpg_actor.py             Deterministic actor (obs → tanh → action)
├── maddpg_critic.py            Centralized Q(state, joint_action)
├── maddpg_agent.py             Per-agent wrapper (actor, target, noise)
├── replay_buffer.py            Off-policy transition storage
├── context_engineering_block.py 20-dim extended state features
├── train_maddpg.py             Training loop
└── evaluate_maddpg.py          Evaluation + 3-way comparison
```

---

## 6. How to Run Training

Run from `brain/`:

```bash
# Dry-run (stub adapters, no Qdrant / Groq):
python -m context_marl_ac.maddpg.train_maddpg \
    --episodes 50 \
    --dry-run \
    --run-name test_run

# Full training with CEB:
python -m context_marl_ac.maddpg.train_maddpg \
    --episodes 500 \
    --benchmark-path context_marl_ac/results/benchmark_splits/train.jsonl \
    --use-ceb \
    --run-name maddpg_ceb_v1 \
    --checkpoint-every 50

# Resume from checkpoint:
python -m context_marl_ac.maddpg.train_maddpg \
    --episodes 200 \
    --checkpoint-path context_marl_ac/results/maddpg/checkpoints/best_reward.pt \
    --use-ceb
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 200 | Training episodes |
| `--dry-run` | off | Use stub adapters |
| `--use-ceb` | off | 20-dim CEB state (vs 14-dim base) |
| `--benchmark-path` | auto | Path to train.jsonl |
| `--checkpoint-path` | "" | Resume from .pt file |
| `--checkpoint-every` | 50 | Periodic save frequency |
| `--run-name` | maddpg_run_01 | Output file prefix |
| `--results-dir` | results/maddpg/ | Output directory |

---

## 7. How to Run Evaluation

```bash
# Evaluate single mode (greedy MADDPG with CEB):
python -m context_marl_ac.maddpg.evaluate_maddpg \
    --mode maddpg \
    --checkpoint context_marl_ac/results/maddpg/checkpoints/best_reward.pt \
    --benchmark-path context_marl_ac/results/benchmark_splits/test.jsonl

# Compare all three modes:
python -m context_marl_ac.maddpg.evaluate_maddpg \
    --mode compare_all \
    --checkpoint context_marl_ac/results/maddpg/checkpoints/best_reward.pt \
    --benchmark-path context_marl_ac/results/benchmark_splits/test.jsonl

# Dry-run comparison:
python -m context_marl_ac.maddpg.evaluate_maddpg \
    --mode compare_all \
    --dry-run \
    --n-questions 5
```

**Modes:**

| Mode | Description |
|------|-------------|
| `maddpg` | MADDPG greedy policy with CEB (20-dim state) |
| `maddpg_no_ceb` | MADDPG greedy policy without CEB (14-dim base) |
| `discrete_marl` | Fixed smoke policy: hybrid_rerank → medium_filter → strict_citations → verify |
| `compare_all` | Runs all three and writes comparison_summary.csv |

---

## 8. Output Files

All outputs go to `results/maddpg/` (configurable with `--results-dir`).

### Training outputs

| File | Description |
|------|-------------|
| `checkpoints/best_reward.pt` | Best checkpoint by episode reward |
| `checkpoints/ep_NNNN.pt` | Periodic episode checkpoints |
| `metrics/episode_metrics_{run}.csv` | Per-episode: reward, steps, LLM calls, status, citation_support, latency |
| `metrics/action_params_log_{run}.csv` | Per-step: agent, discrete_action, raw_action values, mapped params, reward |
| `trajectories/trajectories_{run}.jsonl` | Full step-by-step episode trajectories |
| `aggregate_metrics_{run}.json` | Run-level aggregate statistics |

### Evaluation outputs

| File | Description |
|------|-------------|
| `eval_{mode}.jsonl` | Per-question results (answer, citations, verification, trace) |
| `aggregate_metrics.json` | Per-mode aggregate: pass_rate, citation, steps, latency, failure_rate |
| `comparison_summary.csv` | Side-by-side numeric comparison (only with compare_all) |

---

## 9. Metrics Produced

### Per-episode (training)
- total_reward, num_steps, num_llm_calls, final_status
- verification_pass (0/1), citation_support_rate
- latency_seconds, token_usage, buffer_size

### Per-question (evaluation)
- verification_pass, citation_support, num_unsupported_claims
- num_steps, num_llm_calls, latency_seconds, token_usage
- selected_evidence_count, verifier_decision, final_status

### Aggregate
- verification_pass_rate, mean_citation_support, mean_unsupported_claims
- mean_steps, mean_llm_calls, mean_latency, mean_token_usage
- mean_evidence_count, failure_rate

---

## 10. Safe Defaults / Fallback

If the MADDPG actor produces an invalid output (NaN, wrong shape, exception),
`map_agent_params` returns the following defaults without raising:

| Agent | Defaults |
|-------|---------|
| retriever | top_k=10, hybrid (alpha=0.5), rerank_threshold=0.5 |
| grader | relevance_threshold=0.5, keep_ratio=0.7, strictness=0.5 → medium_filter |
| generator | temperature=0.3, citation_strictness=0.7, max_tokens=512 → strict_citations |
| verifier | support=0.6, confidence=0.7 → verify_answer |
| rewriter | strength=0.5, expansion=0.5 → keyword_rewrite |

`select_discrete_action` always falls back to `valid_actions[0]` if no
preference rule matches.

---

## 11. Assumptions and Limitations

1. **Shared observation**: All MADDPG actors observe the same global state
   vector (14 or 20-dim) rather than per-agent partial observations. This
   simplifies the off-policy update (no need to store per-agent obs) at the
   cost of not enforcing partial observability at the neural level.

2. **Sequential execution**: Only one agent acts per step (stage-gated). The
   joint action stored in the replay buffer has real values only for the active
   agent; inactive agents are padded with zeros. The critic thus primarily
   learns Q-values conditioned on the active agent's action.

3. **Adapter-level params not injected**: Continuous params like `top_k`,
   `temperature`, and `max_tokens` are logged but not yet wired into the
   existing adapter APIs (`retriever_adapter.py`, `llm_adapter.py`). They
   currently influence only discrete action selection. Injecting them into
   adapters would require modifying those files, which is future work.

4. **No LLM training**: MADDPG trains only the small actor/critic networks
   (PyTorch MLPs). The underlying LLM (Groq) and retriever (Qdrant/BGE-M3)
   are not fine-tuned.

5. **Warmup**: The replay buffer requires `WARMUP_STEPS = 500` transitions
   before network updates begin. In dry-run mode (fast, stub episodes) this
   fills quickly; with real LLM calls each episode is slower.

6. **Policy mode flag**: The `policy_mode` distinction (`discrete_marl` vs
   `maddpg_continuous`) lives only in training/evaluation scripts. No flag is
   set in `config.py` to avoid touching shared config.
