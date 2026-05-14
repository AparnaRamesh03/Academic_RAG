# Architecture Context for Defense Diagram
## Stage-Constrained Academic RAG with Context-Engineered MADDPG Control

---

## 1. Architecture Flow (left to right)

```
Academic PDFs
  → Ingestion / Chunking / Metadata tagging
  → Embeddings (BGE-M3 dense + BM25 sparse)
  → Qdrant Vector Index
  → [EPISODE START: Benchmark Question]
  → ContextState initialized (from_question)
  → Context Engineering Block (14-dim base → 20-dim CEB)
  → Stage Detector / Action Masking (get_action_mask)
  → MARL/MADDPG Control Layer
      ├── Decentralized actors (per agent)
      └── Centralized critic Q(s, a_joint)
  → Active Agent executes RAG action (with injected params)
      ├── Retriever → retrieved_chunks, retrieval_scores
      ├── Rewriter  → rewritten_query
      ├── Grader    → graded_chunks, selected_evidence
      ├── Generator → generated_answer, citation_candidates
      └── Verifier  → verification_result, final_status
  → Reward computed (cooperative, shared)
  → Transition stored in replay buffer
  → Critic + actors updated off-policy (DDPG rule)
  → Episode ends: accepted / rejected / timeout / abstained
  → Evaluation: Token F1, ROUGE-L, faithfulness, failure rate
```

---

## 2. Component Mapping

### 2a. Ingestion / Indexing
- **Purpose**: Convert PDFs to searchable chunks with metadata
- **Files**: `brain/final_arch/reranker_shared.py`, Qdrant Docker container (port 6333)
- **Inputs**: Academic PDFs (Transformer, BERT, ResNet, TabNet, Norwegian curriculum papers)
- **Outputs**: Dense embeddings (BGE-M3) + BM25 sparse index in Qdrant

### 2b. ContextState
- **Purpose**: Central mutable episode state shared by all agents and the critic
- **File**: `brain/context_marl_ac/schemas/context_state.py`
- **Class**: `ContextState` (dataclass)
- **Inputs**: benchmark question dict (`from_question`)
- **Outputs**: all downstream observations, reward inputs, logging

### 2c. Context Engineering Block (CEB)
- **Purpose**: Extends 14-dim global state to 20-dim richer observation
- **File**: `brain/maddpg/context_engineering_block.py`
- **Function**: `build_ceb_features(state)`
- **Inputs**: live ContextState
- **Outputs**: 20-dim float vector sent to each actor and the critic

### 2d. Feature Encoder (base 14-dim)
- **Purpose**: Encodes ContextState into normalized 14-dim vector
- **File**: `brain/context_marl_ac/context_engineering/feature_encoder.py`
- **Function**: `encode_features(state)`
- **Inputs**: ContextState
- **Outputs**: 14-dim float vector

### 2e. Action Masking / Stage Detector
- **Purpose**: Enforces the stage-gated pipeline; prevents illegal transitions
- **File**: `brain/context_marl_ac/marl/action_masking.py`
- **Function**: `get_action_mask(agent, state)`, `get_valid_actions(agent, state)`
- **Inputs**: agent name + ContextState
- **Outputs**: binary mask over full action set; only valid actions are non-zero

### 2f. MARL Environment
- **Purpose**: Orchestrates episode loop; dispatches agent actions; computes rewards
- **File**: `brain/context_marl_ac/marl/marl_env.py`
- **Class**: `MARLEnv`
- **Key methods**: `reset(question_dict)`, `step(agent, action, params)`, `get_mask(agent)`

### 2g. MADDPG Control Layer
- **Purpose**: Continuous-parameter control within valid stages
- **Files**: `brain/maddpg/` (standalone package)
- **Inputs**: state features (14 or 20-dim), valid action mask
- **Outputs**: continuous action in [-1,1]^d → numeric params + discrete action

### 2h. RAG Agents
- **Purpose**: Execute real retrieval / grading / generation / verification via LLM adapters
- **Files**: `brain/context_marl_ac/agents/*.py`
- **Inputs**: ContextState + action name + optional maddpg_params
- **Outputs**: mutated ContextState

### 2i. LLM Adapter
- **Purpose**: Groq API calls (llama-3.3-70b-versatile / llama-3.1-8b-instant)
- **File**: `brain/context_marl_ac/adapters/llm_adapter.py`
- **Note**: LLM is NOT trained; it is a fixed inference endpoint

### 2j. Reward Function
- **Purpose**: Computes shared cooperative reward per step
- **File**: `brain/context_marl_ac/marl/reward.py`
- **Function**: `calculate_reward(state, action, is_terminal, gold_answer, gold_chunks)`

---

## 3. Context Engineering Block — Exact Content

### Base 14-dim features (from `feature_encoder.py`)
| Dim | Feature | Source in ContextState |
|---|---|---|
| 1 | query_type_id (0–6, normalized) | `query_type` |
| 2 | query_complexity_id (0–2, normalized) | `query_complexity` |
| 3 | retrieval_confidence (avg score) | `retrieval_scores` |
| 4 | num_retrieved_chunks (capped 20) | `retrieved_chunks` |
| 5 | graded_relevance_ratio | `graded_chunks / retrieved_chunks` |
| 6 | selected_evidence_count (capped 10) | `selected_evidence` |
| 7 | citation_support_rate | `citation_support_rate` |
| 8 | unsupported_claim_count (capped 5) | `unsupported_claims` |
| 9 | verification_failed flag | `verification_result.decision` |
| 10 | retry_count (capped 5) | `retry_count` |
| 11 | latency_so_far (capped 60s) | `latency_so_far` |
| 12 | num_steps (capped 10) | `num_steps` |
| 13 | num_llm_calls (capped 15) | `num_llm_calls` |
| 14 | previous_action_intensity | `len(previous_actions)/10` |

### CEB extra 6-dim features (from `context_engineering_block.py`)
| Dim | Feature | Computation |
|---|---|---|
| 15 | source_diversity | unique source_files / len(retrieved_chunks) |
| 16 | evidence_coverage | len(selected_evidence) / len(retrieved_chunks) |
| 17 | step_fraction | num_steps / MAX_STEPS_PER_EPISODE |
| 18 | llm_call_fraction | num_llm_calls / MAX_LLM_CALLS_PER_EPISODE |
| 19 | query_length_norm | len(user_query) / 300, capped at 1 |
| 20 | requires_multiple_sources | binary flag from ContextState |

### Where CEB sits
- **Receives**: live ContextState after each env.step()
- **Sends to**: each MADDPG actor's `select_action(obs)` call and the centralized critic
- **Used only in MADDPG mode**: discrete MARL uses only the 14-dim base features

---

## 4. ContextState — All Fields

```python
# Query metadata
question_id:               str       # Q001, Q002, ...
original_query:            str       # fixed benchmark question
user_query:                str       # active query (may be rewritten)
rewritten_query:           str       # last rewrite result
expected_sources:          List[str] # ground-truth source filenames
query_type:                str       # factual/conceptual/comparison/section_specific/multi_hop/definition/summarization
query_complexity:          str       # low/medium/high
requires_multiple_sources: bool
requires_strict_citation:  bool

# Retrieval
retrieved_chunks:          List[Dict]  # {text, metadata, score}
retrieval_scores:          List[float]

# Grading / Evidence
graded_chunks:             List[Dict]
selected_evidence:         List[Dict]  # evidence_pack_item list → passed to Generator

# Citation
citation_candidates:       List[Dict]  # {source_file, page_number, section_header, excerpt, content_type}

# Generation
generated_answer:          str

# Verification
verification_result:       Dict        # {decision: PASS|FAIL, overall_feedback, claims}
unsupported_claims:        List[str]
citation_support_rate:     float       # fraction of claims supported

# Episode bookkeeping
previous_actions:          List[Dict]  # [{agent, action}, ...]
retry_count:               int
latency_so_far:            float       # wall-clock seconds
token_usage:               int
num_llm_calls:             int
num_steps:                 int
final_status:              str         # pending/accepted/rejected/abstained/generation_failed/timeout/error
done:                      bool

# MADDPG injection point
maddpg_params:             Optional[Dict]  # injected before each env.step(); None in discrete MARL

# Debug
grader_output:             Dict
```

---

## 5. MARL Control Layer

### Centralized (shared across agents)
- **Critic**: `MADDPGCritic` — `Q(s, a_joint)` where input is `(state_dim + 16)` → hidden(128) → LayerNorm → ReLU → scalar
- **Replay buffer**: `ReplayBuffer(capacity=50,000)` — off-policy; all agents share one buffer
- **Reward**: `calculate_reward()` — cooperative; same scalar shared by all agents at each step
- **Stage detector / action masking**: `get_action_mask()` — enforced globally; actors cannot override

### Decentralized (per agent)
- **5 actors**: one `MADDPGAgentWrapper` per agent (retriever, rewriter, grader, generator, verifier)
- Each actor: `MADDPGActor(obs_dim → hidden(128) → action_dim)` with Tanh output
- Each actor observes: **full global state** (not local obs) — all 14 or 20 dims
- Each actor outputs: continuous vector in [-1,1]^{action_dim}; mapped to RAG params + discrete action

### Shared
- Observation: all actors receive the same global state features
- Update rule: DDPG off-policy with target networks + Polyak soft update (τ=0.005)
- Gradient clip: 1.0

### Role summary
| Component | Role | Centralized? |
|---|---|---|
| Critic | Q(s, a_joint) evaluation | Yes |
| Actor (each agent) | Continuous action → RAG params | No (per agent) |
| Action masking | Stage-gate enforcement | Yes (shared logic) |
| Reward | Cooperative return signal | Yes (shared) |
| Replay buffer | Experience storage | Yes (shared) |
| OUNoise | Exploration during training | No (per actor) |

---

## 6. Specialized RAG Agents

### Retriever (`retriever_agent.py`)
- **Role**: Fetch evidence chunks from Qdrant; inject MADDPG top_k param
- **Actions**: `dense_retrieve`, `sparse_retrieve`, `hybrid_retrieve`, `hybrid_rerank`, `retrieve_more`
- **MADDPG param used**: `top_k` (range 5–30), `dense_sparse_weight`, `rerank_threshold`, `source_diversity`
- **Default on main path**: `hybrid_rerank` forced by action mask
- **Outputs to ContextState**: `retrieved_chunks`, `retrieval_scores`, `selected_evidence`

### Rewriter (`rewriter_agent.py`)
- **Role**: Reformulate query for better retrieval (recovery path only)
- **Actions**: `no_rewrite`, `simple_rewrite`, `keyword_rewrite`, `expanded_rewrite`, `multi_query_rewrite`
- **MADDPG params**: `rewrite_strength`, `query_expansion_weight`, `source_focus_weight`
- **Only activated**: after `verifier.request_rewrite`
- **Outputs to ContextState**: `user_query` (rewritten), `rewritten_query`

### Grader (`grader_agent.py`)
- **Role**: Filter/score retrieved chunks for relevance; select evidence
- **Actions**: `keep_all`, `loose_filter`, `medium_filter`, `strict_filter`, `rerank_only`
- **MADDPG params**: `relevance_threshold`, `evidence_keep_ratio`, `strictness_score`
- **Outputs to ContextState**: `graded_chunks`, `selected_evidence`

### Generator (`generator_agent.py`)
- **Role**: Produce grounded answer from selected evidence
- **Actions**: `generate_answer`, `generate_with_strict_citations`, `generate_short_answer`, `abstain_request_more_evidence`, `regenerate`
- **MADDPG params**: `temperature`, `citation_strictness`, `max_tokens` (128–1024), `answer_detail_level`
- **Abstain**: blocked by mask when evidence exists; signals back to retriever
- **Outputs to ContextState**: `generated_answer`, `citation_candidates`

### Verifier (`verifier_agent.py`)
- **Role**: Check answer grounding; trigger recovery or accept
- **Actions**: `verify_answer`, `request_regeneration`, `request_more_retrieval`, `request_rewrite`
- **MADDPG params**: `support_threshold`, `confidence_threshold`
- **Terminal action**: `verify_answer` → sets `final_status` to `accepted` or `rejected`
- **Recovery**: up to `MAX_RECOVERY_ATTEMPTS = 2`
- **Outputs to ContextState**: `verification_result`, `unsupported_claims`, `citation_support_rate`, `final_status`

---

## 7. MADDPG Extension

### Actor/Critic files
| File | Role |
|---|---|
| `brain/maddpg/maddpg_actor.py` | MLP actor: obs_dim → hidden(128) → action_dim, Tanh |
| `brain/maddpg/maddpg_agent.py` | Wrapper: actor + target_actor + OUNoise + optimizer |
| `brain/maddpg/maddpg_critic.py` | Centralized critic: (state+joint_action) → Q scalar |
| `brain/maddpg/replay_buffer.py` | Off-policy experience replay, capacity=50,000 |
| `brain/maddpg/continuous_action_mapper.py` | [-1,1]^d → numeric params + discrete action name |
| `brain/maddpg/context_engineering_block.py` | 14→20 dim state enrichment |
| `brain/maddpg/train_maddpg.py` | Standalone training loop |
| `brain/maddpg/live_maddpg_runner.py` | Train + eval with live LLM (Groq + Qdrant) |

### How MADDPG extends discrete MARL
1. Same `MARLEnv`, same action masking, same reward — nothing broken
2. Before each `env.step()`, the active actor runs one forward pass → raw action `[-1,1]^d`
3. `map_agent_params(agent, raw)` clamps and scales to numeric RAG params
4. `select_discrete_action(agent, params, valid_actions)` picks the best valid discrete action from params
5. Params are injected into `state.maddpg_params` before `agent.act()` is called
6. Each agent reads `state.maddpg_params` to override default RAG parameters
7. Transition `(s, a_joint, r, s')` stored in shared replay buffer
8. DDPG update every 4 env steps once buffer ≥ 50 transitions (WARMUP_STEPS)

### Stage constraint preservation
- Action masking runs **before** actor output is used for discrete selection
- `select_discrete_action(agent, params, valid_actions)` receives only the **already-masked** valid set
- Actor can output any continuous vector; only valid actions survive the mask

### Target networks + soft update
- Each actor has a frozen target_actor (Polyak τ=0.005)
- One shared critic has a frozen t_critic
- OUNoise (σ=0.15) added during training; disabled during eval (`explore=False`)

### Joint action vector
- 16-dim: retriever(4) | rewriter(3) | grader(3) | generator(4) | verifier(2)
- Only the active agent's dims are filled; inactive agent dims are zero-padded
- Used as input to the centralized critic alongside the state features

---

## 8. Continuous Control Parameters by Agent

### Retriever (4-dim action)
| Parameter | Range | Effect |
|---|---|---|
| `dense_sparse_weight` | [0,1] | Blend between BGE-M3 dense and BM25 sparse |
| `top_k` | 5–30 | Number of chunks to retrieve |
| `rerank_threshold` | [0,1] | Threshold for CrossEncoder reranking |
| `source_diversity` | [0,1] | Encourages multi-source retrieval |

### Rewriter (3-dim action)
| Parameter | Range | Effect |
|---|---|---|
| `rewrite_strength` | [0,1] | How aggressively to reformulate |
| `query_expansion_weight` | [0,1] | Weight for adding related terms |
| `source_focus_weight` | [0,1] | Focus retrieval on specific sources |

### Grader (3-dim action)
| Parameter | Range | Effect |
|---|---|---|
| `relevance_threshold` | [0,1] | Minimum score to keep a chunk |
| `evidence_keep_ratio` | [0.1,1] | Fraction of chunks to keep |
| `strictness_score` | [0,1] | Maps to loose/medium/strict filter action |

### Generator (4-dim action)
| Parameter | Range | Effect |
|---|---|---|
| `temperature` | [0,1] | LLM generation temperature |
| `citation_strictness` | [0,1] | Selects strict-citation action |
| `max_tokens` | 128–1024 | Response length budget |
| `answer_detail_level` | [0,1] | Drives short vs. detailed answer |

### Verifier (2-dim action)
| Parameter | Range | Effect |
|---|---|---|
| `support_threshold` | [0,1] | Minimum citation support to accept |
| `confidence_threshold` | [0,1] | Confidence gate on verification pass |

---

## 9. Training Flow (per episode)

```
1. Sample benchmark question (train.jsonl, 38 questions)
2. env.reset(question_dict) → ContextState initialized
3. Each actor: reset OUNoise
4. Loop until done:
   a. CEB: build 14-dim base + 6 CEB features → 20-dim obs
   b. Stage detector: get_action_mask() → valid actions for each agent
   c. Find first agent with non-zero mask → active_agent
   d. active_agent.select_action(obs, explore=True) → raw continuous action
   e. map_params(raw) → numeric RAG params
   f. select_discrete_action(agent, params, valid_actions) → discrete action
   g. build_joint_action_vector({agent: raw}) → 16-dim joint vec
   h. env.step(agent, discrete_action, params) → new ContextState, reward, done
   i. new obs computed
   j. buffer.push(Transition(obs, joint_vec, reward, next_obs, done, ...))
   k. total_steps += 1
   l. if len(buffer) >= WARMUP_STEPS and total_steps % UPDATE_EVERY == 0:
        _ddpg_update(agents, critic, t_critic, critic_optim, buffer, device)
5. Log episode metrics (reward, steps, latency, citation_support, buffer_size)
6. If ep_reward > best_reward: save best_*.pt checkpoint
7. If ep_idx % checkpoint_every == 0: save periodic ep_XXXX.pt checkpoint
```

### DDPG update details
- Sample batch of 256 from buffer
- Critic update: minimize MSE Bellman error with target networks
- Actor update (each agent): maximize Q(s, a_joint) for the active agent only
- Soft update: τ=0.005 for all target networks
- Gradient clipping: 1.0

---

## 10. Evaluation Flow

### Load checkpoint
- `torch.load(best_*.pt)` → restore actor + target_actor + critic weights
- `explore=False` → OUNoise disabled; pure greedy policy

### Run benchmark (test.jsonl, 9 questions)
- Same episode loop as training but no buffer push, no update, no noise
- All 9 questions evaluated sequentially
- Results saved to `maddpg_{variant}_live.jsonl`

### Comparison variants
| Variant | Description |
|---|---|
| `discrete_marl` | Baseline: hand-coded heuristic + learned Q-values; no continuous params |
| `maddpg_no_ceb` | MADDPG with 14-dim base state (no Context Engineering Block) |
| `maddpg_ceb` | MADDPG with 20-dim CEB state (full system) |

- Discrete MARL baseline loaded from `results/final_eval/learned_eval_*.jsonl` (pre-existing results)
- All three compared in `comparison_summary.csv` and `aggregate_metrics.json`

---

## 11. Metrics and Outputs

### NLP quality metrics
| Metric | Computation |
|---|---|
| Token F1 | token overlap F1 between predicted and gold answer |
| ROUGE-L | LCS-based recall/precision against gold answer |
| Correctness | same as Token F1 (single source of truth) |
| Faithfulness | = citation_support_rate from VerifierAgent |
| Citation Support | fraction of answer claims with source citations |
| Source Precision | retrieved_sources ∩ expected_sources / retrieved_sources |
| Source Recall | retrieved_sources ∩ expected_sources / expected_sources |
| Verification Pass | 1 if final_status == "accepted" |
| Unsupported Claims | count of claims flagged by verifier with no source |

### Efficiency metrics
| Metric | Field |
|---|---|
| Latency (s) | `state.latency_so_far` |
| LLM Calls | `state.num_llm_calls` |
| Token Usage | `state.token_usage` |
| Steps | `state.num_steps` |
| Failure Rate | fraction with status in {rejected, error, timeout, abstained} |

### Output files
| File | Contents |
|---|---|
| `aggregate_metrics.json` | Per-policy mean metrics across all eval questions |
| `comparison_summary.csv` | Same as above, CSV format |
| `episode_metrics.csv` | Per-question results for all policies combined |
| `metrics/ep_metrics_{variant}.csv` | Per-episode training log (reward, steps, buffer_size, latency) |
| `metrics/action_params_{variant}.csv` | Per-step raw actions + mapped params |
| `trajectories/trajectories_{variant}.jsonl` | Full step-by-step episode traces |
| `checkpoints/best_{variant}.pt` | Best-reward checkpoint |
| `checkpoints/{variant}_ep{N:04d}.pt` | Periodic checkpoint every N episodes |
| `results_interpretation.md` | Auto-generated narrative comparison |

---

## 12. Honest Scope / Missing Pieces

### Fully implemented and tested
- Complete 5-agent MARL pipeline (retriever → grader → generator → verifier, with rewriter recovery)
- Stage-gated action masking with correct recovery paths (max 2 retries)
- MADDPG actor-critic architecture (actors, critic, replay buffer, target networks, soft update)
- Context Engineering Block (14→20 dim)
- Continuous-to-discrete action mapping with safe clamping and fallbacks
- Live LLM evaluation via Groq API (real inference, not stubs)
- Hybrid retrieval via Qdrant (BGE-M3 dense + BM25 sparse + CrossEncoder reranker)
- Reward function with 9 components (step cost, latency, citation, verification, retrieval F1, hallucination penalties)
- Checkpoint saving + loading
- Full evaluation pipeline comparing 3 variants

### Training status as of defense
- **Only ~40 episodes actually trained** (2 runs, each killed by Groq rate limits)
- WARMUP_STEPS=50 → first ~8 episodes are exploration-only (no gradient updates)
- Best checkpoint evaluated was near-random (ep2 of second run, buffer=13)
- ep40 checkpoint exists from first run (~32 actual gradient updates) — evaluated separately
- **300–500 episodes needed for policy convergence** — not yet achieved
- The results shown are from an undertrained policy; they demonstrate the pipeline is functional, not that MADDPG outperforms the baseline

### Dry-run / stub mode
- `--dry-run` flag bypasses Groq and Qdrant with deterministic stubs
- All live results used `cfg.DRY_RUN = False` with real LLM calls
- Dry-run does not produce meaningful metric values

### Not implemented / future extensions
- Multi-document PDF ingestion pipeline (Qdrant index pre-populated; ingestion script not in this codebase)
- No query analyzer agent (query_type/complexity set from benchmark metadata, not inferred at runtime)
- `requires_strict_citation` not set dynamically by any agent
- CEB features 19–20 (`query_length_norm`, `requires_multiple_sources`) rely on benchmark metadata, not inferred
- No hyperparameter sweep; single learning rate/batch size configuration
- No distributed training; single-machine CPU training only
- MADDPG no-CEB variant: only 20 episodes trained (even fewer than CEB); not competitive

---

## 13. Defense-Ready Wording

### What this system does
> "We extend a stage-constrained cooperative MARL RAG system with a continuous control layer based on MADDPG. Five specialized agents — Retriever, Rewriter, Grader, Generator, and Verifier — operate inside a fixed stage-gated pipeline enforced by action masking. The MADDPG extension does not change the workflow structure; instead, it learns continuous RAG parameters (top_k, temperature, grading strictness, citation requirements, verification thresholds) that adapt per query. The Context Engineering Block enriches the state representation from 14 to 20 dimensions by adding source diversity, evidence coverage, and budget-fraction features that capture query-level context unavailable in the base state."

### What MADDPG adds over discrete MARL
> "Where the discrete baseline uses fixed or heuristically-chosen parameters, MADDPG actors learn which parameter configurations — high top_k for multi-hop queries, strict citation for factual questions, low temperature for precise definitions — maximize the cooperative reward signal. This is done without any additional LLM calls; the actor forward pass costs ~0.1–0.5 ms on CPU."

### Honest limitation statement
> "The training results presented here are from approximately 40 episodes — sufficient to demonstrate end-to-end pipeline functionality with live LLM inference, but insufficient for policy convergence. Meaningful performance comparison requires 300–500 training episodes. The current evaluation results are consistent with a lightly-explored policy rather than a converged one."

### What CEB adds over no-CEB
> "The Context Engineering Block adds six features computed directly from the live episode state: source diversity among retrieved chunks, evidence coverage ratio, step and LLM-call budget fractions, query length, and multi-source requirement flag. These features are not derivable from the 14-dim base state and provide the actor with per-query context that is precisely the signal needed to adapt retrieval and grading parameters to query complexity."
