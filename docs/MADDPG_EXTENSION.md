# MADDPG-style Continuous-Control Extension (Stage-Conditioned)

This project implements a **MADDPG-style continuous-control architecture for a
staged RAG environment**. It is inspired by MADDPG/DDPG, but because the RAG
workflow is **stage-gated** (only one agent acts per step) rather than
simultaneous-action, the centralised critic is **stage-conditioned**:

```
Q(state, active_agent_one_hot, discrete_action_one_hot, padded_continuous_action) -> R
```

This is **not** textbook MADDPG. Textbook MADDPG trains
`Q(state, joint_action_of_all_agents)` and updates every agent's actor at every
step. In our environment that is wasteful: at any moment exactly one agent is
active, so the joint vector is mostly zeros and the gradient signal is diluted.
The stage-conditioned critic instead receives only what is actually executing,
and per-agent actors are updated only on batch rows where that agent was active.

## File layout

All code lives under `brain/maddpg/`:

```
brain/maddpg/
  stage_utils.py               agent ordering, one-hots, active-agent detection
  continuous_action_mapper.py  raw [-1,1]^d  ->  numeric RAG params + discrete action
  maddpg_actor.py              deterministic actor (tanh head)
  maddpg_agent.py              actor + target actor + OUNoise + optimizer
  maddpg_critic.py             StageConditionedCritic (default) + legacy MADDPGCritic
  replay_buffer.py             Transition with active_agent / next_active_agent fields
  context_engineering_block.py 20-dim CEB state features
  noise.py                     OU + Gaussian noise
  trainer.py                   StageConditionedMADDPGTrainer (single source of truth)
  train_maddpg.py              CLI: dry / live / offline-replay training
  live_maddpg_runner.py        end-to-end live train + eval comparison
  evaluate_maddpg.py           checkpoint-required evaluation
  tests/test_smoke.py          runnable smoke checks
```

## How to run

From `brain/`:

```bash
# Dry-run smoke test (no LLM, exits nonzero if no gradient updates happen)
python -m maddpg.train_maddpg \
    --train-mode dry --episodes 5 \
    --batch-size 4 --warmup-steps 4 --update-every 1 --min-updates 1 \
    --run-name smoke_stage_conditioned

# Live training
python -m maddpg.train_maddpg \
    --train-mode live --episodes 200 --use-ceb \
    --batch-size 64 --warmup-steps 50 --update-every 4 --min-updates 100 \
    --run-name maddpg_ceb_live

# Live train + eval (both no-CEB and CEB)
python -m maddpg.live_maddpg_runner --episodes 30 --n-eval 9

# Evaluate a trained checkpoint (checkpoint REQUIRED by default)
python -m maddpg.evaluate_maddpg \
    --mode maddpg_ceb \
    --checkpoint results/maddpg/checkpoints/best_reward.pt \
    --benchmark-path results/benchmark_splits/test.jsonl \
    --n-questions 9

# Offline-replay training (no LLM): load a previously-saved replay JSONL
python -m maddpg.train_maddpg --train-mode offline-replay \
    --replay-in path/to/transitions.jsonl --offline-gradient-steps 10000

# Smoke tests
python -m maddpg.tests.test_smoke
```

## Architecture details

### Stage-conditioned critic

```
in_dim = state_dim
       + NUM_AGENTS                  (5)
       + MAX_DISCRETE_ACTIONS        (5)
       + MAX_CONTINUOUS_ACTION_DIM   (4)
```

State dim is 14 (base global features) or 20 (Context Engineering Block).
The continuous action slot is **padded** to `MAX_CONTINUOUS_ACTION_DIM = 4`
because different agents have different action dims (retriever=4, generator=4,
grader=3, rewriter=2, verifier=1).

### DDPG-style update

For each gradient step (`trainer.update()`):

1. Sample a batch of transitions.
2. Build `(state, agent_one_hot, discrete_one_hot, action_padded)` per row.
3. For each row, roll the target actor of `next_active_agent` to produce the
   next padded continuous action and the discrete one-hot derived from it.
   When `done` or `next_active_agent is None`, all three slots are zero.
4. Critic loss = `MSE(current_q, reward + gamma * (1 - done) * target_q)`.
5. For each agent, filter to batch rows where that agent was active and
   compute `actor_loss = -critic(state, agent_oh, disc_oh, pad(actor(state))).mean()`.
   The discrete one-hot is treated as non-differentiable execution context.
6. Soft-update target actors and target critic with τ.

### Replay transition schema

`Transition` (see `replay_buffer.py`) stores per-step:

- `state_features`, `next_state_features` — 14- or 20-dim feature vectors
- `active_agent`, `active_agent_id`, `valid_actions`
- `raw_action` (agent-native dim), `action_padded` (MAX_CONTINUOUS_ACTION_DIM)
- `mapped_params`, `discrete_action`, `discrete_action_one_hot`
- `reward`, `done`
- `next_active_agent`, `next_active_agent_id`, `next_valid_actions`
- `question_id`, `step`, `final_status`, `metrics_snapshot`

Plus legacy fields (`joint_action`, `stage`, etc.) kept only for log-compat;
the trainer never reads them.

### Error-path transitions

If `env.step` raises, the trainer pushes a **terminal negative transition**
(reward = `--error-penalty`, default `-1.0`, `done=True`, `next_active_agent=None`)
into the buffer instead of crashing. The episode ends with `final_status="error"`
and the error message is captured in `metrics_snapshot.error_message`.

### Retriever action masking

The retriever can now choose **any** retrieval strategy
(`dense_retrieve`, `sparse_retrieve`, `hybrid_retrieve`, `hybrid_rerank`) on
the main retrieval path. Previously the mask hard-coded `hybrid_rerank` which
made the actor's continuous params meaningless on this stage. `retrieve_more`
remains a recovery-only action after `verifier.request_more_retrieval`.

## Continuous-param wiring table

| Agent     | Param                   | Range       | Used in execution? | Where                                                                               |
|-----------|-------------------------|-------------|--------------------|-------------------------------------------------------------------------------------|
| retriever | `top_k`                 | 5–30        | yes                | `retriever_agent.act()` → `retrieve_*(query, top_k)`                                |
| retriever | `dense_sparse_weight`   | 0.0–1.0     | yes (selector)     | `_sel_retriever` picks dense / sparse / hybrid                                      |
| retriever | `rerank_threshold`      | 0.0–1.0     | yes (selector)     | `_sel_retriever` picks `hybrid_rerank` when ≥ 0.5                                   |
| retriever | `source_diversity`      | 0.0–1.0     | yes (post-filter)  | `retriever_agent._enforce_source_diversity` caps chunks per source                  |
| rewriter  | `rewrite_strength`      | 0.0–1.0     | yes (selector)     | `_sel_rewriter` picks `multi_query` / `keyword` / `simple_rewrite`                  |
| rewriter  | `query_expansion_weight`| 0.0–1.0     | yes (selector)     | `_sel_rewriter` picks `expanded_rewrite` when ≥ 0.6                                 |
| grader    | `relevance_threshold`   | 0.0–1.0     | yes (post-filter)  | `grader_agent.act()` drops chunks with `score < threshold`                          |
| grader    | `evidence_keep_ratio`   | 0.1–1.0     | yes                | `grader_agent.act()` trims to top-N fraction                                        |
| grader    | `strictness_score`      | 0.0–1.0     | yes (selector)     | `_sel_grader` picks `strict` / `medium` / `loose_filter`                            |
| generator | `temperature`           | 0.0–1.0     | yes                | `generator_agent.act()` → `generate_answer(temperature=...)`                        |
| generator | `max_tokens`            | 128–1024    | yes                | `generator_agent.act()` → `generate_answer(max_tokens=...)`                         |
| generator | `citation_strictness`   | 0.0–1.0     | yes (selector)     | `_sel_generator` picks `generate_with_strict_citations` when ≥ 0.65                 |
| generator | `answer_detail_level`   | 0.0–1.0     | yes                | `_map_generator` multiplicatively scales `max_tokens` by 0.5 + detail               |
| verifier  | `support_threshold`     | 0.0–1.0     | yes                | `verifier_agent.act()` overrides PASS/FAIL via citation rate                        |

Params that were previously declared but not actually consumed in execution
(`source_focus_weight`, `confidence_threshold`) have been removed from the
mapper, defaults, and action dims. Rewriter is now 2-dim and verifier 1-dim;
total joint action dim drops from 16 to 14.

## Aggregate metrics file (sample from dry-run)

A real `aggregate_metrics_*.json` looks like:

```json
{
  "run_name": "smoke_stage_conditioned",
  "architecture": "maddpg_style_continuous_control",
  "critic_type": "stage_conditioned",
  "use_ceb": false,
  "state_dim": 14,
  "episodes": 5,
  "total_env_steps": 20,
  "total_gradient_updates": 17,
  "trained": true,
  "best_reward": 0.475,
  "mean_reward": 0.475,
  "verification_pass_rate": 1.0,
  "mean_citation_support": 1.0,
  "failure_rate": 0.0,
  "last_losses": {
    "critic_loss": 0.0225,
    "actor_loss_retriever": -0.3017,
    "actor_loss_generator": 0.0908,
    "actor_loss_verifier": -0.5586
  },
  "hyperparameters": {
    "batch_size": 4, "warmup_steps": 4, "update_every": 1, "min_updates": 1,
    "gamma": 0.99, "tau": 0.005, "actor_lr": 0.001, "critic_lr": 0.001,
    "noise_sigma": 0.15, "grad_clip": 1.0, "error_penalty": -1.0,
    "seed": 0, "hidden_dim": 32
  }
}
```

## Limitations / what this is NOT

- **Discrete action selection is non-differentiable.** The mapper picks the
  discrete action from the continuous params; gradient does not flow through
  that selection. The actor primarily learns continuous RAG parameter control.
- **Stage gating is preserved.** Action masking from the discrete MARL system
  is unchanged (other than letting the retriever pick its retrieval strategy).
- **Not textbook MADDPG.** No joint-action critic, no fully simultaneous-action
  multi-agent setup. The framing is "MADDPG-style continuous control for a
  stage-gated environment."
- **Production-readiness is not claimed here.** The dry-run results show the
  training loop performs real gradient updates and produces a deterministic
  policy. Per-question quality must be evaluated separately against the
  discrete baseline; the live runner emits an honest `results_interpretation.md`
  that distinguishes trained from untrained variants.

## Changelog

- **Converted full-joint MADDPG into stage-conditioned MADDPG-style
  continuous-control architecture.** Replaced
  `Q(s, joint_action[16])` with `Q(s, agent_oh, discrete_oh, action_padded)`.
- Centralised all DDPG logic in `trainer.StageConditionedMADDPGTrainer`;
  `train_maddpg.py`, `live_maddpg_runner.py`, `evaluate_maddpg.py` are thin
  orchestration layers.
- New `replay_buffer.Transition` carries `active_agent`/`next_active_agent`,
  padded action, discrete one-hot, valid action lists, JSON round-trip for
  offline-replay training.
- All hyperparameters surfaced through CLI flags
  (`--batch-size --warmup-steps --update-every --min-updates --seed --gamma
   --tau --actor-lr --critic-lr --noise-sigma --grad-clip --error-penalty`).
- `--train-mode {dry,live,offline-replay}` added; `--replay-in/--replay-out`
  and `--offline-gradient-steps` enable detached gradient training over a
  saved trajectory dump.
- Error transitions (terminal negative) pushed to the buffer on `env.step`
  exceptions instead of killing the run.
- Retriever masking opened up to all retrieval strategies on the main path;
  `retrieve_more` still recovery-only.
- Removed un-wired params (`source_focus_weight`, `confidence_threshold`);
  wired `source_diversity`, `relevance_threshold`, `answer_detail_level`.
- `evaluate_maddpg.py` now hard-errors on missing checkpoint unless
  `--allow-random-policy` is passed; always writes checkpoint metadata into
  `aggregate_metrics.json`.
- Removed personal Desktop `.env` paths from all loaders.
- Smoke tests in `brain/maddpg/tests/test_smoke.py` cover critic shape,
  replay schema, error-transition push, eval checkpoint requirement, param
  mapper ranges, and end-to-end dry-run training.
