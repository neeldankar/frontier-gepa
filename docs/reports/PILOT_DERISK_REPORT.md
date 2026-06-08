# De-risk pilot report (Experiment-3 gating)

Numbers only. No GO/NO-GO interpretation — the operator calls the gate
from the trajectory below.

## Resolved config (locked, executed verbatim)

| field | value |
|---|---|
| arm | `random` (uniform draw from D_feedback) |
| seed | 0 |
| N | 40 |
| b (minibatch) | 3 |
| A (accept_batch) | 20 |
| D_feedback | 150 |
| D_pareto | 75 |
| test | 300 (carved, NOT evaluated this pilot) |
| acceptance | strict improvement (`>`) of new A-sum over cached parent A-sum |
| task model | `together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo` @ temp 0.6 |
| reflection model | `openai/gpt-4.1-mini` |
| dspy | 3.2.1 (pinned) |
| gepa | 0.1.1 (pinned) |
| frozen IFBench difficulty-table SHA-256 | `e7878d44450e7e67fa58d58b1664899da79ebf3d7298ba3d384a7b558c649365` (verified at launch) |
| run dir | `results/derisk_pilot/` |

Substrate / sampler / proposer / metric: byte-identical to the smoke
path — `src/ifbench_substrate.ifbench_substrate()`,
`src/band_sampler.BandBatchSampler(target_band="random")`,
`src/decoupled_proposer.DecoupledReflectiveMutationProposer`,
`src/ifbench_feedback.metric_fn`. Held-out test eval and D4 drift
instrumentation were intentionally skipped (loop-internal accept signal
only).

## Headline numbers

| metric | value |
|---|---|
| total wall | **9,880.73 s ≈ 2 h 44 min** |
| iterations completed | 40 / 40 |
| total adapter evals | 2,205 |
| accepted candidates (incl. seed) | 15 (seed + 14 accepts) |
| **first accept iteration** | **iter 8** |
| **cumulative accepts at N=40** | **14** |
| accepted iters | 8, 10, 12, 16, 18, 20, 24, 28, 30, 33, 34, 36, 38, 39 |
| starting parent A=20 sum (seed candidate) | **9.2333 / 20 (46.17 %)** |

## Parent A-score trajectory at each accept

Reading directly off `state.full_program_trace[i]` fields
`accept_batch_parent_score` and `accept_batch_new_score` (written by
`src/decoupled_proposer.py:275-276` on every iter that produced a
proposal). Sums are out of A=20.

| iter | parent A (before accept) | new A (at accept) | gain |
|---:|---:|---:|---:|
| 8 | 9.2333 | 9.2500 | +0.0167 |
| 10 | 8.5833 | 8.8500 | +0.2667 |
| 12 | 8.5833 | 9.5333 | +0.9500 |
| 16 | 8.5833 | 8.6500 | +0.0667 |
| 18 | 9.3333 | 9.6833 | +0.3500 |
| 20 | 8.5833 | 9.9833 | +1.4000 |
| 24 | 9.3333 | 9.5833 | +0.2500 |
| 28 | 8.5833 | 9.2333 | +0.6500 |
| 30 | 9.2500 | 9.7000 | +0.4500 |
| 33 | 8.8500 | 10.5667 | +1.7167 |
| 34 | 8.8500 | 9.0000 | +0.1500 |
| 36 | 8.6500 | 10.0000 | +1.3500 |
| 38 | 8.8500 | 8.9833 | +0.1333 |
| 39 | 8.8500 | 9.7500 | +0.9000 |

`parent A (before accept)` is the parent-on-A sum that the proposer
read at the iter where the accept fired (the engine's
`subsample_scores_before`). The proposer picks the parent via the
Pareto candidate selector each iter, so this value reflects whichever
parent was selected for THAT iteration, not a single monotonic
trajectory.

## Per-iteration log (all 40 iters)

| iter | accepted | parent A | new A | new prog idx |
|---:|:---:|---:|---:|---:|
| 1 | n | 9.2333 | 8.3667 | — |
| 2 | n | 9.2333 | 9.1667 | — |
| 3 | n | 9.2333 | 8.3833 | — |
| 4 | n | 9.2333 | 8.4500 | — |
| 5 | n | 9.2333 | 9.2333 | — |
| 6 | n | 9.2333 | 8.4000 | — |
| 7 | n | 9.2333 | 7.6000 | — |
| 8 | **Y** | 9.2333 | 9.2500 | 1 |
| 9 | n | 9.2333 | 8.4333 | — |
| 10 | **Y** | 8.5833 | 8.8500 | 2 |
| 11 | n | 8.8500 | 7.9833 | — |
| 12 | **Y** | 8.5833 | 9.5333 | 3 |
| 13 | n | 9.3333 | 7.5333 | — |
| 14 | n | 9.3333 | 9.1833 | — |
| 15 | n | 8.5833 | 7.9667 | — |
| 16 | **Y** | 8.5833 | 8.6500 | 4 |
| 17 | n | 9.3333 | 7.8500 | — |
| 18 | **Y** | 9.3333 | 9.6833 | 5 |
| 19 | n | 10.0167 | 9.6333 | — |
| 20 | **Y** | 8.5833 | 9.9833 | 6 |
| 21 | n | 9.2333 | 8.0500 | — |
| 22 | n | 8.6500 | 8.4833 | — |
| 23 | n | 9.3333 | 8.3167 | — |
| 24 | **Y** | 9.3333 | 9.5833 | 7 |
| 25 | n | 9.2500 | 8.2333 | — |
| 26 | n | 9.2333 | 8.6000 | — |
| 27 | n | 9.3333 | 8.6500 | — |
| 28 | **Y** | 8.5833 | 9.2333 | 8 |
| 29 | n | 9.2500 | 7.6500 | — |
| 30 | **Y** | 9.2500 | 9.7000 | 9 |
| 31 | n | 9.2333 | 8.7167 | — |
| 32 | n | 9.2500 | 8.0167 | — |
| 33 | **Y** | 8.8500 | 10.5667 | 10 |
| 34 | **Y** | 8.8500 | 9.0000 | 11 |
| 35 | n | 8.8500 | 8.3167 | — |
| 36 | **Y** | 8.6500 | 10.0000 | 12 |
| 37 | n | 10.0167 | 9.2333 | — |
| 38 | **Y** | 8.8500 | 8.9833 | 13 |
| 39 | **Y** | 8.8500 | 9.7500 | 14 |
| 40 | n | 8.6500 | 8.4667 | — |

## Artefacts

| path | role |
|---|---|
| `results/derisk_pilot/summary.json` | summary above |
| `results/derisk_pilot/per_iter.json` | per-iter rows (written incrementally each iter end) |
| `results/derisk_pilot/parent_a_trajectory.json` | trajectory above |
| `results/derisk_pilot/run_log.txt` | engine event log |
| `results/derisk_pilot/run_log_stderr.txt` | task-LM warnings (truncation only; no throttle / 429 / backoff) |
| `results/derisk_pilot/candidates.json` | the 15 candidates kept by the engine (seed + 14 accepts) |
| `results/derisk_pilot/candidate_tree.html` | accept lineage |
| `results/derisk_pilot/gepa_state.bin` | resumable state (engine checkpoint) |
| `results/derisk_pilot/shared_rng.pkl` | shared-RNG snapshot |
| `results/derisk_pilot/_smoke/` | the N=2 smoke that preceded this run |

## What was NOT run / changed

- No held-out test eval (D_test carved but un-evaluated).
- No D4 temp-0 drift instrumentation.
- No HotpotQA artefacts modified.
- No Experiment-3 code, schedules, or cells built. The
  `EXPERIMENT3_CURRICULUM_HANDOFF.md` was loaded as context only.
- No tuning to chase accepts (no `max_tokens` change, no temperature
  change, no minibatch growth, no acceptance-rule relaxation).
- No interpretation of GO/NO-GO. The operator calls the gate.
