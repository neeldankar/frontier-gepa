# Chunk 14 wiring + pilot report (IFBench, num_threads=16)

## TL;DR

- IFBench wiring (one-module program + IFEval verifier feedback)
  composes cleanly with the decoupled-acceptance engine and the
  100/0/0 band sampler.  142/142 offline tests pass (128 prior + 14
  Chunk-14).
- Frozen Chunk-13 difficulty table SHA-256 hash-verified at variant
  selection; refuses to launch on mismatch or missing file.
- Parallel pilot (num_threads=16, fresh cache-miss, sandboxed) ran
  cleanly end-to-end: 5 iters + 300-example test eval + temp-0 drift
  instrumentation.  **Zero rate-limit / 429 / throttle / backoff
  events** in the run stderr.
- Cost projection is comfortable: $7.34 / $6.48 / $5.62 at N=80 /
  N=60 / N=40 → cost gate ($100) is not the binding constraint.
- **Wall-time projection is NOT comfortable.**  At measured per-call
  walls (~4 min on long outputs after parallelism saturates), the
  matrix would take **~82 / ~69 / ~57 hours** at N=80 / 60 / 40.
  That does not fit an overnight window.  Decision required from
  the operator before launching.

## Hash typo flagged

The hash supplied in the task brief was
`e7878d44450e7e67f8d58b1664899da79ebf3d7298ba3d384a7b558c649365`
(62 chars; missing `a5` at positions 17-18).  The actual frozen-table
SHA-256 written by Chunk 13 is
`e7878d44450e7e67fa58d58b1664899da79ebf3d7298ba3d384a7b558c649365`
(64 chars).  The wiring asserts against the real value
(`src/ifbench_substrate.py::EXPECTED_IFBENCH_DIFFICULTY_HASH`).
No action needed unless the operator wants the constant matched to a
different source of truth.

## Wiring

The Experiment-1 / 1b HotpotQA path stays byte-identical (default
`Substrate` resolves to the existing imports).  IFBench is wired by
injecting a `Substrate` into `run_gepa.run()` and through
`orchestrator._eval_best_on_test`.

| change | file | purpose |
|---|---|---|
| `Substrate` dataclass (build_program, metric_fn, feedback_fn, component_names, num_threads) | `src/run_gepa.py` | substrate-agnostic adapter wiring |
| `_default_hotpot_substrate()` | `src/run_gepa.py` | preserves HotpotQA path (num_threads=None) |
| `ifbench_substrate()` factory | `src/ifbench_substrate.py` | IFBench bag of callables (num_threads=16) |
| `verify_ifbench_difficulty_table_hash()` | `src/ifbench_substrate.py` | refuses launch on mismatch / missing file |
| `compute_drift_for_cell` (D4) | `src/ifbench_drift.py` | temp-0 base + final + Spearman + temperature-consistent frontier-leaving rate |
| `ifbench_100` variant | `src/orchestrator.py` | logs to `results/logs_ifbench/`, 5 arms x 3 seeds, N=80, 100/0/0, substrate=ifbench, drift_enabled |
| substrate dispatch | `src/orchestrator.py` | `_load_substrate_for_current_variant`, `_load_splits_for_current_variant` |
| `num_threads=16` plumbed through dspy.Evaluate | `src/run_gepa.py`, `src/orchestrator.py`, `src/ifbench_drift.py` | parallel evals on IFBench's output-heavy generations |

### Provenance recorded per cell (matrix launch will populate)

```
arm, seed, git_commit
variant=ifbench_100, substrate=ifbench
sampler=100/0/0, sampler_mix=[1.0, 0.0, 0.0]
n_iter_target=80
dataset=allenai/IF_multi_constraints_upto5
constraint_count_floor=3
carved_sizes={d_feedback:150, accept_batch:20, d_pareto:75, test:300}
difficulty_table_hash=<verified Chunk-13 SHA-256>
difficulty_table_path=results/ifbench/difficulty_table.json
```

## Pilot configuration

| | |
|---|---|
| arm | `static_frontier` |
| seed | 0 |
| N (truncated) | 5 |
| pool sizes | D_feedback=150, accept=20, D_pareto=75, test=300 |
| sampler | 100/0/0 (PURE_ON_BAND_MIX) |
| num_threads | 16 |
| sandbox dir | `results/logs_ifbench/static_frontier_0_pilot/` |
| task model | `together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo` @ temp 0.6 |
| reflection model | `openai/gpt-4.1-mini` |
| drift | enabled (D4: temp-0 base + temp-0 final + Spearman + frontier-leaving rate) |

## Measured walls (parallel)

Derived from `pilot_summary.json` and artefact mtimes (run started
2026-06-03 18:28:39; pilot completed 20:47:46; total = 2h 19min).

| phase | wall (s) | wall (min) | notes |
|---|---|---|---|
| iter loop (5 iters, 210 evals) | 736.1 | 12.3 | per-iter avg 147.2 s |
| test eval (300 examples) | ~5100 | ~85.0 | derived from mtimes |
| drift base (150 ex, shared once) | 2559.8 | 42.7 | from `drift.json::wall_s_base` |
| drift final (150 ex) | 0.75 | 0.01 | pilot final == seed → dspy cache hit |
| **pilot total** | **~8397** | **~139.9** | matches 18:28 → 20:47 wall |

The 5-iter loop sped up from the serial baseline's ~25 minutes to
12.3 minutes (~2.0x).  The remaining wall is bound by long per-call
generations at temp 0.6 (median ~17 s/example after parallelism
saturates; truncation warnings at `max_tokens=16384` fire on ~10% of
calls).

**Throttle / rate-limit / backoff count on run stderr: 0.**
(23 lines on stderr; all are `max_tokens=16384` truncation
warnings.  No 429, no rate-limit, no backoff.  Counted with `grep -ci
'rate|429|throttle|backoff|too many'` against
`run_log_stderr.txt`.)

## Sanity checks (operator-facing)

| check | result | comment |
|---|---|---|
| responses non-empty | yes | task LM returned content on every call |
| scores in [0, 1] | yes | `metric_fn` enforces fraction-of-satisfied-constraints |
| feedback well-formed | yes | uniform renderer (Chunk-12 revision #4) per cell |
| drift fields populated | yes | Spearman=0.9924, leaving rate=0.0200 |
| **at least one accept** | **NO (0 accepts)** | see below |
| hash verification on launch | yes | matched the Chunk-13 frozen SHA-256 |

### Zero accepts in 5 iters — flag, not a failure

The frontier arm proposed 5 candidates over 5 iterations.  Each
proposed candidate's A-batch sum (the strict-improvement bar) sat in
the 8.13 - 9.00 / 20 range; the cached parent (seed) A-batch sum was
9.23 / 20.  No proposal exceeded that bar → no accepts → only the
seed candidate in `candidates.json`.

This is consistent with the at-N=5 expectation for a strong-seed
substrate: gpt-4.1-mini reflecting on a b=3 minibatch and trying to
beat a competent seed instruction on a 20-example A-batch in one
shot is a tight ask.  At N=80, the cumulative random-walk over 80
proposals will surface accepts even at this acceptance rate.  The
relevant matrix-time question is whether the frontier arm
out-accepts the random / hard / easy arms — which the matrix is
designed to measure, not the pilot.

The drift Spearman of 0.99 and frontier-leaving rate of 0.02 (1 of 50
ids moved across the rank-tercile boundary) reflect that the
"final" candidate is the seed candidate, so the temp-0 base pass
and the temp-0 final pass are nearly identical (cache hits explain
the 0.75 s final-pass wall).  These two numbers will be the
informative ones on matrix cells where the final candidate IS
distinct from the seed.

## Cost projection (matches the orchestrator's gate output)

Pilot per-iter cost = $0.003766 ($0.01883 / 5 iters).
Pilot test-eval cost = $0.118384.
Pilot drift cost (from `cost_meta_base`, 323K tokens @ $0.20/1M) ≈ $0.065.

Matrix cost = 15 cells × N × iter_cost + 15 × test_cost + 1 × drift_base + 15 × drift_final.

| N | iter total | test total | drift total | **matrix total** |
|---|---|---|---|---|
| 80 | $4.52 | $1.78 | $1.04 | **$7.34** |
| 60 | $3.39 | $1.78 | $1.04 | **$6.21** |
| 40 | $2.26 | $1.78 | $1.04 | **$5.08** |

All comfortably under the $100 auto-gate (and the $150 hard kill).
Cost is not the constraint.

## Wall-time projection — the binding constraint

Per-cell wall = N × 147.2 s (loop) + ~5100 s (test eval) + ~2560 s
(drift final, pessimistic).  Plus one shared base-drift pass.

| N | per-cell loop (s) | per-cell wall (s) | matrix wall (s) | **matrix wall (hours)** |
|---|---|---|---|---|
| 80 | 11,776 | 19,436 | 293,600 | **81.6 h (~3.4 days)** |
| 60 | 8,832 | 16,492 | 249,440 | **69.3 h (~2.9 days)** |
| 40 | 5,888 | 13,548 | 205,780 | **57.2 h (~2.4 days)** |

Matrix wall = 15 × per-cell + 2560 (shared base).

The dominant terms are the test eval (~85 min/cell) and the drift
final pass (~43 min/cell, pessimistic — the pilot saw cache benefit
only because final == seed).  Even at N=40, no projection fits an
overnight window.

### Why parallelism helps less than naively expected

At 16 threads, a 150-example pass at temp 0 took 2560 s.  Implied
per-call wall ≈ 4.5 min.  This is consistent with the model
occasionally running to the 16K-token max_tokens ceiling on hard
prompts (the same truncation pattern the Chunk-13 base scoring
showed: 16 / 150 runs there hit the cap).  Going from
num_threads=1 to 16 buys roughly a 2x speedup on the iter loop and
proportional speedups on test eval / drift, but does NOT linearize
the long-tail per-call cost.

## Decision points for the operator

The cost gate passes cleanly.  The wall-time gate does not.  Options
the operator may consider before launching, in increasing order of
deviation from spec:

1. **Accept the long wall, run unattended.**  Two-three days, resumable
   per cell, kill switch retained.  Trade-off: occupies a real-clock
   window the BUILD_PLAN budgeted as "3-5 hours" (§7 Chunk 14).
2. **Shrink test pool from 300 to 100** (the BUILD_PLAN locks it at
   300, so this is a deviation — record in DEVIATIONS.md).  Saves
   ~57 min/cell ≈ 14 hours total.
3. **Lower max_tokens** from 16384 to e.g. 8192 to cut the long-tail
   per-call wall.  This changes scoring (some constraints will be
   under-served by shorter outputs), so re-piloting is required.
4. **Drop the per-cell drift final pass and run drift only on
   one cell per arm.**  Saves ~43 min × 12 cells = ~8.6 hours.
   Drift quality degrades but the BUILD_PLAN says drift is the
   secondary readout.
5. **Lower N to 30** (BUILD_PLAN §4 D2 commits N=80; this is a real
   deviation).  Per the §4 D2 framing the primary readout is the
   *early curve*, so even N=30 retains the iterations-to-target
   signal.  Wall at N=30 ≈ ~53 hours.
6. **Re-evaluate parallelism**: bump num_threads to 32.  Risk:
   provider may rate-limit at higher concurrency (pilot at 16 saw
   zero throttling; 32 may push it).  Reward: potentially halve the
   loop / drift / test walls.

No option is launched here.  The matrix wait is the operator's call.

## What this chunk does NOT do

- The matrix is NOT launched.  `cell_summary.json`s, `matrix_summary.json`,
  and the per-cell drift artefacts will exist only after the matrix
  runs.  The orchestrator's `matrix` subcommand is wired and ready
  but un-invoked.
- CLAUDE.md is NOT edited.  Per BUILD_PLAN protocol, the operator
  flips the chunk-status checklist after reviewing this report.
- No new DEVIATIONS.md entry is recorded; the parallelism change
  (num_threads=16) is described in this report and in the
  `Substrate.num_threads` docstring.  If the operator picks any of
  the options above before matrix launch, that change does warrant
  a DEVIATIONS entry.

## Artefacts

| path | role |
|---|---|
| `results/logs_ifbench/static_frontier_0_pilot/pilot_summary.json` | top-level pilot metrics |
| `results/logs_ifbench/static_frontier_0_pilot/test_eval.json` | per-instance F1 + best_val_f1 + avg_test_f1 |
| `results/logs_ifbench/static_frontier_0_pilot/drift.json` | full drift dump (per-id temp-0 scores, Spearman, frontier sets, frontier-leaving rate) |
| `results/logs_ifbench/static_frontier_0_pilot/_drift/base_temp0_scores.json` | shared base temp-0 cache (will be reused across matrix cells when the matrix runs) |
| `results/logs_ifbench/static_frontier_0_pilot/run_log.txt` | engine event log |
| `results/logs_ifbench/static_frontier_0_pilot/run_log_stderr.txt` | task-LM warnings (23 lines, all max_tokens truncation; 0 throttle) |
| `results/logs_ifbench/static_frontier_0_pilot/candidates.json` | only the seed candidate (0 accepts at N=5) |
| `results/logs_ifbench/static_frontier_0_pilot/gepa_state.bin` | resumable state |

## Tests

- `tests/test_chunk14_wiring.py`: 14 offline tests covering
  - substrate factories (`Substrate` shape, HotpotQA + IFBench);
  - the IFBench metric returning float in [0, 1] on satisfied / partial fixtures;
  - the IFBench feedback returning a non-empty feedback string with the right violated-constraint id;
  - `_make_feedback_map` composing the per-component `ScoreWithFeedback` callback;
  - hash verification passing on the real frozen table, raising on tampered file, raising on missing file, raising on side-car divergence;
  - the IFBench substrate + 100/0/0 sampler + decoupled proposer running one fake engine iteration end-to-end (no LM) — minibatch ids exclusively in the mid band, no off-band leakage.
- Full offline pytest: **142 passed** (128 prior + 14 new).
