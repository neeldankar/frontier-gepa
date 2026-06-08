# Findings: frontier-band reflection selection for GEPA

Two independent feasibility probes, not a controlled cross-dataset
comparison. Substrates, binning rules, and N's differ. Each result
stands on its own; together they constrain the hypothesis.

## Headline

1. **HotpotQA, N=44, 5 arms × 3 seeds, 70/15/15.** Clean null.
   Between-arm endpoint test-F1 mean spans 0.0121, dwarfed by a
   within-arm seed spread of 0.0395. The frontier-band hypothesis is
   not supported by these data at this scale.

2. **HotpotQA, N=44, 3 static arms × 3 seeds, 100/0/0
   (Experiment 1b).** The off-band-leakage mechanism is real and
   measurable: static_easy's accepts/cell collapse from 5.3 → 0.0
   when the 30% off-band exposure is removed. Removing the leakage
   does NOT manufacture a frontier effect. The cleaner null is still
   a null.

3. **IFBench, designed and built end-to-end; matrix not executed.**
   The Chunk-13 difficulty table cleared the §4 D3 continuity gate
   (rank terciles, frontier = 50 / 150, GO). The Chunk-14 wiring is
   complete and a parallel-eval pilot validated the pipeline end-to-end
   (5 iters + test eval + temp-0 drift). The matrix was NOT launched
   because of a **substrate-cost diagnosis at pilot time**: at the run
   temperature (0.6) and max_tokens=16384, the IF_multi_constraints_upto5
   model output is long enough that even 16-thread parallelism leaves
   per-call wall at ~17 s on average. The 15-cell matrix projects to
   **57 h at N=40, 69 h at N=60, 82 h at N=80** — intractable on this
   hardware in the available window. The cost projection ($5-7) sits
   well under the $100 gate; the binding constraint is wall time, not
   spend.

## What each result means

### HotpotQA (Experiment 1)

5 arms × 3 seeds × N=44, 70/15/15, decoupled acceptance on a fixed
20-example A-batch, strict-improvement. Held-out test = 300.

Endpoint test F1, mean (seed-min, seed-max):

| arm | mean | seed range |
|---|---|---|
| random | 0.632 | [0.615, 0.652] |
| static_easy | 0.626 | [0.611, 0.658] |
| static_frontier | 0.629 | [0.609, 0.665] |
| static_hard | 0.635 | [0.620, 0.665] |
| vanilla_coupled_gepa | 0.639 | [0.621, 0.655] |

Three of four paired contrasts against static_frontier have 95% CIs
that include zero; the one that excludes zero
(`frontier − vanilla = -0.0097`) puts frontier on the worse side and
is still smaller than the seed spread. Full numbers and figures in
[`../experiments/exp1_bands/analysis_chunk7/FINDINGS.md`](../experiments/exp1_bands/analysis_chunk7/FINDINGS.md).

Diagnosed mechanisms behind the null:

1. **Formatting frontier with bounded bidirectional gains.** The
   base-system F1 on D_feedback is sharply bimodal: 82% mass at
   exactly 0 or 1, only 31 strictly-partial. The strict partials are
   format and granularity near-misses. Optimization has little room
   to move and which band you reflect on cannot produce much
   separation.

2. **Off-band leakage.** Under 70/15/15, every static arm sees 30%
   off-band exposure. Static_easy accepts 5.3 prompt edits per cell
   on average (one seed accepts 9 times) despite supposedly drawing
   from the "nothing to correct" F1==1 band. The cross-arm contrast
   was diluted by design.

### HotpotQA Experiment 1b (isolation)

100/0/0 re-run of the three static arms, reusing Experiment 1's
random and vanilla cells unchanged (Chunk-8 byte-identical
regression test proved this is sound). Same N=44, same frozen
difficulty table, separate logs at `../experiments/exp1_bands/logs_hotpot_100/`.

| arm | 70/15/15 mean accepts | 100/0/0 mean accepts | per-seed under 100/0/0 |
|---|---|---|---|
| static_easy | 5.3 | **0.0** | [0, 0, 0] |
| static_frontier | 7.3 | 7.3 | [7, 10, 5] |
| static_hard | 3.3 | **7.7** | [11, 8, 4] |

The static_easy collapse to 0.0/cell is the direct mechanism test:
the 30% off-band exposure was the entire driver of static_easy's
Experiment-1 activity.

Endpoint test F1 between-arm range under 100/0/0 collapsed from
0.0121 to 0.0069. Largest within-arm seed spread under the
mix-of-regimes layout is 0.0395, 5.7× the between-arm range. The
cross-arm contrast did not sharpen into anything that beats seed
noise; the cleaner null is still a null.

Side finding worth flagging honestly: static_hard's accepts more than
doubled under 100/0/0 (3.3 → 7.7). The reflection LM evidently has
more material to propose against on F1==0 prompts than the a-priori
framing predicted. This is a side finding of the isolation chunk,
not a frontier-band claim.

Full numbers and figures in
[`../experiments/exp1_bands/analysis_chunk10/FINDINGS_hotpot_100.md`](../experiments/exp1_bands/analysis_chunk10/FINDINGS_hotpot_100.md).

### IFBench (Experiment 2) — substrate cost, not result

The IFBench probe was designed to escape mechanism 1 (HotpotQA's
bimodal formatting frontier) by moving to a substrate where the
frontier is populated and the per-row feedback is actionable (the
fraction of verifier-checked constraints satisfied). Built end-to-end:

- Disjoint pools carved from `allenai/IF_multi_constraints_upto5`
  (constraint-count floor 3; 48,463 usable rows; sizes
  150/20/75/300; Chunk 11 redo).
- One-module program (`prompt` → `response`) with a minimal seed,
  vendored IFEval verifiers (54 instruction IDs, pinned to upstream
  commit `ebca9f7d…`), uniform reflective feedback string naming the
  violated constraints (Chunk 12).
- Frozen Chunk-13 difficulty table: rank terciles, frontier = 50,
  GO verdict (BUILD_PLAN §4 D3 continuity gate passed cleanly with
  middle-tercile partial fraction = 1.00, extreme-mass fraction =
  0.20). See [`reports/CHUNK13_REPORT.md`](reports/CHUNK13_REPORT.md).
- Chunk-14 wiring: substrate dispatch through the existing
  orchestrator + GEPA runner; HotpotQA path byte-identical
  (default substrate). 142/142 offline tests pass (128 prior + 14
  Chunk-14).
- D4 temp-0 drift instrumentation implemented and exercised at pilot
  time (Spearman + temperature-consistent frontier-leaving rate).
- SHA-256 of the frozen difficulty table verified at variant
  selection; the runner refuses to launch on mismatch.

The matrix was NOT launched. Pilot diagnosis below.

#### Substrate-cost diagnosis (the reason no matrix ran)

A parallel-eval pilot (`static_frontier`, seed=0, N=5, full pools,
`num_threads=16`, sandboxed) ran end-to-end in ~2h 20min. **Zero
rate-limit / 429 / throttle / backoff events** on the run stderr.
The pipeline works.

Measured walls:

| phase | wall |
|---|---|
| 5-iter loop (210 evals) | 12.3 min |
| test eval (300 examples) | ~85 min |
| drift base (150 examples, shared once across cells) | 42.7 min |
| drift final (cache hit; pilot final == seed) | 0.75 s |
| **pilot total** | **~2 h 20 min** |

Implied per-call wall at temp 0.6 with 16-thread parallelism:
**~17 s/example** on average. The truncation rate at
`max_tokens=16384` was 23/210 on the iter loop alone — the model
occasionally runs to the 16K-token ceiling on hard prompts and pays
several minutes per such call. Parallelism gives ~2× speedup on the
loop but does not linearize the long-tail per-call cost.

Matrix projection at the measured per-call wall:

| N | matrix cost | matrix wall |
|---|---|---|
| 40 | **$5.08** | **~57 h (~2.4 days)** |
| 60 | $6.21 | ~69 h (~2.9 days) |
| 80 | $7.34 | ~82 h (~3.4 days) |

The $100 cost gate passes cleanly at every N. The 4-5 hour
overnight window the BUILD_PLAN budgeted (§7 Chunk 14) is missed at
every N. Mitigations exist — lower `max_tokens`, shrink the test
pool, skip per-cell drift, lower N — but each is a real deviation
from spec, and several change the metric (lower `max_tokens`
changes scoring, so the frozen Chunk-13 table is no longer the
substrate's binding). Re-piloting under any spec change is
required for honesty.

Full pilot numbers in [`CHUNK14_PILOT_REPORT.md`](CHUNK14_PILOT_REPORT.md).

#### What the IFBench probe tells us, without the matrix

The substrate-cost finding is itself the IFBench probe's result for
this build.

- The Chunk-13 GO verdict at frontier=50 confirms that
  `IF_multi_constraints_upto5` has the populated continuous frontier
  the HotpotQA distractor substrate lacked. The design hypothesis
  ("move to a substrate where the frontier is populated") was
  validated at the difficulty-table layer.
- The pilot validated that the program, the verifier path, the
  100/0/0 sampler, the decoupled acceptance, the test eval, the
  drift instrumentation, and the hash-verified table all compose
  correctly end-to-end. There is no wiring defect blocking the
  matrix; only wall-time-to-completion.
- The per-call cost at temp 0.6 + max_tokens=16384 + output-heavy
  generations is a property of the *substrate-task-model pair*, not a
  bug in the matrix machinery. A larger or faster task model, a
  smaller max_tokens, or a smaller test pool would each plausibly
  fit; choosing among them is operator judgment, not engineer
  judgment.

## Cross-probe synthesis

The two probes constrain the hypothesis from different directions.

- **HotpotQA tells us:** at this scale and on this substrate, band
  selection does not move endpoint F1 above seed noise. The
  isolation variant rules out off-band leakage as the explanation;
  the remaining diagnosed mechanism is the bounded bidirectional
  formatting gains on a bimodal substrate.
- **IFBench tells us:** the substrate the hypothesis really wants —
  populated continuous frontier, actionable per-row feedback — is
  also the substrate whose per-call wall is large enough that the
  15-cell matrix at N=80 does not fit on commodity inference within
  the build window. The lever cannot be tested on the substrate it
  was designed for without either better hardware or a methodology
  deviation.

Both probes are honest reads:

- The HotpotQA null is supported by 15 cells of actual results
  across two sampling regimes.
- The IFBench finding is supported by an end-to-end pilot whose
  per-call wall measurement is the load-bearing number, plus a
  Chunk-13 GO verdict that confirms the substrate has the shape
  the hypothesis needs.

Neither probe rejects the frontier-band hypothesis. The HotpotQA
result says "not at this scale, on this substrate, against seed
noise"; the IFBench result says "the matrix that would test it
properly does not fit in the available window". The combined read
is: **the lever is not falsified, but a credible falsification or
confirmation requires either a different substrate-task-model pair
than the one currently used, or substantially more hardware than
this build had available.**

## Pointers

### HotpotQA Experiment 1

- [`../experiments/exp1_bands/analysis_chunk7/FINDINGS.md`](../experiments/exp1_bands/analysis_chunk7/FINDINGS.md) — full narrative + interview-ready summary
- [`../experiments/exp1_bands/analysis_chunk7/iteration_curves.png`](../experiments/exp1_bands/analysis_chunk7/iteration_curves.png) — best-so-far val F1 per arm
- [`../experiments/exp1_bands/analysis_chunk7/endpoint_test_f1.png`](../experiments/exp1_bands/analysis_chunk7/endpoint_test_f1.png) — forest plot, paired-hierarchical bootstrap CIs
- [`../experiments/exp1_bands/analysis_chunk7/iterations_to_target.png`](../experiments/exp1_bands/analysis_chunk7/iterations_to_target.png) — iterations-to-target sweep
- [`../experiments/exp1_bands/analysis_chunk7/summary.json`](../experiments/exp1_bands/analysis_chunk7/summary.json) — every number, bootstrap seed 20260602

### HotpotQA Experiment 1b (isolation)

- [`../experiments/exp1_bands/analysis_chunk10/FINDINGS_hotpot_100.md`](../experiments/exp1_bands/analysis_chunk10/FINDINGS_hotpot_100.md) — accept-count collapse + paired contrasts
- [`../experiments/exp1_bands/analysis_chunk10/accept_collapse.png`](../experiments/exp1_bands/analysis_chunk10/accept_collapse.png) — the leakage mechanism, visualised
- [`../experiments/exp1_bands/analysis_chunk10/iteration_curves_isolation.png`](../experiments/exp1_bands/analysis_chunk10/iteration_curves_isolation.png) — 70/15/15 vs 100/0/0
- [`../experiments/exp1_bands/analysis_chunk10/endpoint_test_f1_isolation.png`](../experiments/exp1_bands/analysis_chunk10/endpoint_test_f1_isolation.png) — endpoint F1 under both mixes
- [`../experiments/exp1_bands/analysis_chunk10/summary.json`](../experiments/exp1_bands/analysis_chunk10/summary.json) — every isolation number

### IFBench Experiment 2

- [`reports/CHUNK13_REPORT.md`](reports/CHUNK13_REPORT.md) — difficulty table + continuity gate (GO)
- [`../experiments/exp2_ifbench/diagnostic_chunk13/histogram.png`](../experiments/exp2_ifbench/diagnostic_chunk13/histogram.png) — score distribution with rank-tercile boundaries
- [`../experiments/exp2_ifbench/difficulty_table.sha256`](../experiments/exp2_ifbench/difficulty_table.sha256) — frozen-table hash (verified at every cell launch)
- [`CHUNK14_PILOT_REPORT.md`](CHUNK14_PILOT_REPORT.md) — wiring + pilot + substrate-cost diagnosis
- [`../experiments/exp2_ifbench/pilot/logs_ifbench/static_frontier_0_pilot/pilot_summary.json`](../experiments/exp2_ifbench/pilot/logs_ifbench/static_frontier_0_pilot/pilot_summary.json) — pilot numbers
- [`../experiments/exp2_ifbench/pilot/logs_ifbench/static_frontier_0_pilot/drift.json`](../experiments/exp2_ifbench/pilot/logs_ifbench/static_frontier_0_pilot/drift.json) — D4 temp-0 Spearman + frontier-leaving rate

### Design + protocol

- [`DEVIATIONS.md`](DEVIATIONS.md) — six forced substitutions, each with cause and impact (HotpotQA distractor substrate, Qwen2.5-7B substitution, D_feedback growth, value-bin binning, 100/0/0 isolation variant, IFBench substrate pivot)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Chunk-1 GEPA source recon and the hook plan (no engine fork)
- [`gepa_band_selection_handoff.md`](gepa_band_selection_handoff.md) — original design spec
- [`CLAUDE.md`](CLAUDE.md) — build plan, chunk-by-chunk status, locked parameters

## What is not in this writeup

- A full IFBench matrix readout. Not run; the substrate-cost
  diagnosis is the IFBench result.
- A re-piloted IFBench cell at lower `max_tokens` or smaller test
  pool. Either would change the metric or the spec; the operator's
  call, not the build's.
- A claim that the frontier-band hypothesis is false. The HotpotQA
  null at this scale, even with leakage removed, is consistent with
  either "no effect" or "an effect smaller than 0.0395 seed noise on
  a bimodal substrate"; neither rejects the hypothesis at the level
  of evidence collected here.
