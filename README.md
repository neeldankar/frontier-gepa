# frontier-gepa

A GEPA feasibility probe testing whether the **difficulty band of the
examples GEPA reflects on** changes its sample efficiency, at matched
iteration budget. Built as two independent experiments:

- **Experiment 1 + 1b (HotpotQA):** 5 arms × 3 seeds × N=44, decoupled
  acceptance on a fixed 20-example A-batch, 100/0/0 isolation re-run
  of the three static arms. Substrate is HotpotQA distractor (the §9
  fallback after the hosted ColBERTv2 index proved unreachable); task LM
  is Qwen2.5-7B-Instruct-Turbo on Together (the §15 forced
  substitution); reflection LM is `gpt-4.1-mini`. Both matrices ran;
  total matrix spend ~$8.62.

- **Experiment 2 (IFBench):** the populated-frontier substrate
  (`allenai/IF_multi_constraints_upto5`). Designed and built end-to-end:
  one-module program, vendored IFEval verifiers, frozen Chunk-13
  difficulty table (continuity gate GO at frontier=50), full Chunk-14
  wiring with D4 temp-0 drift instrumentation, and a parallel-eval pilot
  that confirmed the pipeline works. **Matrix NOT executed** because of
  a substrate-cost diagnosis at pilot time — see headline #3 below.

## Headline

1. **HotpotQA, 5 arms × 3 seeds, 70/15/15.** Clean null. Between-arm
   endpoint test-F1 mean spans 0.0121, dwarfed by a within-arm seed
   spread of 0.0395. The frontier-band hypothesis is not supported by
   these data at this scale.

2. **HotpotQA, 100/0/0 isolation.** The off-band-leakage mechanism is
   real and measurable: static_easy's accepts/cell collapse from 5.3
   → 0.0 when 30% off-band exposure is removed. Removing the leakage
   does NOT manufacture a frontier effect. The cleaner null is still a
   null.

3. **IFBench, designed and built; matrix intractable.** Chunk-13
   difficulty table cleared the continuity gate (rank terciles,
   frontier = 50/150, GO). Chunk-14 wiring complete; pilot validated
   the pipeline end-to-end. At measured per-call wall (~17 s/example
   on temp-0.6 output-heavy generations hitting the 16K max_tokens
   ceiling), the 15-cell matrix projects to **57 h at N=40, 69 h at
   N=60, 82 h at N=80**. The $100 cost gate passes ($5–7 projected);
   the binding constraint is wall time, not spend. We do not ship a
   matrix at a deviation cost (lower max_tokens / smaller test pool /
   smaller N / skip drift) without operator-level methodology
   guidance.

The full cross-probe writeup is in [`FINDINGS.md`](FINDINGS.md).

## Combined writeup

- [`FINDINGS.md`](FINDINGS.md) — cross-probe narrative, with the
  IFBench substrate-cost diagnosis presented as the IFBench result.

## Per-probe pointers

### HotpotQA Experiment 1 (70/15/15, 5 arms × 3 seeds)

- [`results/analysis_chunk7/FINDINGS.md`](results/analysis_chunk7/FINDINGS.md) — narrative + interview-ready summary
- [`results/analysis_chunk7/iteration_curves.png`](results/analysis_chunk7/iteration_curves.png) — best-so-far val F1 per arm
- [`results/analysis_chunk7/endpoint_test_f1.png`](results/analysis_chunk7/endpoint_test_f1.png) — forest plot (paired-hierarchical bootstrap CIs)
- [`results/analysis_chunk7/iterations_to_target.png`](results/analysis_chunk7/iterations_to_target.png) — iterations-to-target at T=0.63
- [`results/analysis_chunk7/summary.json`](results/analysis_chunk7/summary.json) — every number, bootstrap seed 20260602

### HotpotQA Experiment 1b (100/0/0 isolation)

- [`results/analysis_chunk10/FINDINGS_hotpot_100.md`](results/analysis_chunk10/FINDINGS_hotpot_100.md) — accept-count collapse + paired contrasts
- [`results/analysis_chunk10/accept_collapse.png`](results/analysis_chunk10/accept_collapse.png) — the leakage mechanism, visualised
- [`results/analysis_chunk10/iteration_curves_isolation.png`](results/analysis_chunk10/iteration_curves_isolation.png) — 70/15/15 vs 100/0/0 curves
- [`results/analysis_chunk10/endpoint_test_f1_isolation.png`](results/analysis_chunk10/endpoint_test_f1_isolation.png) — endpoint F1 under both mixes
- [`results/analysis_chunk10/summary.json`](results/analysis_chunk10/summary.json) — every isolation number

### IFBench Experiment 2 (designed, built, pilot-validated; matrix NOT run)

- [`CHUNK13_REPORT.md`](CHUNK13_REPORT.md) — base scoring + continuity gate (GO; frontier=50, rank terciles)
- [`results/ifbench/diagnostic_chunk13/histogram.png`](results/ifbench/diagnostic_chunk13/histogram.png) — score distribution with chosen boundaries
- [`results/ifbench/difficulty_table.sha256`](results/ifbench/difficulty_table.sha256) — frozen-table hash (verified at variant selection)
- [`CHUNK14_PILOT_REPORT.md`](CHUNK14_PILOT_REPORT.md) — wiring report + parallel-eval pilot + substrate-cost diagnosis
- [`results/logs_ifbench/static_frontier_0_pilot/pilot_summary.json`](results/logs_ifbench/static_frontier_0_pilot/pilot_summary.json) — pilot numbers
- [`results/logs_ifbench/static_frontier_0_pilot/drift.json`](results/logs_ifbench/static_frontier_0_pilot/drift.json) — D4 temp-0 Spearman + frontier-leaving rate

## Design and protocol

- [`DEVIATIONS.md`](DEVIATIONS.md) — six forced substitutions from the
  §15 settled-decisions list, each with cause, decision date, and impact
  on the science.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Chunk-1 GEPA source recon and
  the hook plan (no engine fork).
- [`CLAUDE.md`](CLAUDE.md) — build plan, chunk-by-chunk status, locked
  parameters.
- [`gepa_band_selection_handoff.md`](gepa_band_selection_handoff.md) —
  original design spec.

## Reproducing

```
uv venv --python 3.10 .venv
uv pip install -r requirements.txt
cp .env.example .env  # fill in TASK_MODEL + TOGETHER_API_KEY + OPENAI_API_KEY
.venv/bin/python -m pytest tests/ -m "not integration"     # 142 offline tests

# HotpotQA Experiment 1 (~$5, 1.5-2 hours)
.venv/bin/python -m src.score_d_feedback                    # builds difficulty_table.json
.venv/bin/python -m src.diagnostic_chunk5                   # GO/NO-GO + figure
.venv/bin/python -m src.orchestrator all                    # runs the 70/15/15 matrix
.venv/bin/python -m src.analysis_chunk7                     # offline; rebuilds FINDINGS + figures

# HotpotQA Experiment 1b (100/0/0 isolation, ~$3, 1-1.5 hours)
.venv/bin/python -m src.orchestrator matrix --variant hotpot_100  # 3 static arms × 3 seeds
.venv/bin/python -m src.analysis_chunk10                    # offline; rebuilds isolation FINDINGS

# IFBench Experiment 2 (Chunk-13 base scoring + continuity gate; Chunk-14 NOT run)
.venv/bin/python -m src.score_ifbench_d_feedback            # ~$0.20, builds ifbench difficulty table
.venv/bin/python -m src.diagnostic_chunk13                  # continuity gate + figure
.venv/bin/python -m src.orchestrator pilot --variant ifbench_100 --pilot-n-iter 5
                                                            # validates pipeline; ~$0.14, ~2h 20min
# .venv/bin/python -m src.orchestrator matrix --variant ifbench_100
#   NOT recommended without methodology guidance; see CHUNK14_PILOT_REPORT.md
#   for the substrate-cost diagnosis (57-82h projected wall at N=40-80).
```

Pinned versions in `requirements.txt`: `dspy==3.2.1`, `gepa==0.1.1`.
