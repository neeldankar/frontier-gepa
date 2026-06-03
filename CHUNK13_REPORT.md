# Chunk 13 report: IFBench difficulty table + continuity diagnostic

## Verdict: GO (chosen binning: rank terciles, frontier = 50)

The base IFBench program was scored on the 150-example D_feedback pool
at the run temperature (0.6). The continuity gate (BUILD_PLAN §4 D3)
selected **rank terciles** -- both clauses pass and pass cleanly. The
table is frozen and hashed; Chunk 14 is unblocked.

## Continuity gate readout

| metric | value | threshold | clause ok? |
|---|---|---|---|
| middle-tercile partial fraction | **1.0000** | >= 0.80 | yes |
| extreme-mass fraction | **0.2000** | < 0.50 | yes |
| rank-terciles stand | **True** | both clauses | yes |
| chosen binning | **rank_terciles** | -- | -- |
| frontier size (chosen) | **50** | >= 20 | yes |
| **verdict** | **GO** | -- | -- |

The chosen middle tercile contains zero ties at the extremes (all 50
ids are strict partials) and the overall extreme-mass fraction is 0.20
-- well clear of the 0.50 ceiling. This is the populated, continuous
frontier Experiment 2 was designed to test against.

## Score distribution (n = 150)

| | count | share |
|---|---|---|
| score == 0.0 | 18 | 12.0 % |
| 0 < score < 1 | 120 | 80.0 % |
| score == 1.0 | 12 | 8.0 % |

For comparison, the HotpotQA distractor substrate (Experiment 1, n=150)
sat at 82 % extreme mass with 31 strict partials. IFBench sits at 20 %
extreme mass with 120 strict partials -- a quantitatively different
regime, which is the whole reason for the gate switching to rank
terciles here.

## Per-band composition under the chosen (rank-tercile) binning

Boundary values: hard | mid at score = **0.3333**;  mid | easy at score = **0.6000**.

| band | size | zero | strict-partial | one |
|---|---|---|---|---|
| hard | 50 | 18 | 32 | 0 |
| mid (frontier) | 50 | 0 | **50** | 0 |
| easy | 50 | 0 | 38 | 12 |

The frontier is exactly the slice the method-under-test wants: every id
is a partial success, and the band sits where the gradient of the score
is non-trivially negotiable in both directions.

## Frozen artefacts

| path | role |
|---|---|
| `results/ifbench/difficulty_table.json` | the frozen rank-tercile table |
| `results/ifbench/difficulty_table.sha256` | `e7878d44450e7e67fa58d58b1664899da79ebf3d7298ba3d384a7b558c649365` |
| `results/ifbench/continuity_gate.json` | gate fractions, verdict, both-binnings frontier sizes |
| `results/ifbench/d_feedback_scores.json` | per-id score map (resumable) |
| `results/ifbench/d_feedback_records.json` | per-id prompt/response/violation log |
| `results/ifbench/diagnostic_chunk13/histogram.png` | distribution with chosen boundaries |
| `results/ifbench/diagnostic_chunk13/summary.json` | machine-readable summary |

The Chunk-14 matrix orchestrator MUST hash-verify
`results/ifbench/difficulty_table.json` against
`results/ifbench/difficulty_table.sha256` before any cell runs, mirroring
the Chunk-9 invariant. The difficulty table is now frozen and no
further API call should re-score D_feedback on the table path.

## Run economics

- Wall time: 3386.7 s (~56.4 min) for 150 examples, single LM call each.
- Per-example wall-clock: median ~5 s; long tail from `max_tokens=16384`
  truncation cases (one example hit 296.9 s after the model spiralled
  into repetition on a hard prompt -- the standard IFBench failure mode
  the dataset is designed to surface, not a substrate defect).
- Truncation events: 16 / 150 (10.7 %) hit `max_tokens=16384`. These
  are real generations the verifier scored against; truncation just
  means the model used the entire output budget. No retries, no
  re-scoring -- per BUILD_PLAN, the table is the one-shot frozen
  reading of base-system difficulty.
- Mean response length: 5149 chars; max 95,647 chars (one of the
  truncated runs).
- Mean constraint count per row: 3.69 (range 3..5; the floor enforced
  by the Chunk-11 loader).
- Mean satisfied / row: 1.60 of 3.69 -- consistent with the 0.43 mean
  score.

Rough token spend cannot be exactly priced without the dspy history
buffer, but at ~3000 input + 5000 output tokens per call (rough
estimate from response lengths) at ~$0.20 / 1M blended on
together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo, the run is on the order of
**$0.20**, well inside the Chunk-13 budget allowance of <$1.

## Tests

- `tests/test_chunk13.py`: 24 offline tests covering
  - rank-tercile correctness (equal thirds when divisible by 3,
    remainder to mid, disjointness, deterministic tie-breaking,
    out-of-range / empty),
  - value-bin partition (reused),
  - continuity gate (continuous -> terciles; classic bimodal -> value
    bins; semi-bimodal -> value bins; partial-OK-but-extreme-FAIL
    falls back; extreme-OK-but-partial-FAIL falls back; frontier-size
    GO criterion at and above threshold, NO-GO below),
  - `build_chosen_table` dispatching the right table per regime,
  - frozen-table immutability, SHA-256 stability and round-trip,
  - PURE_ON_BAND_MIX (100/0/0) draws against an IFBench-style
    rank-tercile table emit zero off-band leakage on the mid and
    hard targets.
- Full offline pytest: **128 passed** (104 prior + 24 new).
- Live live-API path: this chunk's scoring run is itself the live
  integration -- 150/150 examples scored, the diagnostic ran, the GO
  verdict printed. No matrix launch.

## What this chunk does NOT do (deferred)

- Chunk 14 (the IFBench matrix and drift instrumentation) is NOT
  launched here. The table is frozen and the gate verdict is GO; the
  matrix orchestrator wiring (the per-cell loop, the temp-0 drift
  passes, the cost gate against $100 / $150 / $300) is Chunk 14's job.
- CLAUDE.md is NOT edited. Per `Working protocol`, the operator flips
  the build plan checklist after reviewing this report.
- Chunk 12's `CHUNK12_REPORT.md` and `DEVIATIONS.md` entry 6 already
  document the substrate pivot to `IF_multi_constraints_upto5`; no new
  DEVIATIONS entry is needed for Chunk 13 because the design plays out
  exactly as D3 specified (rank-tercile path on a continuous
  distribution).
