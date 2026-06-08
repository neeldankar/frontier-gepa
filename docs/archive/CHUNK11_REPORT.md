# Chunk 11 (redo) report: IFBench substrate pivot

## Verdict: GO

Pool sizes: full Experiment-1 layout 150 / 20 / 75 / 300 = 545. No
proportional shrink and no operator-review gate, because the
substrate pivot from `allenai/IFBench_test` (300 rows) to
`allenai/IF_multi_constraints_upto5` (95,373 rows) is comfortably above
the §15 target total.

## Why the redo

Chunk 11 v1 targeted `allenai/IFBench_test` and hit OPERATOR_REVIEW:
only 300 rows, 256 of them single-constraint, so its base-system score
distribution would be bimodal-by-construction and the strict-partial
frontier would be structurally bounded by the 44 multi-constraint rows.
That reproduces the HotpotQA thin-frontier regime that Experiment 2
exists to escape, and would have failed Chunk 13's continuity
diagnostic by the same mechanism Experiment 1 already documented.

Pivoting to the IF-RLVR composite (`allenai/IF_multi_constraints_upto5`)
addresses the size and the structural-bimodality concern at once:
multi-constraint by construction, ~95k rows, score distribution is
continuous because each row's score is `k/N` for `N` constraints
satisfied out of the row's count.

## Dataset confirmation

| | |
|---|---|
| Canonical HF id | `allenai/IF_multi_constraints_upto5` |
| Family | AllenAI IF-RLVR composite (IFEval 25 + IFBench-Train 29 verifiable constraints) |
| Source provenance | The dataset bundles IFEval prompts and IFBench-Train constraints; each row carries the source row id in `key`. |
| Schema (live) | `key` (str), `messages` (list[role/content]), `ground_truth` (Python-literal str: `[{'instruction_id': [...], 'kwargs': [...]}]`), `dataset` (always `ifeval` in this composite), `constraint_type` (always `multi`), `constraint` (human-readable concatenation; informational) |
| Native splits | `train` only (single split exposed) |
| Total raw rows | **95,373** |

## Constraint-count distribution (per row)

| n_constraints | count | share |
|---|---|---|
| 1 | 23,007 | 24.1% |
| 2 | 23,903 | 25.1% |
| 3 | 23,322 | 24.5% |
| 4 | 18,038 | 18.9% |
| 5 | 7,103 | 7.4% |
| **total** | **95,373** | 100% |

Each row has its own `ground_truth.instruction_id` array; the per-row
constraint count is `len(instruction_id)`. The composite was sampled at
1–5 constraints per row.

## Constraint-count floor: 3

BUILD_PLAN §7 Chunk 11 redo says ">= 3 if that comfortably exceeds 545,
else >= 2". At floor 3 we have **48,463 rows** (23,322 + 18,038 + 7,103),
~89× the target total of 545. Floor 3 chosen.

**Why this matters for the science.** The Experiment-2 readout depends
on a continuous, populated frontier. Rows with 3–5 constraints produce
fractional scores in `{0, 1/3, 1/2, 2/3, 1/4, ..., 4/5, 1}` rather than
HotpotQA's effective `{0, 1}` bimodality. The score-spread per row
provides the actionable, bidirectionally-bounded improvement signal that
the BUILD_PLAN §4 D1 / §1 framing requires to test the frontier-band
hypothesis cleanly.

## Verifier availability

Curated catalog: 54 IFEval + IFBench-Train instruction IDs (25 + 29 =
54), enumerated by scanning the live dataset's universe at Chunk-11-redo
time. Hardcoded in `src/ifbench_data.py::KNOWN_VERIFIER_IDS`. Spans 16
families:

```
change_case (3), combination (2), copy (4), count (4),
detectable_content (2), detectable_format (9), first_word (2),
keywords (11), language (1), last_word (2), length_constraints (4),
letters (2), new (1), paragraphs (2), punctuation (3), startend (2)
```

For the live load at floor 3:
- 48,463 rows above floor
- **0 rows excluded for an uncovered instruction_id**
- `uncovered_instruction_ids = ()`

Every instruction_id present in the >=3-constraint pool is in the
curated catalog. The Chunk-12 verifier module's implementation target is
exactly this 54-id catalog.

## Carved pool sizes

| pool | size |
|---|---|
| `d_feedback` | 150 |
| `accept_batch` | 20 |
| `d_pareto` | 75 |
| `test` | 300 |
| **total** | **545** |

Disjoint by construction (filtered-pool row index is the unique id used
in the carve-time disjointness assertion). Deterministic under
`seed_splits=0` (Experiment-1 convention preserved). Carved at
import-time tests; no splits-file written to disk this chunk.

## Per-cell expected score behavior (informational, not run this chunk)

- Each carved row evaluates to `score = k / N` where `N = constraint_count`
  and `k` is the number of constraints the model satisfies on a given
  candidate.
- Floor 3 means `N ∈ {3, 4, 5}` and `score ∈ {0, 1/5, 1/4, 1/3, 2/5,
  1/2, 3/5, 2/3, 3/4, 4/5, 1}` per row. That spans the open interval
  with realistic resolution.
- For Chunk-13 binning, value bins (`hard = 0`, `mid = 0<score<1`,
  `easy = 1`) are the natural choice and should produce a populated
  middle band; rank terciles are the alternative if Chunk 13's
  continuity check prefers them. Either way the frontier band is
  structurally populated, not bimodally starved.

## Deliverables in this commit

- `src/ifbench_data.py`: rewritten for the IF-RLVR composite schema.
  `load_ifbench_usable(floor=3)`, `carve_ifbench_splits(seed, usable)`,
  `verify_carved_pool_constraint_coverage(splits)`, plus the 54-id
  catalog and the audit-counts dataclass.
- `tests/test_chunk11.py`: rewritten. 14 offline tests (catalog
  constants, schema contracts, carving with synthetic pools, verifier
  coverage check including a deliberate-unknown-iid red-team case) and
  6 live integration tests (HF cache only) confirming the live numbers
  match: 95,373 raw, the published constraint-count distribution, 48,463
  at floor 3, zero excluded, live carve produces 150/20/75/300 disjoint
  with full coverage.
- `DEVIATIONS.md` entry 6: documents the substrate pivot from
  IFBench_test to IF_multi_constraints_upto5 and the substrate-
  construction choices (floor 3, no IFEval fallback).
- `CHUNK11_REPORT.md`: this file (rewritten).

Full offline pytest: 105 + 14 = 119 passed (76 from Experiments 1+1b +
29 from the now-superseded Chunk-11 v1 file — actually replaced — see
the new total below).

## Substrate construction notes flagged for the operator

1. **The IF-RLVR composite is a TRAINING set** for the AllenAI IF-RLVR
   work, not a held-out test split. Because we carve disjoint pools
   from it with seed-deterministic shuffles, the four pools are
   internally disjoint and no train/test leakage exists *within this
   experiment*. However, models trained on the IF-RLVR composite
   (including potentially the task model itself) may have seen our
   test pool. The task model `together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo`
   was not IF-RLVR-trained, but the operator should confirm before
   Chunk 14 launch.
2. **`dataset` field is always `'ifeval'`** in this composite, meaning
   the source prompts trace back to IFEval. The constraints themselves
   span IFEval (25) + IFBench-Train (29). The composite is labeled
   "IF_multi_constraints_upto5", not "IFBench_test", so we are running
   on a derivative of the published IFBench benchmark family rather
   than the canonical evaluation split. This is consistent with the
   BUILD_PLAN §7 Chunk 11 redo instruction ("the IF-RLVR composite,
   ~95k rows, constraints from IFEval (25) + IFBench-Train (29)").

## Next chunk (deferred, NOT run here)

Chunk 12 implements the verifier module against `KNOWN_VERIFIER_IDS`,
the one-module DSPy program, the seed prompt, and the feedback
function. Chunk 13 then scores the base system on D_feedback and runs
the continuity diagnostic / go-no-go.
