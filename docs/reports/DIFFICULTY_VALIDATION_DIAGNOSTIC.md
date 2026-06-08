# Difficulty-validation diagnostic (three sub-tasks)

Diagnostic-only follow-up on the v1 difficulty-validation pass
(`results/difficulty_validation/`, [`DIFFICULTY_VALIDATION.md`](DIFFICULTY_VALIDATION.md)).
Numbers only. No pass/fail call. No experimental wiring change, no
loader change, no schedule arms, no Experiment 3.

## Task 1 — `per_example.json` shape

Inspected `results/difficulty_validation/per_example.json` (75 rows
from the v1 pass). Per-row keys:

```
constraint_count, id, score, source_key
```

Sample row 0:
```
{"id": 15352, "source_key": "ai2-adapt-dev/tulu_v3.9_aya_100k_5454",
 "constraint_count": 3, "score": 0.3333333333333333}
```

**The file stores only the aggregate fraction score** (`score = k / N`
returned by `ifbench_feedback.metric_fn`). **No per-verifier
satisfaction booleans are present** — neither the satisfied
instruction-id list nor the violated instruction-id list is recorded.

→ The strict re-score in task 2 cannot be done OFFLINE from the
existing artefact; it requires a re-run with a metric that captures
satisfied / violated breakdowns.

## Task 2 — Strict re-score

### Method

Re-ran the **same 75 examples** (identical bucket-sample seed = 42,
so the same `{id, source_key}` pairs as v1) through the base IFBench
seed program at `dspy.LM(cache=False)` with `num_threads=48`. The
recording metric wraps `ifbench_feedback.score_and_feedback` and
records both `satisfied` and `violations` lists per id while
returning the fraction score to `dspy.Evaluate`. Two metrics
computed from a single set of fresh predictions:

- `fraction_score = k / N` (the existing IFBench metric)
- `strict_score = 1.0 if k == N else 0.0` (all-or-nothing)

The fresh predictions differ from v1's (independent temp-0.6
samples) so the fraction means below differ slightly from the v1
numbers; reporting both metrics from the SAME run keeps them
comparable.

### Per-bucket (cc, n, fraction_mean, strict_mean) — cc = 3, 4, 5

| cc | n | fraction_mean | fraction_std | strict_mean | strict_std | n all satisfied |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 25 | **0.4267** | 0.2909 | **0.04** | 0.1960 | 1 |
| 4 | 25 | **0.5700** | 0.2182 | **0.12** | 0.3250 | 3 |
| 5 | 25 | **0.5120** | 0.2197 | **0.04** | 0.1960 | 1 |

`strict_mean` is the proportion of examples in the bucket where the
model satisfied **every** verifier. `strict_n_all_satisfied` is the
absolute count (rounded; equals `strict_mean × n`).

### Concurrency-load + wall

| | |
|---|---|
| scoring wall | **166.57 s ≈ 2 min 47 s** |
| examples scored | 75 |
| num_threads | 48 |
| throttle / 429 / backoff events on stderr | **0** |

Throttle scan: strict grep for any of `litellm.*rate`,
`litellm.*429`, `litellm.*throttle`, `litellm.*backoff`,
`RateLimitError`, `TooManyRequests`, `429 Too Many` on the captured
stderr. Zero hits. The loose `rate` grep returned only the HF-Hub
auth notice ("rate limits") — false positive, excluded.

## Task 3 — Floor investigation

### Where `CONSTRAINT_COUNT_FLOOR=3` is defined

`src/ifbench_data.py` lines 78–80:

```python
# Constraint-count floor for the substrate. Chunk-11-redo requirement:
# >=3 if that comfortably exceeds 545, else >=2. Live count at >=3 is 48,463 rows.
CONSTRAINT_COUNT_FLOOR: int = 3
```

The constant is consumed by `load_ifbench_usable`
(`src/ifbench_data.py:226`, default arg `constraint_count_floor=CONSTRAINT_COUNT_FLOOR`)
and applied as a hard cut in the loader (`src/ifbench_data.py:253`):

```python
cc = len(iids)
raw_hist[cc] += 1
if cc < constraint_count_floor:
    continue
```

`raw_hist` is incremented BEFORE the filter, so the loader's
`constraint_count_histogram_raw` field always reflects the
unfiltered universe regardless of which floor is passed.

### Why floor = 3 (cited from existing artefacts, not re-derived)

Quoted from `CHUNK11_REPORT.md::Constraint-count floor: 3`:

> BUILD_PLAN §7 Chunk 11 redo says ">= 3 if that comfortably exceeds
> 545, else >= 2". At floor 3 we have **48,463 rows**
> (23,322 + 18,038 + 7,103), ~89× the target total of 545. Floor 3
> chosen.
>
> **Why this matters for the science.** The Experiment-2 readout
> depends on a continuous, populated frontier. Rows with 3–5
> constraints produce fractional scores in
> `{0, 1/3, 1/2, 2/3, 1/4, ..., 4/5, 1}` rather than HotpotQA's
> effective `{0, 1}` bimodality. The score-spread per row provides
> the actionable, bidirectionally-bounded improvement signal that
> the BUILD_PLAN §4 D1 / §1 framing requires to test the frontier-
> band hypothesis cleanly.

### Raw cc = 1 and cc = 2 counts (live verification)

The runner called `load_ifbench_usable(constraint_count_floor=1)`
and read `.constraint_count_histogram_raw`. Live result, written to
`results/difficulty_validation/strict_summary.json::raw_constraint_count_histogram`:

| constraint_count | live count | CHUNK11_REPORT count | match? |
|---:|---:|---:|---|
| 1 | **23,007** | 23,007 | yes |
| 2 | **23,903** | 23,903 | yes |
| 3 | 23,322 | 23,322 | yes |
| 4 | 18,038 | 18,038 | yes |
| 5 | 7,103 | 7,103 | yes |
| **total** | **95,373** | 95,373 | yes |

Both buckets the v1 difficulty-validation pass reported as `n=0`
have substantial raw populations:
- **cc = 1: 23,007 rows in the raw IFBench source** (24.1 % share)
- **cc = 2: 23,903 rows in the raw IFBench source** (25.1 % share)

These rows are excluded from the carved test split (and from the
other carved pools) by the `CONSTRAINT_COUNT_FLOOR = 3` filter at
carve time. The loader was not modified.

## Artefacts

| path | role |
|---|---|
| `src/run_strict_rescore.py` | runner: recording metric + raw histogram verification |
| `results/difficulty_validation/per_example_detailed.json` | per-id `{satisfied_ids, violated_ids, fraction_score, strict_score, …}` |
| `results/difficulty_validation/strict_per_bucket.json` | per-bucket fraction + strict aggregates |
| `results/difficulty_validation/strict_summary.json` | summary + live raw histogram (cc=1..5) |

Existing files referenced (unchanged):

- [`DIFFICULTY_VALIDATION.md`](DIFFICULTY_VALIDATION.md) — v1 pass
- `results/difficulty_validation/per_example.json` — v1 per-id aggregate-only scores
- `src/ifbench_data.py` — loader (no edits)
- [`CHUNK11_REPORT.md`](CHUNK11_REPORT.md) — original raw-histogram + floor rationale
