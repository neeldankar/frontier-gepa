# Difficulty-axis validation (IFBench, base seed program)

Numbers only. No pass/fail call.

## What was measured

Scored the IFBench base seed program (`src/ifbench_program.build_program()`)
on examples from the carved test split, bucketed by the
`constraint_count` field already present on every IFBench example.
No optimization, no GEPA loop, no schedule arms, no new experimental
wiring. Reuses `src/ifbench_substrate.ifbench_substrate()` +
`src/ifbench_feedback.metric_fn` unchanged.

| | |
|---|---|
| task model | `together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo` @ temp 0.6 |
| sampler | n/a (no GEPA loop) |
| concurrency | `dspy.Evaluate(num_threads=48)` |
| LM cache | **disabled** (`dspy.LM(cache=False)`) so the 48-thread fan-out exercises real HTTP load. Without this the Chunk-14 pilot's test eval had already cached the seed candidate's responses for every test-split id and the pass would have completed in ~1.6 s with zero outbound calls. |
| bucket cap | 25 per `constraint_count` (target 125 total) |
| bucket-sample seed | 42 |
| draw source | the carved test split (300 examples; Chunk-11 loader filters `constraint_count >= 3` at carve time) |
| frozen IFBench difficulty-table SHA-256 | `e7878d44450e7e67fa58d58b1664899da79ebf3d7298ba3d384a7b558c649365` (verified at launch) |
| dspy | 3.2.1 (pinned) |
| gepa | 0.1.1 (pinned; not used here) |

## Per-bucket results

| constraint_count | n | mean_score | std_score (population) | source |
|---:|---:|---:|---:|---|
| 1 | 0 | — | — | not present in carved test split |
| 2 | 0 | — | — | not present in carved test split |
| 3 | 25 | 0.4267 | 0.2752 | carved test split (157 available, sampled 25) |
| 4 | 25 | 0.5500 | 0.2345 | carved test split (105 available, sampled 25) |
| 5 | 25 | 0.4480 | 0.2282 | carved test split (38 available, sampled 25) |

Buckets 1 and 2 are absent from the test split because the Chunk-11
loader (`src/ifbench_data.py::CONSTRAINT_COUNT_FLOOR = 3`) excludes
`constraint_count < 3` rows at carve time. The bucket-1/2 entries are
recorded with `n=0`; they were not synthesised from elsewhere.

## Monotonicity

Strictly decreasing across buckets-with-data (cc=3 → cc=4 → cc=5)?
**No.** The break occurs at the **cc=3 → cc=4** transition:

| from cc | to cc | mean_from | mean_to | direction |
|---:|---:|---:|---:|---|
| 3 | 4 | 0.4267 | 0.5500 | **up (break)** |
| 4 | 5 | 0.5500 | 0.4480 | down |

Only the second adjacent pair (cc=4 → cc=5) drops as the hypothesis
predicts. The first pair (cc=3 → cc=4) rises by 0.123 mean score.

(Bucket population std exceeds the inter-bucket mean range, but per
your direction no pass/fail call is made; the means above are the
numbers requested.)

## Wall time + concurrency-load check

| | |
|---|---|
| scoring wall | **148.54 s ≈ 2 min 29 s** |
| examples scored | 75 |
| num_threads | 48 |
| throttle / 429 / backoff events on stderr | **0** |

Throttle scan was a strict grep for any of
`litellm.*rate`, `litellm.*429`, `litellm.*throttle`, `litellm.*backoff`,
`RateLimitError`, `TooManyRequests`, `429 Too Many` on the captured
stderr. Zero hits. (A loose grep for "rate" also returned only the
HF-Hub auth notice "rate limits" — a false positive, not a litellm
event.)

Stderr did surface **two `dspy.adapters.json_adapter` parse warnings**
(IDs 37646 cc=4 and 34595 cc=5) where the JSON adapter fell back to
ChatAdapter. Both IDs still produced valid predictions — their final
scores in `per_example.json` are 0.75 and 0.40 respectively. The
errors are evaluator-internal warnings, not throttle events.

The trailing "cannot schedule new futures after shutdown" tracebacks
on stderr are dspy's parallelizer teardown noise (same pattern the
Chunk-14 pilot saw). They fire after the evaluator returned and after
`summary.json` was written; no scores were lost to them.

## Artefacts

| path | role |
|---|---|
| `results/difficulty_validation/per_example.json` | per-id `{id, source_key, constraint_count, score}` |
| `results/difficulty_validation/per_bucket.json` | per-bucket `{constraint_count, n, mean_score, std_score}` |
| `results/difficulty_validation/summary.json` | top-level summary + monotonicity_breaks |
| `src/run_difficulty_validation.py` | the runner (NO experimental wiring, NO schedule arms, NO --cells flag) |
