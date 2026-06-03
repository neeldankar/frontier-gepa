# Chunk 11 report: IFBench dataset gate (STOP for operator review)

## Verdict

`OPERATOR_REVIEW` — the proposed `D_feedback = 83` is below the
`D_FEEDBACK_REVIEW_FLOOR = 90` flag in BUILD_PLAN §7 Chunk 11. Rank
terciles project to 27, comfortably above the `TERCILE_NOGO_FLOOR = 20`
but uncomfortably close. **Carving is blocked** until the operator
acknowledges the gate (the loader refuses unless
`allow_review_threshold=True` is passed).

## Dataset confirmation

| | |
|---|---|
| Canonical HF id | `allenai/IFBench_test` |
| Family | AllenAI multi-constraint IFBench (Pyatkin et al. 2025) |
| Paper | NeurIPS 2025; arXiv 2505.07591 |
| Split used | `train` (single split exposed by IFBench_test) |
| Is this IFEval? | **No.** Distinct dataset; IFEval is Google's. |

Neither `gepa==0.1.1` nor `dspy==3.2.1` ships an IFBench loader, so
`src/ifbench_data.py` rolls its own. The schema is preserved verbatim so
Chunk 12's verifiers can consume it: `key`, `prompt`,
`instruction_id_list`, `kwargs`.

## Usable count

| | |
|---|---|
| Raw rows | **300** |
| Usable rows (prompt + ≥1 constraint) | **300** |
| Rows with **single** constraint | 256 |
| Rows with **multiple** (≥2) constraints | 44 |

All 300 raw rows passed the usable filter. Note that "multi-constraint" in
the dataset's family name refers to IFBench's contribution at the
**benchmark** level (its 58 OOD constraints distinguishing it from
IFEval). It does not mean every individual row carries multiple
constraints — only 44 of 300 (15%) do.

## Proportional shrink

Target pool layout from BUILD_PLAN §15 / §4 D1: `150:20:75:300` = 545.
With only 300 usable rows, all four pools scale by `300/545 ≈ 0.5505`,
with the integer rounding distributed by largest-fractional-part first:

| pool | target | shrunk | scale factor |
|---|---|---|---|
| d_feedback | 150 | **83** | 0.553 |
| accept_batch | 20 | 11 | 0.550 |
| d_pareto | 75 | 41 | 0.547 |
| test | 300 | 165 | 0.550 |
| **total** | 545 | **300** | exact fit |

Verified by tests (`tests/test_chunk11.py`):

- `test_300_usable_matches_chunk11_report_numbers`
- `test_sums_match_capped_usable[300]`
- `test_ratios_are_preserved_within_rounding`
- `test_operator_review_when_d_feedback_below_floor`
- `test_split_sizes_match_shrunk_proportions`

## Gate analysis

The gate logic per BUILD_PLAN §7 Chunk 11 (last bullet):

| check | value | threshold | result |
|---|---|---|---|
| Dataset family is multi-constraint AllenAI? | yes | required | pass |
| `D_feedback` | 83 | `>= 90` for GO | **fail (< 90 → OPERATOR_REVIEW)** |
| Rank tercile size | 27 | `> 20` for not-NO_GO | pass |

The Chunk-13 GO/NO-GO downstream looks at the FRONTIER band's count
(rank-tercile = 27 here, or whatever the value-bin frontier is on
IFBench's distribution). 27 is above the NO-GO floor of 20 by 7
instances — narrow margin.

## Why this is operator-decidable, not auto-NO-GO

The BUILD_PLAN explicitly distinguishes these two outcomes:

> If the resulting D_feedback would be small enough that rank terciles
> fall near the 20-instance NO-GO floor (flag if D_feedback < ~90,
> terciles ~30), STOP and report for operator review rather than
> proceeding.

We hit the "review" threshold (D_feedback = 83 < 90; tercile = 27 ~ 30),
not the "NO-GO" floor (tercile would have to be ≤ 20). The operator's
trade-off is between:

1. **Proceed with shrunk pools** (83 / 11 / 41 / 165). Smaller
   D_feedback means more sampling variance in the difficulty table and
   thinner per-band sampling at b=3 over N=80. The IFBench frontier may
   or may not survive the size shrink, depending on Chunk-13's
   continuity diagnostic.
2. **Pivot the substrate** to a larger IFBench variant (e.g.
   `allenai/IF_multi_constraints_upto5` for IF-RLVR training data, ~6k
   rows but mixed source with IFEval) or accept that Experiment 2 will
   not run on IFBench.
3. **Reduce N for IFBench** below the BUILD_PLAN §4 D2's N=80 to
   compensate for thinner pools. Honest framing: this would dilute the
   already-secondary endpoint readout further.

I am not authorized to make any of those calls. STOP and report.

## Side observations worth flagging

- **15% multi-constraint share.** Only 44 / 300 rows have ≥2
  constraints. If the scientific question requires multi-constraint
  prompts specifically (rather than the benchmark family), the usable
  count drops to 44 and the dataset is effectively unworkable. This
  reading is mentioned because the BUILD_PLAN's word "multi-constraint"
  is ambiguous between (a) the family name, vs (b) a per-row property.
- **No native dev/test split**, only `train`. We carve from the full
  300 regardless. Consistent with Experiment 1's "carve our own"
  methodology.
- **Difficulty-table size implication.** A rank-tercile of 27 is small
  but not pathological for the BUILD_PLAN §4 D3 logic. Value-based bins
  on IFBench's score distribution may produce different band sizes; the
  continuity diagnostic in Chunk 13 will decide between rank and value
  bins.

## What was committed in this chunk

- `src/ifbench_data.py`: loader, constants, `compute_proportional_sizes`,
  `gate_decision`, `carve_ifbench_splits` (gated, refuses without
  `allow_review_threshold=True` under OPERATOR_REVIEW; always refuses
  under NO_GO).
- `tests/test_chunk11.py`: 29 offline tests + 6 live integration tests
  (HF cache only, no API). All pass.
- `CHUNK11_REPORT.md`: this file.

**Not committed** (operator-gated): any carved splits to disk. The
Chunk-13 difficulty table and any downstream artifacts also wait on the
operator decision.

## Decision needed

Acknowledge the operator-review gate and authorize one of:

1. **PROCEED** with the shrunk 83/11/41/165 pools — pass
   `allow_review_threshold=True` to `carve_ifbench_splits` in Chunks 13+.
2. **PIVOT** to a different IFBench variant or a different substrate.
3. **PIVOT** to a different N or pool layout for Experiment 2.
4. **ABANDON** Experiment 2.

The Chunk-13 continuity diagnostic and Chunk-14 matrix will not run
until the operator chooses.
