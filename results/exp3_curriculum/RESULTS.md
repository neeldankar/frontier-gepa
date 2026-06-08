# Exp-3 Curriculum: Results

Aggregated from the 12 cell summaries in `results/exp3_curriculum/<arm>/seed<seed>.json`
(4 arms x seeds {0,1,2}, T=40). This file states only what the figure and tables
show. Comparison against the pre-registered predictions is in the writeup, separately.

Primary endpoint: accept dynamics (P1 iteration-to-first-accept, P2 cumulative accepts).
Secondary: held-out test F1.

## Primary figure

`cumulative_accepts.png` -- cumulative accepts vs iteration, one bold line per arm
(mean of 3 seeds) with the three per-seed curves shown as thin lines of the same
color. Dotted verticals mark the phase boundaries (iters 0-13 / 14-26 / 27-39). The
right panel zooms to phase 1 (iters 0-13).

Reading the curves at the end of phase 1 (iteration 13), cumulative accepts:

| arm | seed0 | seed1 | seed2 | mean |
| --- | --- | --- | --- | --- |
| random | 1 | 2 | 0 | 1.00 |
| easy_to_hard | 0 | 0 | 0 | 0.00 |
| hard_to_easy | 2 | 2 | 1 | 1.67 |
| static_medium | 2 | 3 | 2 | 2.33 |

`easy_to_hard` is at 0 cumulative accepts through iteration 13 for all three seeds
(per-seed [0, 0, 0]); its first accepted iteration per seed is
[15, 15, 21].

## Primary table 1: iteration-to-first-accept

| arm | seed0 | seed1 | seed2 | mean |
| --- | --- | --- | --- | --- |
| random | 1 | 2 | 25 | 9.33 |
| easy_to_hard | 15 | 15 | 21 | 17.00 |
| hard_to_easy | 1 | 10 | 1 | 4.00 |
| static_medium | 5 | 4 | 1 | 3.33 |

## Primary table 2: total accepts (at iteration 40)

| arm | seed0 | seed1 | seed2 | mean |
| --- | --- | --- | --- | --- |
| random | 3 | 4 | 3 | 3.33 |
| easy_to_hard | 5 | 2 | 3 | 3.33 |
| hard_to_easy | 6 | 4 | 1 | 3.67 |
| static_medium | 5 | 4 | 2 | 3.67 |

## Secondary table: final held-out test F1 (300-example test split)

Labeled secondary; per the pre-registration this endpoint is reported, not treated as
discriminating. Per-seed values and the within-arm range are shown because the spread
is the relevant context.

| arm | seed0 | seed1 | seed2 | mean | range(max-min) |
| --- | --- | --- | --- | --- | --- |
| random | 0.6299 | 0.6182 | 0.6325 | 0.6269 | 0.0143 |
| easy_to_hard | 0.6214 | 0.6371 | 0.6360 | 0.6315 | 0.0157 |
| hard_to_easy | 0.6391 | 0.6274 | 0.6316 | 0.6327 | 0.0116 |
| static_medium | 0.5985 | 0.6162 | 0.6724 | 0.6290 | 0.0739 |
