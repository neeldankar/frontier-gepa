# Chunk 10 - HotpotQA isolation analysis (the leakage was the mechanism)

**HEADLINE: the off-band leakage was the mechanism behind static_easy's Experiment-1 activity.** Under 70/15/15, static_easy accepted 5.3 prompts per cell on average (per-seed [9, 3, 4]) despite supposedly drawing from the "nothing to correct" F1==1 band. Under 100/0/0, static_easy accepted **0.0** prompts per cell (per-seed [0, 0, 0]). The 30% off-band exposure was the entire driver; remove it and the band-arm behaves exactly as its label says it should.

## D5 iso - accept-count comparison (static arms)

| arm | 70/15/15 mean accepts | 100/0/0 mean accepts | per-seed 70/15/15 | per-seed 100/0/0 |
|---|---|---|---|---|
| `static_easy` | 5.3 | **0.0** | [9, 3, 4] | [0, 0, 0] |
| `static_frontier` | 7.3 | **7.3** | [8, 8, 6] | [7, 10, 5] |
| `static_hard` | 3.3 | **7.7** | [4, 3, 3] | [11, 8, 4] |

The static_easy collapse (5.3 → 0.0, a drop of 5.3 accepts/cell, with per-seed counts going from [9, 3, 4] to [0, 0, 0]) is the direct confirmation that the 30% off-band exposure was the entire driver of static_easy's Experiment-1 activity.

The pattern for the other static arms is not symmetric and is worth flagging honestly: static_frontier's mean accepts stayed essentially flat (7.3 → 7.3), while **static_hard's accepts more than doubled** (3.3 → 7.7). Under 70/15/15, static_hard's minibatch averaged ~2.1 hard + 0.45 mid + 0.45 easy; under 100/0/0 it is 3 hard. More hard examples per minibatch evidently give the reflection LM more material to propose actionable edits against, even though hard examples are F1==0 (complete failures). That contradicts the spec's a-priori framing of static_hard as the reflection-starved arm. It is a side finding of this isolation chunk, not a frontier-band claim.

## D2 iso - endpoint test F1 by arm, 70/15/15 vs 100/0/0

| arm | regime | mean test F1 | 95% CI |
|---|---|---|---|
| `random` | reused (70/15/15) | 0.6319 | [0.5984, 0.6643] |
| `static_easy` | 70/15/15 | 0.6265 | [0.5974, 0.6548] |
| `static_easy` | **100/0/0** | **0.6316** | [0.6031, 0.6598] |
| `static_frontier` | 70/15/15 | 0.6289 | [0.5994, 0.6575] |
| `static_frontier` | **100/0/0** | **0.6369** | [0.6054, 0.6672] |
| `static_hard` | 70/15/15 | 0.6348 | [0.6042, 0.6650] |
| `static_hard` | **100/0/0** | **0.6329** | [0.6022, 0.6626] |
| `vanilla_coupled_gepa` | reused (70/15/15) | 0.6386 | [0.6080, 0.6683] |

## D3 iso - paired contrasts under 100/0/0 (seed-matched; paired bootstrap, 10000 resamples, 95% percentile)

- **static_frontier − static_hard** (100/0/0 vs 100/0/0): +0.0039 [95% CI -0.0129, +0.0292]  CI includes 0
- **static_frontier − random** (100/0/0 vs 70/15/15): +0.0050 [95% CI -0.0023, +0.0100]  CI includes 0
- **static_frontier − static_easy** (100/0/0 vs 100/0/0): +0.0052 [95% CI -0.0105, +0.0167]  CI includes 0
- **static_frontier − vanilla_coupled_gepa** (100/0/0 vs 70/15/15): -0.0017 [95% CI -0.0239, +0.0171]  CI includes 0

Between-arm range collapsed from 0.0121 F1 points (70/15/15 across all 5 arms) to **0.0069** F1 points (100/0/0 for the 3 static arms + reused random and vanilla baselines). Largest within-arm seed spread in the new mix-of-regimes layout is 0.0395, which is 5.7x the between-arm range. The cross-arm contrast did not sharpen into anything that beats seed noise; the cleaner null is still a null.

## What this isolation chunk demonstrates

1. **The off-band-leakage mechanism is real and measurable.** The static_easy accept count goes from 5.3/cell (with 30% off-band leakage) to 0.0/cell (with the leakage removed). That is the direct mechanism test Experiment 1's Chunk-7 analysis hypothesized and Chunk 10 confirms.

2. **Removing the leakage does NOT manufacture a frontier effect on held-out F1.** The cross-arm endpoint contrast under 100/0/0 remains within seed noise; the frontier-band hypothesis is still not supported by these data at this scale, even with the cleaner sampling regime. That is a stronger null than Experiment 1 alone produced, because the obvious confound (leakage) is now ruled out.

3. **Read together with Experiment 1, the result narrows the design space.** Whatever explains the null, it is not the leakage. The remaining candidate from the Chunk-7 diagnosis is mechanism 1: bounded bidirectional formatting gains on a frontier dominated by verbosity / specificity mismatches. Verifying that mechanism needs a different substrate where the frontier is populated and the feedback is actionable — exactly the IFBench probe BUILD_PLAN §7 Chunks 11-15 propose.

