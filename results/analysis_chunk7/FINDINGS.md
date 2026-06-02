# Chunk 7 - findings (the matrix is a null)

**HEADLINE: NULL.** Endpoint test-F1 mean across the five arms ranges only from 0.6265 to 0.6386 (spread 0.0121), while the largest within-arm seed spread is 0.0395. The seed-level noise is comparable to, and in places exceeds, the between-arm separation.

## D2 endpoint test F1 by arm (300-example held-out, mean of 3 seeds; 95% paired-hierarchical bootstrap CI)
- random                   0.6319  [0.5984, 0.6643]
- static_easy              0.6265  [0.5974, 0.6548]
- static_frontier          0.6289  [0.5994, 0.6575]
- static_hard              0.6348  [0.6042, 0.6650]
- vanilla_coupled_gepa     0.6386  [0.6080, 0.6683]

Mean range across arms: 0.6265 to 0.6386 (spread 0.0121).
Largest within-arm seed spread: 0.0395.

## D3 paired contrasts (seed-matched; paired bootstrap, 10000 resamples, 95% percentile)
- **static_frontier - static_hard**: -0.0059 [95% CI -0.0119, +0.0007]  CI includes 0
- **static_frontier - random**: -0.0030 [95% CI -0.0137, +0.0149]  CI includes 0
- **static_frontier - static_easy**: +0.0024 [95% CI -0.0068, +0.0147]  CI includes 0
- **static_frontier - vanilla_coupled_gepa**: -0.0097 [95% CI -0.0190, -0.0003]  **CI excludes 0**

Of the four paired contrasts, 1 has a 95% CI that excludes zero (static_frontier - vanilla_coupled_gepa) and 3 include zero. The one that excludes zero (static_frontier - vanilla_coupled_gepa: -0.0097) puts static_frontier on the WRONG side of the contrast (worse, not better), and the magnitude is smaller than the within-arm seed spread (0.0395). Honest read: the frontier-band hypothesis is NOT supported by these data at this scale, and the only CI that excludes zero points away from frontier rather than toward it.

## D4 iterations-to-target ranking flips across thresholds
Ranks (fastest first; arms that never reach are omitted):
- T=0.62: static_easy < vanilla_coupled_gepa < static_hard < random < static_frontier
- T=0.63: vanilla_coupled_gepa < random < static_easy < static_hard < static_frontier
- T=0.64: vanilla_coupled_gepa < static_easy < random < static_hard < static_frontier

Ranking flips across thresholds: True. That itself is a finding -- at this scale, "static_frontier is faster" depends on where you put the finish line, which is consistent with noise rather than a real efficiency edge. (In fact, static_frontier is the SLOWEST arm to reach T=0.63 by mean iteration count, at 30.0 iters; vanilla_coupled_gepa is the fastest at 2.0.)

## D5 acceptance dynamics (the off-band leakage in numbers)
- random: mean 6.3 accepts (per-seed: [6, 7, 6])
- static_easy: mean 5.3 accepts (per-seed: [9, 3, 4])
- static_frontier: mean 7.3 accepts (per-seed: [8, 8, 6])
- static_hard: mean 3.3 accepts (per-seed: [4, 3, 3])
- vanilla_coupled_gepa: mean 7.3 accepts (per-seed: [10, 6, 6])

Note that **static_easy accepts at a non-trivial rate** (5.3 accepts per cell on average, with one seed accepting 9 times). The static_easy minibatch is supposed to be all F1==1 examples ("nothing to correct"); the only reason it accepts is the 15% / 15% off-band draws, which pull from the strictly-partial mid band and the F1==0 hard band on every iteration. That same 30% off-band exposure reaches every static arm, so each arm sees enough partial-band material to drive useful reflection, which dilutes the between-arm contrast the experiment is built around.

## Two structural reasons the null is plausible at this scale
1. **Bounded bidirectional formatting gains.** The Chunk-5 partial inspection showed the strictly-partial cases are dominated by verbosity / specificity mismatches (the partial_verbose mu_f branch). Any instruction edit that tightens the answer to reduce extras necessarily risks dropping middle names or qualifiers and falling into the opposite verbosity failure. The reflection LM gets reliable signal both ways, so the floor is high and the ceiling is bounded -- arms converge to similar endpoints from different angles.
2. **15/15 off-band leakage.** The 70/15/15 mix means every static arm gets 30 percent of its minibatch from off-band instances. With the mid band 31 examples deep and the hard band 50 deep, even static_easy sees several partial instances and several zero instances over N=44 iterations on average, which is enough to drive accepted edits (D5 shows it does, including one static_easy cell with 9 accepts). The "static-easy" arm in this setup is not reflection-starved; it is reflection-diluted. A genuine purity test would use a 100/0/0 draw and accept the coverage-vs-purity tradeoff.

## Interview-ready summary (3-4 sentences)
We ran the 15-cell static-band matrix end-to-end at the locked D_feedback=150 / A=20 / D_pareto=75 / N=44 settings and observed a clean null: arm-mean held-out test F1 spans only 0.0121 F1 points while within-arm seed spread reaches 0.0395. Three of four paired contrasts against static_frontier have 95 percent CIs that include zero; the one that excludes zero (static_frontier - vanilla_coupled_gepa = -0.0097) puts static_frontier on the WORSE side of the contrast and has a magnitude smaller than the within-arm seed spread (0.0395). The cleanest design lesson is the 15/15 off-band leakage: static_easy still accepted prompts because 30 percent of every minibatch is off-band, so each arm gets enough partial-band examples to drive useful reflection and the cross-arm contrast gets diluted. A sharper follow-up swaps the 70/15/15 mix for a 100/0/0 pure-band draw and asks whether the resulting purity gain beats the coverage loss -- that is the genuine test of band selection as a lever.
