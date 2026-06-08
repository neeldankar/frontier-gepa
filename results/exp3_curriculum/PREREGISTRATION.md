# Pre-registration: GEPA Curriculum on HotpotQA (Exp 3)

Status: to be locked (committed, unedited) before the Chunk 5 full matrix runs.
Date drafted: 2026-06-07.

This document states the design, metrics, and predictions before any Exp 3 run is
executed. Once committed it is not edited. Results and interpretation go in a separate
writeup. The point of locking predictions now is to make the outcome informative whichever
way it falls, and to remove the temptation to reframe a null after seeing it.

---

## Research question

GEPA samples its reflection minibatch uniformly at random. Does scheduling the difficulty
of the examples fed to reflection (a curriculum) change GEPA's optimization, at matched
iteration budget, with acceptance decoupled from reflection?

This extends Exp 1/1b. Those tests used static difficulty bands and found that band
concentration changes optimization dynamics (accept rate) but not held-out generalization.
Exp 3 changes the lever from static concentration to a temporal schedule and asks whether
the ordering matters.

---

## Design (frozen before running)

- Substrate: HotpotQA, distractor setting.
- Arms (4): `random` (control, native GEPA uniform sampling over all D_feedback),
  `easy_to_hard` (hypothesis), `hard_to_easy` (anti-curriculum check), `static_medium`
  (static-band analog).
- Seeds: {0, 1, 2}, the same three applied to every arm. 12 runs total.
- Iteration budget: T = 40 per run.
- Minibatch b = 3. Acceptance batch A = 20, frozen once and shared across all 12 runs.
  D_feedback = 150, D_pareto = 75, test = 300. These are locked and will not be changed to
  save time or to rescue a result.
- Difficulty operationalization: empirical base score (seed-program F1 on each D_feedback
  example), binned by value. easy: s >= 0.99 (n=69). medium: 0.01 < s < 0.99 (n=31).
  hard: s <= 0.01 (n=50). Table frozen once at init.
- Schedule (phase-based, boundaries at T/3 and 2T/3 = iters 0-13 / 14-26 / 27-39):
  `easy_to_hard` draws easy then medium then hard; `hard_to_easy` is the reverse;
  `static_medium` draws medium throughout; `random` ignores bins.
- Acceptance is decoupled: reflect on the schedule-selected minibatch, accept iff the
  candidate strictly improves the average A=20 score over the cached parent.

---

## Metrics

Primary (the experiment is about these):
- P1. Iteration-to-first-accept, per run.
- P2. Cumulative accepts over iterations, per run (the accept trajectory), and total
  accepts at iteration 40.

Secondary (reported, not load-bearing):
- S1. Final held-out test F1 on the 300-example test split, one eval at end of run.

A note on power, stated up front: with three seeds the design cannot resolve small effects
on test F1. Exp 1 had a between-arm F1 range of 0.0121 against a within-arm seed spread of
0.0395. That noise lives in GEPA's stochastic trajectory, not in the eval, so a larger test
set would not fix it. We therefore do not treat F1 as a discriminating endpoint and do not
run significance tests on it. The accept metrics had roughly an order of magnitude more
separation in Exp 1b (3.3 to 7.7 across bands), which is why they are primary.

---

## Predictions

These pit two priors against each other. The classical curriculum-learning prior says
easy-first should help. The substrate-mechanics prior says easy-first wastes the early
budget on this substrate, because the easy bin is a pile of already-solved (F1 = 1.0)
examples that give the reflection model almost no failure signal, while the hard bin gives
the clearest signal (this is the mechanism Exp 1b observed: the hard band drove the most
accepts). We predict the substrate-mechanics prior.

- Prediction 1 (primary, high confidence). Iteration-to-first-accept: `hard_to_easy`
  earliest, `easy_to_hard` latest, with `random` and `static_medium` in between.
  Rationale: `hard_to_easy` opens on the hard bin (maximal failure signal); `easy_to_hard`
  opens on the easy bin (ceiling examples, near-zero signal), so its first accept should
  not arrive until the schedule reaches the medium/hard phases (iter 14+).

- Prediction 2 (primary, high confidence). Early accept accumulation, measured as
  cumulative accepts by the end of phase 1 (iteration 13): `hard_to_easy` substantially
  higher than `easy_to_hard`. Same mechanism as P1.

- Prediction 3 (primary, lower confidence). Total accepts at iteration 40: ordered roughly
  `hard_to_easy` >= `random` >= `static_medium`, with `easy_to_hard` recovering during its
  later (medium/hard) phases and finishing near `random`. Flagged lower-confidence because
  the accept bar rises as the parent improves and because path dependence over the budget
  is hard to call in advance. The timing predictions (P1, P2) are the load-bearing ones;
  total accepts may compress across arms.

- Prediction 4 (secondary, the null). Final test F1 will not separate the arms beyond the
  seed noise floor: the between-arm range in mean test F1 will be smaller than the
  within-arm per-seed spread. This would extend Exp 1b's dynamics-vs-outcomes decoupling
  from static concentration to temporal scheduling: scheduling changes how and when the
  optimizer accepts, but not what it generalizes to.

---

## How predictions will be evaluated

Given three seeds, the read is descriptive, not inferential. No p-values.

- P1: report per-arm mean and the three per-seed values of iteration-to-first-accept.
  Counts as confirmed if the predicted ordering (`hard_to_easy` < others < `easy_to_hard`)
  holds in the means and the `hard_to_easy` vs `easy_to_hard` gap exceeds the per-seed
  spread within those arms.
- P2: plot cumulative accepts vs iteration, all arms, with per-seed spread shown. Counts as
  confirmed if `hard_to_easy`'s phase-1 cumulative accepts exceed `easy_to_hard`'s by more
  than the per-seed spread.
- P3: report total accepts per arm (mean and per-seed). Counts as consistent if the ordering
  holds in the means; treated as inconclusive rather than disconfirming if arms compress to
  within the per-seed spread.
- P4: report mean test F1 per arm and the per-seed spread. Counts as confirmed (a clean
  null) if the between-arm range is below the within-arm spread.

If any prediction's ordering reverses beyond the seed spread, that is a disconfirmation and
is reported as such, not explained away. A reversal on P1/P2 (easy-first accepting earlier)
would be a genuine surprise worth investigating, e.g. a formatting-lock-in effect.

---

## Degenerate-outcome guard

If total accepts are near zero across all arms (well below the several-per-run seen in the
pilot and Exp 1b), the run is uninformative about scheduling rather than evidence for any
prediction. That outcome is reported as degenerate, and we revisit the setup before drawing
conclusions.

---

## Commitments

- Every arm and every seed is reported, including nulls and the control.
- A clean null (P4 holds, dynamics shift per P1/P2) is an acceptable and informative outcome,
  not a failure. So is a flat result across all metrics.
- No difficulty-axis, metric, or substrate shopping after seeing results. The primary metric
  is fixed here as the accept dynamics; test F1 is secondary and stays secondary regardless
  of which looks more favorable.
- Locked params (A=20, D_pareto=75, T=40, test=300, the seeds, the frozen difficulty table
  and acceptance batch) are not changed to alter or speed up the result.
- The pivot from the original IFBench constraint-count plan to base-score difficulty on
  HotpotQA is framed honestly in the writeup as diagnosis-driven (the constraint-count axis
  failed validation), not as a planned design.
