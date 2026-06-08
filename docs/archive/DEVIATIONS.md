# DEVIATIONS.md — forced substitutions from the §15 settled decisions

This file logs every place where the live setup differs from
`gepa_band_selection_handoff.md` §15 ("Settled decisions"). Each entry names
the spec setting, the substitution, the cause, the date the substitution
was decided, and what the substitution does to the science.

These are forced substitutions, not redesigns. The experimental design
(four arms over equal-tercile bands, 70/15/15 draw, decoupled accept on
A=20, strict-improvement acceptance, N=44 stop, 3 seeds, Pareto candidate
selection unchanged) is intact.

## 1. Substrate: retrieval-based HotpotQA → HotpotQA distractor

**Spec (§15):** "Substrate: retrieval-based HotpotQA, the paper's four-module
multi-hop program. Distractor only as the Monday fallback."

**Live:** HotpotQA `distractor` config. Each example carries the 10 paragraphs
HotpotQA provides (2 gold + 8 distractor). The program (`src/program.py`)
consumes these directly; no retrieval call is issued. The `create_query_hop2`
module is dropped; the program is now three modules: `summarize1`, `summarize2`,
`final_answer`. The corresponding seed file `prompts/seeds/create_query_hop2.md`
is left in place but unused.

**Cause:** DSPy's hosted ColBERTv2 endpoint at `20.102.90.50:2017/wiki17_abstracts`
is unreachable from this machine. Two `curl` probes returned HTTP 000 (no TCP
connect) within a 20s timeout each. Two consecutive program runs failed at
hop-1 retrieval with `requests.exceptions.ConnectTimeout`. The handoff §9
flagged the index as "known to be flaky" and named distractor as the explicit
fallback.

**Decision date:** 2026-05-30.

**Impact on the science:**
- The optimization surface shrinks from four modules to three. The band-sampled
  reflection minibatch still cycles through the remaining three modules via
  GEPA's round-robin module selector.
- The "frontier band" hypothesis (band-selecting reflection minibatches by
  base-system F1 difficulty improves sample efficiency over random) is
  unchanged in shape. The retrieval failure mode is removed from the system,
  so per-instance F1 is now bounded above by reasoning-over-given-paragraphs
  rather than retrieval+reasoning. The frontier-band population could compress
  modestly if the distractor setting saturates more easily; this is one of the
  things the Chunk-5 diagnostic checks (intermediate F1 mass).
- The reference arm (vanilla coupled GEPA) is unaffected by the substrate
  swap and remains a valid no-decoupling baseline.

**Mitigation if retrieval comes back online:** the Chunk-2 `src/retrieval.py`
file is left in place. Restoring four-module retrieval mode would require
adding back `create_query_hop2` to `program.py` and changing forward to call
`dspy.Retrieve` twice. No other chunk is affected.

## 2. Task model: Qwen3-8B → Qwen2.5-7B-Instruct-Turbo

**Spec (§15):** "Task model Qwen3-8B at temp 0.6 (top-p 0.95, top-k 20)."

**Live:** `together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo` on Together AI,
serverless pay-per-token. Temperature, top-p, max_tokens carried over from the
spec. `top_k=20` is held in `config/experiment.yaml` for Chunk-4/6 to pass
through litellm `extra_body` once the runner needs it.

**Cause:** Qwen3-8B is not available on any major US serverless provider
(June 2026 catalog snapshot):
- Together AI: returns `BadRequestError: Together_aiException - Unable to
  access non-serverless model Qwen/Qwen3-8B. Please ... create and start a
  new dedicated endpoint`.
- Fireworks AI: model page lists "serverless: Not supported"; the official
  `docs.fireworks.ai/serverless/pricing` Qwen3-family serverless entry is
  `fireworks/qwen3p6-plus` only.
- DeepInfra: catalog lists `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-32B`,
  `Qwen/Qwen3-235B-A22B`. No 8B dense.

The serverless pay-per-token billing model is required at the $300 budget
cap: provisioning a dedicated Qwen3-8B endpoint adds hourly carrying cost
that the cap cannot absorb across 15 runs without a separate budget request.

**Decision date:** 2026-06-01.

**Why this substitution and not a larger Qwen3:**
We prioritize capability tier (~8B) over model generation. A 30B+ task model
would compress the F1 difficulty distribution toward the extremes (most
HotpotQA-distractor questions become easy for a 30B model), starving the
middle tercile that the "frontier band" arm is built around. Section 6's
GO/NO-GO check on the F1 distribution is exactly the gate for this concern;
holding the size tier near the spec maximizes the chance of a populated
middle tercile.

**Impact on the science:**
- Paper-comparability against the GEPA paper's Qwen3-8B HotpotQA numbers
  (baseline F1 ~42.3, GEPA ~62.3) is weakened. Absolute F1 is no longer
  directly comparable; relative-arm comparisons within this experiment are
  unaffected.
- The "frontier band" claim is robust to this substitution: the experiment
  asks whether band selection beats random selection holding the task model
  constant across arms. As long as the F1 distribution on D_feedback under
  Qwen2.5-7B has a populated middle tercile, the design works.
- The framing in §14 ("the work Dimakis is on", multi-turn BFCL as the named
  extension) is unchanged. The substitution is a budget/availability concession,
  not a methodological choice.

**Mitigation:** if Qwen3-8B becomes serverless on any provider before Chunk 6,
switch `TASK_MODEL` in `.env` and re-score the base system on D_feedback for
the new difficulty table. No code changes are required.

## 3. D_feedback: 100 → 150 (§5 fallback)

**Spec (§15):** "D_feedback 100 ... 100/20/75/300. Disjoint, fixed seeds."

**Spec (§5, Fix 1 / band-size guidance):** "If you want larger bands for more
reflection diversity, grow D_feedback to 150 and take thirds of 50. The
non-negotiable is that the three band arms are equal size."

**Live:** D_feedback grown from 100 to 150 by appending 50 additional
HotpotQA-dev examples drawn from positions past the original 495-position
shuffled block. The original 100 D_feedback ids, the 20 accept_batch ids,
the 75 D_pareto ids, and the 300 test ids are bit-identical to the n=100
layout (verified by shuffle position arithmetic; the original block uses
indices `[0:100, 100:120, 120:195, 195:495]` of the seeded shuffle, the
extra 50 use `[495:545]`). All 545 ids remain disjoint by construction.

`config/experiment.yaml` updated: `splits.d_feedback: 150`. `src/data.py`
extended with `ORIGINAL_D_FEEDBACK = 100` constant; `load_splits` now picks
up the extra ids from the post-original block when the configured size
exceeds the original. A=20, D_pareto=75, test=300, b=3, N=44 unchanged.

**Cause:** Chunk-5 diagnostic on n=100 returned 18 strictly-partial
instances overall (18% of the pool); the middle tercile held 52.9%
strictly partial -- just above the 50% GO line. With only 18 partial
instances across the entire pool, the 70/15/15 band sampler would have
revisited the same partials many times over 44 iterations, and per-
trajectory feedback diversity would be thin. The §5 fallback grows the
pool to give the frontier band more distinct partial instances.

**Decision date:** 2026-06-01.

**Impact on the science:**
- Frontier-band reflection sees ~70% of 132 = ~92 trajectory-instance
  draws over the run; with the n=100 partial pool of 18, that's ~5x
  revisits per partial. With n=150 we get 31 partials (the actual
  observed count), or ~3x revisits -- still revisited but with more
  distinct instances per revisit class.
- The accept_batch / D_pareto / test composition is bit-identical; no
  effect on the acceptance bar or the per-iteration D_pareto cost.
- Per-arm rollouts grow proportionally only in the seed-evaluation step
  (D_feedback scoring during base-system measurement). The optimization
  loop's per-iteration cost is unchanged (b=3, A=20, P=75).
- The §6 GO/NO-GO gate verdict moves from "borderline GO at 52.9%" to
  "clean GO at 62.0% strictly partial in mid."

**Mitigation:** none required -- this IS the mitigation. If 62.0% turns
out to still be too thin in Chunk 7 analysis (e.g., if seed-pair
variance dominates the arm contrast), the further §5 lever is to grow
D_feedback to 180 or 210 (preserving equal-tercile sizes 60/60/60 or
70/70/70). No code change required beyond bumping the config.

## 4. Band definition: equal-rank terciles → value-based bins

**Spec (§15):** "Bins: equal terciles by base-system F1 rank on D_feedback."

**Spec (§5, GO/NO-GO):** "confirm the F1 distribution has real intermediate
mass (a meaningful fraction of instances with F1 roughly between 0.2 and
0.8)."

**Live:** Frontier band definition changed from "middle tercile by F1 rank"
to value-based partition:

  - `hard`:  F1 == 0.0       (complete failure)
  - `mid`:   0 < F1 < 1       (frontier band; the method under test)
  - `easy`:  F1 == 1.0        (complete success)

Band keys (`'easy'` / `'mid'` / `'hard'`) are preserved so nothing downstream
renames. Band sizes are unequal by design; on the n=150 D_feedback at
Qwen2.5-7B-Turbo distractor, the observed sizes are **50 / 31 / 69**
(hard / mid / easy).

**Cause:** Chunk-5 diagnostic on n=150 showed a bimodal three-cluster F1
distribution: 50 hard-zeros, 69 easy-ones, 31 strictly-partial. Equal-
rank terciles forced the middle tercile to size 50, padding the frontier
with 19 F1==1 instances and 0 F1==0 instances; only 62% of "mid" was
actually frontier. That conflated "frontier" (band identity) with
"already-mastered" (signal-free for reflection), softening the
contrast the experiment is built around.

Value-based bins make the frontier band literally the set of strictly-
partial instances. Frontier draws are guaranteed to deliver examples the
reflection LM can meaningfully act on; easy and hard draws are guaranteed
NOT to. The 70/15/15 mix and the random arm are unaffected (the random
arm draws uniformly over all 150 ids, so its natural composition is
~33/21/46 -- the band-fraction equals the size-fraction, which is the
correct behavior; uniform-over-bands would have over-sampled the small
frontier and is now ruled out by a regression test).

**Decision date:** 2026-06-01 (pre-launch, diagnostic-driven).

**Impact on the science:**
- The four arms are still distinguishable by what they reflect on:
  random covers the full distribution naturally (~33/21/46 expected),
  static-easy concentrates on F1==1 ("nothing to fix"), static-frontier
  concentrates on 0<F1<1 ("rich actionable signal"), static-hard
  concentrates on F1==0 ("opaque/unfixable"). These are the same
  semantic claims §4 made for the equal-tercile design, now sharpened.
- "% partial in mid" is no longer a meaningful GO/NO-GO axis; with
  value bins it is 100% by construction. The §6 gate is replaced with
  a count: GO if |mid| >= 20 (so that b=3 over N=44 = 132 trajectory-
  instance touches has enough distinct frontier instances to avoid
  collapsing toward repeated draws of the same few). Observed |mid| =
  31; gate passed.
- Acceptance, D_pareto evaluation, and the Pareto candidate selector
  are unchanged; the bin definition only affects what `BandBatchSampler`
  serves to the reflection step.

**Mitigation:** none required -- this IS the mitigation. If a future
substrate / task model swap yields a single-mode F1 distribution where
value-based binning collapses one of the bands toward zero, the §5
fallback "grow D_feedback further" still applies. If |mid| ever falls
below GO_MIN_FRONTIER_COUNT under a new run, the verdict will say so.

## 5. Sampling mix: 70/15/15 → 100/0/0 for the Experiment-1b isolation variant

**Spec (§15, DEFAULT_MIX):** "Band draw mix 70% target tercile, 15/15
off-band."

**Live (Experiment 1b, three static arms only):** PURE_ON_BAND_MIX =
(1.0, 0.0, 0.0). The three static arms (static_easy, static_frontier,
static_hard) are re-run at this mix into `results/logs_hotpot_100/`.
Random and vanilla cells are NOT re-run; the Chunk-8 regression test
`test_random_arm_byte_identical_under_70_15_15_and_100_0_0` proves the
random-arm draw sequence is bit-identical under both mixes for the same
seed, so Experiment 1's `results/logs/random_{0,1,2}/` and
`vanilla_coupled_gepa_{0,1,2}/` cells are reused without re-running.

The frozen `results/difficulty_table.json` from Experiment 1 (n=150,
value bins, SHA-256 `bfeaf82ce55eb7917d8d3c786b8a4fb2d9d4f80b759fe124ad0b62f16c1fb47d`)
is reused byte-unchanged; no re-scoring. Each Chunk-9 cell's
provenance.json records that hash to prove it.

**Cause:** Experiment 1's Chunk-7 analysis
(`results/analysis_chunk7/FINDINGS.md`) diagnosed two mechanisms behind
the null. Mechanism 2 was off-band leakage: the 70/15/15 mix's 30% off-
band exposure was sufficient to push several strictly-partial and
several F1==0 instances into every static arm's reflection minibatch
over N=44, including `static_easy` which still accepted 5.3 prompt
edits per cell on average despite drawing from a band that should have
been "nothing to correct." That diluted the cross-arm contrast the
experiment was built around. 100/0/0 removes the leakage entirely.

**Decision date:** 2026-06-02. Pre-launch and isolation-driven; not
a result-aware change.

**Impact on the science:**
- The three static arms now sample purely from their target band, so
  the band-arm contrast is the unconfounded comparison the original
  design intended.
- **Static_easy is expected to flatline** under 100/0/0 (zero or near-
  zero accepts per cell) because its all-F1==1 target band cannot
  drive the `skip_perfect_score` branch into proposing anything. That
  collapse (from Experiment 1's 5.3 accepts/cell to ~0) IS the
  validation that off-band leakage was the mechanism behind
  static_easy's Experiment-1 activity. The Chunk 8 engine-level test
  `test_pure_on_band_all_perfect_minibatch_no_exception` confirms the
  no-exception path.
- Random and vanilla are not re-run; the chunk-8 byte-identical
  regression test is the sound basis for the reuse.
- The Chunk-10 analysis (deferred) reads both `results/logs/` and
  `results/logs_hotpot_100/` and recomputes paired contrasts.

**Mitigation:** none required -- this is the isolation variant. If a
future variant changes the difficulty table, the SHA-256 stored in
each cell's provenance.json surfaces the drift.

## 6. Experiment-2 substrate: IFBench_test → IF_multi_constraints_upto5

**Spec (BUILD_PLAN.md §4 D1 / §7 Chunk 11 original):** "Confirm the
dataset is the multi-constraint AllenAI IFBench (Pyatkin et al. 2025),
the same instruction-following benchmark used in the GEPA paper, NOT
Google IFEval."

**Spec (BUILD_PLAN.md §7 Chunk 11 REDO, 2026-06-02):** "We pivot to the
multi-constraint composite. Load `allenai/IF_multi_constraints_upto5`
(the IF-RLVR composite, ~95k rows, up to 5 constraints per instruction,
constraints from IFEval (25) + IFBench-Train (29))."

**Live for Experiment 2:** `allenai/IF_multi_constraints_upto5`. 95,373
rows. Per-row constraint-count distribution
`{1: 23007, 2: 23903, 3: 23322, 4: 18038, 5: 7103}`. Carve at
constraint_count floor 3 (48,463 usable, ~89× the 545 target).

**Cause:** The first Chunk-11 cut targeted `allenai/IFBench_test` and
hit the operator-review gate: only 300 rows total, and 256 of those
single-constraint, so the score distribution would be
bimodal-by-construction and the strict-partial frontier structurally
bounded by the 44 multi-constraint rows. That would reproduce the
HotpotQA thin-frontier regime that Experiment 2 was designed to escape,
and would NO-GO at Chunk 13's continuity diagnostic for the same reason
Experiment 1 reached a null on a thin frontier.

The IF-RLVR composite addresses both axes at once:
- **Size**: 95k rows >> 545, so no proportional shrink.
- **Continuity**: at floor 3, every row produces a score in
  `{0, 1/5, 1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 1}`. This is a
  populated continuous frontier by construction, not bimodal at `{0, 1}`.

**Decision date:** 2026-06-02. Pre-launch, diagnostic-driven (the
constraint-count distribution of `IFBench_test` was the diagnostic).

**Impact on the science:**
- Experiment 2 evaluates on a *derivative* of the published IFBench
  benchmark family rather than the canonical `IFBench_test` evaluation
  split. The composite was assembled for IF-RLVR training data, but is
  used here purely for its constraint richness; we carve disjoint
  pools, so no train/test leakage exists within Experiment 2.
- Verifier inputs (the 25 IFEval + 29 IFBench-Train instruction IDs =
  54 total) are the same verifiable types the BUILD_PLAN specified, so
  the per-row scores are computable by IFBench's official verifiers
  exactly as on the canonical split. Chunk 12 implements those 54
  verifiers; Chunk 11 enumerated the catalog and confirmed the live
  pool's instruction IDs are all in it (0 uncovered).
- The score axis is `score = k/N` for `N = constraint_count` and `k =
  satisfied`. With `N ∈ {3, 4, 5}`, the strict-partial range spans `k/N
  ∈ (0, 1)` with realistic resolution and a populated middle band.
- The substrate choice does not change the §15 design (5 arms × 3
  seeds, decoupled acceptance on a fixed accept-batch, b=3,
  100/0/0 sampling, etc.). Only the dataset under test changes.

**Caveat flagged for operator review (CHUNK11_REPORT §"Substrate
construction notes"):**
- `IF_multi_constraints_upto5` is a TRAINING set for the IF-RLVR
  work. The task model
  `together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo` was not IF-RLVR-trained,
  but the operator should confirm before Chunk 14 launch.
- All rows carry `dataset='ifeval'` in the composite metadata; the
  source prompts trace to IFEval. The constraints themselves span
  IFEval + IFBench-Train. This is consistent with BUILD_PLAN §7 Chunk
  11 redo.

**Mitigation:** none required — this is the substrate the experiment
runs on. The Chunk-13 continuity diagnostic and the Chunk-14 matrix
will use the Chunk-11-redo carve. Reverting to `IFBench_test` would
re-introduce the thin-frontier problem the redo solves.

## Verification

All four substitutions were verified end-to-end before this file was
committed:

- Structural smoke (`src/smoke_chunk2.py`) green: split sizes
  150/20/75/300 after the §5 fallback (was 100/20/75/300), disjoint
  (545 unique HotpotQA ids), reproducible under seed_splits=0, three
  named predictors loaded with seed instructions verbatim.
- One-rollout LM smoke green: example `5abf63f15542997ec76fd3ea` returned a
  partial-credit answer ("1922" vs gold "October 1922"). Gold supporting
  titles (Socialist Revolutionary Party, Russian Civil War) both present in
  the 10-paragraph context, distributed one per hop bucket.
- Full pytest (15/15 passing): 10 offline contract tests + 5 integration
  tests including `test_gold_paragraphs_in_context`.
- Chunk-5 diagnostic on n=150 (equal terciles): 50 / 69 / 31 (zero /
  one / strictly partial); middle tercile 0 / 19 / 31 (62.0% partial);
  GO verdict at 62.0%.
- Chunk-5 diagnostic on n=150 (value bins, post-entry-4): mid = 31
  strictly-partial only (no zeros, no ones); hard = 50 (all F1==0);
  easy = 69 (all F1==1). GO verdict at |mid|=31 (gate is >=20).
- Chunk-4 offline tests on the value bins: 37/37 passing, including
  per-arm 70/15/15 distribution checks on the unequal 50/31/69 band
  layout AND a regression test that the random arm reflects the natural
  ~33/21/46 composition (not uniform-over-bands).

API spend during Chunk-2 verification: under 1¢ (one probe call of 31 tokens,
one rollout under 10 LM calls). API spend during Chunk-5 scoring at n=150:
~$0.05 total across the 150 examples (the original 100 mostly hit the
dspy.LM disk cache from the n=100 run; only the 50 new ids paid the real
per-example cost).
