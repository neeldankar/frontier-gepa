# Experiment 3 Handoff: Complexity-Scheduled Curriculum for GEPA

Read this before doing anything in this phase. It explains where the project stands, the one
task to run first, the decision gate that follows, and the full design to build only if that
gate passes. Verify every file path and symbol against the actual repo before acting; names
below are from prior context and may need confirmation.

---

## 1. Where the project stands

- **Experiment 1 (HotpotQA, band selection, 70/15/15):** complete. Null. Between-arm test-F1
  range 0.0121, dwarfed by 0.0395 within-arm seed spread.
- **Experiment 1b (HotpotQA, 100/0/0 isolation):** complete. Cleaner null; confirmed off-band
  leakage was the mechanism behind Exp 1's one marginal contrast; static-easy accepts collapsed
  to 0. Side finding: the hard band drove the *most* accepts (concentrated failures give the
  reflection LM clearer signal), but that did not convert to held-out test F1.
- **Experiment 2 (IFBench, band selection, built):** wired and pilot-validated, matrix NOT run.
  Diagnosed intractable on a laptop: IFBench requires long constraint-satisfying generations
  (~17s/call, model hits the 16K max_tokens ceiling), so the 300-example test eval alone is
  ~85 min/cell and the 15-cell matrix projects to 57-82h. Cost is trivial; wall time is the wall.
- **Shipped:** HotpotQA writeup + README pushed at commit `bcced15`.

This phase (Experiment 3) is a **pivot, not a continuation**. Own that framing: the band nulls
surfaced that static band selection is not curriculum learning, and that realization motivated
this experiment. Do not present the arc as if it were planned from the start.

---

## 2. The conceptual frame (why Experiment 3 exists)

Experiments 1 and 2 tested **static band selection**: score every example once, assign it to a
fixed band, draw from that band for the whole run. The question was "which difficulty band is
most informative to reflect on?" That is a stationary selection policy.

**Curriculum learning is progressive**: start on simple material, build competence, move to
harder material. The lever is the *schedule*, not the band. Experiment 3 tests that.

The substrate matters. HotpotQA has no built-in complexity axis (every question is structurally
two-hop; difficulty must be estimated by scoring, and that estimate goes stale as the prompt
improves). Multi-constraint instruction following does have one: **the number of constraints per
example is printed in the data, known a priori, intrinsic, and stable across the run.** One
constraint is structurally simpler than five. That is the gradient curriculum learning needs, and
it removes the frozen-difficulty-table-goes-stale problem entirely. The schedule is over constraint
count; no scoring is required to run it.

---

## 3. PHASE GATE: run this de-risk pilot FIRST

Do not build the scheduler, a new data loader, the matrix, or any RunPod setup yet. Everything
downstream is gated on this one cheap run.

**The question this answers:** can GEPA find *any* accepts on IFBench with our task model, given
enough iterations? The Exp 2 pilot got 0 accepts, but only at N=5, which is below the rollout
regime where even the GEPA paper saw IFBench move (the paper needed ~32 train rollouts to match
GRPO and ~79 to reach optimal on IFBench; 5 iterations at b=3 is ~15 train rollouts). So 0-at-N=5
is uninformative. We need a longer single cell.

**The run:**
- Use the **existing Experiment 2 IFBench wiring**: the Substrate injection path, the frozen
  Chunk-13 difficulty table, the existing >=3-constraint data. No new code.
- Run **one cell: the random arm** (uniform full-pool draw), seed 0, **N=40 iterations**.
  Random, not the frontier-only config that flatlined: random draws across all difficulties
  including the hard examples that 1b showed produce the clearest accept signal. If random
  cannot move, nothing will. The >=3-constraint data is a conservative test (harder than the
  full 1-5 range Exp 3 will use), so a positive result here transfers.
- `num_threads=16`.
- To save ~85 min, **skip the held-out test eval and the drift passes** if they disable cleanly.
  We need the optimization-loop trace only, not test F1.

**Report, then STOP:**
- the accept trajectory: at which iteration the first accept appears, cumulative accepts by iter 40;
- the parent score at each accept point, so we can see whether the prompt is actually climbing;
- loop wall time.

Expected wall: ~1.5-1.7h on the laptop at 16 threads.

---

## 4. Decision rule (operator call after the pilot)

- **GO** if accepts accumulate, roughly 3+ by iter 40 with the parent score climbing. GEPA moves
  on IFBench; the full Experiment 3 matrix is worth building, and RunPod is worth setting up.
- **NO-GO** if it flatlines near 0 accepts through iter 40. IFBench does not optimize on this
  model; ship HotpotQA as the result and present Experiment 3 as a *design* (the pitch plus this
  handoff), not a run. A clean, well-motivated design is a strong interview artifact on its own.

Stop and surface the trajectory for the operator. Do not auto-proceed to building Exp 3.

---

## 5. Experiment 3 full design (build ONLY on GO)

### 5.1 Difficulty axis and proxy validation
Difficulty = number of constraints per example (1-5), taken from the data. No scoring needed for
the schedule. **Cheap validation to include:** in a one-pass base scoring across tiers, confirm
mean base score falls as constraint count rises. If it does, the proxy is empirically validated on
this substrate; report it. (We concede true difficulty is unobservable; constraint count is a
principled a-priori heuristic, and this check is defensive armor, not a correctness gate.)

### 5.2 Data (CHANGE from Experiment 2)
Exp 2 filtered to >=3 constraints. **Exp 3 needs the full 1-5 constraint range** so there are tiers
to schedule over. Build disjoint train/feedback/pareto/test splits drawing across all five tiers,
with enough feedback-pool examples per tier to sustain the schedule. Source dataset unchanged:
`allenai/IF_multi_constraints_upto5`. Keep the verifier wiring from Exp 2.

### 5.3 Tier-degeneracy guard (apply before locking the schedule)
Check the per-tier base score distribution. If the 1-constraint tier is near-ceiling for the model
(the static-easy degeneracy that killed that arm in Exp 1), the early schedule phase will spin
without signal. Mitigation: either start the schedule at 2 constraints, or define tiers by
base-score bins instead of raw count so the easy tier retains headroom. Decide this from the data,
not in advance.

### 5.4 Arms (4)
- **random** (GEPA as shipped): uniform full-pool draw throughout. Control.
- **easy-to-hard** (the hypothesis): low-constraint early, high-constraint late. Traditional curriculum.
- **hard-to-easy**: reverse schedule. Anti-curriculum check.
- **static-medium**: always draw 3-constraint examples. The static-band analog from Exp 1/2,
  here to show whether the *schedule* beats picking one fixed level.

### 5.5 Schedule mechanism
Each iteration, draw the **reflection minibatch** from the tier determined by run position. Example
linear schedule at N=80: iters 1-25 -> tier 1, 25-50 -> tier 2, 50-80 -> tiers 3-5 (adjust the
easy boundary per the 5.3 guard). The schedule changes **only what GEPA reflects on**. It does NOT
change the acceptance batch or the test set, both of which stay fixed and identical across all arms
(this is the decoupling carried over from Exp 1: reflect on the scheduled minibatch, accept on
strict improvement over the cached parent on the fixed A=20 batch).

### 5.6 Analysis requirement: stratified test eval (addresses the endpoint confound)
Easy-to-hard ends by adapting to high-constraint examples; hard-to-easy ends on low-constraint
examples; the test set is multi-constraint. So an aggregate win for easy-to-hard could be the
schedule OR just finishing on examples resembling the test set's hard end (the recency effect).
**Mitigation:** score the held-out set bucketed by constraint count and report per-tier test
performance alongside the aggregate for every arm. That lets us see *where* each arm gains and
separate schedule shape from endpoint match. This is an analysis addition, not new pipeline.

### 5.7 Locked params (carry over from Exp 1/2 unless noted)
- task model `together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo`, temp 0.6
- reflection model gpt-4.1-mini
- minibatch b=3; A=20 (accept batch); D_feedback, D_pareto, test sizes per the Exp 2 setup,
  re-derived for the 1-5 split
- decoupled strict-improvement acceptance on the fixed A=20 batch
- pinned dspy==3.2.1, gepa==0.1.1
- 3 seeds per arm for the full matrix (4 arms x 3 seeds = 12 cells), disjoint fixed seeds
- compare on the iteration axis

### 5.8 Prediction (pre-register this before running the matrix)
Easy-to-hard outperforms random and hard-to-easy on sample efficiency (iterations-to-target,
early-curve separation). Hard-to-easy worst. If easy-to-hard also beats static-medium, that is
evidence the schedule itself matters, not just the average difficulty reflected on. A null
(easy-to-hard does not beat random) is a clean negative result and reported as such.

---

## 6. Infrastructure: RunPod cell distribution (build ONLY on GO, after the design is wired)

The matrix is wall-bound, not cost-bound. The cells are independent, so distribute them across
cheap CPU instances that still call Together AI (no GPU needed; inference stays on Together's side).

- Add a `--cells <range>` flag to the orchestrator so each machine runs a disjoint slice.
- 3 CPU instances, ~4 cells each (12 cells total), projects to ~6h wall vs ~57-82h serial.
- Each instance: clone repo, drop `.env` with API keys, install deps, run its slice.
- Aggregate: rsync each instance's `results/logs_*` back into one tree, then run the analysis
  on the combined set.
- **Watch:** 3 instances x 16 threads = 48 concurrent Together requests. The Exp 2 pilot saw 0
  throttling at 16; 48 may hit rate limits. If backoff events climb, drop to 8-12 threads per
  instance or use more instances at lower concurrency.

---

## 7. Guardrails

- **Pilot first.** No scheduler, no new loader, no matrix, no RunPod until the pilot is GO and the
  operator says proceed.
- **Stop at each gate** and surface results for operator review. Do not auto-advance phases.
- **Do not disturb the shipped HotpotQA paths.** Exp 1/1b reproducibility must stay byte-identical;
  the default substrate must keep resolving to the existing HotpotQA imports.
- **Do not flip CLAUDE.md** per the standing protocol.
- **Disclose deviations** in DEVIATIONS.md as they arise (the 1-5 data range, any schedule-boundary
  adjustment from the degeneracy guard, etc.).
- When in doubt about a design choice, stop and ask the operator rather than guessing.
