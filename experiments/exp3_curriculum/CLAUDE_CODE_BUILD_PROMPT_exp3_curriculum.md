# Claude Code Build Prompt: GEPA Curriculum on HotpotQA (Exp 3)

Paste this into Claude Code in the `frontier-gepa` repo. Work through the chunks in
order. Each chunk has a verification gate. Do not start a chunk until the previous
chunk's gate passes. Stop and report at each gate rather than chaining straight through.

---

## Context

This repo implements a decoupled-acceptance variant of the GEPA prompt optimizer
(arXiv 2507.19457). Prior work (Exp 1/1b static difficulty bands on HotpotQA) produced a
clean null on held-out test F1 but showed that difficulty-band concentration changes
optimization dynamics (accept rate) without changing generalization.

Exp 3 tests whether a progressive easy-to-hard schedule (curriculum) over the reflection
minibatch changes optimization dynamics, measured on the iteration axis. The primary
endpoint is NOT test F1. It is accept dynamics. See "Metrics" below. The full design
rationale lives in `docs/archive/EXPERIMENT3_CURRICULUM_HANDOFF.md` after Chunk 0; read
it if you need the why, but this prompt is the source of truth for the what.

---

## Hard constraints (do not violate; these are known failure modes)

1. **Do not auto-upgrade dependencies.** `dspy==3.2.1`, `gepa==0.1.1`, bridged by the
   existing `_PatchedDspyAdapter`. If something looks like it needs a newer version, stop
   and ask. Do not touch the pinned versions in `requirements.txt`.
2. **Value-based difficulty bins, not equal-rank terciles.** The F1 distribution is
   bimodal (mostly 0.0 and 1.0 with a thin band of partials). Equal-rank terciles pad the
   middle band with perfect examples. Bin by value (thresholds below), not by rank.
3. **Decouple reflection from acceptance.** Reflect on the schedule-selected minibatch
   (b=3 drawn from the scheduled bin). Accept on strict improvement over the cached parent
   on a FIXED shared A=20 batch. Reflection input and accept test are different sets.
4. **Freeze the A=20 acceptance batch once at init and share it across all 12 runs.** The
   accept bar must be identical across arms and seeds. Do not redraw it per arm or per
   seed.
5. **Freeze the difficulty table once at init.** Score the pool once with the seed
   program, store, never recompute mid-run. Per-instance scores come from an explicit
   scoring pass, not from validation logs (there is no free validation-log signal).
6. **The schedule is a static function of the iteration index only.** Never adaptive,
   never tracking a "current learner." GEPA is a Pareto population with stochastic parent
   selection; there is no single learner to track.
7. **Matched comparison.** Same three seeds {0, 1, 2} applied to all four arms. Same
   frozen A=20 batch and same frozen difficulty table across everything.

---

## Locked params

Task model `together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo` at temp 0.6. Reflection model
`gpt-4.1-mini`. Minibatch b=3. A=20. D_feedback=150, D_pareto=75, test=300. Iteration
budget T=40. Substrate: HotpotQA distractor setting (ColBERTv2 retrieval is unreliable;
distractor is the active fallback). Three seeds per arm.

---

## Arms (4)

- `random` (control): native GEPA uniform sampling over all D_feedback examples. Do NOT
  bin this arm. This is the honest baseline; binning it would change what "random" means.
- `easy_to_hard` (hypothesis): schedule sweeps the reflection draw from the easy bin to
  the hard bin over the iteration budget.
- `hard_to_easy` (anti-curriculum check): the reverse sweep.
- `static_medium` (static-band analog): always draw from the medium bin.

---

## Metrics

Primary (this is what the experiment is about):
- **Cumulative accepts over iterations.** Each iteration, log a binary: did the candidate
  strictly improve the A=20 average over the cached parent? Sum over T iterations.
- **Iteration-to-first-accept.** Index of the first accepted iteration.

Secondary (report, do not depend on):
- **Held-out test F1** on the 300-example test split, evaluated once at the end of each
  run.

---

## Bin definitions (value-based, from base score)

Let `s` be the seed-program base score (F1) for an example.
- easy bin: `s >= 0.99`
- hard bin: `s <= 0.01`
- medium bin: `0.01 < s < 0.99`

Draw b=3 from the selected bin uniformly with replacement (the medium bin is small, so
replacement is required). If any selected bin has fewer than b distinct members, still
sample with replacement and log a warning; do not silently fall back to another bin.

## Schedule (default: phase-based, T=40)

Define phase boundaries at T/3 and 2T/3 (iterations 0-13, 14-26, 27-39).
- `easy_to_hard`: phase 1 -> easy bin, phase 2 -> medium bin, phase 3 -> hard bin.
- `hard_to_easy`: phase 1 -> hard bin, phase 2 -> medium bin, phase 3 -> easy bin.
- `static_medium`: all phases -> medium bin.
- `random`: ignore bins, sample uniformly over all examples every iteration.

Implement this as a pure function `bin_for_iteration(arm, iter, T) -> bin_name | None`
(None means "uniform over all"). Keep it parameterized so the phase version can be swapped
for a soft-mixing version later without touching the runner.

---

# CHUNK 0 — Repo cleanup (~30 min)

Goal: make the repo legible without losing anything. Move, do not delete.

Do:
- Create `docs/` and `docs/archive/`.
- `git mv` into `docs/archive/`: `BUILD_PLAN.md`, `ARCHITECTURE.md`, `DEVIATIONS.md`,
  all `CHUNK*_REPORT.md`, `PILOT_DERISK_REPORT.md`, `DIFFICULTY_VALIDATION.md`,
  `DIFFICULTY_VALIDATION_DIAGNOSTIC.md`, `EXPERIMENT3_CURRICULUM_HANDOFF.md`,
  `gepa_band_selection_handoff.md`.
- Keep at root: `README.md`, `CLAUDE.md`, `FINDINGS.md`, `LICENSE`, `requirements.txt`,
  `pytest.ini`, `.gitignore`, `.env.example`.
- Reorganize `results/` by substrate. Proposed target layout (adapt names to what's
  actually there, and log every move in the commit message):
  ```
  results/
    difficulty_table.json          # keep; frozen input artifact
    hotpotqa/
      exp1_static_bands/           # logs_hotpot_100, analysis_chunk*, smoke_chunk4, etc.
    ifbench/
      pilot/                       # derisk_pilot, logs_ifbench
      difficulty_validation/       # difficulty_validation, diagnostic_chunk5
    exp3_curriculum/               # EMPTY for now; this experiment writes here
  ```
- Grep the codebase and configs for any hardcoded paths into the moved directories and fix
  them. Run the existing test suite to confirm nothing path-broke.
- Update `README.md` with the new layout (one short section).

Gate: `pytest` passes (or fails only on pre-existing unrelated failures, which you should
list). `git status` shows only renames/moves plus the README edit. Report the move map.

---

# CHUNK 1 — Difficulty table + bins (~35 min)

Goal: a frozen, value-binned difficulty table over D_feedback.

Do:
- Check whether `results/difficulty_table.json` was built with the current seed program on
  the current 150-example D_feedback split. If yes, reuse it. If you cannot confirm
  provenance, regenerate it: score every D_feedback example once with the seed program,
  store `{example_id: base_score}`. One scoring pass, then freeze.
- Add a `bins.py` (or extend the existing difficulty module) that assigns each example to
  easy/medium/hard using the value thresholds above, and exposes
  `get_bin_members(bin_name) -> list[example_id]`.
- Print and persist a bin summary: count per bin, and the score histogram. Confirm the
  medium bin has at least b=3 members; if not, stop and report (the experiment is not
  runnable as designed and we need to revisit thresholds).

Gate: bin counts printed, medium bin >= 3, difficulty table provenance stated (reused vs
regenerated). Commit the frozen table and bin summary.

---

# CHUNK 2 — Schedule module (~30 min)

Goal: the four arms as pure, tested functions.

Do:
- Implement `bin_for_iteration(arm, iter, T)` exactly as specified under "Schedule."
- Implement `sample_minibatch(arm, iter, T, rng, bins, all_ids, b=3)` that returns b
  example ids: for `random`, uniform over `all_ids`; otherwise uniform with replacement
  over the scheduled bin. Use the passed `rng` (seeded) for all randomness; no global
  random state.
- Unit tests: for each non-random arm, assert the bin sequence over iters 0..39 matches
  the phase pattern; assert `random` ignores bins; assert seeded determinism (same seed ->
  same draws); assert the medium-bin-with-replacement path works.

Gate: `pytest` on the new schedule tests passes. No network or model calls in these tests.

---

# CHUNK 3 — Curriculum runner + per-iteration accept logging (~45 min, densest chunk)

Goal: one function that runs a single (arm, seed) and emits the metrics.

Do:
- Wire `sample_minibatch` into the GEPA reflection loop so the reflection minibatch comes
  from the schedule, NOT from native uniform sampling (except for the `random` arm).
- Keep the decoupled acceptance: after reflection produces a candidate, evaluate it on the
  FROZEN shared A=20 batch, compare to the cached parent's A=20 score, accept iff strictly
  greater.
- Per iteration, append a record to a per-run JSON log:
  ```json
  {
    "iter": 0,
    "bin_sampled": "easy",
    "minibatch_ids": ["..."],
    "parent_a20": 0.46,
    "candidate_a20": 0.46,
    "accepted": false
  }
  ```
- At end of run, write a run summary:
  ```json
  {
    "arm": "easy_to_hard",
    "seed": 0,
    "T": 40,
    "first_accept_iter": 8,
    "cumulative_accepts": 14,
    "final_test_f1": 0.41,
    "iterations": [ ...the per-iter records... ]
  }
  ```
- `final_test_f1` is a single eval on the 300-example test split at the very end. Cache the
  parent's A=20 score so you are not recomputing it every iteration.

Gate: code compiles, types/interfaces line up, dry-run with a mocked model (no real calls)
produces a well-formed JSON log with the right schema. Do not run a real model yet.

---

# CHUNK 4 — Smoke test (~30 min build, short run)

Goal: cheap end-to-end validation before committing to the full matrix. This is the GO/
NO-GO gate for the expensive runs.

Do:
- Run ONE arm (`hard_to_easy`, since it should accept earliest), seed 0, with a short
  budget (T=8). Real models. Write to `results/exp3_curriculum/smoke/`.
- Verify: the per-iter log fills correctly, bins sampled match the schedule, at least the
  plumbing for an accept event fires (an accept is likely but not guaranteed at T=8; what
  must work is the logging and the A=20 comparison).
- Record wall time per iteration. Project the full-matrix wall: per-iter-wall * 40 * 12,
  plus 12 test-F1 evals. Report that projection.

Gate: clean smoke log, per-iteration wall recorded, full-matrix wall projected. STOP here
and report the projection. Do not start Chunk 5 until I confirm GO and the
pre-registration is locked.

---

# CHUNK 5 — Full matrix orchestration (~25 min build, long run)

Goal: run all 4 arms x 3 seeds = 12 runs, resumably.

Do:
- Orchestration script that iterates arms {random, easy_to_hard, hard_to_easy,
  static_medium} x seeds {0, 1, 2}, T=40, writing each run to
  `results/exp3_curriculum/<arm>/seed<seed>.json`.
- Make it resumable: skip a run whose summary JSON already exists and is complete. Log
  start/end timestamps per run.
- Print a monitoring note (the existing pattern: `caffeinate -dims` in a separate
  terminal, `tail -f` on the active run log).

Gate: orchestration dry-run (mocked) confirms all 12 cells are enumerated and the
skip-if-complete logic works. Then launch the real matrix. Report when all 12 summaries
exist.

---

# CHUNK 6 — Analysis + plots (~40 min)

Goal: the figures and tables that answer the question.

Do:
- Aggregate the 12 run summaries. For each arm, average over its 3 seeds and also keep the
  per-seed spread.
- Primary figure: cumulative accepts vs iteration, one line per arm, with per-seed spread
  shown (shaded band or thin per-seed lines). This is the headline.
- Primary table: iteration-to-first-accept and total accepts, per arm (mean and per-seed
  values).
- Secondary table: final test F1 per arm (mean and per-seed spread), clearly labeled as
  secondary.
- Write the numbers into a short `results/exp3_curriculum/RESULTS.md` with no
  interpretation beyond stating what the curves and tables show. Interpretation and the
  pre-registered-prediction comparison happen in the writeup, separately.

Gate: figure and both tables generated from the run summaries, `RESULTS.md` written.
Report the headline numbers.

---

## Notes for the agent

- After each chunk, commit with a clear message and stop at the gate.
- If any hard constraint appears to conflict with the existing code, stop and ask rather
  than working around it.
- The pre-registration (the predicted accept-curve shapes and the predicted F1 null) is a
  separate document that must exist before Chunk 5 runs. Do not generate it yourself; it
  is being written on the advisory side.
