# frontier-gepa

Controlled experiments on **which examples GEPA should reflect on** — does the
difficulty (or failure composition) of the examples fed to GEPA's reflective mutation
change its optimization, at matched budget? Four studies, two substrates (HotpotQA
distractor, IFBench `allenai/IF_multi_constraints_upto5`). Task LM
`together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo` @ temp 0.6; reflection LM `gpt-4.1-mini`.
Stack pinned at `dspy==3.2.1`, `gepa==0.1.1` (bridged by a small `_PatchedDspyAdapter`).

Across all four studies the result converges: example selection reliably changes
optimization dynamics (when and how often GEPA accepts) but not outcomes (total accepts
over budget, held-out test F1). This holds across two substrates and three distinct levers,
static difficulty bands, temporal curriculum schedule, and failure composition. The
mechanism is substrate-independent: GEPA maintains a Pareto population with a fixed
acceptance bar, so there is no single learner for a selection strategy to shape. A clean
null, arrived at by pre-registered predictions and reported in full, is the finding.

## The four studies

1. **Exp 1 / 1b — static difficulty bands (HotpotQA).** 5 arms × 3 seeds, N=44,
   decoupled acceptance on a fixed 20-example A-batch, plus a 100/0/0 isolation re-run.
   → `experiments/exp1_bands/`
2. **Exp 2 — IFBench substrate (designed, diagnosed, pilot-validated; matrix not run).**
   One-module program, vendored IFEval verifiers, frozen difficulty table (continuity
   gate GO), parallel-eval pilot; full matrix shelved on a wall-time diagnosis.
   → `experiments/exp2_ifbench/`
3. **Exp 3 — curriculum schedule (HotpotQA).** 4 arms × 3 seeds, T=40; a temporal
   easy→hard / hard→easy / static-medium / random schedule over the reflection minibatch.
   Primary endpoint = accept dynamics. → `experiments/exp3_curriculum/`
4. **Phase 0 — active example selection (IFBench).** Does failure composition
   (`addressability`) predict whether reflecting on an example generalizes, beyond base
   score? → `experiments/phase0_selection/`

## Headline findings

1. **Exp 1 (70/15/15).** Clean null on held-out test F1 — between-arm mean spans 0.0121
   vs a within-arm seed spread of 0.0395.
2. **Exp 1b (100/0/0 isolation).** The off-band-leakage mechanism is real:
   `static_easy` accepts/cell collapse 5.3 → 0.0 when 30% off-band exposure is removed —
   but removing it does not manufacture a frontier effect. The cleaner null is still null.
3. **Exp 2 (IFBench).** Difficulty table cleared the continuity gate (rank terciles,
   frontier 50/150). Matrix shelved: output-heavy temp-0.6 generations hit the 16K-token
   ceiling, projecting ~57–82 h wall for the 15-cell matrix (spend was fine; wall was not).
4. **Exp 3 (curriculum).** Schedule changes accept *dynamics* but not generalization.
   Iteration-to-first-accept (mean): `hard_to_easy` 4.0, `static_medium` 3.3,
   `random` 9.3, `easy_to_hard` 17.0 — `easy_to_hard` sits at **0 cumulative accepts
   through phase 1 (iters 0–13) across all seeds** while the others rise earlier. Test F1
   is a null (the prediction comparison lives in the writeup, not the results file).
5. **Phase 0 (selection).** Failure composition does **not** predict reflection
   generalization beyond base score: binary strict-improvement is near-degenerate (42/45
   improved; LR p=0.75), and the continuous after-margin OLS gives addressability
   t=0.44, p=0.66 (R² 0.004→0.008). The base-score×addressability correlation is ~0
   (Pearson 0.036), so the scalar carried independent variance but no predictive signal.

Cross-probe narrative: [`docs/FINDINGS.md`](docs/FINDINGS.md).

## Repository layout

```
README.md  LICENSE  requirements.txt  pytest.ini  .gitignore  .env.example
config/  prompts/  src/  tests/

docs/
  ARCHITECTURE.md        GEPA source recon + the hook plan (no engine fork)
  CLAUDE.md              build plan, locked parameters, guardrails
  FINDINGS.md            cross-probe writeup
  DEVIATIONS.md          forced substitutions from the settled-decisions list
  BUILD_PLAN.md          original chunked build plan
  handoffs/              original design specs (band-selection, exp3 curriculum)
  prereg/                pre-registrations (exp3 curriculum)
  reports/               per-chunk session reports + validation diagnostics

experiments/
  exp1_bands/            HotpotQA static bands: frozen difficulty table + d_feedback,
                         per-cell run summaries (logs/, logs_hotpot_100/),
                         analysis_chunk7 (70/15/15) + analysis_chunk10 (100/0/0) figures
  exp2_ifbench/          IFBench: frozen table (+sha256), diagnostic_chunk13,
                         difficulty_validation, pilot/ run summaries
  exp3_curriculum/       curriculum spec, schedule, per-(arm,seed) summaries,
                         RESULTS.md + cumulative_accepts.png
  phase0_selection/      inspection + active-selection validation scripts, cached
                         results, reports (type_landscape, active_selection, margin)
```

Note: per-cell **raw run artifacts** (`generated_best_outputs_valset/`, `gepa_state.bin`,
`shared_rng.pkl`, `candidate_tree.html`) were pruned for a portfolio-sized repo and remain
in git history; the per-cell `cell_summary.json` / `run_log*` / `test_eval.json`, the
analysis figures, and all frozen inputs are kept. The active Exp-3 modules (`bins`,
`curriculum_runner`, `orchestrate_exp3`, `analysis_exp3`) and the IFBench substrate
read/write under `experiments/`; the older Exp-1/2 scripts (`orchestrator`, `score_*`,
`analysis_chunk*`, `diagnostic_chunk*`) predate this layout and still use a legacy
`results/` path when re-run. The committed artifacts under `experiments/` are the archived
record of the runs that already happened.

## Per-study pointers

- **Exp 1:** [`experiments/exp1_bands/analysis_chunk7/FINDINGS.md`](experiments/exp1_bands/analysis_chunk7/FINDINGS.md),
  `iteration_curves.png`, `endpoint_test_f1.png`, `summary.json`.
- **Exp 1b:** [`experiments/exp1_bands/analysis_chunk10/FINDINGS_hotpot_100.md`](experiments/exp1_bands/analysis_chunk10/FINDINGS_hotpot_100.md),
  `accept_collapse.png`, `summary.json`.
- **Exp 2:** [`docs/reports/CHUNK13_REPORT.md`](docs/reports/CHUNK13_REPORT.md),
  [`docs/reports/CHUNK14_PILOT_REPORT.md`](docs/reports/CHUNK14_PILOT_REPORT.md),
  [`experiments/exp2_ifbench/diagnostic_chunk13/histogram.png`](experiments/exp2_ifbench/diagnostic_chunk13/histogram.png).
- **Exp 3:** [`experiments/exp3_curriculum/RESULTS.md`](experiments/exp3_curriculum/RESULTS.md),
  [`cumulative_accepts.png`](experiments/exp3_curriculum/cumulative_accepts.png),
  [`docs/prereg/PREREGISTRATION_exp3_curriculum.md`](docs/prereg/PREREGISTRATION_exp3_curriculum.md),
  [spec](experiments/exp3_curriculum/CLAUDE_CODE_BUILD_PROMPT_exp3_curriculum.md).
- **Phase 0:** [`experiments/phase0_selection/PHASE0_REPORT.md`](experiments/phase0_selection/PHASE0_REPORT.md),
  [`type_landscape.md`](experiments/phase0_selection/type_landscape.md),
  [`active_selection_report.md`](experiments/phase0_selection/active_selection_report.md),
  [`active_selection_margin_report.md`](experiments/phase0_selection/active_selection_margin_report.md).

## Design and protocol
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — GEPA source recon + the hook plan (no engine fork).
- [`docs/CLAUDE.md`](docs/CLAUDE.md) — build plan, locked parameters, guardrails.
- [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md), [`docs/handoffs/`](docs/handoffs/),
  [`docs/reports/`](docs/reports/).

## Reproducing

```
uv venv --python 3.10 .venv
uv pip install -r requirements.txt
cp .env.example .env  # TASK_MODEL + TASK_MODEL_API_KEY (Together) + OPENAI_API_KEY
.venv/bin/python -m pytest tests/ -m "not integration"      # offline test suite

# Exp 3 curriculum (HotpotQA): score base system, build value bins, run the 4x3 matrix, analyse
.venv/bin/python -m src.score_d_feedback                     # frozen difficulty table
.venv/bin/python -m src.bins                                 # value-based bins + gate
.venv/bin/python -m src.orchestrate_exp3 --workers 4         # 4 arms x 3 seeds, T=40
.venv/bin/python -m src.analysis_exp3                        # figures + RESULTS.md

# Phase 0 selection (IFBench): inspection + active-selection validation (reuses cached scores)
.venv/bin/python experiments/phase0_selection/inspect_ifbench_selection.py
.venv/bin/python experiments/phase0_selection/active_selection_validation.py   # pre-flight (no LLM)
.venv/bin/python experiments/phase0_selection/active_selection_validation.py --run
```

Earlier studies (Exp 1/1b/2) used `src.orchestrator` / `src.score_ifbench_d_feedback` /
`src.diagnostic_chunk*`; see [`docs/reports/`](docs/reports/) for their session reports.

Caveat: the committed frozen tables and summaries make these command blocks illustrative
of the runs that already happened (a re-run reproduces, it is not required to read the
results). The older `score_*` and `diagnostic_chunk*` scripts still write to the legacy
`results/` path rather than `experiments/`.

Pinned: `dspy==3.2.1`, `gepa==0.1.1`.
