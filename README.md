# frontier-gepa

A 2-day GEPA feasibility probe testing whether the **difficulty band of the
examples GEPA reflects on** changes its sample efficiency, at matched
iteration budget. Substrate is HotpotQA distractor (the §9 fallback after
the hosted ColBERTv2 index proved unreachable); program is the 3-module
distractor variant (`summarize1`, `summarize2`, `final_answer`); task LM is
Qwen2.5-7B-Instruct-Turbo on Together (the §15 forced substitution; Qwen3-8B
is not serverless on any major provider), reflection LM is `gpt-4.1-mini`;
5 arms × 3 seeds × N=44 iterations × A=20 strict-improvement decoupled
acceptance × D_pareto=75 validation × 300-example held-out test. Total
matrix spend: ~$8.62.

## Headline

**Null at this scale.** Between-arm endpoint test-F1 mean spans only
**0.0121** F1 points (random 0.632 / static_easy 0.626 / static_frontier
0.629 / static_hard 0.635 / vanilla_coupled_gepa 0.639), dwarfed by a
within-arm seed spread reaching **0.0395**. Three of four paired
contrasts against `static_frontier` have 95% CIs that include zero; the
one that excludes zero (frontier − vanilla_coupled_gepa = −0.0097) puts
frontier on the worse side and is still smaller than the seed spread.
The **frontier-band hypothesis is not supported by these data at this
scale.**

## The design lesson

The 70/15/15 mix sends **30% of every minibatch off-band**. At our
observed band sizes (50 hard / 31 mid / 69 easy out of 150), that off-
band exposure is enough to push several strictly-partial instances into
every static arm's reflection set over N=44 iterations — including
`static_easy`, which still accepts 5.3 prompt edits per cell on average
(one seed accepts 9 times) despite supposedly drawing from the
"nothing to correct" band. The intended cross-arm contrast was diluted
by design. **A sharper follow-up runs a 100/0/0 pure-on-band draw** and
asks whether the resulting purity gain beats the coverage loss — that
is the genuine test of band selection as a lever.

## Pointers

- [`results/analysis_chunk7/FINDINGS.md`](results/analysis_chunk7/FINDINGS.md) — narrative + interview-ready 3–4 sentence summary
- [`results/analysis_chunk7/iteration_curves.png`](results/analysis_chunk7/iteration_curves.png) — D1 best-so-far val F1 per arm
- [`results/analysis_chunk7/endpoint_test_f1.png`](results/analysis_chunk7/endpoint_test_f1.png) — D2 forest plot (test F1 + paired-hierarchical bootstrap CI)
- [`results/analysis_chunk7/iterations_to_target.png`](results/analysis_chunk7/iterations_to_target.png) — D4 iterations-to-target at T=0.63
- [`results/analysis_chunk7/summary.json`](results/analysis_chunk7/summary.json) — every number, bootstrap seed = 20260602
- [`DEVIATIONS.md`](DEVIATIONS.md) — four forced substitutions from the §15 settled-decisions list (substrate, task model, D_feedback size, band definition), each with cause and impact
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Chunk-1 GEPA source recon and the hook plan (no engine fork)
- [`CLAUDE.md`](CLAUDE.md) — build plan, chunk-by-chunk status, locked parameters
- [`gepa_band_selection_handoff.md`](gepa_band_selection_handoff.md) — original design spec

## Reproducing

```
uv venv --python 3.10 .venv
uv pip install -r requirements.txt
cp .env.example .env  # fill in TASK_MODEL + TOGETHER_API_KEY + OPENAI_API_KEY
.venv/bin/python -m pytest tests/ -m "not integration"     # 68 offline tests
.venv/bin/python -m src.score_d_feedback                    # ~$0.05, builds difficulty_table.json
.venv/bin/python -m src.diagnostic_chunk5                   # GO/NO-GO verdict + figure
.venv/bin/python -m src.orchestrator all                    # ~$10, 2-3 hours, runs the full matrix
.venv/bin/python -m src.analysis_chunk7                     # offline; rebuilds FINDINGS + figures
```

Pinned versions in `requirements.txt`: `dspy==3.2.1`, `gepa==0.1.1`.
