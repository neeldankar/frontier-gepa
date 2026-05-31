# CLAUDE.md — Frontier-Band Reflection Selection for GEPA

## What this project is

A controlled experiment testing whether the difficulty band of the examples GEPA reflects on matters, and specifically whether the partial-success "frontier" band is the most informative band to reflect on, at matched treatment exposure. It is a 2-day feasibility probe for a research interview, not a powered study or a product. Build for a clean, honest, interpretable result.

The full design spec is in `gepa_band_selection_handoff.md`. Read it before writing code. Locked parameters are in `PREREGISTRATION.md`. This file is your orientation, the build plan, and the guardrails.

## Roles and workflow

- Prompts (the four module seeds, the feedback string template) are designed in a separate Claude conversation and handed to you as files in `prompts/`. Use them. Do not invent your own; if a prompt is missing, leave a clearly marked placeholder and flag it.
- You (Claude Code) own all implementation: data, program wiring, the feedback computation, the GEPA modification, the runner, and analysis.
- Do not redesign the experiment. If you think the spec is wrong, say so and stop, do not silently change it.

## Working protocol (read this every session)

- Do ONE chunk per session. Touch only that chunk's files. When the done-criterion is met, run the tests, commit, and stop for review. Start the next chunk in a fresh session. This keeps context small and the build reviewable.
- Define interfaces before implementations so chunks compose. When a chunk depends on another's output, read that output (a file), do not reimplement it.
- Write the tests named in each chunk. Keep them fast and offline where possible.
- Update the chunk status checklist below and append findings to `ARCHITECTURE.md` as you learn how gepa is structured.
- Real API spend starts at Chunk 5 (base-system scoring) and is large at Chunk 6 (the runs). Do not launch the full matrix without confirming the pilot's per-rollout cost against the $300 cap.

## Guardrails: do not re-introduce these (each was rejected for a reason)

- Do NOT recouple acceptance onto the reflection minibatch. Acceptance runs on a fixed shared accept batch, identical across all arms, strict improvement over a cached parent.
- Do NOT add any refresh, windowing, or incremental update to the difficulty table. It is scored once on the base system and frozen for the whole experiment.
- Keep the pools disjoint: D_feedback, the accept batch, D_pareto, and test never overlap. No leakage.
- Use equal terciles (33/33/33 by F1 rank), not 25/50/25.
- Stop each run on a fixed iteration count N, not on a rollout budget.
- Do NOT customize the reflection meta-prompt. Use the gepa default. (It is constant across arms, so it cannot move the result, and customizing breaks fidelity to stock GEPA.)
- Plot and compare on the iteration axis, not total rollouts.

## File structure

See the tree the project was scaffolded from in Chunk 0. Source lives under `src/`, prompts under `prompts/`, results under `results/`.

## Locked parameters (mirror of PREREGISTRATION.md)

- Splits: D_feedback 100, fixed accept batch 20, D_pareto 75, held-out test 300. Disjoint, fixed seeds.
- Minibatch size b = 3. Band draw mix 70% target tercile, 15/15 off-band. Random arm draws uniformly.
- Bins: equal terciles by base-system F1 rank on D_feedback.
- Stop: N = 44 reflection iterations per run.
- Accept: strict improvement (>) of proposed candidate over cached parent on the fixed accept batch.
- Arms: random, static-easy, static-frontier, static-hard. Plus one vanilla coupled GEPA reference (random draw, acceptance on the minibatch as shipped).
- Seeds: 3. Runs: 4 arms x 3 + reference = 15.
- Task model: Qwen3-8B at temp 0.6 (set in config, provider via env, do not hardcode). Substrate: retrieval-based HotpotQA; distractor is the fallback only if retrieval infra is a quagmire.

## Build plan (chunks)

Status: `[ ]` not started, `[~]` in progress, `[x]` done.

- [x] **Chunk 0, scaffold and env.** Create the structure, requirements.txt (dspy, gepa, the provider SDK, datasets, matplotlib), config/experiment.yaml with the locked params, .env.example, .gitignore. Done: `import dspy, gepa` works and the config loads.
- [x] **Chunk 1, GEPA source recon (the gate).** Read the installed gepa and dspy.GEPA source. Locate: where the reflection minibatch is sampled from D_feedback, where a mutation is accepted or rejected, and where per-instance D_pareto scores are stored. Write ARCHITECTURE.md with the exact functions/classes and a decision: can the reflection sampler and the acceptance test each be overridden via subclass or callback, or must the loop be forked. This decides Chunk 4's approach. Done: ARCHITECTURE.md committed with a concrete hook plan. No code changes. If a clean override is not possible within a focused effort, flag it and stop, the fallback is a coupled two-condition smoke.
- [~] **Chunk 2, data and program.** Disjoint splits with fixed seeds, HotpotQA loading, retrieval setup (dspy ColBERTv2 hosted index, with a local fallback noted), and the four-module multi-hop DSPy program (summarize1, create_query_hop2, summarize2, final_answer) using the seeds in `prompts/seeds/`. Done: the program runs on one example and returns an answer plus per-hop retrieved titles; splits are disjoint and reproducible.
- [ ] **Chunk 3, feedback function.** Implement `mu_f(example, trace) -> (score, feedback_string)` exposing these fields and using the template in `prompts/feedback_template.md`: per hop the generated query and retrieved doc titles; the gold supporting-fact titles; the retrieval gap (gold titles not retrieved); whether the gold answer string appears in the retrieved or summarized context (substring check); optionally whether the hop-2 query references the bridge entity. Done: unit tests on 2-3 examples show the gap and substring fields populate and the string reads as actionable.
- [ ] **Chunk 4, GEPA modification.** Per ARCHITECTURE.md: the frozen difficulty table (score the base system once on D_feedback, equal-tercile bins), the four band samplers (70/15/15 mix), hook them into the reflection draw, decouple acceptance onto the fixed batch (strict improvement, cached parent), and stop on N iterations. Done: `run(arm, seed)` does N iterations and returns per-iteration logs; a tiny-slice smoke produces accepted candidates.
- [ ] **Chunk 5, diagnostic and go/no-go.** Score the base system on D_feedback, plot the F1 distribution, and verify the middle tercile is populated and not dominated by zero-F1 (retrieval-miss) instances. Done: a diagnostic figure plus a printed GO/NO-GO verdict. If NO-GO, stop and report; do not spend the main budget.
- [ ] **Chunk 6, runner, pilot, logging.** Orchestrate the 15 runs with per-iteration logging (validation F1, accept events, cumulative rollouts) and final held-out test F1, resumable, saved to results/logs/. Run the pilot first (one arm, one seed, 50 examples) to validate per-rollout cost and plumbing. Done: one command runs the matrix; the pilot confirms cost is within budget.
- [ ] **Chunk 7, analysis.** Performance-vs-iteration curves per arm (compared at equal iteration counts), held-out test F1 with bootstrap CIs across the 3 seeds, and iterations-to-target with the target set from the pilot. Done: figures and a results table written to results/.

## Style

Direct, no fluff, no em dashes. Prefer small, tested, composable modules over a monolith. Comment the non-obvious (the gepa hooks, the band sampler, the decoupling), not the obvious.
