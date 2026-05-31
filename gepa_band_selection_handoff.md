# Handoff: Frontier-Band Reflection Selection for GEPA (build spec, self-contained)

## 0. How to use this document (for the receiving Claude)

You are a build and analysis partner for a research probe I am running before a round-2 interview at Bespoke Labs. The design below is settled through several hard review rounds. Your job is to help me **build and run it**, not to re-open it. Do not re-litigate anything in the "Settled decisions" list unless something here concretely breaks. If you spot a real new hole, say so directly.

The bar this is judged against is **not** a publishable paper or a summer project. It is "a sharp, defensible research direction that demonstrates capability and taste in a 45-minute conversation, backed by a small honest result." Execution is upside, not required.

The GEPA paper is **not attached**. Everything you need from it is inlined in Section 2. If you think you need a paper fact that is not here, ask me rather than guessing.

Working style I want from you: direct, no hedging, no praise padding, no em dashes, specific over general. Surface limitations rather than reassure. Use the actual current date when reasoning about anything time-sensitive.

Symbols used throughout:
- `D_feedback` = reflection pool (instances the reflection minibatch is drawn from).
- `D_pareto` = validation set used for candidate selection and final selection. Size `P`.
- `A` = the fixed shared acceptance batch (decoupled accept test). Size 20.
- `B` = total rollout budget per run (now an outcome, not a cap; see Fix 2).
- `N` = number of optimization iterations per run (the stopping rule; see Fix 2).
- `b` = reflection minibatch size = 3.
- `r` = acceptance rate (observed, ~0.3 from the paper's search trees).
- `mu` = the scalar eval metric (here, answer F1 in [0,1]).
- `mu_f` = GEPA's feedback function (returns a score plus natural-language feedback).
- A "rollout" = one full multi-hop execution of the system on one instance plus its evaluation (~4 LLM module calls plus 2 retrievals for this program).

---

## 1. Interview context (why the design looks the way it does)

- I am a rising senior at UC Berkeley (applied math, stats/ML). Planning a Stats MS then a CS PhD. Research is my weakest dimension, which is why this matters.
- Round 2 is with **Vishnu Suresh**: MTS research engineer at Bespoke, UIUC CS, Berkeley RISE Lab (advised by Joseph Gonzalez), **curated the Berkeley Function Calling Leaderboard (BFCL)**, was a DATA 140 (probability) TA so he is stats-grounded. Domain: tool use, function calling, evaluation methodology. He knows GEPA's world well. He is junior and not the final decision-maker; he likely wants someone he can engage with intellectually.
- **Mahesh Sathiamoorthy (CEO)** is the decision-maker. Thesis: post-training signal quality is the new moat. He filters hard for "did you actually build this by hand," so a working pipeline is real signal.
- **Why GEPA:** GEPA's author list includes **Alexandros Dimakis, a Bespoke co-founder** (the paper is Berkeley-led; affiliations are UC Berkeley, Stanford, Notre Dame, Databricks, MIT, BespokeLabs.ai). Phrase it in the room as "the GEPA work Dimakis is on," not "your paper." GEPA sits on Bespoke's signal-quality thesis (rich language feedback beats scalar reward).
- **Substrate decision (why HotpotQA, not BFCL):** BFCL is Vishnu's lane and the maximal-fit substrate, but multi-turn BFCL is expensive to evaluate and he knows that better than anyone, so "I validated on HotpotQA, BFCL is the natural extension I would run with more budget" is a line he will accept. HotpotQA is standard, he knows it cold, has a naturally populated difficulty band, and the relevance rides on GEPA and on how I reason, not on the dataset. So I validate on HotpotQA and anchor the pitch on the tool-use regime.
- **Build posture:** I am building the feasibility version, not pitching design-only, because the execution signal is real with Mahesh. I have a finished separate project (reward-hack-arena: a 2x2 of SFT/GRPO times verifier/GPT-4o-judge labels on Qwen2.5-3B judging GSM8K) as my safe centerpiece. This GEPA probe is the swing, so an underpowered null is acceptable if framed honestly.
- **Build window:** Monday June 1 and Tuesday June 2 only. Interview Wednesday June 3, 2:30pm PT.

---

## 2. GEPA, everything you need from the paper (it is not attached)

**Paper:** "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning." ICLR 2026 oral. arXiv 2507.19457. Code: `github.com/gepa-ai/gepa`, also exposed as `dspy.GEPA`.

**What it is.** GEPA (Genetic-Pareto) optimizes the *prompts* of a compound LLM system while keeping model weights frozen. It mutates prompts using natural-language reflection on execution traces, and keeps a Pareto frontier of candidates for diversity. Headline results: beats GRPO (which used 24,000 rollouts) by up to ~20 percent while using up to ~35x fewer rollouts, and beats the prior prompt optimizer MIPROv2 by >10 percent.

**Core algorithm (Algorithm 1 in the paper), the loop you will modify:**
```
Inputs: system Phi, dataset D_train, metric mu, feedback fn mu_f, budget B
Hyperparams: minibatch size b, Pareto set size n_pareto

1.  Split D_train into D_feedback and D_pareto   (|D_pareto| = n_pareto)
2.  Candidate pool P = [Phi]; parents A = [None]
3.  for each (x_i, m_i) in D_pareto:             # per-instance scores live on D_pareto
4.      S[Phi][i] = mu(Phi(x_i), m_i)
5.  end
6.  while budget B not exhausted:
7.      k = SelectCandidate(P, S)                # Pareto-based, stochastic (Algorithm 2)
8.      j = SelectModule(Phi_k)                  # round-robin over modules
9.      M = minibatch of size b from D_feedback  # <-- THE SAMPLING STEP I OVERRIDE
10.     gather feedback, scores, traces for Phi_k[j] on M using mu_f
11.     pi'_j = UpdatePrompt(pi_j, feedbacks, traces[j])   # reflection LM proposes a new prompt
12.     Phi' = copy of Phi_k with module j updated to pi'_j
13.     sigma, sigma' = avg score on M (before, after)     # <-- ACCEPT TEST IS ON THE MINIBATCH
14.     if sigma' improved:
15.         add Phi' to P; record parent k
16.         for each (x_i, m_i) in D_pareto:     # validation scoring dominates the budget
17.             S[Phi'][i] = mu(Phi'(x_i), m_i)
18.     end
19. end
20. return Phi* maximizing average score on D_pareto
```

**Two facts from this algorithm that the whole design turns on:**
1. The per-instance scores GEPA logs (`S`) are computed on **D_pareto** (lines 3-5 and 16-18), **not** on D_feedback. The reflection minibatch is drawn from **D_feedback** (line 9). They are disjoint (line 1). So GEPA does not log per-instance difficulty for the reflection pool; you must score that pool yourself.
2. The acceptance test (lines 13-14) runs on **the same minibatch M used for reflection**. So if you bias which instances go into M, you also change the acceptance bar. This is the confound the decoupling fixes (Section 4).

Also note: line 7 re-samples *which candidate to mutate* stochastically from the Pareto frontier every iteration. There is no single "current policy." This is why the dynamic/curriculum version is a category mismatch and is moved to the extension (Section 11).

**Pareto-based candidate selection (Algorithm 2), the diversity mechanism (you do NOT modify this):**
```
SelectCandidate(P, S):
  for each instance i: s*[i] = max over candidates k of S[k][i]
                       Pareto*[i] = { candidates achieving s*[i] on instance i }
  C = unique candidates appearing in any Pareto*[i]
  remove strictly dominated candidates from C
  f[Phi] = number of instances for which Phi is in the pruned Pareto set
  sample a candidate with probability proportional to f[Phi]
```
This is what keeps exploration diverse and dodges local optima. It is held constant across all arms, so the treatment does not touch the diversity machinery.

**Reflection meta-prompt (Appendix C), gist.** The reflection LM is shown the current instruction plus a block of (inputs, outputs, feedback) for the minibatch examples, and asked to write a new instruction: infer the task from the inputs, extract niche or domain-specific facts from the feedback, and include any generalizable strategy. It returns the new instruction in a code block. You can use the `gepa` library's built-in meta-prompt; do not rewrite it.

**Feedback function mu_f.** Returns a numeric score plus `feedback_text`. For HotpotQA the feedback module identifies the set of relevant documents still to be retrieved at each hop and returns that as text. Use the program's existing mu_f.

**The HotpotQA program (Appendix E.1 and L), the system under optimization.** Retrieval-based multi-hop QA, derived from the paper's HoVerMultiHop program with the last hop modified to answer instead of generating another query. Four LLM modules, each an optimizable prompt:
- `summarize1`: (question, passages) -> summary_1
- `create_query_hop2`: (question, summary_1) -> query (the second-hop search query)
- `summarize2`: (question, context, passages) -> summary_2
- `final_answer`: (question, summary_1, summary_2) -> answer

Flow per instance: retrieve with the original question (hop 1) -> summarize1 -> create_query_hop2 -> retrieve with that query (hop 2) -> summarize2 -> final_answer. So one rollout is ~4 LLM module calls plus 2 retrievals. Retrieval is over a Wikipedia ColBERTv2 index (DSPy's hosted index is the default and is known to be flaky). The metric is answer **F1** (and EM); use F1 throughout for a continuous signal.

**Numbers you can rely on (Qwen3-8B, from the paper):**
- HotpotQA: baseline F1 ~42.3, GEPA ~62.3, GRPO ~43.3, MIPROv2 ~55.3. So there is a fat partial-success band between ~0.42 and ~0.62.
- GEPA's total HotpotQA budget was ~6,871 rollouts; GRPO's was 24,000.
- GEPA made only ~64 reflection calls on HotpotQA at that full budget (Table 4). The paper states the **majority of the rollout budget is spent on validation** (scoring candidates on D_pareto), not on reflection. Train-side rollouts to reach the optimum were 79 to 737.
- Minibatch size b = 3 (Appendix E.4). Module selection policy is round-robin.
- Model settings (Appendix E.2): Qwen3-8B at temperature 0.6, top-p 0.95, top-k 20, context window 16384.
- Cost reference (Appendix E.3): all the paper's GPT-4.1-mini experiments cost under $500 total; GEPA alone was $86 across six benchmarks at full budget. So our 15 small runs at ~$222 is in a sane range.
- Generalization gap: GEPA's instruction-only prompts generalize well (low val-to-test gap), which is why instruction optimization beats few-shot here.

---

## 3. One-line summary of the experiment

Test whether **which difficulty band of examples GEPA reflects on** changes its sample efficiency, by replacing GEPA's random reflection minibatch with a frozen-difficulty-band selector, on retrieval-based HotpotQA, with reflection decoupled from acceptance, at matched iteration count. This is a clean static **band** question. It makes no single-learner assumption. The dynamic curriculum version is the named extension, not what is run (see Section 11).

---

## 4. Research question, hypothesis, and the decoupling

**RQ.** GEPA draws its reflection minibatch at random. Does selecting that set by baseline difficulty band (mid-difficulty vs easy vs hard) change sample efficiency versus random, holding the acceptance test fixed and the iteration count equal?

**Hypothesis.** With a size-3 reflection set, which examples you reflect on is high leverage. Already-mastered instances give the reflector little to fix; always-failed instances give opaque, unfixable failures; mid-difficulty (frontier) instances give the richest actionable signal, so each reflection is more informative per iteration. Expected effect: faster improvement per iteration for the frontier band, most clearly versus random and versus the two extreme bands.

**Honest prior.** The effect may be modest or null. GEPA already extracts a lot from any failure, random sampling hits mid-difficulty instances anyway, and the Pareto diversity machinery may matter more than minibatch composition. That uncertainty is why it is worth measuring.

**The decoupling (load-bearing, keep it).** In stock GEPA the minibatch is both the reflection input and the accept/reject test (Algorithm 1 lines 13-14). Biasing the minibatch would therefore change the acceptance bar across arms, confounding the result. So: every arm **reflects** on its band-selected size-3 set, but **accepts or rejects** the proposed mutation on a **fixed shared accept batch A of 20 instances, identical across all arms and all iterations**, using strict improvement (`sigma' > sigma`, not `>=`) of the proposed candidate over its parent on A, with the parent's A-score cached once per candidate. This isolates reflection-signal quality from acceptance dynamics. A=20 is disjoint from D_feedback and D_pareto.

---

## 5. The four arms (with Fix 1 applied: equal-size bands)

All four arms run over **one frozen baseline difficulty table** (Section 6). They are identical in every respect except which band the size-3 reflection set is preferentially drawn from. Everything else (SelectCandidate Pareto sampling, round-robin module selection, the reflection meta-prompt, the fixed accept batch A, D_pareto, the model, the seed) is held constant across arms.

1. **random** (floor): draw the size-3 set uniformly from the full D_feedback pool.
2. **static-easy**: draw mostly from the high-F1 (mastered) band.
3. **static-frontier**: draw mostly from the mid-F1 (frontier) band. This is the method under test.
4. **static-hard**: draw mostly from the low-F1 (too-hard) band.

**Fix 1, equal-size bands (this replaces the earlier 25/50/25).** Bin D_feedback into **equal terciles** by baseline F1: bottom third = too-hard, middle third = frontier, top third = mastered, ~33 instances each. Each band arm draws its size-3 set **70/15/15** from (target band / one off-band / other off-band). Equal band sizes are non-negotiable: with the earlier 25/50/25 split the frontier band held 50 instances versus 25 for easy and hard, so the frontier arm would reflect on ~1.75x more distinct instances, and a win could be attributed to coverage diversity rather than to mid-difficulty. Equal terciles make every band arm reflect on ~31 distinct instances, so difficulty is the only thing that varies. The random arm stays uniform over all 100 as the floor.

If you want larger bands for more reflection diversity, grow D_feedback to 150 and take thirds of 50. The non-negotiable is that the three band arms are equal size.

**Identification.**
- frontier vs random: does selection matter at all (this contrast is allowed to mix difficulty-selection with concentration, since it only claims "selection beats no selection").
- frontier vs easy and frontier vs hard: is it specifically the mid band. Coverage-matched by the equal terciles, so this is clean and difficulty is the only varying factor.
- The mid band is no longer the only arm with any special property, so a frontier win is attributable to mid-difficulty rather than to adaptivity (there is no adaptive arm) or to band size (bands are equal).

**Plus a reference run: vanilla coupled GEPA** (random reflection, acceptance on the minibatch, i.e. stock GEPA), run for the same iteration count N. Purpose: confirm the decoupling does not distort baseline behavior. Compare it to the decoupled-random arm. Read this comparison **on the iteration axis**, because at the same N it consumes less budget than the decoupled arms (its accept test is on b=3 rather than on A=20).

---

## 6. The frozen difficulty table, and the go/no-go

- Score the **base system end to end** once on the full 100-instance D_feedback pool, recording per-instance answer F1. This single scoring does triple duty: it is the difficulty table for all band arms, the frontier-population diagnostic, and the go/no-go gate.
- Bin into equal terciles by F1. Frozen for the whole run. (Static by design. The table is baseline-relative, not current-candidate-relative; that is the deliberate scope, with current-candidate difficulty being the dynamic extension in Section 11.)
- **GO/NO-GO before spending the main budget.** Quantile bins are never empty by construction, so the real failure mode is a bimodal F1 distribution (a pile at 0 and a pile at 1 with little in between), which would make the "frontier" middle tercile a fiction of 0s and 1s. So the operative check is: confirm the F1 distribution has real **intermediate mass** (a meaningful fraction of instances with F1 roughly between 0.2 and 0.8). If the middle tercile is all 0/1, abort and re-pair (switch model, or fall back to the distractor setting). Do not run the 15 cells until this passes.

---

## 7. Decoupling, sampling, and the budget ledger (with Fix 2 applied)

**Acceptance.** Fixed shared batch A = 20 instances, same across all arms and iterations. Accept the proposed candidate iff its F1 on A strictly exceeds the parent's cached F1 on A. The reference arm instead accepts on its b=3 minibatch (stock GEPA).

**Sampling.** Every arm draws minibatch size b = 3 each iteration and runs for the **same iteration count N**. So total reflection rollouts (3N) are identical across arms by construction. Arms differ only in where those rollouts land: each band arm concentrates ~70 percent of its draws on its target tercile, random spreads uniformly. The redistribution of a fixed reflection budget is the treatment, not a budget difference. Coverage across distinct instances is now matched across the three band arms by the equal terciles.

**Fix 2, fix the iteration count N, not the budget B.** Acceptance rate `r` is endogenous to the arm: the hard arm may produce mutations that clear A less often, so it accepts less, spends less on D_pareto validation per iteration, and at a fixed B would run *more* iterations. So you cannot hold both B and N constant. The run-loop **terminates at N iterations**, not at a rollout cap. This makes reflection rollouts (3N) and accept-batch rollouts (20N) identical across arms by construction, and lets B float per arm. Report the total B consumed per arm as the cost number. Set N ~= 44 (chosen so that at r ~= 0.3 the per-arm budget lands near 2,000 rollouts). Compare arms at equal N (the primary axis).

**Per-iteration rollout cost (decoupled arms):**
```
per-iter = b (reflect, 3) + A (accept batch, 20) + r * P (D_pareto on accept; r~0.3, P=75)
         = 3 + 20 + 0.3*75 = 3 + 20 + 22.5 ~= 45.5 rollouts
```
At N ~= 44 that is ~= 2,000 rollouts per decoupled arm. B will float roughly 1,700 (low accept rate) to 2,300 (high accept rate) across arms. Reference arm is cheaper per iteration (b=3 accept instead of A=20), so it consumes less B at the same N. About 90 percent of B is condition-invariant validation plus accept-batch cost, which is exactly why the primary plot must be on the iteration axis, not on total rollouts (on a total-rollout axis the four curves would look nearly identical).

**Consolidated quantities.**
| Quantity | Value |
|---|---|
| D_feedback (reflection pool) | 100 |
| Fixed accept batch A | 20 |
| D_pareto (validation) P | 75 |
| Held-out test | 300 (scored once at the end, not counted in B) |
| Bands | equal terciles, ~33 each |
| Reflection minibatch b | 3 |
| Draw mix per band arm | 70/15/15 |
| Iteration count N (stopping rule) | ~44 |
| Arms | random, static-easy, static-frontier, static-hard |
| Seeds | 3 per arm |
| Reference | vanilla coupled GEPA, 3 seeds |
| Total runs | 4 x 3 + 3 = 15 |

All splits are disjoint (D_feedback, A, D_pareto, test = 495 instances total, drawn from HotpotQA dev ~7.4k). Hold D_pareto identical across all arms, since it sets the dominant cost. Where possible, pair the seed across arms (same seed gives the same SelectCandidate RNG, the same round-robin module schedule, and the same A) to reduce variance, even though trajectories diverge after iteration 1.

---

## 8. Budget ($300 cap)

Assumption: ~$0.006 per multi-hop rollout. Price the **full bundle** (~4 LLM module calls plus 2 retrievals), not a single model call. This assumption is the one thing I cannot close myself; it depends on the provider (Section 9), so validate it in the 50-example pilot before launching.

Rough total: 15 runs at ~2,000 rollouts ~= 30k, plus 300-instance test eval x 15 ~= 4.5k, plus init pool scoring 100 x 15 ~= 1.5k, plus ~1k pilot, ~= 37k rollouts ~= **$222 at $0.006**, fits $300 with buffer. If the confirmed provider cost is meaningfully higher, the lever order is: drop the reference to 1 seed, then seeds 3 -> 2, **before** shrinking P or A, since shrinking those degrades the science faster than dropping a seed.

---

## 9. Open inputs you (the receiving Claude) should get from me

1. **Provider per-rollout cost and rate limit.** Pick a Qwen3-8B endpoint (Together, Fireworks, DeepInfra, or local vLLM if I have the GPU). The per-rollout cost finalizes the ledger; the rate limit decides whether the 15 cells can run in parallel on Tuesday. I will confirm this. Do not assume it.
2. **The retrieval index.** DSPy's hosted ColBERTv2 Wikipedia index is the default and is flaky. I will test it before committing. If it is down, the named fallback is the **distractor** setting of HotpotQA (gold paragraphs provided, no retrieval, but it drops the create_query_hop2 module). Know which index is live before the build starts.

---

## 10. Build plan and the Monday-AM gate

**Weekend (travel, read only).** Read the `gepa` README and skim the loop in source. The paper facts you need are in Section 2.

**Monday AM, the gate (timeboxed ~3 hours).** Read the `gepa` source and answer two yes/no questions:
1. Can the reflection-set draw from D_feedback (Algorithm 1 line 9) be **overridden via a hook or subclass** rather than forking the loop?
2. Can the accept test (lines 13-14) be **split off the minibatch onto the fixed batch A** the same way?

Hard trigger: if past the 3-hour box you are writing into the engine loop rather than overriding a method, **stop**. Fall back to the coupled two-condition smoke (random vs frontier, acceptance left on the minibatch, which reintroduces the acceptance-bar confound and is feasibility-only, not a mechanism result) or to design-only. reward-hack-arena is the safety net, so a clean fallback is acceptable.

**Monday midday.** Stand up the DSPy retrieval-based multi-hop HotpotQA program (the four modules in Section 2). Get a vanilla GEPA run working. Confirm the retrieval index.

**Monday PM.** Implement: the one-time pool scoring and tercile binning; the four band samplers (sort the frozen F1 table, draw 70/15/15 from the target tercile); the accept-batch split. Run the 50-example smoke test end to end (one arm, one seed) to validate plumbing and the per-rollout cost.

**Tuesday AM.** Run the GO/NO-GO diagnostic on the full pool. If it passes, launch the 15 cells (parallelize subject to the rate limit).

**Tuesday PM.** Diagnostics, the iteration-axis curves with bootstrap CIs, rollouts-to-target, the 1-page writeup, a clean repo.

The build is now small because the dynamic machinery is gone: one-time pool scoring, four samplers, and the accept-batch split. The only real build risk is that accept-batch split, which is exactly what the Monday gate checks.

---

## 11. Metrics, analysis, and pre-registration

**Primary axis: reflection iteration.** Plot performance vs iteration, arms compared at equal N. This is where the treatment, if it exists, is visible, and it is matched across arms by Fix 2.
**Cost axis: total rollouts.** Report rollouts-to-target on the total-B axis as the honest cost number.
**Held-out test.** F1 on the 300-instance test set, scored once at the end on each run's returned candidate, with paired bootstrap CIs across the 3 seeds (pair at the per-instance test-eval level: same test items under each arm's final candidate).
**Target for rollouts-to-target.** Set it from the pilot, as the F1 the random arm reaches with margin. Blind fallback ~0.50. Do not pre-set 0.55: the paper's 0.62 came at ~3x this budget, so 0.55 may be unreachable at this N and would leave the metric undefined.

**Pre-register before the main run (do not tune on results):** the tercile bands and their thresholds, the 70/15/15 draw mix, and the strict-improvement accept rule with cached parent. Set the rollouts-to-target threshold after the pilot but before the 15-cell run.

**Power and framing.** This is a **feasibility probe, not a powered result**. With N ~= 44 the curriculum is realized over ~44 size-3 draws, so few-draw noise is real. P = 75 is the dominant variance source: F1 over 75 validation instances injects roughly +/- 6 points of selection noise into the returned candidate and therefore into the test number, the same order as any effect. So a clear, seed-stable separation is suggestive; a tight overlap is "underpowered," never "disproven." Do not let a tight overlap get presented as a result.

---

## 12. Claims framework

**Narrow claim (if frontier wins):** "Static frontier-band selection of GEPA's reflection set improves sample efficiency by X (target reached with Y fewer iterations) versus random, easy, and hard band selection, at equal iteration count, over 3 seeds, on retrieval-based HotpotQA, with acceptance held fixed on a shared batch."

**Mechanism:** frontier vs random says selection matters at all; frontier vs easy and vs hard (coverage-matched terciles) says it is specifically the mid-difficulty band.

**Cannot claim:** anything dynamic or curriculum-tracking (that is the named extension, not run); generality beyond HotpotQA; dominance over Pareto diversity in all regimes; transfer to stock coupled GEPA (the reference arm only partly addresses this); optimality; conceptual novelty (difficulty-aware example selection is not new; the novelty is the lever, selecting the *optimizer's reflection set* rather than in-prompt demonstrations).

**Null framing:** "Reflection-set composition is not GEPA's bottleneck at this scale, which points to the reflection and Pareto-selection steps rather than example choice." Usable and honest.

**Prior art to name (so novelty is positioned honestly):** Active Example Selection for In-Context Learning (EMNLP 2022), Selective Annotation (ICLR 2023), EASE, and especially CAMS (2025, a balanced-difficulty prompt curriculum for multimodal CoT). Lineage: self-paced learning (Kumar 2010), Bengio curriculum learning (2009), RHO-1. The differentiation: all of that selects which examples go into the prompt as demonstrations or which to annotate; this selects which examples the optimizer reflects on. Different lever, different place in the pipeline.

---

## 13. Limitations (own these, do not hide them)

1. **Single QA task.** No generality. The pitch anchors the tool-use regime with multi-turn BFCL as the named extension.
2. **Underpowered.** 3 seeds, N ~= 44, plausibly small effect. Feasibility probe.
3. **P = 75 selection noise** (~+/- 6 points) is the dominant variance source and feeds the power problem.
4. **Static, baseline-relative difficulty.** The table is frozen at the base system's difficulty, not the current candidate's. Deliberate scope; the dynamic version is the extension (item 6).
5. **Decoupling transfer-validity.** Decoupled GEPA is not stock GEPA; the vanilla reference arm only partly addresses whether the decoupling distorts behavior.
6. **The dynamic curriculum is not run, and that is a considered choice, not an omission.** GEPA is a Pareto population with stochastic candidate selection (Algorithm 1 line 7), not a single improving learner, so "a curriculum that tracks the learner" is a category mismatch, not a tuning problem. On top of that, a free incremental difficulty signal harvested from the b=3 minibatch would re-observe each pool instance only ~1.3 times over the run, under heterogeneous Pareto candidates rather than one policy, with single-sample F1 noisiest exactly on boundary instances, so it would collapse toward the static frontier band anyway at this budget. The clean dynamic test is **candidate-relative** (reflect on the instances the specific candidate drawn this iteration is on the boundary for), which is well-posed and sidesteps the no-single-learner problem, but it needs a multi-sample per-candidate estimate on the pool that B = 2,000 cannot buy without leakage. Carry this as the insight, not as a hole.

---

## 14. Pitch framing for the room

Present this as a method, **reflection-set selection for GEPA**, motivated by long-horizon tool-use trajectories (Vishnu's BFCL world) where the reflection signal is richest, validated on HotpotQA for cost and a clean populated band, with multi-turn BFCL as the named extension. Lead with the method and the reasoning, not with HotpotQA mechanics. Frame the GEPA connection as "the work Dimakis is on." Success is not "frontier wins." It is demonstrating that I read a hot method carefully, found a genuinely unoptimized component (the random reflection minibatch), caught that the component doubles as the acceptance gate and controlled for it, connected it to a literature it has not been applied to, designed a coverage-matched controlled comparison, and assessed novelty and power honestly.

**The hardest question to expect, and the prepared answer.** Vishnu (RISE lab, knows GEPA cold) is likely to ask some version of: "GEPA re-samples which candidate to mutate every iteration, so there is no single current policy. Why static bands, and why not a dynamic curriculum that tracks the learner?" The answer, delivered without flinching: there is no single learner to track in a Pareto population, so a learner-tracking curriculum is category-mismatched here; the well-posed dynamic version is candidate-relative difficulty, which needs a per-candidate multi-sample estimate the budget cannot buy without the leakage I deliberately avoided; so I study the well-posed static band question, which makes no single-learner assumption, and I name the candidate-relative dynamic version as the extension. Delivered cleanly, this turns the deepest objection into evidence that I understand exactly where and why the method breaks, which matters more than a win.

---

## 15. Settled decisions (do not re-open unless something here concretely breaks)

- Substrate: retrieval-based HotpotQA, the paper's four-module multi-hop program. Distractor only as the Monday fallback.
- Task model Qwen3-8B at temp 0.6 (top-p 0.95, top-k 20). Reflection model mini-class (gpt-4.1-mini or gpt-4o-mini), held constant.
- Acceptance decoupled onto a fixed shared batch A=20, strict improvement, cached parent.
- Four static band arms over one frozen tercile difficulty table; dynamics is the named extension.
- Equal-size bands (terciles), 70/15/15 draw (Fix 1).
- Stopping rule is N iterations, not B; B floats and is reported as cost (Fix 2).
- Primary axis is reflection iteration; cost axis is total rollouts; test F1 with bootstrap CIs across 3 seeds.
- D_feedback 100, A 20, P 75, test 300, b 3, N ~= 44, 15 runs, ~$222 at the $0.006 assumption.
- The deliverable is the band result. A null is presented as feasibility-suggestive or underpowered, never as "frontier does not help."

If you, the receiving Claude, think one of these is wrong, say so directly and explain what breaks. Otherwise help me build it: start with the Monday-AM gate (the gepa source recon on lines 9 and 13-14), then the DSPy program, then the samplers and the accept-batch split, then the diagnostics and analysis.
