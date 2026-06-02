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

## Verification

Both substitutions were verified end-to-end before this file was committed:

- Structural smoke (`src/smoke_chunk2.py`) green: split sizes 100/20/75/300,
  disjoint (495 unique HotpotQA ids), reproducible under seed_splits=0,
  three named predictors loaded with seed instructions verbatim.
- One-rollout LM smoke green: example `5abf63f15542997ec76fd3ea` returned a
  partial-credit answer ("1922" vs gold "October 1922"). Gold supporting
  titles (Socialist Revolutionary Party, Russian Civil War) both present in
  the 10-paragraph context, distributed one per hop bucket.
- Full pytest (15/15 passing): 10 offline contract tests + 5 integration
  tests including `test_gold_paragraphs_in_context`.

API spend during Chunk-2 verification: under 1¢ (one probe call of 31 tokens,
one rollout under 10 LM calls).
