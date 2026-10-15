# Baseline check: did GEPA beat the seed program on HotpotQA? (2026-09-29)

Repo: `main` at `dbfbaa3`. This file is the only change made in this pass.

Question: did the GEPA-optimized programs in Exp 1, Exp 1b and Exp 3 beat the unoptimized
seed program on HotpotQA test F1? A second, independent read comes from the within-run
trajectory.

Method:
- Every number was recomputed from committed run artifacts: per-instance
  `test_eval.json`, `run_log.txt`, `run_log.json`, and Exp 3 `seed*.json`. No report
  tables were used as sources.
- There are no p-values and no verdicts. Judgment calls are in **OPEN QUESTIONS**.
- The scripts are in the Appendix.

Path shorthand: `E1` = `experiments/exp1_bands/logs`, `E1B` = `experiments/exp1_bands/logs_hotpot_100`,
`E3` = `experiments/exp3_curriculum`.

---

## 1. `git status --short`

```
?? docs/WEBSITE-2026-09-28.md
```

That file is the untracked output of the previous pass. The working tree was otherwise clean.

---

## 2. The baseline

### 2.1 Source
- The source is `E1/static_frontier_0_pilot/test_eval.json`:
  - `:2` `"best_idx": 0`
  - `:3` `"best_val_f1": 0.5963222703222701`
  - `:4` `"avg_test_f1": 0.6316411742379483`
  - `:5` `"n_test": 300`
  - `:6` onward: `per_instance_f1`, 300 values
- The same values are summarized at `E1/static_frontier_0_pilot/pilot_summary.json:15-18`.
- `best_idx: 0` means the best candidate by D_pareto average was candidate 0, the seed. The
  pilot ran 4 iterations (`pilot_summary.json:4-5`), and nothing it proposed beat the seed on
  D_pareto.
- Recomputing the mean of the 300 per-instance values gives **0.6316411742379484**, which
  matches `:4` to floating-point precision.

### 2.2 What it is
- **Split:** the held-out HotpotQA test split, n=300. It is loaded by `src/data.load_splits`
  in both runners (`src/orchestrator.py:204-208`, `src/curriculum_runner.py:378`).
- **Metric:** mean per-instance answer F1 from `substrate.metric_fn`, the same metric the arm
  endpoints use.
- **Grading path, Exp 1/1b** (this also covers the pilot):
  - `_eval_best_on_test` picks the best candidate by D_pareto average (`src/orchestrator.py:450-454`).
  - It scores that candidate with `_PatchedDspyAdapter` + `substrate.metric_fn` over the test
    set (`src/orchestrator.py:467-485`).
- **Grading path, Exp 3:**
  - The same best-by-D_pareto rule (`src/curriculum_runner.py:294-296`).
  - The same adapter class and metric (`src/curriculum_runner.py:392-403`).
  - The test eval runs through that in-run adapter (`src/curriculum_runner.py:298-302`).
- **Differences between the two paths:**
  - Exp 1 builds fresh LM and adapter objects for the test eval; Exp 3 reuses the run's own.
  - Exp 3 committed only the mean `final_test_f1`, not per-instance scores.

### 2.3 Byte-level evidence that it is the same quantity as the arm endpoints

The value 0.6316411742379483 recurs exactly, down to the last digit, everywhere the best
candidate is the seed:
- `E1B/static_frontier_0_pilot/test_eval.json`: `best_idx 0`, and a `per_instance_f1` list
  identical, element for element, to the Exp 1 pilot's.
- `E1B/static_easy_{0,1,2}/test_eval.json:2,4`: `best_idx 0` in all three cells, and the same
  value.
- `E3/hard_to_easy/seed2.json:7-8`: `"final_test_f1": 0.6316411742379483`,
  `"best_candidate_idx": 0`.

So the Exp 1 path, the Exp 1b path and the Exp 3 path all produce the same number for the
same program. The baseline is directly comparable to the arm means, and the comparison
below goes ahead.

### 2.4 Caveat: the baseline is a single cached draw
- The task LM runs at temperature 0.6 (`config/experiment.yaml:54`).
- The dspy LM cache defaults to on (`.venv/lib/python3.10/site-packages/dspy/clients/lm.py:39`,
  `cache: bool = True`). `LMConfig.to_lm` never overrides it (`src/run_gepa.py:103-116`).
- So the seed's test outputs were sampled once and then served from the cache in every
  later evaluation. The exact 16-digit repeats in 2.3 confirm this.
- **The seed program's own resampling variance on test is not measured anywhere in the repo.**
  Each arm endpoint is likewise one draw, for one candidate.
- The seed's D_pareto score is 0.5963 (`E1/random_0/run_log.txt:1`:
  "Base program full valset score: 0.5963222703222701 over 75 / 75 examples"). Its test
  score is 0.6316. Those are different splits, each a single draw.

---

## 3. Per-arm test F1 against the baseline

Each cell's test F1 was recomputed as the mean of its 300 `per_instance_f1` values; each one
matched the cell's `avg_test_f1` at `test_eval.json:4`. Exp 3 values are read from
`E3/<arm>/seed<s>.json:7` (no per-instance data was committed). SD is the sample SD
(ddof=1) across the 3 seeds, and each seed is n=300 test examples. Δ = value − 0.6316.

| study | arm | seed 0 | seed 1 | seed 2 | mean | SD | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean | seeds > baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | **seed program (baseline)** | | | | **0.6316** | — | | | | | |
| Exp 1 | random | 0.6111 | 0.6339 | 0.6506 | 0.6319 | 0.0198 | −0.0205 | +0.0023 | +0.0189 | +0.0002 | 2/3 |
| Exp 1 | static_easy | 0.6267 | 0.6306 | 0.6221 | 0.6265 | 0.0043 | −0.0049 | −0.0010 | −0.0095 | −0.0052 | 0/3 |
| Exp 1 | static_frontier | 0.6260 | 0.6239 | 0.6368 | 0.6289 | 0.0069 | −0.0056 | −0.0078 | +0.0052 | −0.0027 | 1/3 |
| Exp 1 | static_hard | 0.6326 | 0.6232 | 0.6488 | 0.6348 | 0.0130 | +0.0009 | −0.0085 | +0.0171 | +0.0032 | 2/3 |
| Exp 1 | vanilla_coupled_gepa | 0.6450 | 0.6242 | 0.6465 | 0.6386 | 0.0125 | +0.0134 | −0.0075 | +0.0149 | +0.0069 | 2/3 |
| Exp 1b | static_easy | 0.6316 | 0.6316 | 0.6316 | 0.6316 | 0.0000 | 0 | 0 | 0 | 0 | 0/3 (all three are the seed) |
| Exp 1b | static_frontier | 0.6211 | 0.6412 | 0.6483 | 0.6369 | 0.0141 | −0.0105 | +0.0096 | +0.0167 | +0.0052 | 2/3 |
| Exp 1b | static_hard | 0.6340 | 0.6457 | 0.6191 | 0.6329 | 0.0133 | +0.0024 | +0.0141 | −0.0125 | +0.0013 | 2/3 |
| Exp 3 | random | 0.6299 | 0.6182 | 0.6325 | 0.6269 | 0.0076 | −0.0017 | −0.0135 | +0.0009 | −0.0048 | 1/3 |
| Exp 3 | easy_to_hard | 0.6214 | 0.6371 | 0.6360 | 0.6315 | 0.0088 | −0.0102 | +0.0055 | +0.0044 | −0.0001 | 2/3 |
| Exp 3 | hard_to_easy | 0.6391 | 0.6274 | 0.6316 | 0.6327 | 0.0059 | +0.0074 | −0.0042 | 0 (seed) | +0.0011 | 1/3 |
| Exp 3 | static_medium | 0.5985 | 0.6162 | 0.6724 | 0.6290 | 0.0386 | −0.0331 | −0.0155 | +0.0408 | −0.0026 | 1/3 |

Per-cell sources:
- Exp 1: `E1/<arm>_<s>/test_eval.json:4`.
- Exp 1b: `E1B/<arm>_<s>/test_eval.json:4`. Exp 1b re-ran only the three static arms; its
  random and vanilla cells are the Exp 1 cells.
- Exp 3: `E3/<arm>/seed<s>.json:7`.

---

## 4. How many arms clear the baseline, and by how much

- **Arm means:** 6 of 12 arm rows are above 0.6316, 5 are below, and 1 is equal (Exp 1b
  static_easy, which is the seed itself).
- **Size of the gaps:** the largest mean gap above the baseline is +0.0069 (Exp 1 vanilla);
  the largest below is −0.0052 (Exp 1 static_easy). Every arm mean is within 0.007 F1 of
  the seed.
- **Per seed:**
  - **Every arm that produced a non-seed program has seeds on both sides of the baseline**,
    with one exception: Exp 1 static_easy is below it in all 3 seeds.
  - No arm is above the baseline in all 3 seeds.
  - Across all 36 cells, 16 are above the baseline, 16 below, and 4 equal (those 4 returned
    the seed program itself).
- **Per-seed spread:**
  - Within-arm SDs run from 0.0043 to 0.0386.
  - Per-seed Δ runs from −0.0331 to +0.0408.
  - Every arm-mean Δ is smaller in magnitude than that arm's own SD, with two exceptions:
    Exp 1 static_easy (|Δ| 0.0052 vs SD 0.0043), and Exp 1b static_easy (Δ 0, SD 0, because
    all three cells are the seed).
- **Per-instance movement (Exp 1/1b only, derived).** Compared item by item with the seed's
  300 per-instance scores, each best candidate changes the score on about 30–90 items, with
  gains and losses nearly balanced. Examples of improved / worsened item counts:
  - Exp 1 random seeds 0/1/2: 36/43, 44/42, 25/15
  - Exp 1 static_frontier: 21/18, 16/19, 34/33
  - Exp 1b static_frontier: 45/44, 44/34, 41/34

  The optimized programs behave differently from the seed on the test set, and on net
  those changes cancel. Some of this movement is also temperature-0.6 resampling, because
  a new instruction creates new cache keys (§2.4).

---

## 5. Within-run trajectory (second read)

### 5.1 Exp 1 / 1b: D_pareto validation F1 of accepted candidates vs the seed

**Source.** Each cell's `run_log.txt` records:
- the seed's score ("Iteration 0: Base program full valset score: 0.5963222703222701 over 75 / 75 examples", line 1);
- for every accepted candidate, "Individual valset scores for new program: {...}" (75 values), followed by "New program candidate index: k".

The mean of each dict is that candidate's D_pareto val F1. The per-cell maximum of these,
together with the seed, reproduces `cell_summary.json` `best_val_f1` for every cell (checked
by assertion).

D_pareto (75 examples) is disjoint from the A=20 accept batch and from D_feedback. Accept
decisions are made on A=20, not on D_pareto.

Val gains are relative to the seed's 0.5963. "Last" means the last accepted candidate.

| study | arm | seed | n accepted | n above seed on val | best val − seed | last val − seed |
|---|---|---|---|---|---|---|
| Exp 1 | random | 0 | 6 | 6 | +0.0748 | +0.0424 |
| | | 1 | 7 | 6 | +0.0260 | +0.0024 |
| | | 2 | 6 | 4 | +0.0758 | +0.0115 |
| Exp 1 | static_easy | 0 | 9 | 9 | +0.0721 | +0.0467 |
| | | 1 | 3 | 2 | +0.0276 | −0.0047 |
| | | 2 | 4 | 3 | +0.0657 | +0.0394 |
| Exp 1 | static_frontier | 0 | 8 | 3 | +0.0471 | +0.0471 |
| | | 1 | 8 | 4 | +0.0388 | +0.0326 |
| | | 2 | 6 | 5 | +0.0641 | +0.0100 |
| Exp 1 | static_hard | 0 | 4 | 4 | +0.1020 | +0.0397 |
| | | 1 | 3 | 2 | +0.0490 | +0.0490 |
| | | 2 | 3 | 3 | +0.0206 | +0.0206 |
| Exp 1 | vanilla_coupled_gepa | 0 | 10 | 8 | +0.0770 | +0.0030 |
| | | 1 | 6 | 3 | +0.0267 | −0.0049 |
| | | 2 | 6 | 6 | +0.0961 | +0.0623 |
| Exp 1b | static_easy | 0–2 | 0 | 0 | 0 | — |
| Exp 1b | static_frontier | 0 | 7 | 5 | +0.0657 | +0.0317 |
| | | 1 | 10 | 10 | +0.1057 | +0.0428 |
| | | 2 | 5 | 4 | +0.0417 | +0.0355 |
| Exp 1b | static_hard | 0 | 11 | 9 | +0.0937 | +0.0366 |
| | | 1 | 8 | 8 | +0.0813 | +0.0232 |
| | | 2 | 4 | 3 | +0.0580 | +0.0580 |

**Totals (derived):**
- 107 of 134 accepted candidates score above the seed on D_pareto. The median gain is +0.027.
- In all 21 cells with at least one accept, the best candidate beats the seed on D_pareto,
  by +0.021 to +0.106 (mean +0.062).
- The same 21 best candidates average **+0.0013** against the seed on test, and 11 of the
  21 are above it.

**How to read the val numbers:**
- **Individual accepted candidates' val scores are not selected on D_pareto.** They were
  accepted on A=20, so the 107/134 count is not inflated by selection on the 75
  validation items.
- **The "best val − seed" column is selected.** It is the maximum over candidates on the
  same 75 items used to pick the test candidate. `docs/handoffs/gepa_band_selection_handoff.md:242`
  anticipated this: "P = 75 is the dominant variance source: F1 over 75 validation instances
  injects roughly +/- 6 points of selection noise into the returned candidate and therefore
  into the test number".
- **The seed's val score (0.5963) is itself a single cached draw** at temperature 0.6
  (§2.4).

### 5.2 Accept-batch (A=20) scores, Exp 1 and Exp 3
- The seed's A=20 score is 11.238/20 = **0.5619** in Exp 1 (`E1/random_0/run_log.json`,
  record `i: 0`, `accept_batch_parent_score`) and in every Exp 3 cell (the minimum
  `parent_a20` in `E3/<arm>/seed<s>.json`). That is the same batch with the same cached
  outputs.
- **Exp 3:**
  - All 42 accepted candidates are above the seed on A=20, by +0.006 to +0.072.
  - This is guaranteed by the gate: each accept requires beating its parent, and every
    chain of parents starts at the seed (`.venv/.../gepa/core/engine.py:538-541`;
    `src/decoupled_proposer.py:287-294`).
  - It therefore carries no independent information about whether the optimizer improved
    anything.
  - The one non-guaranteed A=20 number: of the 174 proposals whose parent was the seed,
    25 (14%) beat the seed on A=20.
- **Exp 3 has no D_pareto or per-instance test record.** Across git history,
  `results/exp3_curriculum/` and `experiments/exp3_curriculum/` only ever held
  `seed*.json`, which contains per-iteration `parent_a20`, `candidate_a20` and `accepted`,
  plus the final test F1. The Exp 1-style validation read in 5.1 therefore cannot be
  repeated for Exp 3 from committed artifacts.
- One Exp 3 detail: in `E3/hard_to_easy/seed2.json:5-8` the cell accepted one candidate
  (at iteration 1, on A=20), yet the seed is still `best_candidate_idx: 0`. The accepted
  candidate lost to the seed on D_pareto.

---

## 6. Does any committed file already make this comparison?

**No committed file compares the seed program's test F1 with the arm endpoints.**
- `experiments/exp1_bands/analysis_chunk10/FINDINGS_hotpot_100.md:23` reports 0.6316 as
  "`static_easy` | **100/0/0** | **0.6316**". It does not say this is the unoptimized seed.
- `experiments/exp3_curriculum/RESULTS.md:58` lists 0.6316 for hard_to_easy seed 2
  without comment.
- **The one implicit seed comparison in the repo is on validation, not test.**
  `experiments/exp1_bands/analysis_chunk7/summary.json:51`, under `D1_iteration_curves`,
  says: "Best-so-far is the cumulative max of accepted candidates' val scores; if no accept
  yet, value is the seed candidate's val F1." That is plotted in
  `experiments/exp1_bands/analysis_chunk7/iteration_curves.png`, whose curves start at the
  seed's val score and rise from it.
- **The paper-baseline comparison was explicitly set aside.** `docs/DEVIATIONS.md:89-92`:
  "Paper-comparability against the GEPA paper's Qwen3-8B HotpotQA numbers (baseline F1
  ~42.3, GEPA ~62.3) is weakened. Absolute F1 is no longer directly comparable;
  relative-arm comparisons within this experiment are unaffected."
- **The previous pass raised the missing baseline** but did not compute the comparison:
  `docs/WEBSITE-2026-09-28.md`, §2.2 "Unreported baseline" and OPEN QUESTION 7. That file
  is untracked.

**What this pass computes fresh:** the per-arm test comparison (§3–4) and the per-candidate
validation tally (§5.1).

---

## 7. OPEN QUESTIONS

1. **Are the Exp 1 and Exp 3 selection nulls informative?**
   - On test F1, no arm mean is more than 0.007 from the seed, and every non-degenerate arm
     has seeds on both sides.
   - On D_pareto, most accepted candidates beat the seed.
   - Does "GEPA was not learning" follow? Or is it "GEPA learned something that does not
     transfer from 75 validation items to 300 test items at this noise level"?
2. **Resampling noise of the seed.** The baseline is one cached temperature-0.6 draw.
   Should the seed be re-scored on test with the cache off, several times, before any
   conclusion is drawn? The same question applies to D_pareto: 0.5963 may be a low draw,
   given 0.6316 on test.
3. **Val→test gap.** Best-val gains average +0.062, while test gains average +0.0013. How
   much of that is:
   - selection noise on the 75 items (`handoff:242`),
   - a single low draw for the seed's val score,
   - or real overfitting to D_pareto?
   None of the three has been measured.
4. **Per-instance churn.** Best candidates change 30–90 of 300 test items against the seed,
   with net change near 0. Is that instruction effect, or temperature-0.6 resampling caused
   by new cache keys? A paired re-score would separate them.
5. **Exp 3 coverage.** Exp 3 committed no D_pareto or per-instance data. Accept the A=20-only
   within-run read, or re-evaluate the Exp 3 best candidates? Their instructions are not
   committed either, since `seed*.json` has no candidate texts.
6. **Framing.** If the seed baseline is adopted, how should the README, docs/FINDINGS.md and
   the 2026-09-28 synthesis present it? Two options:
   - Selection changed dynamics while the optimizer made no measurable test gain over the
     seed.
   - The optimizer's test gain is below the resolution of this setup.
7. **Vanilla stock GEPA.** The stock-acceptance reference arm is also not separable from the
   seed (+0.0069 mean, one seed below). Is that a statement about GEPA at N=44 on this
   substrate and task model, independent of the selection question?

---

## Appendix: scripts (run from repo root with `.venv/bin/python`)

```python
import json, re, ast, numpy as np
E1="experiments/exp1_bands/logs"; E1B="experiments/exp1_bands/logs_hotpot_100"; E3="experiments/exp3_curriculum"

# 2. Baseline from per-instance scores; identity checks
pil=json.load(open(f"{E1}/static_frontier_0_pilot/test_eval.json"))
base=np.array(pil["per_instance_f1"]); B=base.mean(); print(B, len(base), pil["best_idx"])
print(json.load(open(f"{E1B}/static_frontier_0_pilot/test_eval.json"))["per_instance_f1"]==pil["per_instance_f1"])

# 3-4. Per-arm test F1 vs baseline (Exp 1/1b from per-instance lists; Exp 3 from seed json)
def exp1(root, arm):
    v=[];ud=[]
    for s in range(3):
        pi=np.array(json.load(open(f"{root}/{arm}_{s}/test_eval.json"))["per_instance_f1"])
        v.append(pi.mean()); ud.append(((pi>base+1e-12).sum(), (pi<base-1e-12).sum()))
    return np.array(v), ud
for root,arms in [(E1,["random","static_easy","static_frontier","static_hard","vanilla_coupled_gepa"]),(E1B,["static_easy","static_frontier","static_hard"])]:
    for arm in arms:
        v,ud=exp1(root,arm); print(root[-6:],arm,v.round(4),v.mean()-B,v.std(ddof=1),(v-B).round(4),ud)
for arm in ["random","easy_to_hard","hard_to_easy","static_medium"]:
    v=np.array([json.load(open(f"{E3}/{arm}/seed{s}.json"))["final_test_f1"] for s in range(3)])
    print("exp3",arm,v.round(4),v.mean()-B,v.std(ddof=1),(v-B).round(4))

# 5.1 D_pareto val of accepted candidates from run_log.txt
def parse(path):
    b=None; c={}; pend=None
    for l in open(path):
        m=re.search(r"Base program full valset score: ([\d.]+)",l)
        if m: b=float(m.group(1))
        m=re.search(r"Individual valset scores for new program: (\{.*\})",l)
        if m: pend=float(np.mean(list(ast.literal_eval(m.group(1)).values())))
        m=re.search(r"New program candidate index: (\d+)",l)
        if m and pend is not None: c[int(m.group(1))]=pend; pend=None
    return b,[c[k] for k in sorted(c)]
# per cell: n accepted, n above seed, max(vals)-seed (== cell_summary best_val_f1 - seed), vals[-1]-seed

# 5.2 Exp 3 A20: seed-parented proposals beating the seed; accepted candidates vs seed
SEED=0.5619047619047619
for arm in ["random","easy_to_hard","hard_to_easy","static_medium"]:
    for s in range(3):
        its=json.load(open(f"{E3}/{arm}/seed{s}.json"))["iterations"]
        sp=[i for i in its if i["parent_a20"]==SEED and i["candidate_a20"] is not None]
        print(arm,s,len(sp),sum(i["candidate_a20"]>SEED for i in sp),[round(i["candidate_a20"]-SEED,4) for i in its if i["accepted"]])
```

```sh
# 6. Existing comparisons
git grep -n -i -E "0\.6316|0\.5963|seed program|unoptimi|baseline|seed candidate" HEAD -- '*.md'
grep -n '"source"' experiments/exp1_bands/analysis_chunk7/summary.json
# Exp 3 artifacts ever committed
git log --all --name-only --format='' -- 'results/exp3_curriculum/*' 'experiments/exp3_curriculum/*' | sort -u
```
