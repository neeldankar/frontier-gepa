"""IFBench active-example-selection validation experiment.

Question: does failure composition (which constraint types fail on a cc3 example)
predict whether reflecting on it generalizes to a held-out A=20, over and above base
score? Outcome = the reflected (minibatch-of-1) proposed instruction strictly improves
over the seed on a frozen A=20. Regress: outcome ~ base_score + addressability.

Two phases (operator gate):
  (default)  pre-flight: build the frozen sets, print stats, STOP. No LLM.
  --run      reflect + score each example (real LLM, N=4), then logistic regression.

Scratch experiment; nothing in src/ is modified. Reuses the locked stack
(dspy==3.2.1, gepa==0.1.1, _PatchedDspyAdapter, Qwen2.5-7B-Instruct-Turbo @ temp 0.6,
gpt-4.1-mini). Base scores are read frozen from phase0/ifbench_inspection.json (no
re-scoring); the A=20 before/after outcome eval is fresh (apples-to-apples).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PHASE0 = Path(__file__).resolve().parent
REPO = PHASE0.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

INSPECTION = PHASE0 / "ifbench_inspection.json"
PREFLIGHT = PHASE0 / "active_selection_preflight.json"
RESULTS = PHASE0 / "active_selection_results.jsonl"
REPORT = PHASE0 / "active_selection_report.md"

A_SIZE = 20
CEILING = 0.99
STRAT_SEED = 0
LANGDETECT_IDS = {
    "change_case:english_lowercase",
    "change_case:english_capital",
    "language:response_language",
}

# Locked pre-registered addressability table (parsed from the spec; do not modify).
ADDRESSABILITY: dict[str, int] = {
    **{t: 3 for t in [
        "paragraphs:paragraphs2", "first_word:first_word_answer", "startend:quotation",
        "last_word:last_word_answer", "startend:end_checker", "paragraphs:paragraphs",
        "detectable_content:postscript", "keywords:exclude_word_harder", "keywords:existence",
        "detectable_format:title", "punctuation:no_comma"]},
    **{t: 2 for t in [
        "detectable_format:sentence_hyphens", "keywords:start_end", "keywords:word_once",
        "detectable_format:number_highlighted_sections", "detectable_format:number_bullet_lists",
        "first_word:first_word_sent", "detectable_content:number_placeholders",
        "change_case:english_lowercase", "change_case:english_capital",
        "length_constraints:number_words", "length_constraints:number_sentences",
        "punctuation:punctuation_dot", "keywords:frequency",
        "keywords:word_count_different_numbers", "length_constraints:number_paragraphs",
        "detectable_format:json_format", "keywords:palindrome", "language:response_language",
        "combination:repeat_prompt", "combination:two_responses"]},
    **{t: 1 for t in [
        "length_constraints:nth_paragraph_first_word", "count:lowercase_counting",
        "count:count_increment_word", "last_word:last_word_sent",
        "change_case:capital_word_frequency", "keywords:letter_frequency",
        "letters:letter_counting", "letters:letter_counting2", "detectable_format:square_brackets"]},
    **{t: 0 for t in [
        "count:counting_composition", "detectable_format:bigram_wrapping", "copy:repeat_phrase",
        "keywords:keyword_specific_position", "count:count_unique",
        "keywords:no_adjacent_consecutive", "copy:copying_multiple", "new:copy_span_idx",
        "copy:copy", "copy:copying_simple"]},
}


def _failed_types(record: dict) -> list[str]:
    return [c["instruction_id"] for c in record["per_constraint"] if not c["passed"]]


def _all_types(record: dict) -> list[str]:
    return [c["instruction_id"] for c in record["per_constraint"]]


def _stratified_a20(pool: list[dict], k: int, seed: int) -> list[dict]:
    """Proportional allocation across frozen score levels, largest-remainder, with a
    seeded within-bucket sample. pool entries are inspection records."""
    rng = random.Random(seed)
    by_score: dict[float, list[dict]] = defaultdict(list)
    for r in pool:
        by_score[round(r["score"], 4)].append(r)
    levels = sorted(by_score)
    n = len(pool)
    # ideal proportional counts, then largest-remainder rounding to sum k
    ideal = {s: k * len(by_score[s]) / n for s in levels}
    base = {s: int(math.floor(ideal[s])) for s in levels}
    rem = k - sum(base.values())
    order = sorted(levels, key=lambda s: ideal[s] - base[s], reverse=True)
    for s in order[:rem]:
        base[s] += 1
    chosen: list[dict] = []
    for s in levels:
        bucket = sorted(by_score[s], key=lambda r: r["id"])
        take = min(base[s], len(bucket))
        chosen.extend(rng.sample(bucket, take))
    # if rounding/bucket caps left us short, fill from remaining by id
    if len(chosen) < k:
        chosen_ids = {r["id"] for r in chosen}
        remaining = sorted((r for r in pool if r["id"] not in chosen_ids), key=lambda r: r["id"])
        chosen.extend(rng.sample(remaining, k - len(chosen)))
    return chosen


def preflight() -> dict:
    inspection = json.loads(INSPECTION.read_text())
    by_id = {r["id"]: r for r in inspection}
    assert len(by_id) == len(inspection), "duplicate ids in inspection.json"

    # cc3 subset and cc3-with-failure
    cc3 = [r for r in inspection if r["n_constraints"] == 3]
    cc3_fail = [r for r in cc3 if r["score"] < 1.0 - 1e-9]

    # rule 1: drop reflected examples whose FAILED types include a langdetect verifier
    dropped_ld = [r for r in cc3_fail if set(_failed_types(r)) & LANGDETECT_IDS]
    reflected_pre = [r for r in cc3_fail if not (set(_failed_types(r)) & LANGDETECT_IDS)]

    # A=20 pool: all-150, ceiling-stripped, langdetect-stripped (PRESENCE of any langdetect
    # constraint, so re-eval scoring is deterministic), score-stratified, frozen.
    a20_pool = [
        r for r in inspection
        if r["score"] < CEILING and not (set(_all_types(r)) & LANGDETECT_IDS)
    ]
    a20 = _stratified_a20(a20_pool, A_SIZE, STRAT_SEED)
    a20_ids = {r["id"] for r in a20}

    # disjoint: drop any A=20 member from the reflected set
    overlap = [r for r in reflected_pre if r["id"] in a20_ids]
    reflected = [r for r in reflected_pre if r["id"] not in a20_ids]
    reflected.sort(key=lambda r: r["example_index"])

    # addressability + coverage assert
    missing = sorted({t for r in reflected for t in _failed_types(r) if t not in ADDRESSABILITY})
    assert not missing, f"failed types missing from addressability table: {missing}"

    def addr(r):
        ft = _failed_types(r)
        return sum(ADDRESSABILITY[t] for t in ft) / len(ft)

    reflected_recs = [{
        "id": r["id"], "example_index": r["example_index"],
        "base_score": r["score"], "failed_types": _failed_types(r),
        "addressability": addr(r),
    } for r in reflected]

    # seed instruction
    from src.ifbench_program import build_program
    seed_instruction = build_program().get_module_instruction("answer")

    out = {
        "a20_ids": sorted(a20_ids),
        "a20_scores": sorted(round(r["score"], 4) for r in a20),
        "reflected": reflected_recs,
        "seed_instruction": seed_instruction,
        "dropped_langdetect_ids": [r["id"] for r in dropped_ld],
        "a20_overlap_ids": [r["id"] for r in overlap],
        "config": {"A_SIZE": A_SIZE, "CEILING": CEILING, "STRAT_SEED": STRAT_SEED,
                   "langdetect_ids": sorted(LANGDETECT_IDS)},
    }
    PREFLIGHT.write_text(json.dumps(out, indent=2))

    # ---- print ----
    addrs = [r["addressability"] for r in reflected_recs]
    bases = [r["base_score"] for r in reflected_recs]
    print("=" * 70)
    print("PRE-FLIGHT (no LLM). Frozen sets written to active_selection_preflight.json")
    print("=" * 70)
    print(f"cc3 subset size: {len(cc3)}")
    print(f"cc3 with >=1 failure (score<1.0): {len(cc3_fail)}")
    print(f"dropped for langdetect (failed type in {sorted(LANGDETECT_IDS)}): "
          f"{len(dropped_ld)}  ids={[r['id'] for r in dropped_ld]}")
    print(f"removed for A=20 overlap (disjointness): {len(overlap)}  "
          f"ids={[r['id'] for r in overlap]}")
    print(f"FINAL reflected n: {len(reflected_recs)}")
    print(f"\nA=20 pool size (ceiling+langdetect stripped): {len(a20_pool)}")
    print(f"A=20 chosen ids: {sorted(a20_ids)}")
    print(f"A=20 score distribution: {dict(sorted(Counter(out['a20_scores']).items()))}")
    print(f"\nbase_score distribution (reflected): "
          f"{dict(sorted(Counter(round(b,4) for b in bases).items()))}")
    print(f"addressability: min={min(addrs):.3f} mean={statistics.mean(addrs):.3f} "
          f"max={max(addrs):.3f}")
    ahist = Counter(round(a, 4) for a in addrs)
    print(f"addressability distribution: {dict(sorted(ahist.items()))}")
    print(f"\naddressability table (parsed, {len(ADDRESSABILITY)} types):")
    for v in (3, 2, 1, 0):
        ts = sorted(t for t, x in ADDRESSABILITY.items() if x == v)
        print(f"  [{v}] {', '.join(ts)}")
    # cost projection
    n = len(reflected_recs)
    calls = n * (1 + A_SIZE) + A_SIZE  # capture-trace + A20-after per ex, + one before
    refl = n
    print(f"\nprojection for --run: ~{calls} task-model calls + {refl} reflection calls "
          f"(N=4). At ~8-15s/task-call/4-wide => roughly {calls*11/4/60:.0f} min order.")
    print("\nSTOP: review, then run with `--run` to execute the reflection loop.")
    return out


# ---------------- run phase ----------------


def _build_adapter():
    import dspy
    from src.ifbench_substrate import ifbench_substrate
    from src.retry import retryable
    from src.run_gepa import _PatchedDspyAdapter, _make_feedback_map, load_lm_configs_from_env

    substrate = ifbench_substrate()
    task_cfg, refl_cfg = load_lm_configs_from_env()
    task_lm = task_cfg.to_lm()
    dspy.settings.configure(lm=task_lm)
    reflection_lm_obj = refl_cfg.to_lm()

    @retryable(max_attempts=5, base_delay=1.0, max_delay=30.0, label="reflection_lm")
    def reflection_lm_callable(x):
        if isinstance(x, str):
            return reflection_lm_obj(prompt=x)
        return reflection_lm_obj(messages=x)

    program = substrate.build_program()
    adapter = _PatchedDspyAdapter(
        student_module=program,
        metric_fn=substrate.metric_fn,
        feedback_map=_make_feedback_map(substrate),
        failure_score=0.0,
        num_threads=4,
        add_format_failure_as_feedback=True,
        rng=random.Random(0),
        reflection_lm=reflection_lm_callable,
        warn_on_score_mismatch=False,
    )
    return adapter


def _mean(scores) -> float:
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores) if scores else 0.0


def run() -> int:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from src.ifbench_data import carve_ifbench_splits

    pf = json.loads(PREFLIGHT.read_text())
    reflected = pf["reflected"]
    a20_ids = pf["a20_ids"]
    seed_cand = {"answer": pf["seed_instruction"]}

    d_feedback = carve_ifbench_splits(seed=0).splits[0]
    by_id = {ex["id"]: ex for ex in d_feedback}
    a20_examples = [by_id[i] for i in a20_ids]
    assert all(i in by_id for i in a20_ids), "A=20 id missing from carve d_feedback"

    adapter = _build_adapter()

    # before = one fresh seed eval on the frozen A=20 (reused for all examples)
    t0 = time.perf_counter()
    before = _mean(adapter.evaluate(a20_examples, seed_cand, capture_traces=False).scores)
    print(f"[before] seed A=20 mean = {before:.4f}  ({time.perf_counter()-t0:.0f}s)")

    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"resuming: {len(done)} already done; {len(reflected)-len(done)} to run")

    for k, rec in enumerate(reflected):
        rid = rec["id"]
        if rid in done:
            continue
        ts = time.perf_counter()
        ex = by_id[rid]
        try:
            ev = adapter.evaluate([ex], seed_cand, capture_traces=True)
            rd = adapter.make_reflective_dataset(seed_cand, ev, ["answer"])
            new = adapter.propose_new_texts(seed_cand, rd, ["answer"])
            proposed = new.get("answer")
            if not proposed:
                raise ValueError("propose_new_texts returned empty 'answer'")
            after = _mean(adapter.evaluate(a20_examples, {"answer": proposed},
                                           capture_traces=False).scores)
            outcome = bool(after > before)
            err = None
        except Exception as e:  # noqa: BLE001 -- log and continue
            proposed, after, outcome, err = None, None, None, f"{type(e).__name__}: {e}"

        row = {
            "id": rid, "example_index": rec["example_index"],
            "base_score": rec["base_score"], "failed_types": rec["failed_types"],
            "addressability": rec["addressability"],
            "before": before, "after": after, "outcome": outcome,
            "proposed_instruction": proposed, "error": err,
        }
        with RESULTS.open("a") as f:
            f.write(json.dumps(row) + "\n")
        msg = f"err={err}" if err else f"after={after:.4f} outcome={outcome}"
        print(f"[{k+1}/{len(reflected)}] id={rid} addr={rec['addressability']:.2f} "
              f"base={rec['base_score']:.3f} {msg}  ({time.perf_counter()-ts:.0f}s)")

    return analyze()


# ---------------- analysis ----------------


def _fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6, iters: int = 100):
    """IRLS logistic regression. X includes intercept column. Returns (beta, ll, cov, ok)."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu)
        WX = X * W[:, None]
        H = X.T @ WX + ridge * np.eye(p)
        g = X.T @ (y - mu) - ridge * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return beta, _loglik(X, y, beta), None, False
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    W = mu * (1 - mu)
    H = X.T @ (X * W[:, None]) + ridge * np.eye(p)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = None
    ok = bool(np.max(np.abs(beta)) < 25)  # crude separation guard
    return beta, _loglik(X, y, beta), cov, ok


def _loglik(X, y, beta) -> float:
    eta = np.clip(X @ beta, -30, 30)
    return float(np.sum(y * eta - np.log1p(np.exp(eta))))


def _chi2_sf_1df(stat: float) -> float:
    if stat <= 0:
        return 1.0
    return math.erfc(math.sqrt(stat / 2.0))


def analyze() -> int:
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    usable = [r for r in rows if r["outcome"] is not None]
    n = len(usable)
    errors = [r for r in rows if r["outcome"] is None]
    y = np.array([1.0 if r["outcome"] else 0.0 for r in usable])
    base = np.array([r["base_score"] for r in usable])
    addr = np.array([r["addressability"] for r in usable])
    before = usable[0]["before"] if usable else None

    L = []
    L.append("# IFBench active-selection validation: results\n")
    L.append("Outcome = proposed instruction (reflect on a single cc3 example, gpt-4.1-mini) "
             "strictly improves the seed's frozen A=20 mean. Regress outcome ~ base_score + "
             "addressability (logistic, numpy IRLS; LR test for the addressability term).\n")
    L.append(f"- reflected examples scored: {n} (errors/skipped: {len(errors)})")
    L.append(f"- seed A=20 'before' mean: {before}")
    if n:
        L.append(f"- outcome base rate (improved): {int(y.sum())}/{n} = {y.mean():.3f}")

    degenerate = (n == 0) or (y.sum() == 0) or (y.sum() == n)
    if degenerate:
        L.append("\n**Degenerate outcome (all-0 or all-1, or n=0): logistic regression not "
                 "estimable. Reporting the base rate only.**")
        print("\n".join(L[-6:]))
    else:
        zb = (base - base.mean()) / (base.std() or 1.0)
        za = (addr - addr.mean()) / (addr.std() or 1.0)
        X_null = np.ones((n, 1))
        X_red = np.column_stack([np.ones(n), zb])
        X_full = np.column_stack([np.ones(n), zb, za])
        b_null, ll_null, _, _ = _fit_logistic(X_null, y)
        b_red, ll_red, _, ok_red = _fit_logistic(X_red, y)
        b_full, ll_full, cov_full, ok_full = _fit_logistic(X_full, y)
        lr = 2 * (ll_full - ll_red)
        p_lr = _chi2_sf_1df(lr)
        mcf_red = 1 - ll_red / ll_null if ll_null else float("nan")
        mcf_full = 1 - ll_full / ll_null if ll_null else float("nan")
        addr_se = math.sqrt(cov_full[2, 2]) if cov_full is not None else float("nan")
        addr_z = b_full[2] / addr_se if addr_se else float("nan")
        addr_pw = math.erfc(abs(addr_z) / math.sqrt(2)) if addr_se else float("nan")

        L.append("\n## Logistic regression (regressors standardized)\n")
        L.append("| model | terms | coefs | logLik | McFadden R2 |")
        L.append("| --- | --- | --- | --- | --- |")
        L.append(f"| reduced | 1, base_score | {np.round(b_red,3).tolist()} | {ll_red:.3f} | {mcf_red:.4f} |")
        L.append(f"| full | 1, base_score, addressability | {np.round(b_full,3).tolist()} | {ll_full:.3f} | {mcf_full:.4f} |")
        L.append("")
        L.append(f"- addressability coef (standardized): {b_full[2]:.4f} "
                 f"(Wald se {addr_se:.4f}, z {addr_z:.3f}, p {addr_pw:.4f})")
        L.append(f"- **Likelihood-ratio test (full vs reduced, 1 dof): "
                 f"LR={lr:.4f}, p={p_lr:.4f}**")
        if not (ok_red and ok_full):
            L.append("- WARNING: possible quasi-separation (large coefficients); treat the "
                     "fit with caution.")
        L.append("")
        finding = ("addressability ADDS predictive power over base_score "
                   f"(LR p={p_lr:.4f} < 0.05)" if p_lr < 0.05 else
                   "addressability does NOT add predictive power over base_score "
                   f"(LR p={p_lr:.4f} >= 0.05); the scalar base score is sufficient")
        L.append(f"## Finding\n{finding}.")
        print("\n".join(L[-12:]))

    if errors:
        L.append("\n## Errors\n" + "\n".join(f"- id={r['id']}: {r['error']}" for r in errors))
    REPORT.write_text("\n".join(L))
    print(f"\nwrote {REPORT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="execute the reflection loop (real LLM)")
    ap.add_argument("--analyze-only", action="store_true", help="re-run regression on existing results")
    args = ap.parse_args()
    if args.analyze_only:
        return analyze()
    if args.run:
        if not PREFLIGHT.exists():
            print("No preflight file; run pre-flight first (no flag).")
            return 1
        return run()
    preflight()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
