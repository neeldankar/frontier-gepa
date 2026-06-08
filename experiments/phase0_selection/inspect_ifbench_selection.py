"""Phase-0 inspection for the IFBench active-selection follow-up.

INSPECTION ONLY. No GEPA loop, no reflection, no experiment arm, no LLM calls.
We reuse the Chunk-13 cached seed-program responses (Qwen2.5-7B-Instruct-Turbo
@ temp 0.6, seed_splits=0, 150 D_feedback from allenai/IF_multi_constraints_upto5)
and recompute the two missing pieces -- the feedback string and the per-constraint
pass/fail -- deterministically via the IFEval verifiers (no model).

Answers two questions (reports numbers; makes NO GO/NO-GO call):
  (A) Is the base-score distribution viable or degenerate? (bimodal / unimodal-near-0.5)
  (B) Is structured failure info already in the feedback (free proxy) or vague?

Run: .venv/bin/python phase0/inspect_ifbench_selection.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE0 = Path(__file__).resolve().parent
REPO = PHASE0.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ifbench_feedback import _verify_one, score_and_feedback  # noqa: E402

RECORDS_PATH = REPO / "results" / "ifbench" / "d_feedback_records.json"
TABLE_PATH = REPO / "results" / "ifbench" / "difficulty_table.json"

INSPECTION_JSON = PHASE0 / "ifbench_inspection.json"
HIST_PNG = PHASE0 / "ifbench_base_score_hist.png"
SMOKE_TXT = PHASE0 / "smoke20_feedback.txt"
REPORT_MD = PHASE0 / "PHASE0_REPORT.md"

FLOOR = 0.01
CEIL = 0.99
N_SMOKE = 20

# Regex for the rendered feedback's "Violated" block: lines like
#   "  - length_constraints:number_paragraphs (num_paragraphs=7)"
# capture the instruction_id (first token before optional " (").
_VIOLATED_LINE = re.compile(r"^\s*-\s+(\S+)")


def parse_failed_from_feedback(feedback_text: str) -> list[str]:
    """Extract failed instruction_ids from the feedback STRING alone (regex,
    no LLM). Only lines inside the 'Violated (N):' block are bullet lines."""
    out: list[str] = []
    in_violated = False
    for line in feedback_text.splitlines():
        s = line.strip()
        if s.startswith("Violated ("):
            in_violated = s != "Violated (0): none."
            continue
        if s.startswith("Satisfied (") or s.startswith("Score "):
            in_violated = False
            continue
        if in_violated:
            m = _VIOLATED_LINE.match(line)
            if m:
                out.append(m.group(1))
    return out


def main() -> int:
    records = json.loads(RECORDS_PATH.read_text())  # dict, insertion = D_feedback order
    table = json.loads(TABLE_PATH.read_text())
    table_scores = [float(s) for s in table["scores"]]

    items = list(records.items())  # [(id_str, record), ...] in file order
    n = len(items)

    # Verifiers known to be non-deterministic across runs (langdetect seeds its
    # RNG from the environment, so language:response_language can flip run-to-run).
    NONDET_IDS = {"language:response_language"}

    results: list[dict] = []
    score_match_cached = 0
    cache_mismatches: list[dict] = []
    for i, (id_str, rec) in enumerate(items):
        response = str(rec.get("response", "") or "")
        out = score_and_feedback(rec, {"response": response})
        score = float(out["score"])

        per_constraint = []
        for iid, kw in zip(rec["instruction_id_list"], rec["kwargs_list"]):
            per_constraint.append(
                {"instruction_id": iid, "kwargs": kw, "passed": _verify_one(iid, kw, response)}
            )

        if abs(score - float(rec["score"])) < 1e-9:
            score_match_cached += 1
        else:
            cache_mismatches.append({
                "example_index": i, "id": rec["id"],
                "cached_score": float(rec["score"]), "recomputed_score": score,
                "constraint_ids": list(rec["instruction_id_list"]),
                "nondeterministic_constraints": [
                    c for c in rec["instruction_id_list"] if c in NONDET_IDS
                ],
            })

        results.append({
            "example_index": i,
            "id": rec["id"],
            "n_constraints": int(rec["constraint_count"]),
            "score": score,
            "n_satisfied": len(out["satisfied"]),
            "n_violations": len(out["violations"]),
            "per_constraint": per_constraint,
            "feedback_text": out["feedback"],
            "feedback_len": len(out["feedback"]),
            "violations_ids": [v["instruction_id"] for v in out["violations"]],
        })

    INSPECTION_JSON.write_text(json.dumps(results, indent=2))

    # ---- consistency: multisets (ordering-independent) ----
    cached_scores = [float(r["score"]) for _, r in items]
    rec_scores = [r["score"] for r in results]
    cached_ms = Counter(round(s, 4) for s in cached_scores)
    table_ms = Counter(round(s, 4) for s in table_scores)
    recomp_ms = Counter(round(s, 4) for s in rec_scores)
    cached_eq_table = cached_ms == table_ms
    recomp_eq_cached_ms = recomp_ms == cached_ms

    # Nondeterminism surface: examples whose constraint set includes a
    # non-deterministic verifier (langdetect). This count is run-independent;
    # whether a given run flips depends on langdetect's per-run RNG.
    langdetect_examples = [
        {"example_index": i, "id": rec["id"]}
        for i, (_, rec) in enumerate(items)
        if any(c in NONDET_IDS for c in rec["instruction_id_list"])
    ]

    # ---- (A) distribution ----
    # Use the FROZEN cached scores (the canonical Chunk-13 difficulty-table
    # basis) so the headline numbers are stable; the recompute is only the
    # source for feedback text and is subject to the langdetect flip noted above.
    scores = cached_scores
    n_floor = sum(1 for s in scores if s <= FLOOR)
    n_ceil = sum(1 for s in scores if s >= CEIL)
    n_mid = sum(1 for s in scores if FLOOR < s < CEIL)
    strict_all_or_nothing = sum(1 for s in scores if s >= CEIL)  # all constraints satisfied
    value_counts = Counter(round(s, 4) for s in scores)
    cc_counts = Counter(r["n_constraints"] for r in results)
    mean_s = statistics.mean(scores)
    std_s = statistics.pstdev(scores)
    qs = statistics.quantiles(scores, n=4) if len(scores) >= 2 else [0, 0, 0]

    # Kill-condition logic (descriptive thresholds, stated explicitly):
    #  bimodal: extremes hold most mass AND middle is sparse.
    #  unimodal-near-0.5: tight cluster near 0.5 with little spread.
    frac_extreme = (n_floor + n_ceil) / n
    frac_mid = n_mid / n
    bimodal = frac_extreme >= 0.80 and frac_mid <= 0.20
    near_half_mass = sum(1 for s in scores if 0.4 <= s <= 0.6) / n
    unimodal_near_half = near_half_mass >= 0.80 and std_s < 0.10
    if bimodal:
        kill = "bimodal (mass at 0/1)"
    elif unimodal_near_half:
        kill = "unimodal-near-0.5 (no spread)"
    else:
        kill = "none"

    # histogram (discrete fractional values)
    vals = sorted(value_counts)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(v) for v in vals], [value_counts[v] for v in vals], color="#1f77b4")
    ax.set_xlabel("fractional base score (fraction of constraints satisfied)")
    ax.set_ylabel("count (of 150 D_feedback)")
    ax.set_title(
        f"IFBench seed-program base scores (n={n})\n"
        f"floor(<= {FLOOR})={n_floor}  middle={n_mid}  ceiling(>= {CEIL})={n_ceil}  "
        f"kill={kill}"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HIST_PNG, dpi=150)
    plt.close(fig)

    # ---- (B) feedback parseability demo (round-trip) ----
    parse_examples = []
    for r in results:
        if r["n_violations"] > 0:
            parsed = parse_failed_from_feedback(r["feedback_text"])
            parse_examples.append({
                "example_index": r["example_index"],
                "score": r["score"],
                "feedback_text": r["feedback_text"],
                "parsed_failed_ids": parsed,
                "verifier_violation_ids": r["violations_ids"],
                "roundtrip_match": parsed == r["violations_ids"],
            })
        if len(parse_examples) >= 4:
            break
    roundtrip_all = all(
        parse_failed_from_feedback(r["feedback_text"]) == r["violations_ids"]
        for r in results
    )

    # ---- smoke 20: full raw feedback ----
    smoke_lines = []
    for r in results[:N_SMOKE]:
        pc = " ".join(
            f"{c['instruction_id']}={'P' if c['passed'] else 'F'}" for c in r["per_constraint"]
        )
        smoke_lines.append(
            f"--- idx={r['example_index']} id={r['id']} n_constraints={r['n_constraints']} "
            f"score={r['score']:.4f} feedback_len={r['feedback_len']}\n"
            f"per-constraint: {pc}\n"
            f"feedback_text:\n{r['feedback_text']}\n"
        )
    SMOKE_TXT.write_text("\n".join(smoke_lines))

    # ---- report ----
    md = []
    md.append("# Phase-0 IFBench inspection (reuse of cached Chunk-13 responses)\n")
    md.append("INSPECTION ONLY -- no GEPA, no LLM calls. Feedback + per-constraint pass/fail "
              "recomputed deterministically from cached seed-program responses "
              "(Qwen2.5-7B-Instruct-Turbo @ temp 0.6, seed 0, 150 D_feedback). "
              "**No GO/NO-GO call here -- that is the operator's decision.**\n")
    md.append("## Consistency check\n")
    md.append(f"- recomputed score == cached record score (per-record): {score_match_cached}/{n}")
    md.append(f"- cached scores == difficulty_table scores (multiset, order-independent): "
              f"{cached_eq_table}  -- the frozen table IS these cached scores")
    md.append(f"- recomputed == cached (multiset): {recomp_eq_cached_ms} "
              f"(differs by exactly the {len(cache_mismatches)} flip below)")
    if cache_mismatches:
        md.append(f"- recompute-vs-cache mismatches ({len(cache_mismatches)}): a known "
                  "non-deterministic verifier (langdetect) flips across runs:")
        for m in cache_mismatches:
            md.append(f"  - idx={m['example_index']} id={m['id']} "
                      f"cached={m['cached_score']} -> recomputed={m['recomputed_score']}; "
                      f"nondeterministic constraint(s): {m['nondeterministic_constraints']}")
    md.append("- Note: `records.json` file order != `difficulty_table.scores` order, so an "
              "index-aligned table compare is meaningless; the multiset check above is the "
              "correct one.")
    md.append(f"- Nondeterminism surface: {len(langdetect_examples)}/{n} examples include the "
              "langdetect verifier `language:response_language` "
              f"(ids {[e['id'] for e in langdetect_examples]}). Across two runs of this "
              "script the recompute matched the cache 150/150 and 149/150; the single flip "
              "was id=36653 (score 0.2 vs 0.0). That one verifier is not run-stable; all "
              "others are deterministic.\n")
    md.append("## (A) Base-score distribution\n")
    md.append("(Computed from the frozen cached Chunk-13 scores -- the canonical "
              "difficulty-table basis -- so these are stable run-to-run.)\n")
    md.append(f"- n = {n}")
    md.append(f"- floor (score <= {FLOOR}): {n_floor} ({n_floor/n:.1%})")
    md.append(f"- middle ({FLOOR} < score < {CEIL}): {n_mid} ({n_mid/n:.1%})")
    md.append(f"- ceiling (score >= {CEIL}): {n_ceil} ({n_ceil/n:.1%})")
    md.append(f"- strict all-or-nothing rate (all constraints satisfied, score==1): "
              f"{strict_all_or_nothing} ({strict_all_or_nothing/n:.1%})")
    md.append(f"- mean={mean_s:.4f}  std={std_s:.4f}  quartiles(Q1,Q2,Q3)="
              f"({qs[0]:.3f}, {qs[1]:.3f}, {qs[2]:.3f})")
    md.append(f"- fraction within [0.4,0.6]: {near_half_mass:.1%}")
    md.append(f"- distinct score values ({len(vals)}): "
              + ", ".join(f"{v}:{value_counts[v]}" for v in vals))
    md.append(f"- n_constraints distribution: "
              + ", ".join(f"{k}:{cc_counts[k]}" for k in sorted(cc_counts))
              + f"  (range {min(cc_counts)}-{max(cc_counts)})")
    md.append(f"- **kill condition triggered: {kill}**")
    md.append(f"  - bimodal test (extremes>=80% AND middle<=20%): {bimodal} "
              f"(extremes={frac_extreme:.1%}, middle={frac_mid:.1%})")
    md.append(f"  - unimodal-near-0.5 test ([0.4,0.6]>=80% AND std<0.10): {unimodal_near_half} "
              f"([0.4,0.6]={near_half_mass:.1%}, std={std_s:.4f})\n")
    md.append(f"Histogram: `{HIST_PNG.relative_to(REPO)}`\n")
    md.append("## (B) Feedback parseability (free proxy?)\n")
    md.append("- Structured per-constraint pass/fail is available DIRECTLY from the "
              "verifiers (`score_and_feedback` returns `violations`/`satisfied` lists; "
              "`_verify_one` is a pure boolean check). No LLM, no string parsing needed.")
    md.append("- The rendered feedback STRING also names every failed constraint as "
              "`  - instruction_id (k=v, ...)` under a `Violated (N):` header, so it is "
              "regex-parseable too.")
    md.append(f"- Round-trip parse (regex over feedback string == verifier violation ids) "
              f"matches for ALL {n} examples: {roundtrip_all}")
    md.append("- Caveat: the per-constraint labels are deterministic EXCEPT the langdetect "
              "verifier `language:response_language`, which can flip across runs "
              f"({len(cache_mismatches)}/{n} examples differed from the cached scoring). The "
              "proxy is otherwise exact and LLM-free.\n")
    md.append("### Worked parse examples\n")
    for pe in parse_examples:
        md.append(f"- idx={pe['example_index']} score={pe['score']:.4f}  "
                  f"roundtrip_match={pe['roundtrip_match']}")
        md.append(f"  - parsed failed ids: {pe['parsed_failed_ids']}")
        md.append(f"  - verifier violations: {pe['verifier_violation_ids']}")
        md.append("  - feedback_text:")
        md.append("    ```\n    " + pe["feedback_text"].replace("\n", "\n    ") + "\n    ```")
    md.append(f"\n## Artifacts\n- `{INSPECTION_JSON.relative_to(REPO)}` (150 results)")
    md.append(f"- `{SMOKE_TXT.relative_to(REPO)}` (20 full raw feedbacks)")
    md.append(f"- `{HIST_PNG.relative_to(REPO)}`")
    REPORT_MD.write_text("\n".join(md))

    # ---- stdout ----
    print("=" * 70)
    print("PHASE-0 IFBench inspection (no LLM; recomputed from cached responses)")
    print("=" * 70)
    print(f"consistency: recomputed==cached per-record {score_match_cached}/{n}; "
          f"cached==table(multiset) {cached_eq_table}; "
          f"mismatches={len(cache_mismatches)} "
          f"(langdetect flip: {[m['id'] for m in cache_mismatches]})")
    print(f"\n(A) distribution: floor(<= {FLOOR})={n_floor} ({n_floor/n:.1%})  "
          f"middle={n_mid} ({n_mid/n:.1%})  ceiling(>= {CEIL})={n_ceil} ({n_ceil/n:.1%})")
    print(f"    strict all-or-nothing (score==1) = {strict_all_or_nothing} "
          f"({strict_all_or_nothing/n:.1%})")
    print(f"    mean={mean_s:.4f} std={std_s:.4f}  [0.4,0.6]={near_half_mass:.1%}")
    print(f"    distinct values: " + ", ".join(f"{v}:{value_counts[v]}" for v in vals))
    print(f"    n_constraints: " + ", ".join(f"{k}:{cc_counts[k]}" for k in sorted(cc_counts)))
    print(f"    KILL CONDITION: {kill}")
    print(f"\n(B) structured failure info present (free proxy): YES "
          f"(violations list direct; feedback string regex round-trip all-match={roundtrip_all})")
    print("\n" + "=" * 70)
    print(f"SMOKE: first {N_SMOKE} examples -- full raw feedback_text")
    print("=" * 70)
    print(SMOKE_TXT.read_text())
    print("=" * 70)
    print("Worked parse examples (regex over feedback string vs verifier violations):")
    for pe in parse_examples:
        print(f"  idx={pe['example_index']} match={pe['roundtrip_match']}  "
              f"parsed={pe['parsed_failed_ids']}  verifier={pe['verifier_violation_ids']}")
    print(f"\nwrote: {INSPECTION_JSON.name}, {HIST_PNG.name}, {SMOKE_TXT.name}, {REPORT_MD.name}")
    print("\nNO GO/NO-GO decision made -- that is the operator's call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
