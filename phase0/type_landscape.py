"""Phase-0 constraint-type landscape (descriptive only).

INSPECTION ONLY: no GEPA, no LLM, no re-scoring. Reads the cached
phase0/ifbench_inspection.json (150 records) and tabulates the constraint-type
landscape. Descriptions are written from each verifier's check_following()
logic in src/ifbench_verifiers/instructions.py (read by hand, embedded below),
NOT inferred from the id name.

No taxonomy, no "actionable" grouping, no GO/NO-GO. Just tables + descriptions.

Outputs: phase0/type_landscape.md, phase0/type_frequency.csv
Run: .venv/bin/python phase0/type_landscape.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PHASE0 = Path(__file__).resolve().parent
INSPECTION = PHASE0 / "ifbench_inspection.json"
OUT_MD = PHASE0 / "type_landscape.md"
OUT_CSV = PHASE0 / "type_frequency.csv"

# One-line descriptions read from each verifier's check_following() logic
# (src/ifbench_verifiers/instructions.py), not from the id name.
DESCRIPTIONS: dict[str, str] = {
    "keywords:existence": "response must contain ALL of the given keywords (case-insensitive regex search)",
    "keywords:frequency": "a given keyword must occur < N times ('less than') or >= N times ('at least'), counted by regex",
    "keywords:forbidden_words": "NONE of the given forbidden words may appear (whole-word, case-insensitive)",
    "keywords:letter_frequency": "a given single letter must occur < N or >= N times across the whole response (case-insensitive char count)",
    "language:response_language": "langdetect-detected language of the whole response must equal the given code (langdetect; non-deterministic across runs)",
    "length_constraints:number_sentences": "sentence count must be < N ('less than') or >= N ('at least')",
    "length_constraints:number_paragraphs": "number of paragraphs split on the '***' divider must equal exactly N",
    "length_constraints:number_words": "word count must be < N ('less than') or >= N ('at least')",
    "length_constraints:nth_paragraph_first_word": "must have exactly N paragraphs (split on blank lines) AND the nth paragraph's first word equals a given word",
    "detectable_content:number_placeholders": "must contain >= N bracketed [placeholders] (regex on [...])",
    "detectable_content:postscript": "must contain a postscript section starting with the given marker (e.g. 'P.S.' / 'P.P.S')",
    "detectable_format:number_bullet_lists": "number of markdown bullet lines (lines starting '*' or '-') must equal exactly N",
    "detectable_format:constrained_response": "response must contain one of a fixed set of allowed answer options",
    "detectable_format:multiple_sections": "must contain >= N sections delimited by '<spliter> <number>' (e.g. 'SECTION 1')",
    "detectable_format:number_highlighted_sections": "must contain >= N markdown-highlighted spans (text wrapped in *...* or **...**)",
    "detectable_format:json_format": "the whole response (stripped of ``` fences) must parse as valid JSON",
    "detectable_format:title": "must contain a non-empty title wrapped in << >>",
    "combination:two_responses": "must contain exactly two distinct responses separated by '******'",
    "combination:repeat_prompt": "response must start with the given prompt text verbatim (then answer)",
    "startend:end_checker": "response (stripped of quotes/case) must end with the given phrase",
    "change_case:capital_word_frequency": "count of ALL-CAPS words must be < N ('less than') or >= N (else), nltk tokenized",
    "change_case:english_capital": "the entire response must be uppercase AND langdetect-detected as English (langdetect; non-deterministic)",
    "change_case:english_lowercase": "the entire response must be lowercase AND langdetect-detected as English (langdetect; non-deterministic)",
    "punctuation:no_comma": "the response must contain no commas",
    "startend:quotation": "the response must be wrapped in double-quote characters at both start and end",
    "copy:repeat_phrase": "the given phrase must appear exactly small_n times, each occurrence with exactly one word changed",
    "copy:copy": "the response must equal the given prompt text verbatim (stripped, case-insensitive)",
    "new:copy_span_idx": "the response must equal the substring prompt[n_start:n_end] verbatim (stripped, case-insensitive)",
    "detectable_format:sentence_hyphens": "all sentences must be joined by hyphens with no spaces between them",
    "keywords:no_adjacent_consecutive": "no two adjacent words may start with consecutive letters of the alphabet",
    "detectable_format:square_brackets": "EVERY word in the response must be enclosed in [square brackets]",
    "keywords:word_once": "a given keyword must occur exactly once (regex count == 1)",
    "keywords:word_count_different_numbers": "a given keyword must occur < N or >= N times (same logic as keywords:frequency)",
    "keywords:exclude_word_harder": "the given keyword must not appear as a space-delimited token (' keyword ')",
    "paragraphs:paragraphs": "must have exactly 2 paragraphs separated by the '***' divider",
    "paragraphs:paragraphs2": "must have exactly 2 paragraphs separated by a blank line (\\n\\n)",
    "first_word:first_word_sent": "the first word of EVERY sentence must equal the given word",
    "first_word:first_word_answer": "the first word of the whole response must equal the given word",
    "last_word:last_word_sent": "the last word of EVERY sentence (punctuation stripped) must equal the given word",
    "last_word:last_word_answer": "the last word of the whole response (punctuation stripped) must equal the given word",
    "detectable_format:bigram_wrapping": "words taken in non-overlapping pairs must be wrapped so each pair starts with '<<' and ends with '>>'",
    "copy:copying_simple": "the response must equal the given prompt verbatim (stripped, case-insensitive)",
    "copy:copying_multiple": "the response must be the given prompt repeated exactly N times, separated by '******'",
    "punctuation:punctuation_dot": "the response must contain no '.' (dot) characters",
    "punctuation:punctuation_exclamation": "the response must contain no '!' characters",
    "count:lowercase_counting": "the number of all-lowercase words must be <= N",
    "letters:letter_counting": "the total count of alphabetic letters must be >= N ('at least') or < N ('less than')",
    "letters:letter_counting2": "a given single letter must occur < N or >= N times (case-insensitive char count; same class as keywords:letter_frequency)",
    "count:counting_composition": "must have exactly 3 '***'-split paragraphs, each with exactly n_sent sentences, each sentence exactly n_words words",
    "count:count_unique": "every tokenized word in the response must be unique (no repeated tokens at all)",
    "count:count_increment_word": "keyword1 must occur exactly once AND keyword2 exactly twice",
    "keywords:palindrome": "the response must contain at least one palindrome word",
    "keywords:keyword_specific_position": "the m-th word of the n-th sentence must equal the given keyword",
    "keywords:start_end": "the response's first and last token (nltk) must be the same word",
}


def main() -> int:
    data = json.loads(INSPECTION.read_text())
    n = len(data)

    # per-example score and constraint count
    ex_score = {r["example_index"]: r["score"] for r in data}
    ex_ncons = {r["example_index"]: r["n_constraints"] for r in data}

    # violated / satisfied example membership per id
    fail_examples: dict[str, list[int]] = defaultdict(list)
    sat_examples: dict[str, list[int]] = defaultdict(list)
    for r in data:
        idx = r["example_index"]
        seen_v, seen_s = set(), set()
        for c in r["per_constraint"]:
            iid = c["instruction_id"]
            if c["passed"]:
                if iid not in seen_s:
                    sat_examples[iid].append(idx); seen_s.add(iid)
            else:
                if iid not in seen_v:
                    fail_examples[iid].append(idx); seen_v.add(iid)

    all_ids = sorted(set(fail_examples) | set(sat_examples))
    missing_desc = [i for i in all_ids if i not in DESCRIPTIONS]

    # ---- (1) violation-type frequency table ----
    freq_rows = []
    for iid in all_ids:
        fails = fail_examples.get(iid, [])
        if not fails:
            continue  # this table is violations only; section (3) covers the rest
        scores = [ex_score[i] for i in fails]
        freq_rows.append({
            "instruction_id": iid,
            "n_fail_examples": len(fails),
            "n_satisfied_examples": len(sat_examples.get(iid, [])),
            "mean_score_when_failing": round(statistics.mean(scores), 4),
            "min_score_when_failing": round(min(scores), 4),
            "max_score_when_failing": round(max(scores), 4),
            "description": DESCRIPTIONS.get(iid, "(MISSING DESCRIPTION)"),
        })
    freq_rows.sort(key=lambda r: (-r["n_fail_examples"], r["instruction_id"]))

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(freq_rows[0].keys()))
        w.writeheader()
        w.writerows(freq_rows)

    # ---- (3) full universe ----
    universe = []
    for iid in all_ids:
        nv, ns = len(fail_examples.get(iid, [])), len(sat_examples.get(iid, []))
        kind = "violated-only" if ns == 0 else ("satisfied-only" if nv == 0 else "mixed")
        universe.append((iid, nv, ns, kind))
    sat_only = [u for u in universe if u[3] == "satisfied-only"]
    viol_only = [u for u in universe if u[3] == "violated-only"]

    # ---- (4) per-example structure ----
    vcount = Counter(sum(1 for c in r["per_constraint"] if not c["passed"]) for r in data)
    vtypes = Counter(len({c["instruction_id"] for c in r["per_constraint"] if not c["passed"]}) for r in data)

    # ---- (5) score x n_constraints cross-tab ----
    crosstab: dict[float, Counter] = defaultdict(Counter)
    for r in data:
        crosstab[round(r["score"], 4)][r["n_constraints"]] += 1
    ncons_vals = sorted({r["n_constraints"] for r in data})
    score_vals = sorted(crosstab)

    # ---- write md ----
    L = []
    L.append("# Phase-0 IFBench constraint-type landscape\n")
    L.append(f"Descriptive tabulation of `phase0/ifbench_inspection.json` (n={n}). "
             "INSPECTION ONLY: no GEPA, no LLM, no re-scoring. Type descriptions are read "
             "from each verifier's `check_following()` in "
             "`src/ifbench_verifiers/instructions.py`. No taxonomy, no grouping, no "
             "GO/NO-GO -- tables and descriptions only.\n")
    L.append(f"- distinct constraint types present: {len(all_ids)} (the full catalog)")
    L.append(f"- types appearing in any violations list: {len(freq_rows)}")
    L.append(f"- types appearing only as satisfied (never violated): {len(sat_only)}")
    if missing_desc:
        L.append(f"- WARNING missing descriptions: {missing_desc}")
    L.append("")

    L.append("## (1) Violation-type frequency table\n")
    L.append("Per type that appears in any example's violations: how many of the 150 "
             "examples fail on it, the satisfied-example count for contrast, and the "
             "base-score spread (mean/min/max) of the examples that fail on it. Sorted by "
             "fail count. Also in `type_frequency.csv`.\n")
    L.append("| instruction_id | n_fail | n_sat | mean_score(fail) | min | max | what check_following() checks |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in freq_rows:
        L.append(f"| {r['instruction_id']} | {r['n_fail_examples']} | "
                 f"{r['n_satisfied_examples']} | {r['mean_score_when_failing']} | "
                 f"{r['min_score_when_failing']} | {r['max_score_when_failing']} | "
                 f"{r['description']} |")
    L.append("")

    L.append("## (3) Full type universe\n")
    L.append(f"All {len(all_ids)} distinct types present, with violated/satisfied example "
             "counts and whether each was ever violated.\n")
    L.append("| instruction_id | n_violated | n_satisfied | appears_as | what check_following() checks |")
    L.append("| --- | --- | --- | --- | --- |")
    for iid, nv, ns, kind in sorted(universe, key=lambda u: (-(u[1] + u[2]), u[0])):
        L.append(f"| {iid} | {nv} | {ns} | {kind} | {DESCRIPTIONS.get(iid,'(MISSING)')} |")
    L.append("")
    L.append(f"Satisfied-only (never violated, {len(sat_only)}): "
             + ", ".join(u[0] for u in sat_only))
    L.append("")
    L.append(f"Violated-only (never satisfied, {len(viol_only)}): "
             + ", ".join(u[0] for u in viol_only))
    L.append("")

    L.append("## (4) Per-example structure\n")
    L.append("Distribution of violation COUNT per example (how many constraints each "
             "example fails):\n")
    L.append("| n_violations | n_examples |")
    L.append("| --- | --- |")
    for k in sorted(vcount):
        L.append(f"| {k} | {vcount[k]} |")
    L.append("")
    L.append("Distribution of DISTINCT violation types per example:\n")
    L.append("| n_distinct_violation_types | n_examples |")
    L.append("| --- | --- |")
    for k in sorted(vtypes):
        L.append(f"| {k} | {vtypes[k]} |")
    L.append("")
    L.append(f"(Note: per-example violation count and distinct-type count are identical "
             f"here -> within an example the violated constraints are all distinct types: "
             f"{vcount == vtypes}.)")
    L.append("")

    L.append("## (5) score x n_constraints cross-tab\n")
    L.append("Rows = fractional base score; columns = number of constraints in the example; "
             "cells = example count. Confirms which fractions come from 3/4/5-constraint "
             "examples.\n")
    L.append("| score | " + " | ".join(f"cc={c}" for c in ncons_vals) + " | row_total |")
    L.append("| --- | " + " | ".join("---" for _ in ncons_vals) + " | --- |")
    col_tot = Counter()
    for s in score_vals:
        row = crosstab[s]
        cells = [row.get(c, 0) for c in ncons_vals]
        for c in ncons_vals:
            col_tot[c] += row.get(c, 0)
        L.append(f"| {s} | " + " | ".join(str(x) for x in cells) + f" | {sum(cells)} |")
    L.append("| **total** | " + " | ".join(str(col_tot[c]) for c in ncons_vals)
             + f" | {sum(col_tot.values())} |")
    L.append("")

    OUT_MD.write_text("\n".join(L))

    # ---- stdout ----
    print(f"distinct types present: {len(all_ids)}  (violated-in-any: {len(freq_rows)}, "
          f"satisfied-only: {len(sat_only)}, violated-only: {len(viol_only)})")
    if missing_desc:
        print("MISSING DESCRIPTIONS:", missing_desc)
    else:
        print("all types have a check_following()-derived description")
    print("\ntop violation types (id: n_fail, mean_score_fail [min,max]):")
    for r in freq_rows[:12]:
        print(f"  {r['instruction_id']}: {r['n_fail_examples']} "
              f"mean={r['mean_score_when_failing']} [{r['min_score_when_failing']},"
              f"{r['max_score_when_failing']}]")
    print("\nper-example violation-count distribution:", dict(sorted(vcount.items())))
    print("distinct-violation-type distribution:    ", dict(sorted(vtypes.items())))
    print("\nscore x n_constraints cross-tab:")
    print("  score   " + "  ".join(f"cc{c}" for c in ncons_vals) + "   total")
    for s in score_vals:
        row = crosstab[s]
        print(f"  {s:<6} " + "   ".join(f"{row.get(c,0):>3}" for c in ncons_vals)
              + f"   {sum(row.values())}")
    print(f"\nwrote {OUT_MD.name}, {OUT_CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
