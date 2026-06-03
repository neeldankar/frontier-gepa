"""IFBench data loader, dataset gate, and disjoint splits for Experiment 2.

**Substrate pivot from Chunk-11 v1.** The first cut targeted
`allenai/IFBench_test` and hit OPERATOR_REVIEW: only 300 rows, 256 of
which are single-constraint, so its score distribution would be
bimodal-by-construction and the strict-partial frontier would be
structurally bounded by the 44 multi-constraint rows. That reproduces
the HotpotQA thin-frontier regime Experiment 2 exists to escape. The
substrate is therefore switched to `allenai/IF_multi_constraints_upto5`
(IF-RLVR composite, ~95k rows, up to 5 constraints per instruction;
constraints from IFEval (25) + IFBench-Train (29)). See DEVIATIONS.md
entry 6.

Schema (IF_multi_constraints_upto5):
  - `key`:              source-row provenance string (NOT guaranteed
                        unique across rows; we key disjointness by
                        filtered-pool index instead).
  - `messages`:         chat-style list `[{role, content}]`; for this
                        dataset, all rows have a single role='user'
                        entry.
  - `ground_truth`:     a Python-literal-string list of one dict
                        `[{'instruction_id': [...], 'kwargs': [...]}]`
                        with parallel `instruction_id` and `kwargs`
                        arrays; the constraint count of a row is
                        `len(instruction_id)`.
  - `constraint_type`:  always `'multi'` for this dataset.
  - `constraint`:       human-readable concatenated constraint
                        description (informational; not consumed by
                        verifiers).
  - `dataset`:          always `'ifeval'` for this dataset (the IF-RLVR
                        composite tags its source family).

Per-row record produced by this loader:
  - `id`:               sequential integer in the filtered (>=floor)
                        pool; the canonical disjoint-id field.
  - `source_key`:       raw IF_multi_constraints_upto5 `key`; provenance
                        only.
  - `prompt`:           the user message content.
  - `instruction_id_list`: list of constraint IDs (the verifier names).
  - `kwargs_list`:      parallel list of constraint kwargs.
  - `constraint_count`: `len(instruction_id_list)`.

Constraint-count floor: BUILD_PLAN §7 Chunk-11-redo specifies ">=3 if
that comfortably exceeds 545, else >=2". The live count at >=3 is
48,463 rows (well above 545), so the floor is 3.

Verifier-availability check: every constraint identifier in carved rows
must appear in `KNOWN_VERIFIER_IDS`. That set is the curated catalog of
the 54 IFEval + IFBench-Train instruction IDs canonical for this
dataset (sourced by enumerating the dataset's universe at the time of
Chunk-11-redo, cross-checked against the count user-specified: 25
IFEval + 29 IFBench-Train = 54). If a future dataset version introduces
a new ID, that row is excluded from the pools at carve time and
reported by name. Chunk 12 implements the actual verifier functions
against this exact list.
"""

from __future__ import annotations

import ast
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy
from datasets import load_dataset

# Canonical dataset id and variant. Single source of truth.
IFBENCH_DATASET_ID: str = "allenai/IF_multi_constraints_upto5"
IFBENCH_SPLIT_NAME: str = "train"
IFBENCH_FAMILY: str = (
    "AllenAI IF-RLVR composite (IF_multi_constraints_upto5; "
    "IFEval 25 + IFBench-Train 29 verifiable constraints)"
)

# Constraint-count floor for the substrate. Chunk-11-redo requirement:
# >=3 if that exceeds 545, else >=2. Live count at >=3 is 48,463 rows.
CONSTRAINT_COUNT_FLOOR: int = 3

# Pool layout. Full Experiment-1 target sizes; no proportional shrink
# needed at 95k.
TARGET_SIZES: dict[str, int] = {
    "d_feedback": 150,
    "accept_batch": 20,
    "d_pareto": 75,
    "test": 300,
}
SPLIT_ORDER: tuple[str, ...] = ("d_feedback", "accept_batch", "d_pareto", "test")
TARGET_TOTAL: int = sum(TARGET_SIZES.values())

INPUT_FIELDS: tuple[str, ...] = ("prompt", "instruction_id_list", "kwargs_list")

# The 54-element curated catalog of IFEval + IFBench-Train verifier IDs.
# This is the implementation target for Chunk 12. Sourced by enumerating
# `allenai/IF_multi_constraints_upto5` at Chunk-11-redo time.
KNOWN_VERIFIER_IDS: frozenset[str] = frozenset({
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "combination:repeat_prompt",
    "combination:two_responses",
    "copy:copy",
    "copy:copying_multiple",
    "copy:copying_simple",
    "copy:repeat_phrase",
    "count:count_increment_word",
    "count:count_unique",
    "count:counting_composition",
    "count:lowercase_counting",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:bigram_wrapping",
    "detectable_format:constrained_response",
    "detectable_format:json_format",
    "detectable_format:multiple_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:sentence_hyphens",
    "detectable_format:square_brackets",
    "detectable_format:title",
    "first_word:first_word_answer",
    "first_word:first_word_sent",
    "keywords:exclude_word_harder",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:keyword_specific_position",
    "keywords:letter_frequency",
    "keywords:no_adjacent_consecutive",
    "keywords:palindrome",
    "keywords:start_end",
    "keywords:word_count_different_numbers",
    "keywords:word_once",
    "language:response_language",
    "last_word:last_word_answer",
    "last_word:last_word_sent",
    "length_constraints:nth_paragraph_first_word",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "letters:letter_counting",
    "letters:letter_counting2",
    "new:copy_span_idx",
    "paragraphs:paragraphs",
    "paragraphs:paragraphs2",
    "punctuation:no_comma",
    "punctuation:punctuation_dot",
    "punctuation:punctuation_exclamation",
    "startend:end_checker",
    "startend:quotation",
})
assert len(KNOWN_VERIFIER_IDS) == 54, "verifier catalog size drift"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IFBenchUsable:
    """Result of loading IF_multi_constraints_upto5 and filtering to rows
    with `constraint_count >= CONSTRAINT_COUNT_FLOOR` AND whose
    instruction_ids are all in `KNOWN_VERIFIER_IDS`."""

    dataset_id: str
    family: str
    constraint_count_floor: int
    n_rows_raw: int
    constraint_count_histogram_raw: dict[int, int]
    n_above_floor_pre_verifier_filter: int
    uncovered_instruction_ids: tuple[str, ...]
    n_excluded_for_uncovered_verifier: int
    n_excluded_for_incomplete_kwargs: int
    n_usable: int
    rows: list[dict[str, Any]] = field(default_factory=list)


def _extract_prompt(messages: list[dict[str, str]]) -> str:
    """Pull the user-role content out of the messages list."""
    for m in messages:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _kwargs_complete(iids: list[str], kwargs_list: list[Any]) -> bool:
    """Per-row check: every arg-requiring constraint carries a populated
    kwargs dict (at least one non-None value). No-arg constraints may
    carry None or an all-None placeholder dict; both are acceptable.

    Imports the canonical arg-requiring set from `src.ifbench_verifiers`
    so the loader and the verifier package stay aligned on which
    instruction IDs need real args."""
    from src.ifbench_verifiers import ARG_REQUIRING_VERIFIER_IDS
    for iid, kw in zip(iids, kwargs_list):
        if iid not in ARG_REQUIRING_VERIFIER_IDS:
            continue
        if not isinstance(kw, dict):
            return False
        if not any(v is not None for v in kw.values()):
            return False
    return True


def _parse_ground_truth(gt: str) -> tuple[list[str], list[Any]]:
    """ground_truth is a Python-literal string like
    `"[{'instruction_id': [...], 'kwargs': [...]}]"`. Returns
    (instruction_id_list, kwargs_list). Raises if malformed."""
    parsed = ast.literal_eval(gt)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("ground_truth is not a non-empty list")
    head = parsed[0]
    iids = head.get("instruction_id") or []
    kw = head.get("kwargs") or []
    if not isinstance(iids, list) or len(iids) == 0:
        raise ValueError("instruction_id missing or empty")
    if not isinstance(kw, list):
        raise ValueError("kwargs not a list")
    return list(iids), list(kw)


def load_ifbench_usable(
    constraint_count_floor: int = CONSTRAINT_COUNT_FLOOR,
    hf_cache_dir: str | Path | None = None,
) -> IFBenchUsable:
    """Load the IF-RLVR composite, filter to multi-constraint rows whose
    instruction_ids all map to a known verifier, and report the audit
    counts."""
    ds = load_dataset(
        IFBENCH_DATASET_ID,
        split=IFBENCH_SPLIT_NAME,
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
    )
    n_raw = len(ds)
    raw_hist: Counter[int] = Counter()
    above_floor_raw = 0
    excluded_unknown_iid = 0
    excluded_incomplete_kwargs = 0
    uncovered: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    next_id = 0

    for row in ds:
        try:
            iids, kw = _parse_ground_truth(row["ground_truth"])
        except Exception:
            continue
        cc = len(iids)
        raw_hist[cc] += 1
        if cc < constraint_count_floor:
            continue
        above_floor_raw += 1
        not_known = [iid for iid in iids if iid not in KNOWN_VERIFIER_IDS]
        if not_known:
            excluded_unknown_iid += 1
            for iid in not_known:
                uncovered[iid] += 1
            continue
        # Completeness contract: for every arg-requiring constraint in
        # this row, the dataset must supply a populated kwargs dict.
        # IFEval verifiers silently random-generate missing args, which
        # would corrupt the score. See CHUNK12_REPORT.md.
        if not _kwargs_complete(iids, kw):
            excluded_incomplete_kwargs += 1
            continue
        prompt = _extract_prompt(row["messages"])
        if not prompt.strip():
            continue
        rows.append({
            "id": next_id,
            "source_key": row["key"],
            "prompt": prompt,
            "instruction_id_list": iids,
            "kwargs_list": kw,
            "constraint_count": cc,
        })
        next_id += 1

    return IFBenchUsable(
        dataset_id=IFBENCH_DATASET_ID,
        family=IFBENCH_FAMILY,
        constraint_count_floor=constraint_count_floor,
        n_rows_raw=n_raw,
        constraint_count_histogram_raw=dict(sorted(raw_hist.items())),
        n_above_floor_pre_verifier_filter=above_floor_raw,
        uncovered_instruction_ids=tuple(sorted(uncovered)),
        n_excluded_for_uncovered_verifier=excluded_unknown_iid,
        n_excluded_for_incomplete_kwargs=excluded_incomplete_kwargs,
        n_usable=len(rows),
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Carving disjoint pools
# ---------------------------------------------------------------------------


SplitTuple = tuple[
    list[dspy.Example], list[dspy.Example], list[dspy.Example], list[dspy.Example]
]


def _record_to_example(record: dict[str, Any]) -> dspy.Example:
    return dspy.Example(
        id=record["id"],
        source_key=record["source_key"],
        prompt=record["prompt"],
        instruction_id_list=record["instruction_id_list"],
        kwargs_list=record["kwargs_list"],
        constraint_count=record["constraint_count"],
    ).with_inputs(*INPUT_FIELDS)


def _assert_disjoint(splits: list[list[dspy.Example]]) -> None:
    seen: dict[int, str] = {}
    for split_idx, items in enumerate(splits):
        for ex in items:
            eid = ex["id"]
            if eid in seen:
                raise AssertionError(
                    f"id {eid!r} appears in both {seen[eid]!r} and "
                    f"{SPLIT_ORDER[split_idx]!r}"
                )
            seen[eid] = SPLIT_ORDER[split_idx]


@dataclass(frozen=True)
class CarveResult:
    splits: SplitTuple
    sizes: dict[str, int]
    seed: int
    constraint_count_floor: int
    n_usable_pool: int
    uncovered_instruction_ids: tuple[str, ...]


def carve_ifbench_splits(
    seed: int = 0,
    usable: IFBenchUsable | None = None,
) -> CarveResult:
    """Carve the four disjoint pools from the usable >=floor verifier-
    covered pool using a deterministic shuffle. No operator gate this
    time: 48,463 rows at >=3 is comfortably above the 545 target."""
    if usable is None:
        usable = load_ifbench_usable()
    if usable.n_usable < TARGET_TOTAL:
        raise RuntimeError(
            f"usable pool ({usable.n_usable}) smaller than target total "
            f"({TARGET_TOTAL}); cannot carve at constraint_count_floor="
            f"{usable.constraint_count_floor}"
        )

    rng = random.Random(seed)
    indices = list(range(usable.n_usable))
    rng.shuffle(indices)
    chosen = indices[:TARGET_TOTAL]

    cursor = 0
    split_lists: list[list[dspy.Example]] = []
    for key in SPLIT_ORDER:
        n = TARGET_SIZES[key]
        sl = [_record_to_example(usable.rows[i]) for i in chosen[cursor : cursor + n]]
        split_lists.append(sl)
        cursor += n

    _assert_disjoint(split_lists)
    return CarveResult(
        splits=tuple(split_lists),  # type: ignore[arg-type]
        sizes=dict(TARGET_SIZES),
        seed=seed,
        constraint_count_floor=usable.constraint_count_floor,
        n_usable_pool=usable.n_usable,
        uncovered_instruction_ids=usable.uncovered_instruction_ids,
    )


def verify_carved_pool_constraint_coverage(splits: SplitTuple) -> tuple[bool, list[str]]:
    """Verify every instruction_id in the carved pools is in
    `KNOWN_VERIFIER_IDS`. Returns (all_covered, uncovered_list)."""
    uncovered: set[str] = set()
    for items in splits:
        for ex in items:
            for iid in ex["instruction_id_list"]:
                if iid not in KNOWN_VERIFIER_IDS:
                    uncovered.add(iid)
    return (len(uncovered) == 0, sorted(uncovered))
