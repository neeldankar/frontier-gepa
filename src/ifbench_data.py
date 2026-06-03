"""IFBench data loader and dataset-gate check (BUILD_PLAN.md §7 Chunk 11).

Substrate: AllenAI IFBench (Pyatkin et al. 2025, NeurIPS 2025;
arXiv 2505.07591), the multi-constraint instruction-following benchmark. NOT
Google IFEval. The canonical Hugging Face id is `allenai/IFBench_test`.
The IF-RLVR training set (`allenai/IF_multi_constraints_upto5`) is
out-of-scope for the Experiment-2 evaluation substrate.

Neither the installed `gepa==0.1.1` nor `dspy==3.2.1` ships an IFBench
data loader, so this module rolls its own. The per-row schema preserves
what the IFBench verifiers will need in Chunk 12:
  - `key`:                  stable id (integer in IFBench_test, kept as
                            string for cross-system compatibility)
  - `prompt`:               the instruction prompt the model sees
  - `instruction_id_list`:  list of constraint identifier strings (the
                            constraint family / kind, e.g.
                            `count:keywords_multiple`); IFBench's official
                            verifiers consume this plus `kwargs`.
  - `kwargs`:               list of dicts (one per constraint), each with
                            verifier-specific keyword arguments (mostly
                            nulls; the non-null entries name the
                            constraint payload).

The split layout mirrors Experiment 1's methodology: four disjoint
pools (d_feedback, accept_batch, d_pareto, test) sliced from a single
deterministic shuffle of the usable rows. When the dataset is smaller
than the §15 target total (D_feedback=150, A=20, D_pareto=75, test=300
= 545), this module shrinks all four pools proportionally on the
150:20:75:300 ratio and records the actual sizes. It never falls back
to IFEval.

Operator-review gate (BUILD_PLAN §7 Chunk 11, last bullet):
  If the resulting D_feedback would be small enough that rank terciles
  approach the 20-instance NO-GO floor (`D_feedback < D_FEEDBACK_REVIEW_FLOOR`,
  default 90 → tercile ~30), `gate_decision()` returns "OPERATOR_REVIEW"
  rather than "GO". `carve_ifbench_splits()` then refuses to carve unless
  `allow_review_threshold=True` is passed explicitly. The runtime caller
  must acknowledge the gate before any matrix work proceeds in Chunk 14.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
from datasets import load_dataset

# Canonical dataset id and variant. Single source of truth.
IFBENCH_DATASET_ID: str = "allenai/IFBench_test"
IFBENCH_SPLIT_NAME: str = "train"  # IFBench_test exposes a single "train" split
IFBENCH_FAMILY: str = "AllenAI multi-constraint IFBench (Pyatkin et al. 2025)"

# Target Experiment-1 pool layout. Total = 545.
TARGET_SIZES: dict[str, int] = {
    "d_feedback": 150,
    "accept_batch": 20,
    "d_pareto": 75,
    "test": 300,
}
SPLIT_ORDER = ("d_feedback", "accept_batch", "d_pareto", "test")
TARGET_TOTAL: int = sum(TARGET_SIZES.values())

# Operator-review thresholds (BUILD_PLAN §7 Chunk 11).
D_FEEDBACK_REVIEW_FLOOR: int = 90
TERCILE_NOGO_FLOOR: int = 20

INPUT_FIELDS = ("prompt", "instruction_id_list", "kwargs")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IFBenchUsable:
    """Container for the result of the dataset-gate count step. Wraps the
    loaded rows plus the audit fields we want to commit to provenance."""

    dataset_id: str
    family: str
    n_rows_raw: int
    n_usable: int
    constraint_count_histogram: dict[int, int]
    rows: list[dict[str, Any]]


def _is_usable(row: dict[str, Any]) -> bool:
    """A row is 'usable' iff it has a non-empty prompt AND a non-empty
    `instruction_id_list`."""
    prompt = row.get("prompt") or ""
    ids = row.get("instruction_id_list") or []
    if not (isinstance(prompt, str) and prompt.strip()):
        return False
    if not (isinstance(ids, list) and len(ids) >= 1):
        return False
    return True


def _row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    """Project a raw IFBench row down to the fields downstream code reads.
    Coerces `key` to string so split-level disjointness checks key on the
    same type Experiment-1 used."""
    return {
        "id": str(row["key"]),
        "prompt": row["prompt"],
        "instruction_id_list": list(row["instruction_id_list"]),
        "kwargs": list(row.get("kwargs") or []),
    }


def load_ifbench_usable(
    hf_cache_dir: str | Path | None = None,
) -> IFBenchUsable:
    """Load `allenai/IFBench_test`, filter to usable rows, and report the
    counts the dataset gate needs. Network call to the HF hub on first
    run; subsequent runs hit the local HF cache."""
    ds = load_dataset(
        IFBENCH_DATASET_ID,
        split=IFBENCH_SPLIT_NAME,
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
    )
    n_raw = len(ds)
    usable_rows: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    for row in ds:
        if not _is_usable(row):
            continue
        usable_rows.append(_row_to_record(row))
        n_constraints = len(row["instruction_id_list"])
        counts[n_constraints] = counts.get(n_constraints, 0) + 1
    return IFBenchUsable(
        dataset_id=IFBENCH_DATASET_ID,
        family=IFBENCH_FAMILY,
        n_rows_raw=n_raw,
        n_usable=len(usable_rows),
        constraint_count_histogram=dict(sorted(counts.items())),
        rows=usable_rows,
    )


# ---------------------------------------------------------------------------
# Proportional shrink + gate decision
# ---------------------------------------------------------------------------


def compute_proportional_sizes(
    usable: int,
    target_sizes: dict[str, int] = TARGET_SIZES,
) -> dict[str, int]:
    """Return per-pool sizes that fit within `usable` examples and preserve
    the 150:20:75:300 ratio as closely as the integer rounding allows.

    Strategy: floor each pool's raw fractional size, then distribute the
    remainder (largest fractional parts first) so the four sizes sum
    exactly to `min(usable, TARGET_TOTAL)`. When `usable >= TARGET_TOTAL`,
    returns `TARGET_SIZES` unchanged.
    """
    total_target = sum(target_sizes.values())
    if usable >= total_target:
        return dict(target_sizes)

    scale = usable / total_target
    raw: dict[str, float] = {k: v * scale for k, v in target_sizes.items()}
    floored: dict[str, int] = {k: int(math.floor(v)) for k, v in raw.items()}
    deficit = usable - sum(floored.values())
    # Distribute the remainder by largest fractional part first; tie-break
    # on the SPLIT_ORDER tuple so the assignment is deterministic.
    fracs = sorted(
        ((k, raw[k] - floored[k]) for k in target_sizes),
        key=lambda kv: (-kv[1], SPLIT_ORDER.index(kv[0])),
    )
    sizes = dict(floored)
    for i in range(deficit):
        sizes[fracs[i][0]] += 1
    return sizes


@dataclass(frozen=True)
class GateDecision:
    verdict: str  # "GO", "OPERATOR_REVIEW", or "NO_GO"
    reason: str
    sizes: dict[str, int]
    rank_tercile_size: int  # floor(d_feedback / 3); the Chunk-13 NO-GO concern


def gate_decision(usable_count: int) -> GateDecision:
    """Apply BUILD_PLAN §7 Chunk 11's gate logic to the proposed pool sizes.

    - "NO_GO" if the resulting d_feedback is small enough that rank
      terciles would fall AT the 20-instance NO-GO floor or below.
    - "OPERATOR_REVIEW" if d_feedback would be below
      `D_FEEDBACK_REVIEW_FLOOR` (default 90) but the terciles are still
      above the NO-GO floor.
    - "GO" otherwise.

    The verdict is advisory; the caller is responsible for honoring it.
    """
    sizes = compute_proportional_sizes(usable_count)
    d_feedback = sizes["d_feedback"]
    tercile = d_feedback // 3
    if tercile <= TERCILE_NOGO_FLOOR:
        return GateDecision(
            verdict="NO_GO",
            reason=(
                f"projected D_feedback={d_feedback} would put rank terciles "
                f"at {tercile} <= TERCILE_NOGO_FLOOR ({TERCILE_NOGO_FLOOR})"
            ),
            sizes=sizes,
            rank_tercile_size=tercile,
        )
    if d_feedback < D_FEEDBACK_REVIEW_FLOOR:
        return GateDecision(
            verdict="OPERATOR_REVIEW",
            reason=(
                f"projected D_feedback={d_feedback} is below the review "
                f"floor ({D_FEEDBACK_REVIEW_FLOOR}); rank terciles would be "
                f"{tercile}, above the NO-GO floor but uncomfortably close"
            ),
            sizes=sizes,
            rank_tercile_size=tercile,
        )
    return GateDecision(
        verdict="GO",
        reason=f"D_feedback={d_feedback} >= {D_FEEDBACK_REVIEW_FLOOR}; rank tercile={tercile}",
        sizes=sizes,
        rank_tercile_size=tercile,
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
        prompt=record["prompt"],
        instruction_id_list=record["instruction_id_list"],
        kwargs=record["kwargs"],
    ).with_inputs(*INPUT_FIELDS)


def carve_ifbench_splits(
    seed: int = 0,
    *,
    usable: IFBenchUsable | None = None,
    allow_review_threshold: bool = False,
) -> tuple[SplitTuple, GateDecision]:
    """Carve four disjoint pools from `usable.rows` with a deterministic
    shuffle, returning the splits and the gate decision they were carved
    under. Refuses if the gate is OPERATOR_REVIEW unless
    `allow_review_threshold=True` is passed (the Chunk-11 contract). Always
    refuses on NO_GO."""
    if usable is None:
        usable = load_ifbench_usable()
    gate = gate_decision(usable.n_usable)
    if gate.verdict == "NO_GO":
        raise RuntimeError(
            f"IFBench dataset gate: NO_GO. {gate.reason}. Refusing to carve."
        )
    if gate.verdict == "OPERATOR_REVIEW" and not allow_review_threshold:
        raise RuntimeError(
            f"IFBench dataset gate: OPERATOR_REVIEW. {gate.reason}. "
            f"Pass allow_review_threshold=True to acknowledge and proceed."
        )

    sizes = gate.sizes
    rng = random.Random(seed)
    indices = list(range(usable.n_usable))
    rng.shuffle(indices)
    total_needed = sum(sizes.values())
    if total_needed > usable.n_usable:
        raise RuntimeError(
            f"requested total {total_needed} > usable {usable.n_usable}"
        )
    chosen = indices[:total_needed]

    cursor = 0
    split_lists: list[list[dspy.Example]] = []
    for key in SPLIT_ORDER:
        n = sizes[key]
        sl = [_record_to_example(usable.rows[i]) for i in chosen[cursor : cursor + n]]
        split_lists.append(sl)
        cursor += n

    _assert_disjoint(split_lists)
    return tuple(split_lists), gate  # type: ignore[return-value]


def _assert_disjoint(splits: list[list[dspy.Example]]) -> None:
    seen: dict[str, str] = {}
    for split_idx, items in enumerate(splits):
        for ex in items:
            eid = ex["id"]
            if eid in seen:
                raise AssertionError(
                    f"id {eid!r} appears in both {seen[eid]!r} and "
                    f"{SPLIT_ORDER[split_idx]!r}"
                )
            seen[eid] = SPLIT_ORDER[split_idx]
