"""Value-based difficulty bins for Exp 3 (curriculum).

Reads the frozen difficulty table (``results/difficulty_table.json``) and
assigns each D_feedback example to easy / medium / hard by VALUE thresholds,
per the Exp-3 spec (CLAUDE_CODE_BUILD_PROMPT_exp3_curriculum.md, "Bin
definitions"):

    easy   : base_score >= 0.99
    hard   : base_score <= 0.01
    medium : 0.01 < base_score < 0.99

Bin by value, not rank (hard constraint #2): the HotpotQA F1 distribution is
bimodal (mostly 0.0 / 1.0 with a thin partial band), so rank terciles would
pad the medium band with perfect examples. For HotpotQA's discrete F1 these
thresholds coincide with the strict {==0, partial, ==1} split, but we keep the
threshold form so the spec is honoured literally and the module is robust to
near-extreme scores on other substrates.

``example_id`` is the integer list index into D_feedback, matching the DataId
gepa assigns (gepa wraps ``list[DataInst]`` in ListDataLoader and uses the list
index as DataId; see src/difficulty.py). This is the same id space as
``difficulty.DifficultyTable.scores``. The frozen table is reused as-is; bins
are derived deterministically from its ``scores`` array and nothing is
re-scored.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

BinName = Literal["easy", "medium", "hard"]
BIN_NAMES: tuple[BinName, BinName, BinName] = ("easy", "medium", "hard")

# Value thresholds (Exp-3 spec). easy at the top, hard at the bottom, medium
# is the strictly-interior frontier band.
EASY_MIN: float = 0.99
HARD_MAX: float = 0.01

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TABLE_PATH = REPO / "results" / "difficulty_table.json"
SUMMARY_PATH = REPO / "results" / "exp3_curriculum" / "bins_summary.json"

# Minimum members the medium (frontier) bin must have for the experiment to be
# runnable as designed: b=3 draws (with replacement, but we still require the
# bin be non-degenerate).
MIN_MEDIUM_MEMBERS: int = 3


def assign_bin(score: float) -> BinName:
    """Map a single base score to its Exp-3 difficulty bin.

    Raises if ``score`` is outside [0, 1] so a metric bug surfaces here rather
    than as a silent miscategorisation later.
    """
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"assign_bin: score {score!r} must be in [0, 1]")
    if score >= EASY_MIN:
        return "easy"
    if score <= HARD_MAX:
        return "hard"
    return "medium"


@dataclass(frozen=True)
class Bins:
    """Immutable value-based bin assignment over D_feedback.

    Attributes:
      - scores:  tuple[float, ...] indexed by D_feedback list index (DataId).
      - members: Mapping[BinName, tuple[int, ...]], per-bin id lists, each
                 sorted ascending by id for determinism.
    """

    scores: tuple[float, ...]
    members: Mapping[BinName, tuple[int, ...]]

    def get_bin_members(self, bin_name: BinName) -> list[int]:
        """Return the D_feedback ids in ``bin_name`` (sorted ascending)."""
        if bin_name not in BIN_NAMES:
            raise ValueError(
                f"get_bin_members: unknown bin {bin_name!r}; expected one of {BIN_NAMES}"
            )
        return list(self.members[bin_name])

    def bin_of(self, example_id: int) -> BinName:
        return assign_bin(self.scores[example_id])

    @property
    def all_ids(self) -> list[int]:
        return list(range(len(self.scores)))

    def counts(self) -> dict[str, int]:
        return {b: len(self.members[b]) for b in BIN_NAMES}

    def histogram(self) -> dict[str, int]:
        """Score histogram (rounded score value -> count), sorted by value.

        Keys are strings so the dict round-trips cleanly through JSON.
        """
        c = Counter(round(float(s), 6) for s in self.scores)
        return {repr(v): c[v] for v in sorted(c)}

    def summary(self) -> dict[str, Any]:
        return {
            "n": len(self.scores),
            "thresholds": {"easy_min": EASY_MIN, "hard_max": HARD_MAX},
            "counts": self.counts(),
            "histogram": self.histogram(),
            "medium_min_required": MIN_MEDIUM_MEMBERS,
            "medium_ok": len(self.members["medium"]) >= MIN_MEDIUM_MEMBERS,
        }


def build_bins(scores: Sequence[float]) -> Bins:
    """Build a :class:`Bins` from a per-id sequence of base scores."""
    if len(scores) == 0:
        raise ValueError("build_bins: scores must be non-empty.")
    fscores = tuple(float(s) for s in scores)
    by_bin: dict[BinName, list[int]] = {b: [] for b in BIN_NAMES}
    for i, s in enumerate(fscores):
        by_bin[assign_bin(s)].append(i)
    members = {b: tuple(sorted(by_bin[b])) for b in BIN_NAMES}
    return Bins(scores=fscores, members=MappingProxyType(members))


def load_bins(table_path: Path | str = DEFAULT_TABLE_PATH) -> Bins:
    """Load the frozen difficulty table and derive Exp-3 value bins from its
    ``scores`` array. The table is read-only; nothing is re-scored."""
    data = json.loads(Path(table_path).read_text())
    return build_bins([float(s) for s in data["scores"]])


def main() -> int:
    """Print and persist the bin summary; gate on the medium bin size.

    Run: ``.venv/bin/python -m src.bins``
    Returns a non-zero exit code (and prints a STOP banner) if the medium
    bin has fewer than ``MIN_MEDIUM_MEMBERS`` members.
    """
    bins = load_bins()
    summary = bins.summary()
    counts = summary["counts"]

    print(f"Difficulty table: {DEFAULT_TABLE_PATH}")
    print(f"n = {summary['n']}  (D_feedback)")
    print(f"thresholds: easy s>={EASY_MIN}, hard s<={HARD_MAX}, medium otherwise")
    print("bin counts:")
    for b in BIN_NAMES:
        print(f"  {b:<7} {counts[b]}")
    print("score histogram (value: count):")
    for v, n in summary["histogram"].items():
        print(f"  {v}: {n}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {SUMMARY_PATH}")

    if counts["medium"] < MIN_MEDIUM_MEMBERS:
        print(
            "\n*** STOP: medium bin has "
            f"{counts['medium']} < {MIN_MEDIUM_MEMBERS} members. "
            "The experiment is not runnable as designed; revisit thresholds. ***"
        )
        return 1
    print(
        f"\nGATE OK: medium bin has {counts['medium']} >= {MIN_MEDIUM_MEMBERS} members."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
