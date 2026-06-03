"""Frozen difficulty table for D_feedback.

Bins D_feedback ids into easy / mid / hard. The table is constructed
once (Chunk 5 for HotpotQA, Chunk 13 for IFBench, scoring D_feedback on
the base system) and never mutated thereafter: each draw of a reflection
minibatch reads it; nothing writes to it. The frozen-difficulty
principle from §15 is preserved.

The data structure is intentionally minimal -- a dataclass with frozen=True
and explicit factory + validation -- so accidental mutation surfaces as a
runtime error, not a silent science breakage.

DataId in these experiments is the integer list index of the D_feedback
list, because gepa wraps a `list[DataInst]` in ListDataLoader and uses
list index as DataId (see `gepa/core/data_loader.py:50`).

Two binning rules are supported (BUILD_PLAN §4 D3 governs which to use):

  Value-based bins (`build_difficulty_table`):
    - 'hard':  score == 0.0
    - 'mid':   0.0 < score < 1.0
    - 'easy':  score == 1.0
    Right for bimodal distributions where the strict-partial set is the
    natural frontier. HotpotQA Experiment 1 uses this rule
    (DEVIATIONS.md entry 4).

  Rank terciles (`build_rank_tercile_table`):
    - Sort ids by score ascending (ties broken by id ascending).
    - Bottom n//3 -> 'hard', top n//3 -> 'easy', remainder -> 'mid'.
    Extras land in 'mid' so the frontier is the largest band when n is
    not divisible by 3. Right for continuous distributions where exact
    extremes do not anchor a meaningful partition. IFBench Experiment 2
    will use this rule iff the Chunk-13 continuity gate
    (`continuity_gate`) selects it.

The dispatcher `build_chosen_table` runs the gate and returns the
appropriate table plus the gate diagnostics. The gate is BUILD_PLAN §4
D3 verbatim: rank terciles stand if the middle tercile is at least 80%
strictly-partial AND less than 50% of the count sits at exact 0 or 1.
Otherwise fall back to value bins. NO-GO at the diagnostic layer if the
frontier band has fewer than 20 instances; the table itself is still
constructible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

BandName = Literal["easy", "mid", "hard"]
BAND_NAMES: tuple[BandName, BandName, BandName] = ("easy", "mid", "hard")

BinningRule = Literal["value_bins", "rank_terciles"]

# BUILD_PLAN §4 D3 thresholds. The middle-tercile partial fraction must
# be at least this value AND the extreme-mass fraction must be strictly
# less than its threshold, for rank terciles to stand.
GATE_MIDDLE_PARTIAL_MIN: float = 0.80
GATE_EXTREME_MASS_MAX: float = 0.50
# Frontier-size GO criterion (count of mid-band ids).
GATE_FRONTIER_MIN_GO: int = 20


@dataclass(frozen=True)
class DifficultyTable:
    """Immutable per-id F1 + tercile assignment.

    Public attributes:
      - scores:        tuple[float, ...] indexed by D_feedback list index
      - band_for_id:   Mapping[int, BandName], the frozen band assignment
      - ids_in_band:   Mapping[BandName, tuple[int, ...]], the per-band id lists
                       (each is sorted ascending by id for determinism)

    Use :func:`build_difficulty_table` to construct one. Direct construction
    is supported (you need a sequence of per-id scores) but the factory does
    the rank-based tercile bin assignment and the validation in one step.
    """

    scores: tuple[float, ...]
    band_for_id: Mapping[int, BandName]
    ids_in_band: Mapping[BandName, tuple[int, ...]]

    def __post_init__(self) -> None:
        # Defensive: catch malformed construction at boundary.
        n = len(self.scores)
        if n == 0:
            raise ValueError("DifficultyTable: scores must be non-empty.")
        if set(self.band_for_id.keys()) != set(range(n)):
            raise ValueError(
                "DifficultyTable: band_for_id keys must be exactly 0..n-1."
            )
        if set(self.ids_in_band.keys()) != set(BAND_NAMES):
            raise ValueError(
                f"DifficultyTable: ids_in_band must have keys {BAND_NAMES}."
            )
        total = sum(len(ids) for ids in self.ids_in_band.values())
        if total != n:
            raise ValueError(
                f"DifficultyTable: per-band ids total {total} != n_scores {n}."
            )
        # Disjointness across bands.
        seen: set[int] = set()
        for band in BAND_NAMES:
            for i in self.ids_in_band[band]:
                if i in seen:
                    raise ValueError(
                        f"DifficultyTable: id {i} appears in multiple bands."
                    )
                if self.band_for_id[i] != band:
                    raise ValueError(
                        f"DifficultyTable: band_for_id[{i}]={self.band_for_id[i]!r} "
                        f"but it is in ids_in_band[{band!r}]."
                    )
                seen.add(i)

    @property
    def n(self) -> int:
        return len(self.scores)

    def band(self, data_id: int) -> BandName:
        return self.band_for_id[data_id]

    def ids(self, band: BandName) -> tuple[int, ...]:
        return self.ids_in_band[band]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": list(self.scores),
            "ids_in_band": {b: list(self.ids_in_band[b]) for b in BAND_NAMES},
            "version": 1,
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "DifficultyTable":
        data = json.loads(Path(path).read_text())
        if data.get("version") != 1:
            raise ValueError(f"unknown DifficultyTable version {data.get('version')!r}")
        scores = tuple(float(s) for s in data["scores"])
        ids_in_band: dict[BandName, tuple[int, ...]] = {
            b: tuple(int(i) for i in data["ids_in_band"][b]) for b in BAND_NAMES
        }
        band_for_id: dict[int, BandName] = {
            i: band for band in BAND_NAMES for i in ids_in_band[band]
        }
        return cls(
            scores=scores,
            band_for_id=MappingProxyType(band_for_id),
            ids_in_band=MappingProxyType(ids_in_band),
        )


def build_difficulty_table(scores: list[float]) -> DifficultyTable:
    """Construct a value-based DifficultyTable from per-id F1 scores.

    Partition rule:
      hard = {i : scores[i] == 0.0}
      mid  = {i : 0.0 < scores[i] < 1.0}
      easy = {i : scores[i] == 1.0}

    Band sizes are unequal in general; that is by design. Scores must lie in
    [0, 1]; values outside that range raise so a metric bug surfaces here
    rather than as a silent miscategorisation later.

    This replaces the earlier equal-rank-tercile partition. The §5 GO/NO-GO
    diagnostic showed a bimodal three-cluster F1 distribution on the
    distractor substrate, where equal-tercile binning padded the middle band
    with F1==1 examples; the value-based partition makes the middle band
    literally the set of strictly-partial cases. See DEVIATIONS.md entry 4.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("build_difficulty_table: scores must be non-empty.")
    for i, s in enumerate(scores):
        if not (0.0 <= s <= 1.0):
            raise ValueError(
                f"build_difficulty_table: score for id={i} is {s!r}; must be in [0,1]"
            )

    hard_ids = tuple(i for i, s in enumerate(scores) if s == 0.0)
    mid_ids = tuple(i for i, s in enumerate(scores) if 0.0 < s < 1.0)
    easy_ids = tuple(i for i, s in enumerate(scores) if s == 1.0)

    band_for_id: dict[int, BandName] = {}
    for i in hard_ids:
        band_for_id[i] = "hard"
    for i in mid_ids:
        band_for_id[i] = "mid"
    for i in easy_ids:
        band_for_id[i] = "easy"

    return DifficultyTable(
        scores=tuple(float(s) for s in scores),
        band_for_id=MappingProxyType(dict(band_for_id)),
        ids_in_band=MappingProxyType({
            "easy": easy_ids,
            "mid": mid_ids,
            "hard": hard_ids,
        }),
    )


def build_rank_tercile_table(scores: list[float]) -> DifficultyTable:
    """Construct a rank-tercile DifficultyTable.

    Sort ids by (score, id) ascending. Take the bottom ``n//3`` as
    'hard', the top ``n//3`` as 'easy', and the remaining middle slice
    as 'mid'. When n is not divisible by 3, the extras land in 'mid' so
    the frontier is the largest band -- the Chunk-13 design wants the
    method-under-test band to absorb any uneven remainder rather than
    starve.

    Scores must lie in [0, 1]; values outside that range raise so a
    metric bug surfaces here rather than as a silent miscategorisation
    later. Ties are broken deterministically by id ascending, so the
    same scores produce the same partition across runs.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("build_rank_tercile_table: scores must be non-empty.")
    for i, s in enumerate(scores):
        if not (0.0 <= s <= 1.0):
            raise ValueError(
                f"build_rank_tercile_table: score for id={i} is {s!r}; must be in [0,1]"
            )

    n_h = n // 3
    n_e = n // 3
    n_m = n - n_h - n_e

    order = sorted(range(n), key=lambda i: (scores[i], i))
    hard_ids = tuple(sorted(order[:n_h]))
    mid_ids = tuple(sorted(order[n_h : n_h + n_m]))
    easy_ids = tuple(sorted(order[n_h + n_m : n_h + n_m + n_e]))

    band_for_id: dict[int, BandName] = {}
    for i in hard_ids:
        band_for_id[i] = "hard"
    for i in mid_ids:
        band_for_id[i] = "mid"
    for i in easy_ids:
        band_for_id[i] = "easy"

    return DifficultyTable(
        scores=tuple(float(s) for s in scores),
        band_for_id=MappingProxyType(dict(band_for_id)),
        ids_in_band=MappingProxyType({
            "easy": easy_ids,
            "mid": mid_ids,
            "hard": hard_ids,
        }),
    )


def _middle_tercile_partial_fraction(scores: list[float]) -> float:
    """Fraction of the middle-tercile of `scores` (by rank) whose value
    lies strictly in (0, 1). Mirrors the partition rule of
    `build_rank_tercile_table` so the gate measures the actual middle
    tercile that would be installed if rank terciles were chosen."""
    n = len(scores)
    if n == 0:
        return 0.0
    n_h = n // 3
    n_e = n // 3
    n_m = n - n_h - n_e
    if n_m == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: (scores[i], i))
    mid_slice_scores = [scores[i] for i in order[n_h : n_h + n_m]]
    n_partial = sum(1 for s in mid_slice_scores if 0.0 < s < 1.0)
    return n_partial / n_m


def _extreme_mass_fraction(scores: list[float]) -> float:
    """Fraction of `scores` sitting at exactly 0.0 or exactly 1.0
    (count-based, not score-sum-based: 'mass' in BUILD_PLAN §4 D3
    reads as the count of instances at the extremes)."""
    n = len(scores)
    if n == 0:
        return 0.0
    n_extreme = sum(1 for s in scores if s == 0.0 or s == 1.0)
    return n_extreme / n


def continuity_gate(scores: list[float]) -> dict[str, Any]:
    """BUILD_PLAN §4 D3 continuity gate.

    Returns a diagnostics dict naming the chosen binning rule and the
    measured fractions that drove the choice, plus the per-binning
    frontier sizes and a GO/NO-GO verdict.

    Rule (verbatim from D3):
      - Rank terciles stand iff the middle tercile is at least 80%
        strictly-partial (0 < score < 1) AND less than 50% of the count
        sits at exactly 0 or 1.
      - Otherwise fall back to value-based bins.
      - NO-GO at the diagnostic layer iff the frontier band (under the
        chosen binning) has fewer than 20 instances.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("continuity_gate: scores must be non-empty.")

    middle_partial_frac = _middle_tercile_partial_fraction(scores)
    extreme_mass_frac = _extreme_mass_fraction(scores)
    middle_partial_ok = middle_partial_frac >= GATE_MIDDLE_PARTIAL_MIN
    extreme_mass_ok = extreme_mass_frac < GATE_EXTREME_MASS_MAX
    rank_terciles_stand = middle_partial_ok and extreme_mass_ok
    chosen: BinningRule = "rank_terciles" if rank_terciles_stand else "value_bins"

    # Probe both partitions so the operator can compare without rerunning.
    n_frontier_value = sum(1 for s in scores if 0.0 < s < 1.0)
    n_h = n // 3
    n_e = n // 3
    n_frontier_rank = n - n_h - n_e
    n_frontier_chosen = (
        n_frontier_rank if chosen == "rank_terciles" else n_frontier_value
    )

    frontier_ok = n_frontier_chosen >= GATE_FRONTIER_MIN_GO
    verdict = "GO" if frontier_ok else "NO-GO"

    return {
        "n": n,
        "middle_tercile_partial_frac": middle_partial_frac,
        "middle_tercile_partial_threshold": GATE_MIDDLE_PARTIAL_MIN,
        "middle_tercile_partial_ok": middle_partial_ok,
        "extreme_mass_frac": extreme_mass_frac,
        "extreme_mass_threshold": GATE_EXTREME_MASS_MAX,
        "extreme_mass_ok": extreme_mass_ok,
        "rank_terciles_stand": rank_terciles_stand,
        "chosen_binning": chosen,
        "frontier_size_under_rank_terciles": n_frontier_rank,
        "frontier_size_under_value_bins": n_frontier_value,
        "frontier_size_under_chosen": n_frontier_chosen,
        "frontier_min_for_go": GATE_FRONTIER_MIN_GO,
        "frontier_ok": frontier_ok,
        "verdict": verdict,
    }


def build_chosen_table(
    scores: list[float],
) -> tuple[DifficultyTable, dict[str, Any]]:
    """Run the D3 continuity gate, build whichever DifficultyTable it
    selects, and return both. The table is always constructible (the
    verdict is a separate signal); callers act on the gate dict's
    'verdict' field to decide whether to proceed to Chunk 14."""
    gate = continuity_gate(scores)
    if gate["chosen_binning"] == "rank_terciles":
        table = build_rank_tercile_table(scores)
    else:
        table = build_difficulty_table(scores)
    return table, gate


def difficulty_table_sha256(path: Path | str) -> str:
    """SHA-256 of the on-disk difficulty-table JSON. Used by Chunks 9
    and 14 to assert the table was not re-scored between freezing and
    matrix launch."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
