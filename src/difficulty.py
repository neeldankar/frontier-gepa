"""Frozen difficulty table for D_feedback.

Bins D_feedback ids by base-system F1 *value* (not rank). The table is
constructed once (Chunk 5 scores D_feedback on the base system) and never
mutated thereafter: each draw of a reflection minibatch reads it; nothing
writes to it. The frozen-difficulty principle from §15 is preserved.

The data structure is intentionally minimal -- a dataclass with frozen=True
and explicit factory + validation -- so accidental mutation surfaces as a
runtime error, not a silent science breakage.

DataId in this experiment is the integer list index of the D_feedback list,
because gepa wraps a `list[DataInst]` in ListDataLoader and uses list index
as DataId (see `gepa/core/data_loader.py:50`).

Band definition (value-based, as of DEVIATIONS.md entry 4 / 2026-06-01):
  - 'hard':  F1 == 0.0       (complete failure -- no overlap with gold)
  - 'mid':   0.0 < F1 < 1.0   (frontier band; the experiment's method-under-test)
  - 'easy':  F1 == 1.0        (complete success)

HotpotQA F1 is an exact token-overlap ratio, so equality at 0 and 1 is exact;
nothing between is degenerate. Band sizes are unequal in general -- on the
n=150 D_feedback under Qwen2.5-7B-Instruct-Turbo (distractor substrate) we
observe 50 / 31 / 69 (hard / mid / easy). The earlier equal-rank-tercile
binning padded the frontier with F1==1 instances on this distribution; this
value-based partition makes the frontier band literally the set of partial
successes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

BandName = Literal["easy", "mid", "hard"]
BAND_NAMES: tuple[BandName, BandName, BandName] = ("easy", "mid", "hard")


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
