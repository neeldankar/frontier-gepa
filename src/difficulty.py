"""Frozen difficulty table for D_feedback.

Bins D_feedback ids into equal terciles by base-system F1 rank. The table is
constructed once (Chunk 5 scores D_feedback on the base system) and never
mutated thereafter: each draw of a reflection minibatch reads it; nothing
writes to it. This is the §15 settled decision the experiment turns on.

The data structure is intentionally minimal -- a dataclass with frozen=True
and explicit factory + validation -- so accidental mutation surfaces as a
runtime error, not a silent science breakage.

DataId in this experiment is the integer list index of the D_feedback list,
because gepa wraps a `list[DataInst]` in ListDataLoader and uses list index
as DataId (see `gepa/core/data_loader.py:50`).

Band names match the handoff: 'easy' (top tercile by F1), 'mid' (middle
tercile, the "frontier" band), 'hard' (bottom tercile). Equal-tercile
construction: 33/33/34 for the standard 100-instance D_feedback (the
extra instance lands in the mid band when 100 is not divisible by 3).
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
    """Construct an equal-tercile DifficultyTable from per-id F1 scores.

    Ranking rule: ids are sorted by score ascending then by id ascending (for
    a stable break on ties). The bottom third is 'hard', the middle third is
    'mid' (the frontier band), the top third is 'easy'. When n is not
    divisible by 3, the remainder lands in the middle tercile (so 100 ->
    33/34/33). This matches the handoff's "equal terciles" intent and keeps
    the middle band slightly larger when there's a remainder, which slightly
    helps the frontier population without affecting band-arm coverage
    symmetry across the three band arms.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("build_difficulty_table: scores must be non-empty.")

    # Stable rank: ascending score, tie-broken by id ascending.
    indexed = sorted(range(n), key=lambda i: (scores[i], i))

    # Equal-tercile partition with the remainder going to mid.
    third = n // 3
    rem = n - 3 * third
    if rem == 0:
        sizes = (third, third, third)
    elif rem == 1:
        sizes = (third, third + 1, third)
    else:  # rem == 2
        sizes = (third, third + 2, third)
    hard_end = sizes[0]
    mid_end = hard_end + sizes[1]
    # easy_end = n  (implicit)

    hard_ids = tuple(sorted(indexed[:hard_end]))
    mid_ids = tuple(sorted(indexed[hard_end:mid_end]))
    easy_ids = tuple(sorted(indexed[mid_end:]))

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
