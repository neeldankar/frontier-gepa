"""Band-aware reflection minibatch sampler.

Implements ``gepa.strategies.batch_sampler.BatchSampler``. Plugged into the
engine via the ``batch_sampler=`` constructor arg (gepa's BatchSampler is a
one-method Protocol; no subclass needed, see ARCHITECTURE.md §2 Q1).

The four arms differ only in `target_band`:
  - "random":   uniform draw of `b` ids from D_feedback.
  - "easy":     mostly draws from the top-F1 tercile.
  - "mid":      mostly draws from the middle-F1 (frontier) tercile.
  - "hard":     mostly draws from the bottom-F1 tercile.

For the three static-band arms, each of the `b` picks chooses a band by the
mix `(target, off_band_1, off_band_2)` and then samples one id from that
band uniformly. Two regimes are supported:
  - **`DEFAULT_MIX = (0.70, 0.15, 0.15)`** — the Experiment-1 default; 70%
    of draws hit the target band, 15% each off-band.
  - **`PURE_ON_BAND_MIX = (1.0, 0.0, 0.0)`** — the Experiment-1b isolation
    variant. Every draw hits the target band; no off-band leakage. Requires
    `|target_band| >= b`; smaller bands raise `ValueError` at draw time
    rather than silently leaking off-band via the bounded-retry fallback.
Per-call sampling is **without replacement within the same minibatch**
(ARCHITECTURE.md §4.1): if the same id is drawn twice in one call, the
second draw retries until a fresh id appears. Across calls, repeats are
allowed (the run will revisit instances over 44 iters x 3 picks = 132
reflection touches against ~33 ids per band).

The sampler is deterministic given a `random.Random` instance passed at
construction. For the §3 stop-and-resume requirement, the runner snapshots
this RNG's state per iteration (see src/run_gepa.py).
"""

from __future__ import annotations

import random
from typing import Literal

from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.strategies.batch_sampler import BatchSampler

from src.difficulty import BAND_NAMES, BandName, DifficultyTable

TargetBand = Literal["easy", "mid", "hard", "random"]
TARGET_BAND_VALUES: tuple[TargetBand, ...] = ("easy", "mid", "hard", "random")

DEFAULT_MIX: tuple[float, float, float] = (0.70, 0.15, 0.15)
# Experiment-1b isolation variant (BUILD_PLAN.md §7 Chunk 8, §4 D5): pure on-
# band sampling. With target_weight=1.0 and both off-band weights=0.0, every
# draw lands in the target band; the cross-arm contrast is no longer diluted
# by the 30% off-band leakage that drove static_easy's 5.3 accepts/cell in
# Experiment 1.
PURE_ON_BAND_MIX: tuple[float, float, float] = (1.0, 0.0, 0.0)
DEFAULT_MINIBATCH_SIZE: int = 3


def _is_pure_on_band(mix: tuple[float, float, float]) -> bool:
    """True iff `mix` is the pure on-band (100/0/0) regime within float
    tolerance. Off-band weights must be exactly zero for the contract to
    hold; we still permit a tiny tolerance on the target weight to avoid
    surprising the operator with floating-point construction."""
    target, off1, off2 = mix
    return (abs(target - 1.0) < 1e-9) and (off1 == 0.0) and (off2 == 0.0)


def off_bands_for(target: BandName) -> tuple[BandName, BandName]:
    """Return the two off-band names for a static-band arm, in a deterministic
    order (left-to-right by BAND_NAMES). The mix is `(target, off1, off2) =
    (0.70, 0.15, 0.15)`; off1 and off2 are interchangeable but we fix the
    ordering for reproducibility."""
    if target not in BAND_NAMES:
        raise ValueError(f"target must be one of {BAND_NAMES}, got {target!r}")
    return tuple(b for b in BAND_NAMES if b != target)  # type: ignore[return-value]


class BandBatchSampler(BatchSampler):
    """A `gepa.strategies.batch_sampler.BatchSampler` that draws by frozen
    difficulty band.

    Parameters
    ----------
    target_band:
        "random" / "easy" / "mid" / "hard".
    difficulty_table:
        The frozen difficulty table (Chunk-5 output). Required for
        non-random arms. Ignored for "random" (where it may be None).
    rng:
        A `random.Random` instance. The sampler does not own RNG seeding;
        the runner is responsible for seeding once and snapshotting state
        per iteration.
    b:
        Minibatch size. Default 3 per spec.
    mix:
        `(target_weight, off1_weight, off2_weight)`. Default `DEFAULT_MIX`
        (0.70, 0.15, 0.15). Pass `PURE_ON_BAND_MIX` (1.0, 0.0, 0.0) for
        the Experiment-1b isolation variant. Ignored for "random" (which
        always samples uniformly over loader ids).
    """

    def __init__(
        self,
        target_band: TargetBand,
        rng: random.Random,
        difficulty_table: DifficultyTable | None = None,
        b: int = DEFAULT_MINIBATCH_SIZE,
        mix: tuple[float, float, float] = DEFAULT_MIX,
    ):
        if target_band not in TARGET_BAND_VALUES:
            raise ValueError(
                f"target_band must be one of {TARGET_BAND_VALUES}, got {target_band!r}"
            )
        if b <= 0:
            raise ValueError(f"b must be positive, got {b!r}")
        if target_band != "random":
            if difficulty_table is None:
                raise ValueError(
                    f"target_band={target_band!r} requires a difficulty_table"
                )
            if not (abs(sum(mix) - 1.0) < 1e-9):
                raise ValueError(f"mix must sum to 1.0, got {sum(mix)}")
            if any(w < 0 for w in mix):
                raise ValueError(f"mix weights must be non-negative, got {mix}")

        self.target_band: TargetBand = target_band
        self.rng = rng
        self.difficulty_table = difficulty_table
        self.b = b
        self.mix = mix

        if target_band == "random":
            self._band_order: tuple[BandName, ...] | None = None
        else:
            target_named: BandName = target_band  # type: ignore[assignment]
            self._band_order = (target_named,) + off_bands_for(target_named)

    def next_minibatch_ids(
        self,
        loader: DataLoader[DataId, DataId],
        state: GEPAState,
    ) -> list[DataId]:
        all_ids = list(loader.all_ids())
        if len(all_ids) < self.b:
            raise ValueError(
                f"loader has {len(all_ids)} ids but b={self.b}"
            )

        if self.target_band == "random":
            return self._sample_random(all_ids)

        # Under pure on-band sampling the bounded-retry fallback at the end
        # of `_draw_one` would silently leak off-band ids if the target
        # band were smaller than b. Make that case loud rather than
        # corrupting the experiment.
        if _is_pure_on_band(self.mix):
            assert self.difficulty_table is not None
            target_band_size = len(self.difficulty_table.ids(self.target_band))  # type: ignore[arg-type]
            if target_band_size < self.b:
                raise ValueError(
                    f"pure on-band sampling: target band {self.target_band!r} "
                    f"has {target_band_size} ids, smaller than b={self.b}; "
                    "without-replacement draws cannot be satisfied without "
                    "leaking off-band. Grow D_feedback so the target band "
                    "has at least b ids, or relax the mix."
                )

        return self._sample_band(all_ids)

    def _sample_random(self, all_ids: list[DataId]) -> list[DataId]:
        # Random.sample is without-replacement and uses the seeded rng.
        return self.rng.sample(all_ids, self.b)

    def _sample_band(self, all_ids: list[DataId]) -> list[DataId]:
        assert self.difficulty_table is not None and self._band_order is not None
        table = self.difficulty_table
        target, off1, off2 = self._band_order
        weights = self.mix
        bands = (target, off1, off2)

        # Allowed ids are the intersection of D_feedback (loader.all_ids()) with
        # the difficulty table's universe -- a sanity guard. In practice the two
        # are identical (the table is built over D_feedback).
        valid_ids = set(all_ids).intersection(range(table.n))

        picks: list[DataId] = []
        chosen: set[DataId] = set()
        for _ in range(self.b):
            picks.append(self._draw_one(bands, weights, table, chosen, valid_ids))
            chosen.add(picks[-1])
        return picks

    def _draw_one(
        self,
        bands: tuple[BandName, BandName, BandName],
        weights: tuple[float, float, float],
        table: DifficultyTable,
        already_chosen: set[DataId],
        valid_ids: set[DataId],
    ) -> DataId:
        # Sample a band, then an id from that band that hasn't been picked
        # yet this call. If the band is exhausted (all ids already picked),
        # resample the band. Bounded retry to avoid infinite loops when the
        # bands are very small.
        max_attempts = 200
        for _ in range(max_attempts):
            band = self.rng.choices(bands, weights=weights, k=1)[0]
            band_ids = [i for i in table.ids(band) if i in valid_ids and i not in already_chosen]
            if not band_ids:
                continue
            return self.rng.choice(band_ids)
        # If we get here, the bands are pathologically small relative to b.
        # Fall back to any unused valid id rather than crashing the run.
        remaining = [i for i in valid_ids if i not in already_chosen]
        if not remaining:
            raise RuntimeError(
                f"BandBatchSampler: no available ids; loader exhausted? "
                f"chosen={already_chosen} valid={len(valid_ids)}"
            )
        return self.rng.choice(remaining)


def make_sampler(
    target_band: TargetBand,
    seed: int,
    difficulty_table: DifficultyTable | None = None,
    b: int = DEFAULT_MINIBATCH_SIZE,
    mix: tuple[float, float, float] = DEFAULT_MIX,
) -> BandBatchSampler:
    """Convenience factory: build a sampler with a fresh `random.Random(seed)`.

    The runner uses this when starting a new (arm, seed) run; on resume it
    builds the sampler with `rng.setstate(...)` applied separately."""
    return BandBatchSampler(
        target_band=target_band,
        rng=random.Random(seed),
        difficulty_table=difficulty_table,
        b=b,
        mix=mix,
    )
