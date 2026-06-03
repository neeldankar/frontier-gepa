"""Chunk 13 offline gates.

Covers `src/difficulty.py` extensions for the IFBench substrate:

  - `build_rank_tercile_table`: near-equal thirds by rank, remainder
    lands in 'mid', deterministic on ties.
  - `build_difficulty_table` (value-based, reused): still partitions
    by exact 0 / strict-partial / exact 1.
  - `continuity_gate` (BUILD_PLAN §4 D3): selects rank terciles on
    continuous distributions and falls back to value bins on
    semi-bimodal distributions; enforces the frontier-size GO/NO-GO.
  - `build_chosen_table`: returns the right table for each regime.
  - `difficulty_table_sha256`: stable hash; round-trip from disk.
  - `BandBatchSampler` under `PURE_ON_BAND_MIX` works against an
    IFBench-style rank-tercile table (100/0/0 band membership).

These are pure-Python; no API, no dataset load.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.band_sampler import PURE_ON_BAND_MIX, BandBatchSampler
from src.difficulty import (
    BAND_NAMES,
    GATE_EXTREME_MASS_MAX,
    GATE_FRONTIER_MIN_GO,
    GATE_MIDDLE_PARTIAL_MIN,
    DifficultyTable,
    build_chosen_table,
    build_difficulty_table,
    build_rank_tercile_table,
    continuity_gate,
    difficulty_table_sha256,
)


# ---------------------------------------------------------------------------
# Rank-tercile partition
# ---------------------------------------------------------------------------


class TestRankTercilePartition:
    def test_exact_thirds_when_divisible_by_3(self):
        # 150 ids -> 50/50/50 (the IFBench D_feedback target shape).
        rng = random.Random(20260530)
        scores = [rng.uniform(0.05, 0.95) for _ in range(150)]
        t = build_rank_tercile_table(scores)
        assert len(t.ids("hard")) == 50
        assert len(t.ids("mid")) == 50
        assert len(t.ids("easy")) == 50
        # Ordering invariant: every hard score <= every mid score <= every easy score.
        h_max = max(scores[i] for i in t.ids("hard"))
        m_min = min(scores[i] for i in t.ids("mid"))
        m_max = max(scores[i] for i in t.ids("mid"))
        e_min = min(scores[i] for i in t.ids("easy"))
        assert h_max <= m_min
        assert m_max <= e_min

    def test_remainder_lands_in_mid(self):
        # n=151 -> 50/51/50; n=152 -> 50/52/50.
        scores_151 = [i / 150.0 for i in range(151)]
        t = build_rank_tercile_table(scores_151)
        assert (len(t.ids("hard")), len(t.ids("mid")), len(t.ids("easy"))) == (50, 51, 50)
        scores_152 = [i / 151.0 for i in range(152)]
        t = build_rank_tercile_table(scores_152)
        assert (len(t.ids("hard")), len(t.ids("mid")), len(t.ids("easy"))) == (50, 52, 50)

    def test_disjoint_and_complete(self):
        scores = [0.1 * i for i in range(10)] + [0.5] * 5  # n=15, 5/5/5
        t = build_rank_tercile_table(scores)
        all_ids = set(t.ids("hard")) | set(t.ids("mid")) | set(t.ids("easy"))
        assert all_ids == set(range(15))
        assert len(t.ids("hard")) == 5
        assert len(t.ids("mid")) == 5
        assert len(t.ids("easy")) == 5

    def test_ties_broken_deterministically_by_id(self):
        # All scores equal: sort is by id ascending, so partition is fully predictable.
        scores = [0.5] * 12  # 4/4/4
        t = build_rank_tercile_table(scores)
        assert t.ids("hard") == (0, 1, 2, 3)
        assert t.ids("mid") == (4, 5, 6, 7)
        assert t.ids("easy") == (8, 9, 10, 11)

    def test_out_of_range_score_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            build_rank_tercile_table([0.5, 1.1, 0.0])
        with pytest.raises(ValueError, match="must be in"):
            build_rank_tercile_table([-0.1, 0.5, 1.0])

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_rank_tercile_table([])


# ---------------------------------------------------------------------------
# Value-bin partition (reused; light re-verification under Chunk 13)
# ---------------------------------------------------------------------------


class TestValueBinPartition:
    def test_value_bins_split_by_exact_extremes(self):
        scores = [0.0, 0.0, 0.3, 0.5, 0.7, 1.0, 1.0, 1.0]
        t = build_difficulty_table(scores)
        assert sorted(t.ids("hard")) == [0, 1]
        assert sorted(t.ids("mid")) == [2, 3, 4]
        assert sorted(t.ids("easy")) == [5, 6, 7]

    def test_value_bins_all_partial(self):
        scores = [0.25, 0.5, 0.75]
        t = build_difficulty_table(scores)
        assert len(t.ids("hard")) == 0
        assert len(t.ids("easy")) == 0
        assert sorted(t.ids("mid")) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Continuity gate (BUILD_PLAN §4 D3)
# ---------------------------------------------------------------------------


class TestContinuityGate:
    @staticmethod
    def _continuous_scores(n: int = 150, seed: int = 0) -> list[float]:
        """Uniform-in-(0,1) scores: middle tercile is 100% strictly
        partial, extreme mass is 0%."""
        rng = random.Random(seed)
        return [rng.uniform(0.01, 0.99) for _ in range(n)]

    @staticmethod
    def _bimodal_scores(n_zero: int, n_one: int, n_partial: int) -> list[float]:
        """Bimodal: many at 0, many at 1, few partials. Extreme mass
        large; middle tercile diluted by extremes."""
        return (
            [0.0] * n_zero + [0.5] * n_partial + [1.0] * n_one
        )

    def test_continuous_distribution_selects_rank_terciles(self):
        scores = self._continuous_scores(n=150, seed=42)
        gate = continuity_gate(scores)
        assert gate["middle_tercile_partial_frac"] >= GATE_MIDDLE_PARTIAL_MIN
        assert gate["extreme_mass_frac"] < GATE_EXTREME_MASS_MAX
        assert gate["rank_terciles_stand"] is True
        assert gate["chosen_binning"] == "rank_terciles"

    def test_bimodal_distribution_falls_back_to_value_bins(self):
        # 75 at 0, 75 at 1, 0 partials -> extreme mass = 100%, fails.
        scores = self._bimodal_scores(n_zero=75, n_one=75, n_partial=0)
        gate = continuity_gate(scores)
        assert gate["extreme_mass_frac"] >= GATE_EXTREME_MASS_MAX
        assert gate["rank_terciles_stand"] is False
        assert gate["chosen_binning"] == "value_bins"

    def test_semi_bimodal_distribution_falls_back_to_value_bins(self):
        # 60 at 0, 60 at 1, 30 partials: middle tercile (50 of 150) gets
        # split between extremes and partials. Either the partial
        # fraction is < 80% or the extreme mass >= 50% (here both).
        scores = self._bimodal_scores(n_zero=60, n_one=60, n_partial=30)
        gate = continuity_gate(scores)
        assert (
            (gate["middle_tercile_partial_frac"] < GATE_MIDDLE_PARTIAL_MIN)
            or (gate["extreme_mass_frac"] >= GATE_EXTREME_MASS_MAX)
        )
        assert gate["chosen_binning"] == "value_bins"

    def test_partial_ok_but_extreme_mass_fails_falls_back(self):
        # 50 at 0, 50 partials, 50 at 1: sorted, middle tercile (positions
        # 50..99) is entirely strict-partial -> partial_frac=100% OK.
        # Extreme mass = 100/150 = 0.667 -> FAIL. Either-clause-fails
        # triggers the value-bin fallback.
        scores = [0.0] * 50 + [0.5] * 50 + [1.0] * 50
        gate = continuity_gate(scores)
        assert gate["middle_tercile_partial_frac"] == pytest.approx(1.0)
        assert gate["middle_tercile_partial_ok"] is True
        assert gate["extreme_mass_frac"] == pytest.approx(100 / 150)
        assert gate["extreme_mass_ok"] is False
        assert gate["chosen_binning"] == "value_bins"

    def test_extreme_mass_ok_but_partial_fails_falls_back(self):
        # 61 zeros + 89 partials, no ones. Sorted: positions 0..60 are
        # 0.0, 61..149 are 0.5. Middle tercile (50..99) has 11 zeros and
        # 39 partials -> partial_frac = 39/50 = 0.78 < 0.80 -> FAIL.
        # Extreme mass = 61/150 ~= 0.407 < 0.50 -> OK. Either-clause-fails
        # triggers the value-bin fallback.
        scores = [0.0] * 61 + [0.5] * 89
        gate = continuity_gate(scores)
        assert gate["extreme_mass_ok"] is True
        assert gate["middle_tercile_partial_frac"] == pytest.approx(39 / 50)
        assert gate["middle_tercile_partial_ok"] is False
        assert gate["chosen_binning"] == "value_bins"

    def test_frontier_size_go_criterion(self):
        # 150 strict-partial scores -> rank terciles, frontier=50 -> GO.
        gate = continuity_gate([0.5] * 150)
        assert gate["chosen_binning"] == "rank_terciles"
        assert gate["frontier_size_under_chosen"] == 50
        assert gate["frontier_ok"] is True
        assert gate["verdict"] == "GO"

    def test_frontier_size_nogo_under_value_bins(self):
        # 100 zeros, 100 ones, 10 partials -> value bins; frontier=10 -> NO-GO.
        scores = [0.0] * 100 + [1.0] * 100 + [0.5] * 10
        gate = continuity_gate(scores)
        assert gate["chosen_binning"] == "value_bins"
        assert gate["frontier_size_under_chosen"] == 10
        assert gate["frontier_size_under_chosen"] < GATE_FRONTIER_MIN_GO
        assert gate["frontier_ok"] is False
        assert gate["verdict"] == "NO-GO"

    def test_frontier_at_threshold_is_go(self):
        # Construct value-bin frontier of exactly 20.
        scores = [0.0] * 100 + [1.0] * 100 + [0.5] * 20
        gate = continuity_gate(scores)
        assert gate["chosen_binning"] == "value_bins"
        assert gate["frontier_size_under_chosen"] == 20
        assert gate["verdict"] == "GO"


# ---------------------------------------------------------------------------
# Dispatcher: build_chosen_table picks the right table for each regime
# ---------------------------------------------------------------------------


class TestBuildChosenTable:
    def test_continuous_returns_rank_tercile_table(self):
        rng = random.Random(13)
        scores = [rng.uniform(0.01, 0.99) for _ in range(150)]
        table, gate = build_chosen_table(scores)
        assert gate["chosen_binning"] == "rank_terciles"
        assert len(table.ids("hard")) == 50
        assert len(table.ids("mid")) == 50
        assert len(table.ids("easy")) == 50

    def test_bimodal_returns_value_bin_table(self):
        scores = [0.0] * 60 + [1.0] * 60 + [0.5] * 30
        table, gate = build_chosen_table(scores)
        assert gate["chosen_binning"] == "value_bins"
        assert all(table.scores[i] == 0.0 for i in table.ids("hard"))
        assert all(table.scores[i] == 1.0 for i in table.ids("easy"))
        assert all(0.0 < table.scores[i] < 1.0 for i in table.ids("mid"))


# ---------------------------------------------------------------------------
# Hash + frozen table
# ---------------------------------------------------------------------------


class TestDifficultyTableFrozenAndHashed:
    def test_table_save_load_roundtrip_under_rank_terciles(self, tmp_path: Path):
        rng = random.Random(7)
        scores = [rng.uniform(0.01, 0.99) for _ in range(150)]
        t = build_rank_tercile_table(scores)
        path = tmp_path / "difficulty_table.json"
        t.save(path)
        t2 = DifficultyTable.load(path)
        assert t2.scores == t.scores
        for b in BAND_NAMES:
            assert t2.ids(b) == t.ids(b)

    def test_table_is_immutable(self):
        t = build_rank_tercile_table([0.1, 0.5, 0.9])
        with pytest.raises(Exception):
            t.scores = (0.2, 0.5, 0.9)  # type: ignore[misc]
        # Mapping proxies reject mutation.
        with pytest.raises(Exception):
            t.ids_in_band["mid"] = (0, 1, 2)  # type: ignore[index]

    def test_sha256_is_stable_and_round_trips_from_disk(self, tmp_path: Path):
        t = build_rank_tercile_table([0.2, 0.5, 0.8, 0.1, 0.9])
        path = tmp_path / "t.json"
        t.save(path)
        h1 = difficulty_table_sha256(path)
        h2 = difficulty_table_sha256(path)
        assert h1 == h2 and len(h1) == 64
        # Verify it tracks JSON content, not in-memory state.
        manual_h = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        assert h1 == manual_h

    def test_sha256_changes_when_table_changes(self, tmp_path: Path):
        t1 = build_rank_tercile_table([0.2, 0.5, 0.8])
        t2 = build_rank_tercile_table([0.3, 0.5, 0.8])
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        t1.save(p1)
        t2.save(p2)
        assert difficulty_table_sha256(p1) != difficulty_table_sha256(p2)


# ---------------------------------------------------------------------------
# 100/0/0 band membership on an IFBench-style rank-tercile table
# ---------------------------------------------------------------------------


class _StubLoader150:
    """Minimal loader exposing 150 ids: enough for the IFBench-style
    difficulty table built in the tests below. Mirrors the Chunk-4
    test infra (no gepa dependency)."""

    def all_ids(self):
        return list(range(150))

    def __len__(self):
        return 150


class TestPureOnBandIFBenchTable:
    def test_pure_on_band_frontier_only_under_rank_terciles(self):
        """An IFBench-style rank-tercile difficulty table (150 ids, 50
        per band) is sampled under PURE_ON_BAND_MIX with target='mid'.
        Every drawn id must be in the mid band; off-band leakage must be
        zero. This is the IFBench analogue of the Chunk-8 100/0/0
        regression test on HotpotQA's value-bin table."""
        rng = random.Random(20260530)
        scores = [rng.uniform(0.01, 0.99) for _ in range(150)]
        t = build_rank_tercile_table(scores)
        mid_ids = set(t.ids("mid"))
        assert len(mid_ids) == 50

        sampler = BandBatchSampler(
            difficulty_table=t,
            target_band="mid",
            b=3,
            rng=random.Random(0),
            mix=PURE_ON_BAND_MIX,
        )
        loader = _StubLoader150()
        drawn: Counter[int] = Counter()
        for _ in range(10_000):
            for did in sampler.next_minibatch_ids(loader, MagicMock()):
                drawn[did] += 1

        assert set(drawn) <= mid_ids
        assert sum(c for did, c in drawn.items() if did not in mid_ids) == 0
        # Coverage: 10k draws of size 3 over 50 ids should touch every id.
        assert len(drawn) == 50

    def test_pure_on_band_hard_only_under_rank_terciles(self):
        rng = random.Random(20260530)
        scores = [rng.uniform(0.01, 0.99) for _ in range(150)]
        t = build_rank_tercile_table(scores)
        hard_ids = set(t.ids("hard"))
        sampler = BandBatchSampler(
            difficulty_table=t,
            target_band="hard",
            b=3,
            rng=random.Random(1),
            mix=PURE_ON_BAND_MIX,
        )
        loader = _StubLoader150()
        drawn: Counter[int] = Counter()
        for _ in range(1_000):
            for did in sampler.next_minibatch_ids(loader, MagicMock()):
                drawn[did] += 1
        assert set(drawn) <= hard_ids
        assert sum(c for did, c in drawn.items() if did not in hard_ids) == 0
