"""Offline tests for src/schedule.py (Exp-3 curriculum schedule).

Pure functions only: no network, no model calls. All randomness is driven by
an explicitly seeded random.Random instance.
"""

from __future__ import annotations

import logging
import random

import pytest

from src.bins import build_bins, load_bins
from src.schedule import (
    ARMS,
    bin_for_iteration,
    phase_for_iteration,
    sample_minibatch,
)

T = 40

# Expected phase windows for T=40: 0-13, 14-26, 27-39.
PHASE_WINDOWS = [(0, 13), (14, 26), (27, 39)]


def _phase_of(it: int) -> int:
    for p, (lo, hi) in enumerate(PHASE_WINDOWS):
        if lo <= it <= hi:
            return p
    raise AssertionError(it)


class TestPhaseBoundaries:
    def test_windows_for_T40(self):
        assert [phase_for_iteration(i, T) for i in range(T)] == [
            _phase_of(i) for i in range(T)
        ]
        # explicit edges
        assert phase_for_iteration(13, T) == 0
        assert phase_for_iteration(14, T) == 1
        assert phase_for_iteration(26, T) == 1
        assert phase_for_iteration(27, T) == 2
        assert phase_for_iteration(39, T) == 2

    def test_out_of_range_and_bad_T(self):
        with pytest.raises(ValueError):
            phase_for_iteration(-1, T)
        with pytest.raises(ValueError):
            phase_for_iteration(T, T)
        with pytest.raises(ValueError):
            phase_for_iteration(0, 0)


class TestBinForIteration:
    def test_easy_to_hard_sequence(self):
        seq = [bin_for_iteration("easy_to_hard", i, T) for i in range(T)]
        expected = ["easy"] * 14 + ["medium"] * 13 + ["hard"] * 13
        assert seq == expected

    def test_hard_to_easy_sequence(self):
        seq = [bin_for_iteration("hard_to_easy", i, T) for i in range(T)]
        expected = ["hard"] * 14 + ["medium"] * 13 + ["easy"] * 13
        assert seq == expected

    def test_static_medium_sequence(self):
        seq = [bin_for_iteration("static_medium", i, T) for i in range(T)]
        assert seq == ["medium"] * T

    def test_random_ignores_bins(self):
        seq = [bin_for_iteration("random", i, T) for i in range(T)]
        assert seq == [None] * T

    def test_unknown_arm_raises(self):
        with pytest.raises(ValueError):
            bin_for_iteration("static_frontier", 0, T)

    def test_iter_out_of_range_raises_for_all_arms(self):
        for arm in ARMS:
            with pytest.raises(ValueError):
                bin_for_iteration(arm, T, T)
            with pytest.raises(ValueError):
                bin_for_iteration(arm, -1, T)


class TestSampleMinibatch:
    # easy=[0,1,2], medium=[3,4,5,6], hard=[7,8,9]
    SCORES = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

    def _bins(self):
        return build_bins(self.SCORES)

    def _all_ids(self):
        return list(range(len(self.SCORES)))

    def test_seeded_determinism_same_seed(self):
        bins, all_ids = self._bins(), self._all_ids()
        for arm in ARMS:
            a = sample_minibatch(arm, 0, T, random.Random(123), bins, all_ids)
            b = sample_minibatch(arm, 0, T, random.Random(123), bins, all_ids)
            assert a == b, arm

    def test_different_seed_usually_differs(self):
        bins, all_ids = self._bins(), self._all_ids()
        draws = {
            tuple(
                sample_minibatch("random", 0, T, random.Random(s), bins, all_ids)
            )
            for s in range(20)
        }
        # 20 seeds over C(10,3) ordered samples should not collapse to one draw.
        assert len(draws) > 1

    def test_binned_draws_within_scheduled_bin(self):
        bins, all_ids = self._bins(), self._all_ids()
        rng = random.Random(0)
        # easy_to_hard: iter 0 -> easy, iter 20 -> medium, iter 35 -> hard
        easy = sample_minibatch("easy_to_hard", 0, T, rng, bins, all_ids)
        medium = sample_minibatch("easy_to_hard", 20, T, rng, bins, all_ids)
        hard = sample_minibatch("easy_to_hard", 35, T, rng, bins, all_ids)
        assert all(i in bins.get_bin_members("easy") for i in easy)
        assert all(i in bins.get_bin_members("medium") for i in medium)
        assert all(i in bins.get_bin_members("hard") for i in hard)

    def test_static_medium_always_medium(self):
        bins, all_ids = self._bins(), self._all_ids()
        rng = random.Random(7)
        for it in (0, 13, 14, 26, 27, 39):
            draw = sample_minibatch("static_medium", it, T, rng, bins, all_ids)
            assert all(i in bins.get_bin_members("medium") for i in draw)

    def test_returns_b_ids(self):
        bins, all_ids = self._bins(), self._all_ids()
        rng = random.Random(1)
        for b in (1, 3, 5):
            assert len(sample_minibatch("static_medium", 0, T, rng, bins, all_ids, b=b)) == b
        # random is without replacement -> b distinct
        rnd = sample_minibatch("random", 0, T, random.Random(1), bins, all_ids, b=3)
        assert len(rnd) == 3 and len(set(rnd)) == 3

    def test_random_within_all_ids(self):
        bins, all_ids = self._bins(), self._all_ids()
        draw = sample_minibatch("random", 5, T, random.Random(99), bins, all_ids, b=4)
        assert all(i in all_ids for i in draw)

    def test_with_replacement_when_bin_smaller_than_b(self, caplog):
        # medium bin has only 2 distinct members -> must still return b=3 with
        # replacement, and warn.
        scores = [0.0, 0.0, 0.5, 0.5, 1.0]  # medium = [2, 3]
        bins = build_bins(scores)
        all_ids = list(range(len(scores)))
        with caplog.at_level(logging.WARNING):
            draw = sample_minibatch("static_medium", 0, T, random.Random(3), bins, all_ids, b=3)
        assert len(draw) == 3
        assert all(i in (2, 3) for i in draw)
        assert any("with replacement" in r.message for r in caplog.records)

    def test_random_raises_when_too_few_ids(self):
        bins = build_bins([0.0, 1.0])
        with pytest.raises(ValueError):
            sample_minibatch("random", 0, T, random.Random(0), bins, [0, 1], b=3)

    def test_bad_b_raises(self):
        bins, all_ids = self._bins(), self._all_ids()
        with pytest.raises(ValueError):
            sample_minibatch("static_medium", 0, T, random.Random(0), bins, all_ids, b=0)


class TestWithFrozenTable:
    """Sanity against the real frozen Exp-3 bins (offline JSON read)."""

    def test_easy_to_hard_draws_from_real_bins(self):
        bins = load_bins()
        all_ids = bins.all_ids
        rng = random.Random(0)
        easy = sample_minibatch("easy_to_hard", 0, T, rng, bins, all_ids)
        medium = sample_minibatch("easy_to_hard", 20, T, rng, bins, all_ids)
        hard = sample_minibatch("easy_to_hard", 35, T, rng, bins, all_ids)
        assert all(bins.bin_of(i) == "easy" for i in easy)
        assert all(bins.bin_of(i) == "medium" for i in medium)
        assert all(bins.bin_of(i) == "hard" for i in hard)
