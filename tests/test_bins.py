"""Offline tests for src/bins.py (Exp-3 value-based difficulty bins).

No network or model calls. The real-table test reads the committed
results/difficulty_table.json (a static artifact), not a live scoring pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bins import (
    BIN_NAMES,
    EASY_MIN,
    HARD_MAX,
    MIN_MEDIUM_MEMBERS,
    assign_bin,
    build_bins,
    load_bins,
)

REPO = Path(__file__).resolve().parents[1]
TABLE_PATH = REPO / "experiments" / "exp1_bands" / "difficulty_table.json"


class TestAssignBin:
    def test_thresholds_and_edges(self):
        # easy at/above EASY_MIN
        assert assign_bin(1.0) == "easy"
        assert assign_bin(EASY_MIN) == "easy"
        assert assign_bin(0.995) == "easy"
        # hard at/below HARD_MAX
        assert assign_bin(0.0) == "hard"
        assert assign_bin(HARD_MAX) == "hard"
        assert assign_bin(0.005) == "hard"
        # medium strictly interior
        assert assign_bin(0.5) == "medium"
        assert assign_bin(0.989) == "medium"
        assert assign_bin(0.011) == "medium"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            assign_bin(-0.01)
        with pytest.raises(ValueError):
            assign_bin(1.01)


class TestBuildBins:
    def test_partition_is_disjoint_and_complete(self):
        scores = [0.0, 1.0, 0.5, 0.99, 0.01, 0.6666, 1.0, 0.0]
        bins = build_bins(scores)
        members = {b: bins.get_bin_members(b) for b in BIN_NAMES}
        flat = sorted(i for ids in members.values() for i in ids)
        assert flat == list(range(len(scores)))  # complete
        # disjoint
        seen: set[int] = set()
        for ids in members.values():
            for i in ids:
                assert i not in seen
                seen.add(i)
        # correct assignment
        assert members["hard"] == [0, 4, 7]      # 0.0, 0.01, 0.0
        assert members["easy"] == [1, 3, 6]      # 1.0, 0.99, 1.0
        assert members["medium"] == [2, 5]       # 0.5, 0.6666

    def test_members_sorted_ascending(self):
        scores = [0.5, 0.0, 0.5, 0.0, 0.5]
        bins = build_bins(scores)
        assert bins.get_bin_members("medium") == [0, 2, 4]
        assert bins.get_bin_members("hard") == [1, 3]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_bins([])

    def test_unknown_bin_name_raises(self):
        bins = build_bins([0.0, 1.0, 0.5])
        with pytest.raises(ValueError):
            bins.get_bin_members("frontier")  # type: ignore[arg-type]

    def test_all_ids_and_bin_of(self):
        scores = [0.0, 1.0, 0.5]
        bins = build_bins(scores)
        assert bins.all_ids == [0, 1, 2]
        assert bins.bin_of(0) == "hard"
        assert bins.bin_of(1) == "easy"
        assert bins.bin_of(2) == "medium"


class TestRealTable:
    """Lock in the headline counts from the frozen Exp-3 difficulty table."""

    def test_frozen_table_counts(self):
        bins = load_bins(TABLE_PATH)
        counts = bins.counts()
        assert bins.summary()["n"] == 150
        # Provenance-verified value-bin counts (Chunk 5: "31 partials").
        assert counts == {"easy": 69, "medium": 31, "hard": 50}

    def test_medium_bin_gate(self):
        bins = load_bins(TABLE_PATH)
        assert len(bins.get_bin_members("medium")) >= MIN_MEDIUM_MEMBERS

    def test_summary_matches_table_scores(self):
        scores = json.loads(TABLE_PATH.read_text())["scores"]
        bins = load_bins(TABLE_PATH)
        assert bins.summary()["n"] == len(scores)
        # histogram counts sum to n
        assert sum(bins.summary()["histogram"].values()) == len(scores)
