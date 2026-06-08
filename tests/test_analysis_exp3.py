"""Offline unit tests for src/analysis_exp3.py pure helpers."""

from __future__ import annotations

from src.analysis_exp3 import T, _cumulative, _md_table, _mean


def _records(accept_iters):
    return [{"iter": i, "accepted": i in accept_iters} for i in range(T)]


def test_cumulative_step_and_total():
    cum = _cumulative(_records({1, 5, 20}))
    assert len(cum) == T
    assert cum[0] == 0
    assert cum[1] == 1
    assert cum[4] == 1
    assert cum[5] == 2
    assert cum[19] == 2
    assert cum[20] == 3
    assert cum[-1] == 3  # total == number of accepts


def test_cumulative_all_zero():
    assert _cumulative(_records(set())) == [0] * T


def test_cumulative_monotonic_nondecreasing():
    cum = _cumulative(_records({3, 4, 4, 30}))  # dup ignored by set
    assert all(b >= a for a, b in zip(cum, cum[1:]))


def test_mean():
    assert _mean([1, 2, 3]) == 2.0
    assert _mean([0, 0, 0]) == 0.0


def test_md_table_shape():
    md = _md_table(["a", "b"], [["1", "2"], ["3", "4"]])
    lines = md.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 2 |"
    assert lines[3] == "| 3 | 4 |"
