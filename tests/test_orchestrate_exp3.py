"""Offline tests for the Exp-3 matrix orchestrator (Chunk 5 dry-run gate).

No model calls: these test cell enumeration, the skip-if-complete logic, and
the todo/skip plan with synthetic summary files.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.orchestrate_exp3 import (
    ARMS,
    SEEDS,
    T,
    all_cells,
    cell_complete,
    cell_summary_path,
    plan,
)


def _write_summary(out_root: Path, arm: str, seed: int, *, n_iters=T, test_f1=0.5, arm_override=None, t_override=None):
    p = cell_summary_path(out_root, arm, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "arm": arm_override or arm,
        "seed": seed,
        "T": t_override or T,
        "first_accept_iter": 1,
        "cumulative_accepts": 5,
        "final_test_f1": test_f1,
        "iterations": [{"iter": i} for i in range(n_iters)],
    }))


def test_enumerates_12_cells():
    cells = all_cells()
    assert len(cells) == 12
    assert set(cells) == {(a, s) for a in ARMS for s in SEEDS}
    assert len(ARMS) == 4 and tuple(SEEDS) == (0, 1, 2)


def test_plan_all_todo_when_empty(tmp_path):
    todo, skipped = plan(tmp_path, T)
    assert len(todo) == 12 and skipped == []


def test_skip_if_complete(tmp_path):
    _write_summary(tmp_path, "random", 0)
    _write_summary(tmp_path, "easy_to_hard", 2)
    todo, skipped = plan(tmp_path, T)
    assert set(skipped) == {("random", 0), ("easy_to_hard", 2)}
    assert len(todo) == 10
    assert ("random", 0) not in todo


def test_incomplete_summary_not_skipped(tmp_path):
    # Too few iteration records -> not complete -> still todo.
    _write_summary(tmp_path, "hard_to_easy", 1, n_iters=T - 1)
    assert not cell_complete(tmp_path, "hard_to_easy", 1, T)
    todo, _ = plan(tmp_path, T)
    assert ("hard_to_easy", 1) in todo


def test_null_test_f1_not_complete(tmp_path):
    p = cell_summary_path(tmp_path, "static_medium", 0)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "arm": "static_medium", "seed": 0, "T": T,
        "final_test_f1": None, "iterations": [{"iter": i} for i in range(T)],
    }))
    assert not cell_complete(tmp_path, "static_medium", 0, T)


def test_mismatched_arm_or_T_not_complete(tmp_path):
    _write_summary(tmp_path, "random", 1, arm_override="WRONG")
    assert not cell_complete(tmp_path, "random", 1, T)
    _write_summary(tmp_path, "random", 2, t_override=8)
    assert not cell_complete(tmp_path, "random", 2, T)


def test_corrupt_summary_not_complete(tmp_path):
    p = cell_summary_path(tmp_path, "random", 0)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json")
    assert not cell_complete(tmp_path, "random", 0, T)


def test_all_complete_then_nothing_todo(tmp_path):
    for arm, seed in all_cells():
        _write_summary(tmp_path, arm, seed)
    todo, skipped = plan(tmp_path, T)
    assert todo == [] and len(skipped) == 12
