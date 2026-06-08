"""Offline mocked dry-run for the Exp-3 curriculum runner (Chunk 3 gate).

No network or model calls: a StubAdapter returns deterministic scores and
proposes deterministic candidate edits, so the full engine + decoupled
proposer + curriculum sampler + per-iteration logging run end-to-end and we
assert the emitted JSON has the right schema, the schedule alignment holds,
and per-cell files are isolated (concurrency-safe).
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest

from gepa import EvaluationBatch

from src.bins import build_bins
from src.curriculum_runner import EXP3_ARMS, run_cell
from src.schedule import bin_for_iteration

COMPONENTS = ["summarize1", "summarize2", "final_answer"]


class _StubExample(dict):
    @property
    def id(self):
        return self["id"]


def _ex(idx: int) -> _StubExample:
    return _StubExample({"id": f"ex_{idx}", "question": f"q{idx}", "answer": f"a{idx}"})


class StubAdapter:
    """Deterministic offline adapter. Score grows with the number of '|v'
    edits accumulated in the candidate text, so newer candidates beat their
    parents on the A-batch and the engine accepts -- exercising the accept
    path. Capped below 1.0 so skip_perfect_score never fires."""

    def __init__(self, components):
        self.components = components
        self.evaluate_calls = []

    def _score(self, candidate):
        edits = sum(str(candidate.get(c, "")).count("|v") for c in self.components)
        return min(0.05 + 0.1 * edits, 0.95)

    def evaluate(self, batch, candidate, capture_traces=False):
        score = self._score(candidate)
        outputs = [dspy.Prediction(answer="stub") for _ in batch]
        scores = [score for _ in batch]
        trajectories = None
        if capture_traces:
            trajectories = [
                {"prediction": o, "trace": [], "example": ex, "score": s}
                for ex, o, s in zip(batch, outputs, scores)
            ]
        ids = tuple(ex["id"] for ex in batch)
        self.evaluate_calls.append((ids, capture_traces))
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {
            name: [{"Inputs": {"q": "x"}, "Generated Outputs": {"a": "y"}, "Feedback": "z"}]
            for name in components_to_update
        }

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        return {name: f"{candidate.get(name, '')} |v" for name in components_to_update}


def _bins_12():
    # easy=[0,1,2,3], medium=[4,5,6,7], hard=[8,9,10,11]
    scores = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    return build_bins(scores), scores


def _run(arm, seed, tmp_path, T=9):
    bins, scores = _bins_12()
    d_feedback = [_ex(i) for i in range(len(scores))]
    accept_batch = [_ex(1000 + i) for i in range(4)]
    d_pareto = [_ex(2000 + i) for i in range(4)]
    test = [_ex(3000 + i) for i in range(4)]
    out = Path(tmp_path) / "exp3"
    summary = run_cell(
        arm,
        seed,
        d_feedback=d_feedback,
        accept_batch=accept_batch,
        d_pareto=d_pareto,
        test=test,
        bins=bins,
        adapter=StubAdapter(COMPONENTS),
        seed_candidate={c: f"seed_{c}" for c in COMPONENTS},
        run_dir=out / arm / f"seed{seed}",
        summary_path=out / arm / f"seed{seed}.json",
        T=T,
    )
    return summary, out


SUMMARY_KEYS = {
    "arm", "seed", "T", "first_accept_iter", "cumulative_accepts",
    "final_test_f1", "best_candidate_idx", "iterations",
}
RECORD_KEYS = {"iter", "bin_sampled", "minibatch_ids", "parent_a20", "candidate_a20", "accepted"}


class TestSchema:
    def test_summary_written_and_well_formed(self, tmp_path):
        summary, out = _run("easy_to_hard", 0, tmp_path)
        path = out / "easy_to_hard" / "seed0.json"
        assert path.exists()
        on_disk = json.loads(path.read_text())
        assert on_disk == summary
        assert set(summary.keys()) == SUMMARY_KEYS
        assert summary["arm"] == "easy_to_hard"
        assert summary["seed"] == 0
        assert summary["T"] == 9
        assert isinstance(summary["final_test_f1"], float)
        assert 0.0 <= summary["final_test_f1"] <= 1.0

    def test_iteration_records_schema(self, tmp_path):
        summary, _ = _run("easy_to_hard", 0, tmp_path)
        recs = summary["iterations"]
        assert 0 < len(recs) <= summary["T"]
        for r in recs:
            assert set(r.keys()) == RECORD_KEYS
            assert isinstance(r["iter"], int)
            assert isinstance(r["accepted"], bool)
            assert isinstance(r["minibatch_ids"], list) and len(r["minibatch_ids"]) == 3
            for v in (r["parent_a20"], r["candidate_a20"]):
                assert v is None or (0.0 <= v <= 1.0)

    def test_iters_are_zero_based_and_increasing(self, tmp_path):
        summary, _ = _run("easy_to_hard", 0, tmp_path)
        iters = [r["iter"] for r in summary["iterations"]]
        assert iters == sorted(iters)
        assert iters[0] >= 0 and iters[-1] < summary["T"]

    def test_per_iter_jsonl_matches_records(self, tmp_path):
        summary, out = _run("hard_to_easy", 1, tmp_path)
        jsonl = out / "hard_to_easy" / "seed1" / "per_iter.jsonl"
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == len(summary["iterations"])
        assert [json.loads(l) for l in lines] == summary["iterations"]


class TestScheduleAlignment:
    def test_bin_sampled_matches_schedule(self, tmp_path):
        for arm in ("easy_to_hard", "hard_to_easy", "static_medium"):
            summary, _ = _run(arm, 0, tmp_path)
            for r in summary["iterations"]:
                assert r["bin_sampled"] == bin_for_iteration(arm, r["iter"], summary["T"])

    def test_binned_minibatch_within_scheduled_bin(self, tmp_path):
        bins, _ = _bins_12()
        for arm in ("easy_to_hard", "hard_to_easy", "static_medium"):
            summary, _ = _run(arm, 0, tmp_path)
            for r in summary["iterations"]:
                members = bins.get_bin_members(r["bin_sampled"])
                assert all(i in members for i in r["minibatch_ids"])

    def test_random_arm_has_no_bin(self, tmp_path):
        summary, _ = _run("random", 0, tmp_path)
        for r in summary["iterations"]:
            assert r["bin_sampled"] is None
            assert all(0 <= i < 12 for i in r["minibatch_ids"])


class TestAcceptLogic:
    def test_accept_counters_consistent(self, tmp_path):
        summary, _ = _run("easy_to_hard", 0, tmp_path)
        accepted = [r["iter"] for r in summary["iterations"] if r["accepted"]]
        assert summary["cumulative_accepts"] == len(accepted)
        assert summary["first_accept_iter"] == (min(accepted) if accepted else None)

    def test_accept_path_fires(self, tmp_path):
        # The increasing-score stub should produce at least one accept.
        summary, _ = _run("static_medium", 0, tmp_path)
        assert summary["cumulative_accepts"] >= 1
        assert summary["first_accept_iter"] is not None


class TestIsolationAndDeterminism:
    def test_cells_write_distinct_files(self, tmp_path):
        s1, out = _run("easy_to_hard", 0, tmp_path)
        s2, _ = _run("hard_to_easy", 2, tmp_path)
        p1 = out / "easy_to_hard" / "seed0.json"
        p2 = out / "hard_to_easy" / "seed2.json"
        assert p1.exists() and p2.exists() and p1 != p2
        # No cross-contamination of the in-memory summaries.
        assert s1["arm"] == "easy_to_hard" and s2["arm"] == "hard_to_easy"

    def test_seeded_determinism_end_to_end(self, tmp_path):
        a, _ = _run("easy_to_hard", 0, tmp_path / "a")
        b, _ = _run("easy_to_hard", 0, tmp_path / "b")
        a_ids = [r["minibatch_ids"] for r in a["iterations"]]
        b_ids = [r["minibatch_ids"] for r in b["iterations"]]
        assert a_ids == b_ids
        assert a["cumulative_accepts"] == b["cumulative_accepts"]

    def test_different_seed_changes_draws(self, tmp_path):
        a, _ = _run("easy_to_hard", 0, tmp_path / "a")
        b, _ = _run("easy_to_hard", 1, tmp_path / "b")
        a_ids = [r["minibatch_ids"] for r in a["iterations"]]
        b_ids = [r["minibatch_ids"] for r in b["iterations"]]
        assert a_ids != b_ids


class TestValidation:
    def test_unknown_arm_raises(self, tmp_path):
        with pytest.raises(ValueError):
            _run("static_frontier", 0, tmp_path)

    def test_all_four_arms_run(self, tmp_path):
        for arm in EXP3_ARMS:
            summary, _ = _run(arm, 0, tmp_path / arm)
            assert summary["arm"] == arm
            assert len(summary["iterations"]) > 0
