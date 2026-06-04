"""Chunk 14 offline wiring tests.

These cover the IFBench-substrate injection points the BUILD_PLAN §7
Chunk 14 wiring requires:

  - `Substrate` factory: `ifbench_substrate()` returns the IFBench bag
    of callables; the default `_default_hotpot_substrate()` still
    returns the HotpotQA bag so Experiment 1 / 1b call sites are
    unaffected.
  - The IFBench feedback function composes with `_make_feedback_map`
    and produces a ScoreWithFeedback with a non-empty feedback string.
  - The IFBench metric returns a float in [0, 1] on a fixture example.
  - The IFBench program + feedback compose with the decoupled-
    acceptance engine and the band sampler under PURE_ON_BAND_MIX
    (100/0/0): one fake iteration runs end to end without error.
  - The frozen Chunk-13 difficulty-table hash is verified at variant
    selection time; tampering or missing files raise loudly.

No LM calls. No dataset downloads. The end-to-end smoke uses the same
StubAdapter pattern as `tests/test_chunk4.py::TestDecoupledAcceptance`,
but with the IFBench substrate's component names so the wiring is
exercised under the IFBench shape.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import dspy
import pytest

from gepa.core.engine import GEPAEngine
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import MaxCandidateProposalsStopper

from src.band_sampler import PURE_ON_BAND_MIX, BandBatchSampler
from src.decoupled_proposer import DecoupledReflectiveMutationProposer
from src.difficulty import build_rank_tercile_table
from src.ifbench_program import IFBenchAnswer, IFBenchProgram, build_program as ifbench_build_program
from src.ifbench_substrate import (
    EXPECTED_IFBENCH_DIFFICULTY_HASH,
    IFBENCH_DIFFICULTY_PATH,
    ifbench_substrate,
    verify_ifbench_difficulty_table_hash,
)
from src.run_gepa import Substrate, _default_hotpot_substrate, _make_feedback_map


# ---------------------------------------------------------------------------
# Substrate factories
# ---------------------------------------------------------------------------


class TestSubstrateFactories:
    def test_default_hotpot_substrate_shape(self):
        sub = _default_hotpot_substrate()
        assert isinstance(sub, Substrate)
        assert sub.name == "hotpot"
        # HotpotQA is the 3-module program: summarize1 / summarize2 / final_answer.
        assert "final_answer" in sub.component_names
        assert callable(sub.build_program)
        assert callable(sub.metric_fn)
        assert callable(sub.feedback_fn)

    def test_ifbench_substrate_shape(self):
        sub = ifbench_substrate()
        assert isinstance(sub, Substrate)
        assert sub.name == "ifbench"
        assert sub.component_names == ("answer",)
        program = sub.build_program()
        assert isinstance(program, IFBenchProgram)
        # Seed instruction is loaded and non-empty.
        seed = program.get_module_instruction("answer")
        assert isinstance(seed, str) and len(seed) > 0

    def test_default_and_ifbench_substrates_are_independent(self):
        """The two factories must return distinct Substrate instances
        whose component_names do not overlap by accident."""
        hotpot = _default_hotpot_substrate()
        ifbench = ifbench_substrate()
        assert hotpot.name != ifbench.name
        assert hotpot.component_names != ifbench.component_names


# ---------------------------------------------------------------------------
# Feedback map + metric on a fixture
# ---------------------------------------------------------------------------


def _ifbench_fixture_example() -> dspy.Example:
    """A 2-constraint IFBench-shaped example. Constraints are chosen so
    that a long-all-uppercase English response satisfies all of them:

      - punctuation:no_comma (no commas in response)
      - change_case:english_capital (English text in uppercase, needs
        >= 2 sentences for langdetect to classify reliably).
    """
    return dspy.Example(
        id=1,
        source_key="fixture-1",
        prompt="Write a multi-sentence response in all uppercase English "
               "with no commas.",
        instruction_id_list=["punctuation:no_comma", "change_case:english_capital"],
        kwargs_list=[None, None],
        constraint_count=2,
    ).with_inputs("prompt", "instruction_id_list", "kwargs_list")


def _two_sentence_english_caps() -> str:
    """A multi-sentence English string in caps with no commas. Long
    enough for langdetect to classify reliably."""
    return (
        "THE CAT SAT ON THE MAT BY THE WINDOW IN THE MORNING. "
        "A SECOND SENTENCE TO HELP LANGUAGE DETECTION CLASSIFY THIS AS ENGLISH."
    )


class TestIFBenchMetricAndFeedback:
    def test_metric_returns_float_in_unit_interval(self):
        sub = ifbench_substrate()
        ex = _ifbench_fixture_example()
        prediction = dspy.Prediction(response=_two_sentence_english_caps())
        score = sub.metric_fn(ex, prediction)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_metric_returns_one_when_all_satisfied(self):
        sub = ifbench_substrate()
        ex = _ifbench_fixture_example()
        prediction = dspy.Prediction(response=_two_sentence_english_caps())
        score = sub.metric_fn(ex, prediction)
        assert score == pytest.approx(1.0)

    def test_metric_returns_partial_when_one_violated(self):
        sub = ifbench_substrate()
        ex = _ifbench_fixture_example()
        # Insert a comma -> violates punctuation:no_comma, satisfies caps.
        with_comma = _two_sentence_english_caps().replace(" ON ", ", ON ", 1)
        prediction = dspy.Prediction(response=with_comma)
        score = sub.metric_fn(ex, prediction)
        assert score == pytest.approx(0.5)

    def test_feedback_fn_returns_score_and_nonempty_feedback(self):
        sub = ifbench_substrate()
        ex = _ifbench_fixture_example()
        with_comma = _two_sentence_english_caps().replace(" ON ", ", ON ", 1)
        prediction = dspy.Prediction(response=with_comma)
        out = sub.feedback_fn(ex, prediction)
        assert set(out.keys()) >= {"score", "feedback", "violations", "satisfied"}
        assert 0.0 <= float(out["score"]) <= 1.0
        assert isinstance(out["feedback"], str) and len(out["feedback"]) > 0
        # Violation list names the IFBench instruction id.
        violated_ids = {v["instruction_id"] for v in out["violations"]}
        assert "punctuation:no_comma" in violated_ids


class TestFeedbackMap:
    def test_make_feedback_map_yields_score_with_feedback(self):
        from gepa.adapters.dspy_adapter.dspy_adapter import ScoreWithFeedback

        sub = ifbench_substrate()
        feedback_map = _make_feedback_map(sub)
        assert set(feedback_map.keys()) == set(sub.component_names)

        ex = _ifbench_fixture_example()
        prediction = dspy.Prediction(response=_two_sentence_english_caps())
        fn = feedback_map[sub.component_names[0]]
        # The signature matches DspyAdapter's per-predictor callback shape.
        result = fn(
            predictor_output={},
            predictor_inputs={},
            module_inputs=ex,
            module_outputs=prediction,
            captured_trace=None,
        )
        assert isinstance(result, ScoreWithFeedback)
        # ScoreWithFeedback inherits from dspy.Prediction (dspy.Example);
        # dict-style access is the canonical interface. Attribute access
        # returns None for keys stored in the internal field dict.
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["feedback"], str)
        assert len(result["feedback"]) > 0


# ---------------------------------------------------------------------------
# Hash verification (Chunk-13 frozen table)
# ---------------------------------------------------------------------------


class TestDifficultyTableHashVerification:
    def test_passes_on_real_frozen_table(self):
        h = verify_ifbench_difficulty_table_hash()
        assert h == EXPECTED_IFBENCH_DIFFICULTY_HASH

    def test_raises_on_tampered_table(self, tmp_path: Path, monkeypatch):
        # Copy the real table, mutate one score, point the verifier at the copy.
        real = json.loads(IFBENCH_DIFFICULTY_PATH.read_text())
        real["scores"][0] = 0.999  # arbitrary perturbation
        tampered = tmp_path / "difficulty_table.json"
        tampered.write_text(json.dumps(real, indent=2))
        monkeypatch.setattr(
            "src.ifbench_substrate.IFBENCH_DIFFICULTY_PATH", tampered
        )
        # Also need to neuter the side-car check (or point it to a stale file)
        # so the *primary* hash mismatch is the one that fires.
        monkeypatch.setattr(
            "src.ifbench_substrate.IFBENCH_DIFFICULTY_HASH_PATH",
            tmp_path / "nonexistent.sha256",
        )
        with pytest.raises(RuntimeError, match="hash mismatch"):
            verify_ifbench_difficulty_table_hash()

    def test_raises_on_missing_table(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "src.ifbench_substrate.IFBENCH_DIFFICULTY_PATH",
            tmp_path / "does_not_exist.json",
        )
        with pytest.raises(RuntimeError, match="not found"):
            verify_ifbench_difficulty_table_hash()

    def test_raises_when_sidecar_diverges_from_json(self, tmp_path: Path, monkeypatch):
        # Real JSON copy; side-car file holds a wrong hash.
        copy = tmp_path / "difficulty_table.json"
        copy.write_text(IFBENCH_DIFFICULTY_PATH.read_text())
        sidecar = tmp_path / "difficulty_table.sha256"
        sidecar.write_text("0" * 64 + "\n")
        monkeypatch.setattr(
            "src.ifbench_substrate.IFBENCH_DIFFICULTY_PATH", copy
        )
        monkeypatch.setattr(
            "src.ifbench_substrate.IFBENCH_DIFFICULTY_HASH_PATH", sidecar
        )
        with pytest.raises(RuntimeError, match="side-car"):
            verify_ifbench_difficulty_table_hash()


# ---------------------------------------------------------------------------
# Engine end-to-end smoke under IFBench substrate + 100/0/0
# ---------------------------------------------------------------------------


class _IFBenchStubAdapter:
    """StubAdapter shaped for the IFBench substrate (single component
    'answer'). Mirrors `tests/test_chunk4.py::StubAdapter` but yields
    canned scores via `score_fn(example_id, candidate)`; the engine
    never touches the real LM."""

    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.components = ["answer"]
        self.evaluate_calls: list[tuple[tuple, dict[str, str], bool]] = []
        self._propose_counter = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa import EvaluationBatch

        ids = tuple(ex["id"] for ex in batch)
        self.evaluate_calls.append((ids, dict(candidate), capture_traces))
        scores = [float(self.score_fn(ex["id"], candidate)) for ex in batch]
        outputs = [dspy.Prediction(response=f"r_{ex['id']}") for ex in batch]
        if capture_traces:
            traj = [
                [
                    {
                        "predictor_name": "answer",
                        "predictor_inputs": {"prompt": ex["prompt"]},
                        "predictor_output": {"response": f"r_{ex['id']}"},
                        "module_inputs": ex,
                        "module_outputs": dspy.Prediction(response=f"r_{ex['id']}"),
                    }
                ]
                for ex in batch
            ]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traj)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {
            name: [{"Inputs": {"prompt": "x"}, "Generated Outputs": {"response": "y"}, "Feedback": "violations: foo"}]
            for name in components_to_update
        }

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        self._propose_counter += 1
        return {
            name: f"{candidate.get(name, '')} | v{self._propose_counter}"
            for name in components_to_update
        }


def _ifbench_stub_example(idx: int) -> dspy.Example:
    """Tiny IFBench-shaped example with a single no-arg constraint, so
    the StubAdapter's canned scores are well-formed."""
    return dspy.Example(
        id=idx,
        source_key=f"stub-{idx}",
        prompt=f"Stub prompt {idx}.",
        instruction_id_list=["punctuation:no_comma"],
        kwargs_list=[None],
        constraint_count=1,
    ).with_inputs("prompt", "instruction_id_list", "kwargs_list")


class TestIFBenchEngineWiringSmoke:
    def test_decoupled_engine_runs_one_iteration_under_100_0_0(self, tmp_path: Path):
        """End-to-end smoke: build the engine the way `run_gepa.run` builds
        it, but with `ifbench_substrate()` for the seed candidate /
        component names and a `_IFBenchStubAdapter` so no LM is touched.
        One fake iteration must complete. State must advance."""
        sub = ifbench_substrate()
        components = list(sub.component_names)
        seed_candidate = {name: f"seed_{name}" for name in components}

        # Tiny D_feedback, A-batch, D_pareto with disjoint id ranges.
        d_feedback = [_ifbench_stub_example(i) for i in range(12)]
        accept_batch = [_ifbench_stub_example(100 + i) for i in range(4)]
        accept_batch_ids = list(range(4))
        d_pareto = [_ifbench_stub_example(200 + i) for i in range(6)]

        # Canned scores: mid band absorbs ids 4..7 (rank-tercile partition).
        scores = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]

        def score_fn(eid, candidate):
            if 0 <= eid < 12:
                # Non-seed candidate scores 1.0 on mid ids (accept).
                if 4 <= eid < 8 and " | v" in candidate.get("answer", ""):
                    return 1.0
                return scores[eid]
            if 100 <= eid < 104:
                # A-batch: any non-seed beats the seed (parent score 0.3).
                return 1.0 if " | v" in candidate.get("answer", "") else 0.3
            return 0.5  # D_pareto

        diff = build_rank_tercile_table(scores)
        assert set(diff.ids("mid")) == {4, 5, 6, 7}

        adapter = _IFBenchStubAdapter(score_fn=score_fn)
        rng = random.Random(0)
        batch_sampler = BandBatchSampler(
            target_band="mid",
            rng=rng,
            difficulty_table=diff,
            b=3,
            mix=PURE_ON_BAND_MIX,
        )
        candidate_selector = ParetoCandidateSelector(rng=rng)
        module_selector = RoundRobinReflectionComponentSelector()
        exp_tracker = create_experiment_tracker(
            use_wandb=False, wandb_api_key=None, wandb_init_kwargs=None,
            use_mlflow=False, mlflow_tracking_uri=None, mlflow_experiment_name=None,
        )

        class _NoopLogger:
            def log(self, *args, **kwargs):
                pass

        proposer = DecoupledReflectiveMutationProposer(
            logger=_NoopLogger(),
            trainset=d_feedback,
            adapter=adapter,
            candidate_selector=candidate_selector,
            module_selector=module_selector,
            batch_sampler=batch_sampler,
            perfect_score=1.0,
            skip_perfect_score=True,
            experiment_tracker=exp_tracker,
            reflection_lm=None,
            reflection_prompt_template=None,
            custom_candidate_proposer=None,
            callbacks=None,
            accept_batch=accept_batch,
            accept_batch_ids=accept_batch_ids,
        )
        engine = GEPAEngine(
            adapter=adapter,
            run_dir=str(tmp_path),
            valset=d_pareto,
            seed_candidate=seed_candidate,
            perfect_score=1.0,
            seed=0,
            reflective_proposer=proposer,
            merge_proposer=None,
            frontier_type="instance",
            logger=_NoopLogger(),
            experiment_tracker=exp_tracker,
            callbacks=None,
            track_best_outputs=False,
            display_progress_bar=False,
            raise_on_exception=True,
            stop_callback=MaxCandidateProposalsStopper(max_proposals=1),
            val_evaluation_policy=FullEvaluationPolicy(),
            use_cloudpickle=False,
            evaluation_cache=None,
        )
        with exp_tracker:
            state = engine.run()
        # 1 iteration completed (state.i is 0-indexed iteration cursor).
        assert state.i == 0
        # At least the seed candidate is in the program list (engine init).
        assert len(state.program_candidates) >= 1
        # The 100/0/0 sampler only emitted mid-band ids on every minibatch call.
        for ids, _, ct in adapter.evaluate_calls:
            if ct and len(ids) == 3:
                # capture_traces=True + size 3 means it's the minibatch.
                assert set(ids) <= {4, 5, 6, 7}, (
                    f"100/0/0 leaked off-band ids: {ids}"
                )


# ---------------------------------------------------------------------------
# Carved IFBench splits compose with `_validate_invariants`
# ---------------------------------------------------------------------------


class TestLoadIFBenchSplitsHelper:
    def test_load_ifbench_splits_returns_four_tuple(self, monkeypatch):
        """The helper returns the (d_feedback, accept_batch, d_pareto, test)
        4-tuple the orchestrator expects. We monkey-patch
        `carve_ifbench_splits` with a fast in-memory stub so the test
        does not download the HF dataset."""
        from src import ifbench_substrate as ifb

        fake_splits = (
            [dspy.Example(id=i, prompt=f"p{i}",
                          instruction_id_list=["punctuation:no_comma"],
                          kwargs_list=[None], source_key=f"k{i}",
                          constraint_count=1
                          ).with_inputs("prompt", "instruction_id_list", "kwargs_list")
             for i in range(150)],
            [dspy.Example(id=200 + i, prompt=f"p{i}",
                          instruction_id_list=["punctuation:no_comma"],
                          kwargs_list=[None], source_key=f"k{i}",
                          constraint_count=1
                          ).with_inputs("prompt", "instruction_id_list", "kwargs_list")
             for i in range(20)],
            [dspy.Example(id=400 + i, prompt=f"p{i}",
                          instruction_id_list=["punctuation:no_comma"],
                          kwargs_list=[None], source_key=f"k{i}",
                          constraint_count=1
                          ).with_inputs("prompt", "instruction_id_list", "kwargs_list")
             for i in range(75)],
            [dspy.Example(id=600 + i, prompt=f"p{i}",
                          instruction_id_list=["punctuation:no_comma"],
                          kwargs_list=[None], source_key=f"k{i}",
                          constraint_count=1
                          ).with_inputs("prompt", "instruction_id_list", "kwargs_list")
             for i in range(300)],
        )

        class _FakeCarve:
            splits = fake_splits

        def _fake_carve(seed=0):
            return _FakeCarve()

        monkeypatch.setattr("src.ifbench_data.carve_ifbench_splits", _fake_carve)
        out = ifb.load_ifbench_splits(config={"splits": {"seed_splits": 0}})
        assert len(out) == 4
        assert [len(s) for s in out] == [150, 20, 75, 300]
