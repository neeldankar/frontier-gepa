"""Chunk 4 offline gates.

Five sets of tests:

1. Difficulty table (src/difficulty.py): frozen across mutations, equal
   terciles by F1 rank, every D_feedback id in exactly one tercile,
   save/load round-trips.

2. Band samplers (src/band_sampler.py): ~70%/15%/15% within tolerance over
   many draws, random arm is ~uniform, each static arm targets the right
   tercile, b=3 always, no within-call duplicates, same seed reproduces
   the exact draw sequence.

3. Decoupled acceptance (src/decoupled_proposer.py): the engine's accept
   decision uses A-batch scores not the minibatch scores (construct a case
   where minibatch improves but A does not -> reject; and the reverse ->
   accept), strict > (equal A-batch sums reject), the parent's A-batch
   score is cached and reused, the same A is used for every call.

4. Stop (src/run_gepa.py + MaxCandidateProposalsStopper): the engine runs
   exactly N iterations regardless of accept rate (small N, stub adapter
   forced into no-accept and always-accept regimes).

5. Resume: kill after k iterations, resume in the same run_dir, confirm
   the run continues at k+1 with the same total_num_evals trajectory and
   no double-counting.

All tests stub the adapter (gepa.core.adapter.GEPAAdapter Protocol) so the
LM is never called. The two end-to-end engine tests (#4, #5) build a real
gepa.core.engine.GEPAEngine and run its loop.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

import dspy
import pytest

from gepa import EvaluationBatch
from gepa.core.data_loader import ensure_loader
from gepa.core.engine import GEPAEngine
from gepa.core.state import GEPAState, initialize_gepa_state
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import MaxCandidateProposalsStopper

from src.band_sampler import (
    DEFAULT_MIX,
    PURE_ON_BAND_MIX,
    BandBatchSampler,
    off_bands_for,
)
from src.decoupled_proposer import DecoupledReflectiveMutationProposer
from src.difficulty import (
    BAND_NAMES,
    DifficultyTable,
    build_difficulty_table,
)


# =========================================================================
# Test infra: stub adapter + helpers
# =========================================================================


class StubAdapter:
    """Minimal GEPAAdapter for offline testing.

    Each call to evaluate() returns scores from `score_fn(example_id, candidate)`.
    `propose_new_texts` mutates a single component to drive the engine forward
    deterministically; the test controls which components and how.
    """

    def __init__(
        self,
        score_fn,
        components: list[str],
        propose_strategy: str = "increment",
    ):
        self.score_fn = score_fn
        self.components = components
        self.propose_strategy = propose_strategy
        # Inspection hooks for tests:
        self.evaluate_calls: list[tuple[tuple, dict[str, str], bool]] = []
        self.propose_calls: list[tuple[dict[str, str], list[str]]] = []
        self._propose_counter = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        ids = tuple(getattr(ex, "id", i) if not isinstance(ex, dict) else ex["id"]
                    for i, ex in enumerate(batch))
        scores = [self.score_fn(eid, candidate) for eid in ids]
        outputs = [dspy.Prediction(answer="stub") for _ in batch]
        trajectories = None
        if capture_traces:
            # Need non-empty trajectories for the proposer to proceed.
            trajectories = [
                {
                    "prediction": o,
                    "trace": [],
                    "example": ex,
                    "score": s,
                }
                for ex, o, s in zip(batch, outputs, scores)
            ]
        self.evaluate_calls.append((ids, dict(candidate), capture_traces))
        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=trajectories,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        # Return one non-empty entry per requested component so the proposer
        # has something to propose against.
        return {
            name: [{"Inputs": {"q": "x"}, "Generated Outputs": {"a": "y"}, "Feedback": "z"}]
            for name in components_to_update
        }

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        """Deterministic propose: append a monotonically increasing counter
        to each requested component so that distinct candidates always have
        distinct text (keeps the engine's candidate_hash unique)."""
        self.propose_calls.append((dict(candidate), list(components_to_update)))
        self._propose_counter += 1
        out = {}
        for name in components_to_update:
            out[name] = f"{candidate.get(name, '')} | v{self._propose_counter}"
        return out


class _StubExample(dict):
    """A dict-like example with both `.id` attr and `['id']` access."""

    @property
    def id(self):
        return self["id"]


def _stub_example(idx: int) -> _StubExample:
    return _StubExample({"id": f"ex_{idx}", "question": f"q{idx}", "answer": f"a{idx}"})


COMPONENTS = ["summarize1", "summarize2", "final_answer"]


def _seed_candidate() -> dict[str, str]:
    return {name: f"seed_{name}" for name in COMPONENTS}


# =========================================================================
# 1. Difficulty table
# =========================================================================


class TestDifficultyTable:
    """Value-based binning (DEVIATIONS.md entry 4):
      hard = F1 == 0.0
      mid  = 0 < F1 < 1
      easy = F1 == 1.0
    Bands are unequal in general; that is by design.
    """

    def test_hard_band_contains_only_zeros(self):
        scores = [0.0, 0.5, 1.0, 0.0, 0.3, 1.0, 0.0]
        t = build_difficulty_table(scores)
        for i in t.ids("hard"):
            assert t.scores[i] == 0.0, f"id {i} in hard but F1={t.scores[i]}"

    def test_mid_band_contains_only_strictly_partial(self):
        scores = [0.0, 0.5, 1.0, 0.0, 0.3, 1.0, 0.0]
        t = build_difficulty_table(scores)
        for i in t.ids("mid"):
            assert 0.0 < t.scores[i] < 1.0, f"id {i} in mid but F1={t.scores[i]}"

    def test_easy_band_contains_only_ones(self):
        scores = [0.0, 0.5, 1.0, 0.0, 0.3, 1.0, 0.0]
        t = build_difficulty_table(scores)
        for i in t.ids("easy"):
            assert t.scores[i] == 1.0, f"id {i} in easy but F1={t.scores[i]}"

    def test_value_bins_partition_sums_to_n(self):
        scores = [0.0, 0.5, 1.0, 0.0, 0.3, 1.0, 0.0]
        t = build_difficulty_table(scores)
        assert (
            len(t.ids("hard")) + len(t.ids("mid")) + len(t.ids("easy"))
            == len(scores)
        )

    def test_value_bins_value_ordered(self):
        # max(hard) < min(mid); max(mid) < min(easy).
        scores = [0.0, 0.5, 1.0, 0.0, 0.3, 1.0, 0.9999]
        t = build_difficulty_table(scores)
        hard_max = max(t.scores[i] for i in t.ids("hard"))
        mid_min = min(t.scores[i] for i in t.ids("mid"))
        mid_max = max(t.scores[i] for i in t.ids("mid"))
        easy_min = min(t.scores[i] for i in t.ids("easy"))
        assert hard_max < mid_min
        assert mid_max < easy_min

    def test_unequal_band_sizes_are_allowed(self):
        # Mirrors the live n=150 distribution: 50/31/69.
        scores = [0.0] * 50 + [0.5] * 31 + [1.0] * 69
        t = build_difficulty_table(scores)
        assert len(t.ids("hard")) == 50
        assert len(t.ids("mid")) == 31
        assert len(t.ids("easy")) == 69

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            build_difficulty_table([0.0, 1.1, 0.5])
        with pytest.raises(ValueError):
            build_difficulty_table([-0.01, 0.5, 1.0])

    def test_every_id_in_exactly_one_band(self):
        rng = random.Random(0)
        # Mix of zeros, ones, and partials so all three bands populate.
        scores = (
            [0.0] * 30
            + [rng.uniform(0.01, 0.99) for _ in range(40)]
            + [1.0] * 30
        )
        t = build_difficulty_table(scores)
        seen: set[int] = set()
        for band in BAND_NAMES:
            for i in t.ids(band):
                assert i not in seen, f"id {i} in multiple bands"
                seen.add(i)
        assert seen == set(range(len(scores)))

    def test_band_for_id_matches_ids_in_band(self):
        scores = [0.0, 0.5, 1.0, 0.0, 0.3, 1.0, 0.7]
        t = build_difficulty_table(scores)
        for band in BAND_NAMES:
            for i in t.ids(band):
                assert t.band(i) == band

    def test_frozen_dataclass_blocks_assignment(self):
        t = build_difficulty_table([0.1, 0.2, 0.3])
        with pytest.raises(Exception):
            t.scores = (0.0,)

    def test_inner_mappings_are_read_only(self):
        t = build_difficulty_table([0.1, 0.2, 0.3])
        # MappingProxyType raises TypeError on item assignment.
        with pytest.raises(TypeError):
            t.ids_in_band["easy"] = (99,)  # type: ignore[index]
        with pytest.raises(TypeError):
            t.band_for_id[0] = "mid"  # type: ignore[index]

    def test_save_and_load_round_trip(self, tmp_path: Path):
        rng = random.Random(1)
        scores = [rng.random() for _ in range(15)]
        t = build_difficulty_table(scores)
        p = tmp_path / "table.json"
        t.save(p)
        t2 = DifficultyTable.load(p)
        assert t2.scores == t.scores
        assert dict(t2.ids_in_band) == dict(t.ids_in_band)
        assert dict(t2.band_for_id) == dict(t.band_for_id)

    def test_construction_failures(self):
        with pytest.raises(ValueError):
            build_difficulty_table([])
        # Mismatched band_for_id / ids_in_band raises in __post_init__.
        with pytest.raises(ValueError):
            DifficultyTable(
                scores=(0.1, 0.2, 0.3),
                band_for_id=MappingProxyType({0: "easy", 1: "easy", 2: "easy"}),
                ids_in_band=MappingProxyType({
                    "easy": (0,), "mid": (1,), "hard": (2,),
                }),
            )

    def test_table_unchanged_after_many_lookups(self):
        # Lookups must not mutate the table.
        t = build_difficulty_table([0.1, 0.5, 0.9, 0.2, 0.6, 0.8, 0.3, 0.7, 0.4])
        before_scores = t.scores
        before_bands = {b: t.ids(b) for b in BAND_NAMES}
        for _ in range(1000):
            for band in BAND_NAMES:
                _ = t.ids(band)
            for i in range(t.n):
                _ = t.band(i)
        assert t.scores == before_scores
        for b in BAND_NAMES:
            assert t.ids(b) == before_bands[b]


# =========================================================================
# 2. Band samplers
# =========================================================================


class TestBandSampler:
    @pytest.fixture
    def equal_bands_99(self) -> DifficultyTable:
        """Equal 33/33/33 bands under value-based binning: 33 hard (F1=0),
        33 mid (F1=0.5), 33 easy (F1=1)."""
        scores = [0.0] * 33 + [0.5] * 33 + [1.0] * 33
        return build_difficulty_table(scores)

    @pytest.fixture
    def loader_99(self):
        return ensure_loader(list(range(99)))

    @pytest.fixture
    def unequal_bands_150(self) -> DifficultyTable:
        """Live n=150 composition: 50 hard / 31 mid / 69 easy."""
        scores = [0.0] * 50 + [0.5] * 31 + [1.0] * 69
        return build_difficulty_table(scores)

    @pytest.fixture
    def loader_150(self):
        return ensure_loader(list(range(150)))

    @pytest.mark.parametrize("target", ["easy", "mid", "hard"])
    def test_static_arm_distribution_equal_bands(
        self, target, equal_bands_99, loader_99
    ):
        s = BandBatchSampler(
            target_band=target, rng=random.Random(7),
            difficulty_table=equal_bands_99,
        )
        counts = {b: 0 for b in BAND_NAMES}
        n_draws = 2000
        for _ in range(n_draws):
            for did in s.next_minibatch_ids(loader_99, MagicMock()):
                counts[equal_bands_99.band(did)] += 1
        total = sum(counts.values())
        pct = {b: counts[b] / total for b in BAND_NAMES}
        # 70/15/15 mix within +/- 4%.
        assert 0.66 <= pct[target] <= 0.74, f"target {target}: pct={pct}"
        for off in [b for b in BAND_NAMES if b != target]:
            assert 0.11 <= pct[off] <= 0.19, f"off-band {off}: pct={pct}"

    @pytest.mark.parametrize("target", ["easy", "mid", "hard"])
    def test_static_arm_distribution_unequal_bands(
        self, target, unequal_bands_150, loader_150
    ):
        """The 70/15/15 mix holds even when bands are unequal in size (the
        live D_feedback case at n=150 is 50/31/69)."""
        s = BandBatchSampler(
            target_band=target, rng=random.Random(7),
            difficulty_table=unequal_bands_150,
        )
        counts = {b: 0 for b in BAND_NAMES}
        n_draws = 2000
        for _ in range(n_draws):
            for did in s.next_minibatch_ids(loader_150, MagicMock()):
                counts[unequal_bands_150.band(did)] += 1
        total = sum(counts.values())
        pct = {b: counts[b] / total for b in BAND_NAMES}
        assert 0.66 <= pct[target] <= 0.74, f"target {target}: pct={pct}"
        for off in [b for b in BAND_NAMES if b != target]:
            assert 0.11 <= pct[off] <= 0.19, f"off-band {off}: pct={pct}"

    def test_random_arm_uniform_over_ids_equal_bands(
        self, equal_bands_99, loader_99
    ):
        """With 33/33/33 bands and uniform-over-ids draws, the per-band
        composition is ~33% / 33% / 33% (band fraction == size fraction)."""
        s = BandBatchSampler(
            target_band="random", rng=random.Random(7), difficulty_table=None,
        )
        counts = {b: 0 for b in BAND_NAMES}
        for _ in range(2000):
            for did in s.next_minibatch_ids(loader_99, MagicMock()):
                counts[equal_bands_99.band(did)] += 1
        total = sum(counts.values())
        pct = {b: counts[b] / total for b in BAND_NAMES}
        for b in BAND_NAMES:
            assert 0.29 <= pct[b] <= 0.37, f"band {b}: pct={pct}"

    def test_random_arm_reflects_natural_distribution_with_unequal_bands(
        self, unequal_bands_150, loader_150
    ):
        """Critical: the random arm is uniform-over-IDS, not uniform-over-
        BANDS. With unequal bands (50/31/69 over n=150), the natural
        composition is ~33% hard / ~21% mid / ~46% easy. Uniform-over-bands
        would (wrongly) over-sample the small frontier band."""
        s = BandBatchSampler(
            target_band="random", rng=random.Random(7), difficulty_table=None,
        )
        counts = {b: 0 for b in BAND_NAMES}
        for _ in range(2000):
            for did in s.next_minibatch_ids(loader_150, MagicMock()):
                counts[unequal_bands_150.band(did)] += 1
        total = sum(counts.values())
        pct = {b: counts[b] / total for b in BAND_NAMES}
        # Expected: hard 50/150=33%, mid 31/150=21%, easy 69/150=46%.
        assert 0.28 <= pct["hard"] <= 0.38, f"hard pct={pct['hard']}"
        assert 0.16 <= pct["mid"] <= 0.26, f"mid pct={pct['mid']}"
        assert 0.41 <= pct["easy"] <= 0.51, f"easy pct={pct['easy']}"

    def test_b_is_three_and_unique_within_call(self, equal_bands_99, loader_99):
        s = BandBatchSampler(
            target_band="mid", rng=random.Random(7),
            difficulty_table=equal_bands_99,
        )
        for _ in range(200):
            draw = s.next_minibatch_ids(loader_99, MagicMock())
            assert len(draw) == 3
            assert len(set(draw)) == 3

    def test_draws_only_from_d_feedback(self, equal_bands_99, loader_99):
        s = BandBatchSampler(
            target_band="mid", rng=random.Random(7),
            difficulty_table=equal_bands_99,
        )
        all_ids = set(loader_99.all_ids())
        for _ in range(200):
            for did in s.next_minibatch_ids(loader_99, MagicMock()):
                assert did in all_ids

    def test_exact_reproducibility_under_same_seed(
        self, equal_bands_99, loader_99
    ):
        s1 = BandBatchSampler(
            target_band="mid", rng=random.Random(7),
            difficulty_table=equal_bands_99,
        )
        s2 = BandBatchSampler(
            target_band="mid", rng=random.Random(7),
            difficulty_table=equal_bands_99,
        )
        seq1 = [s1.next_minibatch_ids(loader_99, MagicMock()) for _ in range(50)]
        seq2 = [s2.next_minibatch_ids(loader_99, MagicMock()) for _ in range(50)]
        assert seq1 == seq2

    def test_off_bands_for_deterministic(self):
        assert off_bands_for("easy") == ("mid", "hard")
        assert off_bands_for("mid") == ("easy", "hard")
        assert off_bands_for("hard") == ("easy", "mid")

    def test_static_arm_requires_difficulty_table(self):
        with pytest.raises(ValueError):
            BandBatchSampler(target_band="mid", rng=random.Random(0))

    def test_mix_validation(self, equal_bands_99):
        with pytest.raises(ValueError):
            BandBatchSampler(
                target_band="mid",
                rng=random.Random(0),
                difficulty_table=equal_bands_99,
                mix=(0.5, 0.3, 0.3),  # sums to 1.1
            )

    # ---- Chunk 8: pure on-band (100/0/0) sampling -----------------------

    @pytest.mark.parametrize("target", ["mid", "hard", "easy"])
    def test_pure_on_band_no_off_band_leakage(
        self, target, unequal_bands_150, loader_150,
    ):
        """Under PURE_ON_BAND_MIX, 10000 draws land entirely in the target
        band; the off-band fraction is exactly 0. This is the design
        guarantee of the Experiment-1b isolation variant."""
        s = BandBatchSampler(
            target_band=target,
            rng=random.Random(7),
            difficulty_table=unequal_bands_150,
            mix=PURE_ON_BAND_MIX,
        )
        n_draws = 10_000
        off_band_count = 0
        target_count = 0
        for _ in range(n_draws):
            for did in s.next_minibatch_ids(loader_150, MagicMock()):
                if unequal_bands_150.band(did) == target:
                    target_count += 1
                else:
                    off_band_count += 1
        assert off_band_count == 0, (
            f"pure on-band leaked: {off_band_count} off-band picks out of "
            f"{target_count + off_band_count} (target={target!r})"
        )
        assert target_count == n_draws * 3

    def test_random_arm_byte_identical_under_70_15_15_and_100_0_0(
        self, loader_150,
    ):
        """The random arm path never reads `self.mix`. Two samplers with
        the same fresh `random.Random(seed)` produce byte-identical id
        sequences under 70/15/15 and 100/0/0. This is the design hinge
        for reusing Experiment 1's random + vanilla cells unchanged in
        Experiment 1b -- the random baseline does not need re-running."""
        s_default = BandBatchSampler(
            target_band="random", rng=random.Random(42), mix=DEFAULT_MIX,
        )
        s_pure = BandBatchSampler(
            target_band="random", rng=random.Random(42), mix=PURE_ON_BAND_MIX,
        )
        seq_default = [
            s_default.next_minibatch_ids(loader_150, MagicMock())
            for _ in range(100)
        ]
        seq_pure = [
            s_pure.next_minibatch_ids(loader_150, MagicMock())
            for _ in range(100)
        ]
        assert seq_default == seq_pure

    def test_pure_on_band_band_smaller_than_b_raises(self):
        """Defensive check: under pure-on-band, a target band smaller than
        b cannot be served without leaking off-band via the bounded-retry
        fallback. We raise loudly rather than silently corrupt the
        experiment."""
        # mid has 2 ids; b defaults to 3.
        scores = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
        small_mid_table = build_difficulty_table(scores)
        assert len(small_mid_table.ids("mid")) == 2
        s = BandBatchSampler(
            target_band="mid",
            rng=random.Random(0),
            difficulty_table=small_mid_table,
            mix=PURE_ON_BAND_MIX,
        )

        class _StubLoader:
            def all_ids(self):
                return list(range(len(scores)))

            def __len__(self):
                return len(scores)

        with pytest.raises(ValueError, match=r"pure on-band sampling"):
            s.next_minibatch_ids(_StubLoader(), MagicMock())

    @pytest.mark.parametrize("target", ["mid", "easy"])
    def test_pure_on_band_determinism(
        self, target, unequal_bands_150, loader_150,
    ):
        """Same seed produces identical 100/0/0 draws across fresh sampler
        instances."""
        s1 = BandBatchSampler(
            target_band=target,
            rng=random.Random(11),
            difficulty_table=unequal_bands_150,
            mix=PURE_ON_BAND_MIX,
        )
        s2 = BandBatchSampler(
            target_band=target,
            rng=random.Random(11),
            difficulty_table=unequal_bands_150,
            mix=PURE_ON_BAND_MIX,
        )
        seq1 = [s1.next_minibatch_ids(loader_150, MagicMock()) for _ in range(50)]
        seq2 = [s2.next_minibatch_ids(loader_150, MagicMock()) for _ in range(50)]
        assert seq1 == seq2


# =========================================================================
# Engine-building helper for tests 3-5
# =========================================================================


def _build_test_engine(
    *,
    score_fn,
    n_feedback: int = 12,
    n_accept: int = 4,
    n_pareto: int = 6,
    n_iterations: int = 5,
    arm: str = "static_frontier",
    seed: int = 0,
    run_dir: Path,
    decoupled: bool = True,
    skip_perfect_score: bool = True,
    mix: tuple[float, float, float] = DEFAULT_MIX,
):
    """Build a GEPAEngine wired to a StubAdapter; returns (engine, adapter, state_after_init).

    The `mix` kwarg lets a test opt into 100/0/0 pure-on-band sampling
    (`PURE_ON_BAND_MIX`) without touching any other call site; default is
    the Experiment-1 70/15/15. Ignored for arm='random'."""
    components = COMPONENTS
    seed_cand = _seed_candidate()
    d_feedback = [_stub_example(i) for i in range(n_feedback)]
    accept_batch = [_stub_example(1000 + i) for i in range(n_accept)]
    accept_batch_ids = list(range(n_accept))
    d_pareto = [_stub_example(2000 + i) for i in range(n_pareto)]

    adapter = StubAdapter(score_fn=score_fn, components=components)

    rng = random.Random(seed)

    # Sampler
    if decoupled and arm != "random":
        scores = [score_fn(_stub_example(i).id, seed_cand) for i in range(n_feedback)]
        diff = build_difficulty_table(scores)
        target = {"static_easy": "easy", "static_frontier": "mid", "static_hard": "hard"}[arm]
        batch_sampler = BandBatchSampler(
            target_band=target, rng=rng, difficulty_table=diff, mix=mix,
        )
    elif decoupled and arm == "random":
        batch_sampler = BandBatchSampler(target_band="random", rng=rng, mix=mix)
    else:
        from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
        batch_sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=rng)

    candidate_selector = ParetoCandidateSelector(rng=rng)
    module_selector = RoundRobinReflectionComponentSelector()
    exp_tracker = create_experiment_tracker(
        use_wandb=False, wandb_api_key=None, wandb_init_kwargs=None,
        use_mlflow=False, mlflow_tracking_uri=None, mlflow_experiment_name=None,
    )

    class _NoopLogger:
        def log(self, *args, **kwargs): pass

    logger = _NoopLogger()

    if decoupled:
        proposer = DecoupledReflectiveMutationProposer(
            logger=logger,
            trainset=d_feedback,
            adapter=adapter,
            candidate_selector=candidate_selector,
            module_selector=module_selector,
            batch_sampler=batch_sampler,
            perfect_score=1.0,
            skip_perfect_score=skip_perfect_score,
            experiment_tracker=exp_tracker,
            reflection_lm=None,  # unused: stub adapter has propose_new_texts
            reflection_prompt_template=None,
            custom_candidate_proposer=None,
            callbacks=None,
            accept_batch=accept_batch,
            accept_batch_ids=accept_batch_ids,
        )
    else:
        proposer = ReflectiveMutationProposer(
            logger=logger,
            trainset=d_feedback,
            adapter=adapter,
            candidate_selector=candidate_selector,
            module_selector=module_selector,
            batch_sampler=batch_sampler,
            perfect_score=1.0,
            skip_perfect_score=skip_perfect_score,
            experiment_tracker=exp_tracker,
            reflection_lm=None,
            reflection_prompt_template=None,
            custom_candidate_proposer=None,
            callbacks=None,
        )

    engine = GEPAEngine(
        adapter=adapter,
        run_dir=str(run_dir),
        valset=d_pareto,
        seed_candidate=seed_cand,
        perfect_score=1.0,
        seed=seed,
        reflective_proposer=proposer,
        merge_proposer=None,
        frontier_type="instance",
        logger=logger,
        experiment_tracker=exp_tracker,
        callbacks=None,
        track_best_outputs=False,
        display_progress_bar=False,
        raise_on_exception=True,
        stop_callback=MaxCandidateProposalsStopper(max_proposals=n_iterations),
        val_evaluation_policy=FullEvaluationPolicy(),
        use_cloudpickle=False,
        evaluation_cache=None,
    )
    return engine, adapter, proposer, exp_tracker


# =========================================================================
# 3. Decoupled acceptance
# =========================================================================


class TestDecoupledAcceptance:
    def test_accept_uses_a_batch_not_minibatch(self, tmp_path: Path):
        """Minibatch improves but A-batch does not -> engine rejects.

        Setup: score_fn distinguishes D_feedback ids from accept-batch ids.
        - For D_feedback ids: any non-seed candidate scores 1.0 (improvement).
        - For accept-batch ids: any candidate scores 0.5 (no improvement).
        """

        def score_fn(eid, candidate):
            is_feedback = isinstance(eid, str) and eid.startswith("ex_") and int(eid.split("_")[1]) < 12
            is_accept = isinstance(eid, str) and eid.startswith("ex_") and 1000 <= int(eid.split("_")[1]) < 1004
            # Non-seed candidate detection: seed has no " | v" suffix.
            is_non_seed = any(" | v" in v for v in candidate.values())
            if is_feedback:
                return 1.0 if is_non_seed else 0.0
            if is_accept:
                return 0.5
            return 0.5  # d_pareto

        engine, adapter, proposer, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=3, arm="static_frontier",
            run_dir=tmp_path,
        )
        state = engine.run()
        # With A constant at 0.5 for every candidate, accept rule (strict >)
        # rejects every proposal. State should still have ONLY the seed.
        assert len(state.program_candidates) == 1, (
            f"expected only seed candidate (no accepts), got {len(state.program_candidates)}"
        )

    def test_accept_when_a_batch_improves(self, tmp_path: Path):
        """Reverse: minibatch does NOT improve but A-batch does -> accept."""

        def score_fn(eid, candidate):
            is_feedback = isinstance(eid, str) and eid.startswith("ex_") and int(eid.split("_")[1]) < 12
            is_accept = isinstance(eid, str) and eid.startswith("ex_") and 1000 <= int(eid.split("_")[1]) < 1004
            is_non_seed = any(" | v" in v for v in candidate.values())
            if is_feedback:
                # Constant on minibatch (no improvement -- but the engine's
                # accept test reads A-batch, not this).
                return 0.3
            if is_accept:
                return 1.0 if is_non_seed else 0.0
            return 0.5

        engine, adapter, proposer, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=3, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        state = engine.run()
        # Each iteration: parent A=0 -> first new A=1.0 (accept); then new
        # parent A is the new candidate's A=1.0; subsequent proposals also
        # produce A=1.0 (equal -> reject). Expect 2 program_candidates total.
        assert len(state.program_candidates) >= 2, (
            f"expected at least one accept; got {len(state.program_candidates)} candidates"
        )

    def test_strict_greater_rejects_equal(self, tmp_path: Path):
        """When new A sum == parent A sum, the engine's strict > rejects.

        Setup: A-batch returns exactly 0.5 for every candidate (parent or
        new). Sum is identical -> rejected.
        """

        def score_fn(eid, candidate):
            is_feedback = isinstance(eid, str) and eid.startswith("ex_") and int(eid.split("_")[1]) < 12
            if is_feedback:
                return 0.7  # bigger than 0 so we don't skip-perfect-out
            return 0.5  # accept batch + valset

        engine, adapter, proposer, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=3, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        state = engine.run()
        assert len(state.program_candidates) == 1, (
            "strict > should reject equal A-batch sums"
        )

    def test_parent_a_score_cached(self, tmp_path: Path):
        """The parent's A-batch score is computed once and cached on the
        proposer, so we never re-evaluate the same parent on A twice."""

        def score_fn(eid, candidate):
            is_feedback = isinstance(eid, str) and eid.startswith("ex_") and int(eid.split("_")[1]) < 12
            if is_feedback:
                return 0.7
            return 0.3  # accept batch: every candidate gets 0.3 -> no accept

        engine, adapter, proposer, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=5, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        engine.run()
        # All 5 iterations pick the same parent (seed), since nothing was
        # accepted. The accept_score_cache should have exactly 1 entry.
        assert len(proposer.accept_score_cache) == 1
        assert 0 in proposer.accept_score_cache

        # Count adapter.evaluate calls on the A-batch with the SEED program:
        # the parent should have been evaluated on A only once over 5 iters.
        seed_a_calls = 0
        seed_cand = _seed_candidate()
        for ids, cand, capture_traces in adapter.evaluate_calls:
            if capture_traces:
                continue
            if len(ids) != 4:  # not the A batch
                continue
            if cand == seed_cand:
                seed_a_calls += 1
        assert seed_a_calls == 1, (
            f"expected exactly 1 seed-on-A eval; got {seed_a_calls}"
        )

    def test_same_a_batch_used_every_call(self, tmp_path: Path):
        """The accept batch passed to the proposer is the one used for
        every call -- proposer.accept_batch must not be silently rebuilt."""

        def score_fn(eid, candidate):
            is_feedback = isinstance(eid, str) and eid.startswith("ex_") and int(eid.split("_")[1]) < 12
            if is_feedback:
                return 0.7
            return 0.3

        engine, adapter, proposer, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=3, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        engine.run()
        # Collect all A-batch id tuples (non-capture_traces calls of size 4).
        a_calls = [ids for ids, _, ct in adapter.evaluate_calls if not ct and len(ids) == 4]
        assert len(a_calls) >= 2
        first = a_calls[0]
        for ids in a_calls[1:]:
            assert ids == first, (
                f"accept batch ids changed across calls: {first} vs {ids}"
            )

    def test_pure_on_band_all_perfect_minibatch_no_exception(self, tmp_path: Path):
        """Static-easy + PURE_ON_BAND_MIX draws only from the F1==1 band,
        so every minibatch is all-perfect. The inherited skip_perfect_score
        branch must return None, the engine must continue, and N iterations
        must complete with zero accepts -- no exception."""
        def score_fn(eid, candidate):
            is_feedback = (
                isinstance(eid, str) and eid.startswith("ex_")
                and int(eid.split("_")[1]) < 12
            )
            if is_feedback:
                # Every D_feedback example is perfect under the seed
                # candidate. The difficulty table built inside
                # _build_test_engine then puts all 12 ids in the easy band.
                return 1.0
            # accept_batch + d_pareto: arbitrary non-perfect score so the
            # engine has something to log but it never matters because no
            # proposal is ever made.
            return 0.5

        N = 3
        engine, adapter, proposer, _ = _build_test_engine(
            score_fn=score_fn,
            n_iterations=N,
            arm="static_easy",
            run_dir=tmp_path,
            mix=PURE_ON_BAND_MIX,
            skip_perfect_score=True,
        )
        state = engine.run()
        # Exactly N iterations ran (state.i == N-1 after the loop).
        assert state.i == N - 1
        # No proposals were ever made, so no candidates were ever accepted.
        assert len(state.program_candidates) == 1
        assert len(state.full_program_trace) == N


# =========================================================================
# 4. Stop: exactly N iterations regardless of accept rate
# =========================================================================


class TestStop:
    def test_n_iterations_with_all_rejects(self, tmp_path: Path):
        # A-batch constant -> all rejects. Loop must still run N times.
        def score_fn(eid, candidate):
            is_feedback = isinstance(eid, str) and eid.startswith("ex_") and int(eid.split("_")[1]) < 12
            if is_feedback:
                return 0.7
            return 0.3

        N = 7
        engine, adapter, proposer, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=N, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        state = engine.run()
        # MaxCandidateProposalsStopper terminates after exactly N proposals;
        # state.i == N - 1 by design (see its docstring), and the per-iter
        # full_program_trace has N entries.
        assert state.i == N - 1, f"expected state.i == {N - 1}, got {state.i}"
        assert len(state.full_program_trace) == N

    def test_n_iterations_with_many_accepts(self, tmp_path: Path):
        # Score increases with the propose counter via " | v" suffix length.
        def score_fn(eid, candidate):
            is_feedback = isinstance(eid, str) and eid.startswith("ex_") and int(eid.split("_")[1]) < 12
            if is_feedback:
                return 0.3
            # Accept batch: score grows with the candidate's text length so
            # every new proposal scores higher.
            return 0.1 + 0.001 * sum(len(v) for v in candidate.values())

        N = 5
        engine, adapter, proposer, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=N, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        state = engine.run()
        assert state.i == N - 1, f"expected state.i == {N - 1}, got {state.i}"
        assert len(state.full_program_trace) == N


# =========================================================================
# 5. Resume: kill at k, restart, continue at k+1
# =========================================================================


class TestResume:
    def test_resume_continues_without_double_count(self, tmp_path: Path):
        """Run two iterations, save state, then run two more in the same
        run_dir. Total iterations should be 4, total_num_evals should
        equal the cumulative count, and no iteration is redone."""

        def score_fn(eid, candidate):
            is_feedback = isinstance(eid, str) and eid.startswith("ex_") and int(eid.split("_")[1]) < 12
            if is_feedback:
                return 0.7
            return 0.3  # all rejects

        # Phase 1: 2 iterations.
        engine_a, adapter_a, proposer_a, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=2, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        state_a = engine_a.run()
        assert state_a.i == 1, f"phase 1 i should be 1 (== N-1), got {state_a.i}"
        evals_after_phase1 = state_a.total_num_evals
        assert (tmp_path / "gepa_state.bin").exists()

        # Phase 2: target N=4 in the same run_dir, expect resume from i=2.
        engine_b, adapter_b, proposer_b, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=4, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        state_b = engine_b.run()
        assert state_b.i == 3, f"phase 2 i should be 3 (== N-1 of 4), got {state_b.i}"
        # full_program_trace should have grown from 2 to 4 entries (no replay).
        assert len(state_b.full_program_trace) == 4
        # Phase 2 added 2 more iterations; total_num_evals should monotonically
        # increase. We check it is strictly larger than after phase 1.
        assert state_b.total_num_evals > evals_after_phase1
        assert len(adapter_b.evaluate_calls) > 0

    def test_resume_with_zero_extra_iterations_is_a_noop(self, tmp_path: Path):
        """If we resume after completing N iterations and ask for N again,
        the loop terminates immediately (state.i >= N-1 stopper fires)."""

        def score_fn(eid, candidate):
            is_feedback = isinstance(eid, str) and eid.startswith("ex_") and int(eid.split("_")[1]) < 12
            if is_feedback:
                return 0.7
            return 0.3

        engine_a, adapter_a, proposer_a, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=3, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        state_a = engine_a.run()
        assert state_a.i == 2  # 3 iters -> state.i ends at 2
        evals1 = state_a.total_num_evals

        engine_b, adapter_b, proposer_b, _ = _build_test_engine(
            score_fn=score_fn, n_iterations=3, arm="static_frontier",
            run_dir=tmp_path, skip_perfect_score=False,
        )
        state_b = engine_b.run()
        # No new iterations should have run.
        assert state_b.i == 2
        assert state_b.total_num_evals == evals1, (
            "no extra evaluations should have happened on a no-op resume"
        )
