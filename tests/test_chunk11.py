"""Chunk 11 (redo): IFBench data loader, dataset gate, splits.

Pivot per BUILD_PLAN §7 Chunk 11 redo: the substrate is
`allenai/IF_multi_constraints_upto5` (the IF-RLVR composite, ~95k rows,
multi-constraint by construction), NOT `allenai/IFBench_test` (which is
predominantly single-constraint and produced an OPERATOR_REVIEW verdict
in Chunk 11 v1).

Offline tests use synthetic IFBenchUsable fixtures to validate the
carving math, the verifier-coverage check, and the schema contracts.
Live integration tests (marked `integration`, skipped by the default
`pytest -m "not integration"` run) hit the HF cache to confirm the live
dataset still matches the Chunk-11-redo report numbers."""

from __future__ import annotations

import pytest

from src.ifbench_data import (
    CONSTRAINT_COUNT_FLOOR,
    IFBENCH_DATASET_ID,
    IFBENCH_FAMILY,
    INPUT_FIELDS,
    KNOWN_VERIFIER_IDS,
    SPLIT_ORDER,
    TARGET_SIZES,
    TARGET_TOTAL,
    IFBenchUsable,
    carve_ifbench_splits,
    verify_carved_pool_constraint_coverage,
)


# =========================================================================
# Constants / configuration
# =========================================================================


def test_dataset_id_is_if_multi_constraints():
    """Confirm we pivoted to the IF-RLVR composite, NOT IFBench_test, NOT
    Google IFEval."""
    assert IFBENCH_DATASET_ID == "allenai/IF_multi_constraints_upto5"
    assert "IF-RLVR" in IFBENCH_FAMILY
    assert "IFEval 25" in IFBENCH_FAMILY
    assert "IFBench-Train 29" in IFBENCH_FAMILY


def test_constraint_count_floor_is_three():
    assert CONSTRAINT_COUNT_FLOOR == 3


def test_target_sizes_are_full_experiment_1_layout():
    assert TARGET_SIZES == {
        "d_feedback": 150,
        "accept_batch": 20,
        "d_pareto": 75,
        "test": 300,
    }
    assert TARGET_TOTAL == 545
    assert SPLIT_ORDER == ("d_feedback", "accept_batch", "d_pareto", "test")


def test_input_fields_carry_verifier_inputs():
    """Chunk 12 verifiers will need prompt + the parallel constraint
    arrays."""
    assert INPUT_FIELDS == ("prompt", "instruction_id_list", "kwargs_list")


def test_known_verifier_catalog_has_54_ids():
    """The curated registry of IFEval + IFBench-Train verifier IDs the
    Chunk-12 verifier module must implement."""
    assert len(KNOWN_VERIFIER_IDS) == 54
    # All ids follow the canonical `family:name` format.
    for iid in KNOWN_VERIFIER_IDS:
        assert ":" in iid, iid
    # Spot-check a handful of known canonical IDs.
    for iid in (
        "keywords:existence",
        "detectable_format:title",
        "punctuation:no_comma",
        "length_constraints:number_words",
    ):
        assert iid in KNOWN_VERIFIER_IDS


# =========================================================================
# Synthetic usable fixtures for carving / verifier-coverage tests
# =========================================================================


def _synthetic_row(i: int, n_constraints: int = 3, unknown_iid: bool = False) -> dict:
    iids = [f"keywords:existence" for _ in range(n_constraints)]
    if unknown_iid:
        # Plant one out-of-catalog id
        iids[0] = "unknown_family:unknown_check"
    return {
        "id": i,
        "source_key": f"src_{i}",
        "prompt": f"Synthetic prompt {i}. " + ("Add detail. " * 5),
        "instruction_id_list": iids,
        "kwargs_list": [None] * n_constraints,
        "constraint_count": n_constraints,
    }


def _synthetic_usable(n: int) -> IFBenchUsable:
    rows = [_synthetic_row(i, n_constraints=3) for i in range(n)]
    return IFBenchUsable(
        dataset_id=IFBENCH_DATASET_ID,
        family=IFBENCH_FAMILY,
        constraint_count_floor=3,
        n_rows_raw=n + 10,  # not used by the carve path
        constraint_count_histogram_raw={1: 0, 2: 0, 3: n, 4: 0, 5: 0},
        n_above_floor_pre_verifier_filter=n,
        uncovered_instruction_ids=(),
        n_excluded_for_uncovered_verifier=0,
        n_usable=n,
        rows=rows,
    )


# =========================================================================
# Carving
# =========================================================================


class TestCarveSplits:
    def test_full_pool_carve_produces_target_sizes(self):
        u = _synthetic_usable(2000)
        result = carve_ifbench_splits(seed=0, usable=u)
        sizes = {SPLIT_ORDER[i]: len(result.splits[i]) for i in range(4)}
        assert sizes == TARGET_SIZES
        assert result.sizes == TARGET_SIZES

    def test_pools_are_pairwise_disjoint(self):
        u = _synthetic_usable(2000)
        result = carve_ifbench_splits(seed=0, usable=u)
        seen: dict[int, str] = {}
        for name, items in zip(SPLIT_ORDER, result.splits):
            for ex in items:
                assert ex["id"] not in seen, (
                    f"id {ex['id']} appears in {seen[ex['id']]} and {name}"
                )
                seen[ex["id"]] = name
        assert len(seen) == TARGET_TOTAL

    def test_carve_refuses_when_pool_too_small(self):
        u = _synthetic_usable(500)  # < 545
        with pytest.raises(RuntimeError, match=r"smaller than target total"):
            carve_ifbench_splits(seed=0, usable=u)

    def test_determinism_same_seed_same_ids(self):
        u = _synthetic_usable(2000)
        a = carve_ifbench_splits(seed=42, usable=u)
        b = carve_ifbench_splits(seed=42, usable=u)
        for sa, sb in zip(a.splits, b.splits):
            assert [ex["id"] for ex in sa] == [ex["id"] for ex in sb]

    def test_different_seeds_differ(self):
        u = _synthetic_usable(2000)
        a = carve_ifbench_splits(seed=42, usable=u)
        b = carve_ifbench_splits(seed=7, usable=u)
        any_diff = any(
            [ex["id"] for ex in sa] != [ex["id"] for ex in sb]
            for sa, sb in zip(a.splits, b.splits)
        )
        assert any_diff

    def test_every_carved_row_has_prompt_and_constraints_above_floor(self):
        u = _synthetic_usable(2000)
        result = carve_ifbench_splits(seed=0, usable=u)
        for items in result.splits:
            for ex in items:
                assert isinstance(ex["prompt"], str) and ex["prompt"].strip()
                ids = ex["instruction_id_list"]
                assert isinstance(ids, list) and len(ids) >= CONSTRAINT_COUNT_FLOOR
                # parallel arrays
                kws = ex["kwargs_list"]
                assert isinstance(kws, list) and len(kws) == len(ids)

    def test_carve_result_records_floor_and_pool_size(self):
        u = _synthetic_usable(2000)
        result = carve_ifbench_splits(seed=0, usable=u)
        assert result.constraint_count_floor == CONSTRAINT_COUNT_FLOOR
        assert result.n_usable_pool == 2000
        assert result.uncovered_instruction_ids == ()


# =========================================================================
# Verifier-coverage check
# =========================================================================


class TestVerifierCoverage:
    def test_all_carved_rows_have_known_verifiers(self):
        u = _synthetic_usable(2000)
        result = carve_ifbench_splits(seed=0, usable=u)
        all_ok, uncovered = verify_carved_pool_constraint_coverage(result.splits)
        assert all_ok
        assert uncovered == []

    def test_an_unknown_iid_is_flagged(self):
        """Manually inject a row with an unknown instruction_id into the
        carved pool and confirm the coverage check catches it."""
        u = _synthetic_usable(2000)
        result = carve_ifbench_splits(seed=0, usable=u)
        # Mutate a single example to include an unknown id.
        sample = result.splits[0][0]
        sample["instruction_id_list"] = ["unknown_family:unknown_check"] + list(
            sample["instruction_id_list"]
        )
        all_ok, uncovered = verify_carved_pool_constraint_coverage(result.splits)
        assert not all_ok
        assert "unknown_family:unknown_check" in uncovered


# =========================================================================
# Live dataset verification (integration)
# =========================================================================


@pytest.mark.integration
class TestIFMultiConstraintsLive:
    """Verify the live `allenai/IF_multi_constraints_upto5` still matches
    the Chunk-11-redo numbers. HF cache only, no API spend."""

    @pytest.fixture(scope="class")
    def usable(self):
        from src.ifbench_data import load_ifbench_usable
        return load_ifbench_usable()

    def test_dataset_id_and_family(self, usable):
        assert usable.dataset_id == "allenai/IF_multi_constraints_upto5"
        assert "IF-RLVR" in usable.family

    def test_n_rows_raw_is_95373(self, usable):
        assert usable.n_rows_raw == 95373

    def test_raw_constraint_count_distribution(self, usable):
        # Chunk-11-redo measurement: 23007 / 23903 / 23322 / 18038 / 7103.
        assert usable.constraint_count_histogram_raw == {
            1: 23007, 2: 23903, 3: 23322, 4: 18038, 5: 7103,
        }

    def test_usable_pool_at_floor_3(self, usable):
        # >= 3 constraints: 23322 + 18038 + 7103 = 48463.
        assert usable.n_above_floor_pre_verifier_filter == 48463
        # Every iid in IF_multi_constraints_upto5 is in the curated
        # catalog, so the verifier-filter excludes nothing.
        assert usable.n_excluded_for_uncovered_verifier == 0
        assert usable.uncovered_instruction_ids == ()
        assert usable.n_usable == 48463

    def test_pool_comfortably_exceeds_target(self, usable):
        assert usable.n_usable >= TARGET_TOTAL * 10  # 48463 vs 545; ~89x

    def test_live_carve_succeeds(self, usable):
        result = carve_ifbench_splits(seed=0, usable=usable)
        sizes = {SPLIT_ORDER[i]: len(result.splits[i]) for i in range(4)}
        assert sizes == TARGET_SIZES
        all_ok, uncovered = verify_carved_pool_constraint_coverage(result.splits)
        assert all_ok
        assert uncovered == []
