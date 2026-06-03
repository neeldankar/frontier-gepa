"""Chunk 11 offline gates: IFBench data loader, dataset gate, splits.

Most tests are pure (no network) and validate the gate / shrink math on
synthetic counts. A small number of tests live in `TestIFBenchLive` and
hit the HuggingFace cache to verify the real dataset matches what Chunk 11
recorded; they are marked with `integration` and skipped by the default
`pytest -m "not integration"` run."""

from __future__ import annotations

import pytest

from src.ifbench_data import (
    D_FEEDBACK_REVIEW_FLOOR,
    IFBENCH_DATASET_ID,
    IFBENCH_FAMILY,
    INPUT_FIELDS,
    SPLIT_ORDER,
    TARGET_SIZES,
    TARGET_TOTAL,
    TERCILE_NOGO_FLOOR,
    GateDecision,
    IFBenchUsable,
    carve_ifbench_splits,
    compute_proportional_sizes,
    gate_decision,
)


# =========================================================================
# Constants and configuration
# =========================================================================


def test_dataset_id_is_canonical_allenai_ifbench_test():
    """The dataset id must point at the AllenAI multi-constraint IFBench
    family (Pyatkin et al. 2025), NOT Google IFEval."""
    assert IFBENCH_DATASET_ID == "allenai/IFBench_test"
    assert "AllenAI" in IFBENCH_FAMILY
    assert "IFBench" in IFBENCH_FAMILY
    assert "IFEval" not in IFBENCH_FAMILY
    assert "Pyatkin" in IFBENCH_FAMILY


def test_target_sizes_are_locked_to_experiment_1_layout():
    assert TARGET_SIZES == {
        "d_feedback": 150,
        "accept_batch": 20,
        "d_pareto": 75,
        "test": 300,
    }
    assert TARGET_TOTAL == 545
    assert SPLIT_ORDER == ("d_feedback", "accept_batch", "d_pareto", "test")
    assert INPUT_FIELDS == ("prompt", "instruction_id_list", "kwargs")


# =========================================================================
# Proportional shrink math
# =========================================================================


class TestProportionalSizes:
    def test_full_size_returns_target_unchanged(self):
        sizes = compute_proportional_sizes(545)
        assert sizes == TARGET_SIZES

    def test_more_than_target_still_returns_target(self):
        # If the dataset is bigger, we still only carve up to 545.
        sizes = compute_proportional_sizes(1000)
        assert sizes == TARGET_SIZES

    def test_300_usable_matches_chunk11_report_numbers(self):
        """The live IFBench_test count: 300 usable rows. The shrink must
        produce sizes that sum exactly to 300."""
        sizes = compute_proportional_sizes(300)
        assert sum(sizes.values()) == 300
        # Floor of 150 * 300/545 = 82.57 -> 82 + carry; expected exact
        # post-distribution numbers:
        assert sizes == {
            "d_feedback": 83,
            "accept_batch": 11,
            "d_pareto": 41,
            "test": 165,
        }

    @pytest.mark.parametrize("u", [100, 200, 300, 450, 544, 545, 600, 1500])
    def test_sums_match_capped_usable(self, u):
        sizes = compute_proportional_sizes(u)
        assert sum(sizes.values()) == min(u, TARGET_TOTAL)

    def test_ratios_are_preserved_within_rounding(self):
        sizes = compute_proportional_sizes(300)
        # All sizes are within 1 of the perfect proportional value.
        for k, v in sizes.items():
            ideal = TARGET_SIZES[k] * (300 / 545)
            assert abs(v - ideal) < 1.5, (k, v, ideal)

    def test_sizes_are_non_negative(self):
        for u in (50, 100, 200, 300, 545):
            sizes = compute_proportional_sizes(u)
            assert all(v >= 0 for v in sizes.values())


# =========================================================================
# Gate decisions
# =========================================================================


class TestGateDecision:
    def test_constants_match_build_plan(self):
        # The BUILD_PLAN §7 Chunk 11 thresholds the operator was given.
        assert D_FEEDBACK_REVIEW_FLOOR == 90
        assert TERCILE_NOGO_FLOOR == 20

    def test_go_when_full_target_available(self):
        d = gate_decision(545)
        assert d.verdict == "GO"
        assert d.sizes == TARGET_SIZES
        assert d.rank_tercile_size == 50

    def test_go_when_bigger_than_target(self):
        d = gate_decision(1000)
        assert d.verdict == "GO"
        assert d.sizes == TARGET_SIZES

    def test_operator_review_when_d_feedback_below_floor(self):
        """At 300 usable IFBench rows, d_feedback projects to 83 -> below
        the 90-floor for operator review. Tercile is 27 (= 83 // 3) which
        is above the NO-GO floor of 20, so verdict must be OPERATOR_REVIEW
        not NO_GO."""
        d = gate_decision(300)
        assert d.verdict == "OPERATOR_REVIEW"
        assert d.sizes["d_feedback"] == 83
        assert d.rank_tercile_size == 27
        assert "REVIEW" in d.verdict
        assert "D_feedback=83" in d.reason

    def test_no_go_when_tercile_at_or_below_floor(self):
        # Engineered count so that d_feedback // 3 lands at or below 20.
        # d_feedback = 60 -> tercile 20 (== floor) triggers NO_GO.
        # sum of (60, 8, 30, 120) = 218 -> usable=218 needed (sizing math
        # produces d_feedback ~= 60 at usable~218).
        # Easier: directly look up the gate at usable=219 to find the
        # threshold; not exact but a smaller value will definitely NO_GO.
        d = gate_decision(60)
        assert d.verdict == "NO_GO"
        assert d.rank_tercile_size <= TERCILE_NOGO_FLOOR

    def test_gate_decision_returns_immutable_shape(self):
        d = gate_decision(300)
        assert isinstance(d, GateDecision)
        assert isinstance(d.sizes, dict)
        assert d.verdict in {"GO", "OPERATOR_REVIEW", "NO_GO"}


# =========================================================================
# Carving on synthetic usable rows
# =========================================================================


def _synthetic_usable(n: int) -> IFBenchUsable:
    rows = [
        {
            "id": f"row{i}",
            "prompt": f"prompt {i}",
            "instruction_id_list": [f"constr:{i}"],
            "kwargs": [{"keyword": f"kw{i}"}],
        }
        for i in range(n)
    ]
    return IFBenchUsable(
        dataset_id=IFBENCH_DATASET_ID,
        family=IFBENCH_FAMILY,
        n_rows_raw=n,
        n_usable=n,
        constraint_count_histogram={1: n},
        rows=rows,
    )


class TestCarveSplits:
    def test_full_size_carve_produces_target_pools(self):
        u = _synthetic_usable(545)
        splits, gate = carve_ifbench_splits(seed=0, usable=u)
        assert gate.verdict == "GO"
        sizes = {SPLIT_ORDER[i]: len(splits[i]) for i in range(4)}
        assert sizes == TARGET_SIZES

    def test_split_sizes_match_shrunk_proportions(self):
        u = _synthetic_usable(300)
        splits, gate = carve_ifbench_splits(
            seed=0, usable=u, allow_review_threshold=True,
        )
        assert gate.verdict == "OPERATOR_REVIEW"
        sizes = {SPLIT_ORDER[i]: len(splits[i]) for i in range(4)}
        assert sizes == {"d_feedback": 83, "accept_batch": 11, "d_pareto": 41, "test": 165}

    def test_pools_are_pairwise_disjoint(self):
        u = _synthetic_usable(300)
        splits, _ = carve_ifbench_splits(
            seed=0, usable=u, allow_review_threshold=True,
        )
        seen: dict[str, str] = {}
        for name, items in zip(SPLIT_ORDER, splits):
            for ex in items:
                assert ex["id"] not in seen, (
                    f"id {ex['id']} appears in {seen[ex['id']]} and {name}"
                )
                seen[ex["id"]] = name
        assert len(seen) == sum(len(s) for s in splits)

    def test_determinism_same_seed_same_ids(self):
        u = _synthetic_usable(300)
        a, _ = carve_ifbench_splits(seed=42, usable=u, allow_review_threshold=True)
        b, _ = carve_ifbench_splits(seed=42, usable=u, allow_review_threshold=True)
        for split_a, split_b in zip(a, b):
            ids_a = [ex["id"] for ex in split_a]
            ids_b = [ex["id"] for ex in split_b]
            assert ids_a == ids_b

    def test_different_seeds_differ(self):
        u = _synthetic_usable(300)
        a, _ = carve_ifbench_splits(seed=42, usable=u, allow_review_threshold=True)
        b, _ = carve_ifbench_splits(seed=7,  usable=u, allow_review_threshold=True)
        any_diff = any(
            [ex["id"] for ex in sa] != [ex["id"] for ex in sb]
            for sa, sb in zip(a, b)
        )
        assert any_diff, "different seeds should produce different draws"

    def test_per_example_schema_carries_inputs(self):
        u = _synthetic_usable(300)
        splits, _ = carve_ifbench_splits(
            seed=0, usable=u, allow_review_threshold=True,
        )
        # Spot-check first 5 of each split: prompt + non-empty constraint
        # list + parallel kwargs.
        for items in splits:
            for ex in items[:5]:
                assert "prompt" in ex and isinstance(ex["prompt"], str) and ex["prompt"].strip()
                ids = ex["instruction_id_list"]
                assert isinstance(ids, list) and len(ids) >= 1
                kw = ex["kwargs"]
                assert isinstance(kw, list)

    def test_operator_review_blocks_carve_without_override(self):
        u = _synthetic_usable(300)
        with pytest.raises(RuntimeError, match=r"OPERATOR_REVIEW"):
            carve_ifbench_splits(seed=0, usable=u)  # no override

    def test_no_go_always_refuses(self):
        u = _synthetic_usable(60)
        # Even with the override flag set, NO_GO refuses.
        with pytest.raises(RuntimeError, match=r"NO_GO"):
            carve_ifbench_splits(
                seed=0, usable=u, allow_review_threshold=True,
            )


# =========================================================================
# Live dataset verification (integration)
# =========================================================================


@pytest.mark.integration
class TestIFBenchLive:
    """These tests actually load `allenai/IFBench_test` from the HF cache.
    Skipped by `pytest -m "not integration"`. Verify that the live dataset
    still matches the numbers Chunk 11 recorded; if any of these fail in
    the future, the live IFBench release changed and the report must be
    revisited."""

    @pytest.fixture(scope="class")
    def usable(self):
        from src.ifbench_data import load_ifbench_usable
        return load_ifbench_usable()

    def test_dataset_id_and_family(self, usable):
        assert usable.dataset_id == "allenai/IFBench_test"
        assert "AllenAI" in usable.family

    def test_n_rows_raw_is_300(self, usable):
        assert usable.n_rows_raw == 300

    def test_all_300_rows_are_usable(self, usable):
        assert usable.n_usable == 300

    def test_every_row_has_prompt_and_constraint_list(self, usable):
        for row in usable.rows:
            assert isinstance(row["prompt"], str) and row["prompt"].strip()
            assert isinstance(row["instruction_id_list"], list)
            assert len(row["instruction_id_list"]) >= 1

    def test_constraint_count_histogram(self, usable):
        # Chunk-11 measurement: 256 rows have 1 constraint, 44 rows have 2.
        assert usable.constraint_count_histogram == {1: 256, 2: 44}

    def test_live_gate_says_operator_review(self, usable):
        gate = gate_decision(usable.n_usable)
        assert gate.verdict == "OPERATOR_REVIEW"
        assert gate.sizes == {
            "d_feedback": 83, "accept_batch": 11, "d_pareto": 41, "test": 165,
        }
