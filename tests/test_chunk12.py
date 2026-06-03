"""Chunk 12 tests: IFBench program + feedback function + vendored
verifier wiring.

Mostly offline. One LM-using smoke test under `@pytest.mark.integration`
exercises the full program end-to-end on a single carved row.
"""

from __future__ import annotations

import dspy
import pytest

from src.ifbench_data import KNOWN_VERIFIER_IDS
from src.ifbench_feedback import (
    metric_fn,
    score_and_feedback,
)
from src.ifbench_program import (
    COMPONENT_NAMES,
    IFBenchProgram,
    build_program,
    load_seed_instruction,
)
from src.ifbench_verifiers import (
    ARG_REQUIRING_VERIFIER_IDS,
    INSTRUCTION_DICT,
)


# =========================================================================
# Vendored-registry coverage (the Chunk-11 promise)
# =========================================================================


def test_all_known_iids_are_in_official_registry():
    """Every id in KNOWN_VERIFIER_IDS (the Chunk-11 catalog) has a
    verifier class in the vendored INSTRUCTION_DICT. If this fails, the
    Chunk-11 catalog and the vendored registry have drifted."""
    missing = KNOWN_VERIFIER_IDS - set(INSTRUCTION_DICT)
    assert missing == set(), f"catalog ids missing from registry: {sorted(missing)}"


def test_arg_requiring_set_is_subset_of_catalog():
    assert ARG_REQUIRING_VERIFIER_IDS <= KNOWN_VERIFIER_IDS


def test_arg_requiring_set_has_expected_size():
    # Chunk-12 measurement: 35 of the 54 catalog IDs randomly generate
    # missing kwargs when build_description is called without them.
    assert len(ARG_REQUIRING_VERIFIER_IDS) == 35
    assert len(KNOWN_VERIFIER_IDS - ARG_REQUIRING_VERIFIER_IDS) == 19


# =========================================================================
# Program / seed-prompt structure
# =========================================================================


def test_seed_prompt_is_minimal_and_nonempty():
    text = load_seed_instruction()
    assert isinstance(text, str)
    assert len(text) > 0
    # Minimal-by-design: should be short. Hard cap a generous 600 chars
    # so future edits don't accidentally hand-engineer a long prompt.
    assert len(text) < 600, f"seed prompt grew to {len(text)} chars"


def test_program_has_one_named_predictor_called_answer():
    p = build_program()
    names = [n for n, _ in p.named_predictors()]
    assert names == ["answer"]
    assert COMPONENT_NAMES == ("answer",)


def test_program_loads_seed_instruction_into_signature():
    p = build_program()
    loaded = p.get_module_instruction("answer")
    assert loaded == load_seed_instruction()


# =========================================================================
# Helper: build crisp synthetic constraints we can hand-craft responses for
# =========================================================================


def _example(prompt: str, iids: list[str], kwargs_list: list) -> dspy.Example:
    return dspy.Example(
        id=0,
        source_key="syn",
        prompt=prompt,
        instruction_id_list=iids,
        kwargs_list=kwargs_list,
        constraint_count=len(iids),
    ).with_inputs("prompt", "instruction_id_list", "kwargs_list")


def _pred(text: str) -> dspy.Prediction:
    return dspy.Prediction(response=text)


# Three crisp constraints we'll mix:
#   keywords:existence with {"keywords": ["octopus"]}     -> response must contain "octopus"
#   punctuation:no_comma                                  -> response must contain no commas
#   change_case:english_capital                           -> response must be all caps

CONSTR_KEYWORD = ("keywords:existence", {"keywords": ["octopus"]})
CONSTR_NO_COMMA = ("punctuation:no_comma", None)
CONSTR_ALL_CAPS = ("change_case:english_capital", None)
CONSTR_ALL_LOWER = ("change_case:english_lowercase", None)


def test_helper_constraints_are_in_registry():
    for iid, _ in (CONSTR_KEYWORD, CONSTR_NO_COMMA, CONSTR_ALL_CAPS, CONSTR_ALL_LOWER):
        assert iid in INSTRUCTION_DICT


# =========================================================================
# 1. Score 1 when all satisfied (BUILD_PLAN test 1)
# =========================================================================


def test_score_1_when_all_constraints_satisfied():
    ex = _example(
        prompt="(prompt unused by feedback)",
        iids=[CONSTR_KEYWORD[0], CONSTR_NO_COMMA[0], CONSTR_ALL_CAPS[0]],
        kwargs_list=[CONSTR_KEYWORD[1], CONSTR_NO_COMMA[1], CONSTR_ALL_CAPS[1]],
    )
    # Response satisfies all three: contains "octopus" (case-insensitive
    # substring match) AND no commas AND all-caps (long enough for
    # langdetect to confidently classify as English).
    pred = _pred(
        "THE OCTOPUS IS A CLEVER CREATURE THAT LIVES IN THE GREAT OCEAN AND "
        "HUNTS QUIETLY EACH NIGHT WHILE THE OTHER FISH SLEEP IN THE CORAL"
    )
    result = score_and_feedback(ex, pred)
    assert result["score"] == 1.0, (
        f"unexpected score {result['score']} "
        f"with violations {[v['instruction_id'] for v in result['violations']]}"
    )
    assert result["violations"] == []
    assert len(result["satisfied"]) == 3


# =========================================================================
# 2. Score 0 when none satisfied (BUILD_PLAN test 2)
# =========================================================================


def test_score_0_when_no_constraints_satisfied():
    ex = _example(
        prompt="(prompt unused by feedback)",
        iids=[CONSTR_KEYWORD[0], CONSTR_NO_COMMA[0], CONSTR_ALL_CAPS[0]],
        kwargs_list=[CONSTR_KEYWORD[1], CONSTR_NO_COMMA[1], CONSTR_ALL_CAPS[1]],
    )
    # Fails all three: no "octopus" keyword, has commas, has lowercase.
    pred = _pred("a parrot, lives, here")
    result = score_and_feedback(ex, pred)
    assert result["score"] == 0.0
    assert result["satisfied"] == []
    assert len(result["violations"]) == 3
    vio_ids = [v["instruction_id"] for v in result["violations"]]
    assert vio_ids == [CONSTR_KEYWORD[0], CONSTR_NO_COMMA[0], CONSTR_ALL_CAPS[0]]


# =========================================================================
# 3. k of n constraints satisfied returns k/n (BUILD_PLAN test 3)
# =========================================================================


# The change_case:english_capital verifier checks `value.isupper() AND
# langdetect == "en"`; langdetect needs a sentence or two of real
# English to classify confidently, so partial-case tests use longer
# strings.

_LONG_CAPS_NO_OCTOPUS = (
    "A PARROT LIVES HERE IN THE GREAT FOREST AND SINGS EACH MORNING. "
    "THE TREES ARE GREEN AND THE SKY IS BLUE WHEN THE SUN RISES."
)
_LONG_LOWER_OCTOPUS_NO_COMMA = (
    "the octopus is a clever creature that lives in the ocean and hunts at night "
    "while the other fish sleep peacefully in the dark coral reefs nearby"
)
_LONG_LOWER_OCTOPUS_WITH_COMMA = (
    "the octopus, a clever creature, lives in the ocean and hunts at night while "
    "the other fish sleep peacefully in the dark coral reefs nearby"
)


@pytest.mark.parametrize(
    "response,expected_score,expected_violated_iids",
    [
        # All caps, no commas, but no "octopus" -> 2/3 satisfied.
        (_LONG_CAPS_NO_OCTOPUS, 2.0 / 3.0, [CONSTR_KEYWORD[0]]),
        # Contains "octopus", has commas, all lowercase -> 1/3.
        (_LONG_LOWER_OCTOPUS_WITH_COMMA, 1.0 / 3.0,
         [CONSTR_NO_COMMA[0], CONSTR_ALL_CAPS[0]]),
        # Contains "octopus", no commas, all lowercase -> 2/3.
        (_LONG_LOWER_OCTOPUS_NO_COMMA, 2.0 / 3.0, [CONSTR_ALL_CAPS[0]]),
    ],
)
def test_score_k_over_n_partial(response, expected_score, expected_violated_iids):
    ex = _example(
        prompt="(prompt unused)",
        iids=[CONSTR_KEYWORD[0], CONSTR_NO_COMMA[0], CONSTR_ALL_CAPS[0]],
        kwargs_list=[CONSTR_KEYWORD[1], CONSTR_NO_COMMA[1], CONSTR_ALL_CAPS[1]],
    )
    result = score_and_feedback(ex, _pred(response))
    assert result["score"] == pytest.approx(expected_score, abs=1e-9), (
        f"unexpected score {result['score']} for response {response!r}; "
        f"violations: {[v['instruction_id'] for v in result['violations']]}"
    )
    got_violated = [v["instruction_id"] for v in result["violations"]]
    assert sorted(got_violated) == sorted(expected_violated_iids)


# =========================================================================
# 4. Violation list names the correct unsatisfied constraints
#    (BUILD_PLAN test 4) + uniform-renderer correctness (Chunk-12 rev #4)
# =========================================================================


def test_violation_list_names_correct_iids_with_uniform_renderer():
    ex = _example(
        prompt="(prompt unused)",
        iids=[CONSTR_KEYWORD[0], CONSTR_NO_COMMA[0], CONSTR_ALL_CAPS[0]],
        kwargs_list=[CONSTR_KEYWORD[1], CONSTR_NO_COMMA[1], CONSTR_ALL_CAPS[1]],
    )
    # All-caps long-enough-for-langdetect English with no commas; flip
    # only the keyword (no "octopus").
    result = score_and_feedback(ex, _pred(_LONG_CAPS_NO_OCTOPUS))
    assert len(result["violations"]) == 1, (
        f"unexpected violations: {[v['instruction_id'] for v in result['violations']]}"
    )
    v = result["violations"][0]
    assert v["instruction_id"] == "keywords:existence"
    # Feedback string mentions the violated id by name + the kwargs as
    # readable key=value pairs (the uniform renderer; no per-family
    # customization).
    fb = result["feedback"]
    assert "keywords:existence" in fb
    assert "keywords=" in fb
    assert "octopus" in fb
    # Also sanity-check satisfied counts.
    assert len(result["satisfied"]) == 2


# =========================================================================
# 5. metric_fn returns just the score (matches the Chunk-3 shape)
# =========================================================================


def test_metric_fn_returns_score_only():
    ex = _example(
        prompt="(prompt unused)",
        iids=[CONSTR_KEYWORD[0], CONSTR_NO_COMMA[0]],
        kwargs_list=[CONSTR_KEYWORD[1], CONSTR_NO_COMMA[1]],
    )
    val = metric_fn(ex, _pred("OCTOPUS WITH NO COMMAS"))
    assert val == pytest.approx(1.0)
    val_partial = metric_fn(ex, _pred("octopus, has, a comma"))
    assert val_partial == pytest.approx(0.5)


# =========================================================================
# 6. Schema test (Chunk-12 revision #3): loader produces parallel
#    instruction_id_list and kwargs_list
# =========================================================================


@pytest.mark.integration
class TestLoaderSchema:
    @pytest.fixture(scope="class")
    def usable(self):
        from src.ifbench_data import load_ifbench_usable
        return load_ifbench_usable()

    def test_every_carved_example_has_parallel_arrays(self, usable):
        # Spot-check the first 200 carved rows
        for row in usable.rows[:200]:
            iids = row["instruction_id_list"]
            kws = row["kwargs_list"]
            assert isinstance(iids, list) and isinstance(kws, list)
            assert len(iids) == len(kws), (
                f"length mismatch in row id={row['id']}: "
                f"iids={len(iids)} kws={len(kws)}"
            )
            assert len(iids) >= 3, "loader filter floor violated"

    def test_every_carved_row_has_complete_kwargs(self, usable):
        """Chunk-12 revision #1 + #2: every arg-requiring constraint in
        every carved row carries non-None populated kwargs. Replaces the
        self-referential wiring test from the original plan."""
        from src.ifbench_verifiers import ARG_REQUIRING_VERIFIER_IDS as ARG_SET
        violations = []
        for row in usable.rows[:500]:  # cap the scan for test speed
            for iid, kw in zip(row["instruction_id_list"], row["kwargs_list"]):
                if iid not in ARG_SET:
                    continue
                if not isinstance(kw, dict):
                    violations.append((row["id"], iid, "kw is not a dict"))
                elif not any(v is not None for v in kw.values()):
                    violations.append((row["id"], iid, "kw has all-None values"))
        assert violations == [], (
            f"{len(violations)} rows fail kwargs-completeness; first few: {violations[:5]}"
        )


# =========================================================================
# 7. End-to-end LM smoke (BUILD_PLAN test 6)
# =========================================================================


@pytest.mark.integration
class TestProgramEndToEnd:
    @pytest.fixture(scope="class")
    def carved_example(self):
        from src.ifbench_data import carve_ifbench_splits, load_ifbench_usable
        usable = load_ifbench_usable()
        carve = carve_ifbench_splits(seed=0, usable=usable)
        return carve.splits[0][0]

    def test_program_runs_end_to_end_on_one_example(self, carved_example):
        import os

        from dotenv import load_dotenv

        load_dotenv()
        task_model = os.environ.get("TASK_MODEL")
        if not task_model:
            pytest.skip("TASK_MODEL not set")

        lm = dspy.LM(
            model=task_model,
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048,
            api_key=os.environ.get("TASK_MODEL_API_KEY") or None,
            num_retries=3,
        )
        dspy.settings.configure(lm=lm)
        program = build_program()
        out = program(prompt=carved_example["prompt"])
        # Non-empty response
        assert isinstance(out.response, str) and out.response.strip()
        # Score is a valid float in [0,1] (verifier path works end-to-end)
        score = metric_fn(carved_example, out)
        assert 0.0 <= score <= 1.0
        # And the feedback string has the expected structure
        result = score_and_feedback(carved_example, out)
        assert "constraints satisfied" in result["feedback"]
