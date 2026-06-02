"""
tests/test_chunk2.py -- Chunk 2 validation

Offline tests (no API calls, fast):
    pytest tests/test_chunk2.py -m "not integration"

Integration smoke test (1 rollout, costs ~$0.006):
    pytest tests/test_chunk2.py -m integration

Three lines are marked ADAPT -- change those to match the actual src/ module
paths and field names Claude Code used. Do not change the assertions.
"""

import os
import pytest
import yaml
from pathlib import Path

from dotenv import load_dotenv

# ADAPT: match actual module paths
from src.data import load_splits      # must return (d_feedback, accept_batch, d_pareto, test_set)
from src.program import build_program  # must return the DSPy multi-hop program

SEEDS_DIR = Path("prompts/seeds")
CONFIG_PATH = Path("config/experiment.yaml")
FIXED_SEED = 42

# Load .env at module import so the integration fixture has TASK_MODEL etc.
# available. No-op for the offline tests.
load_dotenv(CONFIG_PATH.parent.parent / ".env")


# ---------------------------------------------------------------------------
# 1. Seed files -- are the files there and does the program load them?
# ---------------------------------------------------------------------------

# §9 distractor fallback: create_query_hop2 is dropped because there is no
# retrieval call (the program consumes the provided distractor paragraphs
# directly). prompts/seeds/create_query_hop2.md is left in place but unused.
SEED_MODULES = ["summarize1", "summarize2", "final_answer"]


def test_seed_files_exist():
    for name in SEED_MODULES:
        p = SEEDS_DIR / f"{name}.md"
        assert p.exists(), f"Missing seed file: {p}"


def test_seed_files_nonempty():
    for name in SEED_MODULES:
        text = (SEEDS_DIR / f"{name}.md").read_text().strip()
        assert len(text) > 10, f"Seed file suspiciously short: {name}.md"


def test_program_loads_seeds_from_files():
    """
    The instruction text in the live program must match the seed files exactly.
    If this fails, Claude Code invented its own instructions instead of reading prompts/seeds/.
    This is the most important test in this file.
    """
    program = build_program()
    for name in SEED_MODULES:
        expected = (SEEDS_DIR / f"{name}.md").read_text().strip()
        # ADAPT: replace get_module_instruction with however the program exposes
        # the loaded instruction string for each module (attribute, dict, method, etc.)
        actual = program.get_module_instruction(name).strip()
        assert actual == expected, (
            f"'{name}' does not match its seed file.\n"
            f"  File:   {expected!r}\n"
            f"  Loaded: {actual!r}\n"
            "Claude Code invented or hardcoded this instruction. Fix it to read from prompts/seeds/."
        )


# ---------------------------------------------------------------------------
# 2. Split sizes and disjointness
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def splits():
    return load_splits(seed=FIXED_SEED)


EXPECTED_SIZES = {
    "d_feedback": 150,   # §5 fallback (100 -> 150); see DEVIATIONS.md entry 3
    "accept_batch": 20,
    "d_pareto": 75,
    "test": 300,
}


def test_split_sizes(splits):
    d_feedback, accept_batch, d_pareto, test_set = splits
    actuals = {
        "d_feedback": len(d_feedback),
        "accept_batch": len(accept_batch),
        "d_pareto": len(d_pareto),
        "test": len(test_set),
    }
    for name, expected in EXPECTED_SIZES.items():
        assert actuals[name] == expected, (
            f"{name}: expected {expected}, got {actuals[name]}"
        )


def test_splits_total_545(splits):
    total = sum(len(s) for s in splits)
    assert total == 545, f"Splits total {total}, expected 545 (100+50+20+75+300)"


def test_splits_disjoint(splits):
    """No example may appear in more than one split. Leakage here poisons the experiment."""
    d_feedback, accept_batch, d_pareto, test_set = splits
    names = ["d_feedback", "accept_batch", "d_pareto", "test"]
    # ADAPT: replace ex["id"] with however examples are keyed in the loaded HotpotQA dataset
    id_sets = [set(ex["id"] for ex in s) for s in splits]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            overlap = id_sets[i] & id_sets[j]
            assert not overlap, (
                f"Leakage: {names[i]} and {names[j]} share {len(overlap)} examples. "
                "This must be fixed before any runs."
            )


def test_examples_have_required_fields(splits):
    required = {"question", "answer", "supporting_facts"}
    for split in splits:
        for ex in split[:5]:  # spot-check first 5 of each split
            for field in required:
                assert field in ex, f"Example missing required field '{field}'"


# ---------------------------------------------------------------------------
# 3. Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_reproducible():
    a = load_splits(seed=FIXED_SEED)
    b = load_splits(seed=FIXED_SEED)
    for s_a, s_b in zip(a, b):
        assert [ex["id"] for ex in s_a] == [ex["id"] for ex in s_b], (
            "Same seed produced different splits -- not reproducible"
        )


def test_different_seeds_differ():
    a = load_splits(seed=42)
    b = load_splits(seed=99)
    any_differ = any(
        [ex["id"] for ex in s_a] != [ex["id"] for ex in s_b]
        for s_a, s_b in zip(a, b)
    )
    assert any_differ, "Different seeds produced identical splits -- seeding has no effect"


# ---------------------------------------------------------------------------
# 4. Config matches locked parameters
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_config_locked_params(config):
    # experiment.yaml groups keys (splits, minibatch, stopping) for readability.
    # d_feedback bumped to 150 by the §5 fallback (DEVIATIONS.md entry 3);
    # the rest are §15 locked values.
    assert config["splits"]["d_feedback"] == 150
    assert config["splits"]["accept_batch"] == 20
    assert config["splits"]["d_pareto"] == 75
    assert config["splits"]["test"] == 300
    assert config["minibatch"]["b"] == 3
    assert config["stopping"]["n"] == 44


# ---------------------------------------------------------------------------
# 5. Integration: program smoke test (LM only; no retrieval in distractor mode)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProgramSmoke:

    @pytest.fixture(scope="class")
    def example(self):
        splits = load_splits(seed=FIXED_SEED)
        return splits[0][0]  # first D_feedback example

    @pytest.fixture(scope="class")
    def result(self, example):
        # Configure the task LM exactly as src/smoke_chunk2.py does. Setup, not
        # an assertion; the test contracts (answer present, hop1 titles, gold
        # in context, intermediate fields, hops split) are unchanged.
        import dspy
        config = yaml.safe_load(CONFIG_PATH.read_text())
        task_model = os.environ.get("TASK_MODEL")
        if not task_model:
            pytest.skip("TASK_MODEL is not set; cannot run the integration rollout.")
        api_key = os.environ.get(config["task_model"]["api_key_env"])
        base_url = os.environ.get(config["task_model"]["base_url_env"]) or None
        lm_kwargs = dict(
            model=task_model,
            temperature=float(config["task_model"]["temperature"]),
            top_p=float(config["task_model"]["top_p"]),
            max_tokens=int(config["task_model"]["max_tokens"]),
        )
        if api_key:
            lm_kwargs["api_key"] = api_key
        if base_url:
            lm_kwargs["api_base"] = base_url
        dspy.settings.configure(lm=dspy.LM(**lm_kwargs))

        program = build_program()
        return program(**dict(example.inputs()))

    def test_answer_present_and_nonempty(self, result):
        answer = getattr(result, "answer", result.get("answer", ""))
        assert isinstance(answer, str) and answer.strip(), "Program returned an empty answer"

    def test_hop1_titles_present_and_nonempty(self, result):
        """hop1_titles is required by the Chunk 3 feedback function."""
        assert hasattr(result, "hop1_titles") or "hop1_titles" in result, (
            "hop1_titles missing from program output. "
            "Chunk 3 mu_f cannot compute the retrieval gap without it."
        )
        titles = getattr(result, "hop1_titles", result.get("hop1_titles"))
        assert len(titles) > 0, "Hop-1 had no paragraphs assigned in the context split."

    def test_gold_paragraphs_in_context(self, example):
        """Distractor mode: both gold supporting paragraphs must be in the provided
        10-paragraph context. Replaces the old hop2_titles retrieval check, which
        no longer applies (no retrieval, just first/last-half split of context)."""
        gold_titles = set(example["supporting_facts"]["title"])
        context_titles = set(example["context_titles"])
        assert gold_titles, "Example has no gold supporting titles -- malformed data."
        missing = gold_titles - context_titles
        assert not missing, (
            f"Gold supporting titles missing from the distractor context: {sorted(missing)}. "
            "If this fires, the dataset row is malformed or the loader corrupted context_titles."
        )

    def test_intermediate_fields_present(self, result):
        """summary_1 and summary_2 must be in the output for traces to be complete.
        hop2_query was dropped in the §9 distractor fallback (no create_query_hop2)."""
        for field in ["summary_1", "summary_2"]:
            assert hasattr(result, field) or field in result, (
                f"Intermediate field '{field}' missing. "
                "The feedback function and trace logging both depend on this."
            )

    def test_hops_split_into_two_halves(self, result):
        """In distractor mode, the 10 provided paragraphs are split into a first-half
        hop1 bucket and a second-half hop2 bucket; the two halves must be disjoint."""
        hop1 = frozenset(getattr(result, "hop1_titles", result.get("hop1_titles", [])))
        hop2 = frozenset(getattr(result, "hop2_titles", result.get("hop2_titles", [])))
        assert hop1 and hop2, "One of the hop buckets is empty -- context split is broken."
        overlap = hop1 & hop2
        assert not overlap, f"hop1 and hop2 buckets overlap on titles: {sorted(overlap)}"
