"""
tests/test_chunk2.py -- Chunk 2 validation

Offline tests (no API calls, fast):
    pytest tests/test_chunk2.py -m "not integration"

Integration smoke test (1 rollout, costs ~$0.006):
    pytest tests/test_chunk2.py -m integration

Three lines are marked ADAPT -- change those to match the actual src/ module
paths and field names Claude Code used. Do not change the assertions.
"""

import pytest
import yaml
from pathlib import Path

# ADAPT: match actual module paths
from src.data import load_splits      # must return (d_feedback, accept_batch, d_pareto, test_set)
from src.program import build_program  # must return the DSPy multi-hop program

SEEDS_DIR = Path("prompts/seeds")
CONFIG_PATH = Path("config/experiment.yaml")
FIXED_SEED = 42


# ---------------------------------------------------------------------------
# 1. Seed files -- are the files there and does the program load them?
# ---------------------------------------------------------------------------

SEED_MODULES = ["summarize1", "create_query_hop2", "summarize2", "final_answer"]


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
    "d_feedback": 100,
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


def test_splits_total_495(splits):
    total = sum(len(s) for s in splits)
    assert total == 495, f"Splits total {total}, expected 495"


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
    # ADAPTED: experiment.yaml groups keys (splits, minibatch, stopping) for
    # readability. The locked numbers themselves are unchanged.
    assert config["splits"]["d_feedback"] == 100
    assert config["splits"]["accept_batch"] == 20
    assert config["splits"]["d_pareto"] == 75
    assert config["splits"]["test"] == 300
    assert config["minibatch"]["b"] == 3
    assert config["stopping"]["n"] == 44


# ---------------------------------------------------------------------------
# 5. Integration: program smoke test (requires retrieval + model, ~1 rollout)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProgramSmoke:

    @pytest.fixture(scope="class")
    def result(self):
        splits = load_splits(seed=FIXED_SEED)
        example = splits[0][0]  # first D_feedback example
        program = build_program()
        return program(question=example["question"])

    def test_answer_present_and_nonempty(self, result):
        answer = getattr(result, "answer", result.get("answer", ""))
        assert isinstance(answer, str) and answer.strip(), "Program returned an empty answer"

    def test_hop1_titles_present_and_nonempty(self, result):
        """hop1_titles is required by the Chunk 3 feedback function to compute the retrieval gap."""
        assert hasattr(result, "hop1_titles") or "hop1_titles" in result, (
            "hop1_titles missing from program output. "
            "Chunk 3 mu_f cannot compute the retrieval gap without it."
        )
        titles = getattr(result, "hop1_titles", result.get("hop1_titles"))
        assert len(titles) > 0, (
            "Hop-1 returned no passages. ColBERTv2 may be down -- consider the distractor fallback."
        )

    def test_hop2_titles_present_and_nonempty(self, result):
        """hop2_titles required for the Chunk 3 feedback function."""
        assert hasattr(result, "hop2_titles") or "hop2_titles" in result, (
            "hop2_titles missing from program output."
        )
        titles = getattr(result, "hop2_titles", result.get("hop2_titles"))
        assert len(titles) > 0, "Hop-2 returned no passages."

    def test_intermediate_fields_present(self, result):
        """summary_1, hop2_query, and summary_2 must be in the output for traces to be complete."""
        for field in ["summary_1", "hop2_query", "summary_2"]:
            assert hasattr(result, field) or field in result, (
                f"Intermediate field '{field}' missing. "
                "The feedback function and trace logging both depend on this."
            )

    def test_hops_retrieved_different_passages(self, result):
        """Hop-2 uses a different query, so it should pull at least some different passages."""
        hop1 = frozenset(getattr(result, "hop1_titles", result.get("hop1_titles", [])))
        hop2 = frozenset(getattr(result, "hop2_titles", result.get("hop2_titles", [])))
        assert hop1 != hop2, (
            "Hop-1 and hop-2 retrieved identical passages. "
            "Either create_query_hop2 is not generating a distinct query, "
            "or retrieval is ignoring the query and returning the same results."
        )
