"""Vendored IFBench / IFEval-G constraint verifiers (BUILD_PLAN.md §7 Chunk 12).

This package vendors the AllenAI / Google IFEval-G verifier source so the
Chunk-14 matrix never depends on a network fetch or on the heavy
`open-instruct` PyPI distribution. See PROVENANCE.md for the pinned
upstream commit hash and the modification log.

Public surface:

- :data:`INSTRUCTION_DICT` - mapping from instruction_id string
  (e.g. ``"keywords:existence"``) to the verifier class. Standard IFEval
  pattern: instantiate with ``cls(instruction_id)``, configure with
  ``verifier.build_description(**kwargs)``, then evaluate with
  ``verifier.check_following(response_text) -> bool``.

- :data:`KNOWN_VERIFIER_IDS` - frozen set of the 54 instruction IDs the
  Experiment-2 substrate (``allenai/IF_multi_constraints_upto5``) uses;
  re-exported here so feedback code has one import for both the registry
  and the substrate's allow-list.

- :data:`ARG_REQUIRING_VERIFIER_IDS` - frozen set of the 35 instruction
  IDs whose verifiers RANDOMLY GENERATE the missing arguments when
  ``build_description`` is called without kwargs (e.g.
  ``keywords:existence`` will pick a random keyword if none is supplied).
  Feedback callers MUST pass the dataset's kwargs verbatim for every id
  in this set; otherwise the verifier silently checks against a random
  threshold and corrupts the score. The set was determined empirically
  by scanning the live ``IF_multi_constraints_upto5`` dataset at
  Chunk-12 time (every constraint-instance in the dataset for these 35
  IDs carries a non-None, populated kwargs dict; the remaining 19 IDs
  always carry ``None`` or an all-None placeholder dict). The
  partitioning is documented in CHUNK12_REPORT.md.

This package also pins nltk's data path to a repo-local copy of the
``punkt`` and ``punkt_tab`` English tokenizers so the verifier path is
deterministic and offline; see ``data/nltk_data/``.
"""

from __future__ import annotations

from pathlib import Path

import nltk

# Pin nltk to the repo-local English tokenizer data so Chunk 14's
# overnight matrix cannot fail mid-run on a network fetch.
_NLTK_DATA = Path(__file__).resolve().parent / "data" / "nltk_data"
if str(_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(_NLTK_DATA))

# Re-export the vendored registry. The import order matters because
# `instructions_registry` imports `instructions`, which imports
# `instructions_util`, which imports `nltk`. By the time we get here, the
# nltk.data.path patch above is already in place.
from .instructions_registry import INSTRUCTION_DICT  # noqa: E402

# Mirror of src/ifbench_data.py::KNOWN_VERIFIER_IDS so the feedback code
# can do one import for both the registry and the catalog assertion. The
# two constants are kept identical by Chunk-11 / Chunk-12 tests.
KNOWN_VERIFIER_IDS: frozenset[str] = frozenset({
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "combination:repeat_prompt",
    "combination:two_responses",
    "copy:copy",
    "copy:copying_multiple",
    "copy:copying_simple",
    "copy:repeat_phrase",
    "count:count_increment_word",
    "count:count_unique",
    "count:counting_composition",
    "count:lowercase_counting",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:bigram_wrapping",
    "detectable_format:constrained_response",
    "detectable_format:json_format",
    "detectable_format:multiple_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:sentence_hyphens",
    "detectable_format:square_brackets",
    "detectable_format:title",
    "first_word:first_word_answer",
    "first_word:first_word_sent",
    "keywords:exclude_word_harder",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:keyword_specific_position",
    "keywords:letter_frequency",
    "keywords:no_adjacent_consecutive",
    "keywords:palindrome",
    "keywords:start_end",
    "keywords:word_count_different_numbers",
    "keywords:word_once",
    "language:response_language",
    "last_word:last_word_answer",
    "last_word:last_word_sent",
    "length_constraints:nth_paragraph_first_word",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "letters:letter_counting",
    "letters:letter_counting2",
    "new:copy_span_idx",
    "paragraphs:paragraphs",
    "paragraphs:paragraphs2",
    "punctuation:no_comma",
    "punctuation:punctuation_dot",
    "punctuation:punctuation_exclamation",
    "startend:end_checker",
    "startend:quotation",
})
assert len(KNOWN_VERIFIER_IDS) == 54, "verifier catalog size drift"

# Empirically derived from the live IF_multi_constraints_upto5 dataset
# at Chunk-12 time (2026-06-02). For every id in this set, the dataset
# always carries a non-None, populated kwargs dict; for every id NOT in
# this set, the dataset always carries None or an all-None placeholder
# dict (which is functionally no-arg). The IFEval verifier
# build_description silently random-generates missing args for the
# arg-requiring families, so feedback callers MUST pass the dataset's
# kwargs verbatim for these 35 ids.
ARG_REQUIRING_VERIFIER_IDS: frozenset[str] = frozenset({
    "change_case:capital_word_frequency",
    "combination:repeat_prompt",
    "copy:copy",
    "copy:copying_multiple",
    "copy:copying_simple",
    "copy:repeat_phrase",
    "count:count_increment_word",
    "count:counting_composition",
    "count:lowercase_counting",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:multiple_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "first_word:first_word_answer",
    "first_word:first_word_sent",
    "keywords:exclude_word_harder",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:keyword_specific_position",
    "keywords:letter_frequency",
    "keywords:word_count_different_numbers",
    "keywords:word_once",
    "language:response_language",
    "last_word:last_word_answer",
    "last_word:last_word_sent",
    "length_constraints:nth_paragraph_first_word",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "letters:letter_counting",
    "letters:letter_counting2",
    "new:copy_span_idx",
    "startend:end_checker",
})
assert len(ARG_REQUIRING_VERIFIER_IDS) == 35, "arg-requiring set size drift"
assert ARG_REQUIRING_VERIFIER_IDS <= KNOWN_VERIFIER_IDS, (
    "ARG_REQUIRING_VERIFIER_IDS contains ids not in KNOWN_VERIFIER_IDS"
)

__all__ = [
    "INSTRUCTION_DICT",
    "KNOWN_VERIFIER_IDS",
    "ARG_REQUIRING_VERIFIER_IDS",
]
