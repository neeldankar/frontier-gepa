# Chunk 12 report: IFBench program, feedback, and vendored verifiers

## Verdict: ready for Chunk 13

The one-module DSPy program, the seed prompt, the feedback function,
the vendored verifier registry, and all five revisions the user
specified for this chunk are in place. Offline pytest passes 104/104;
the three live integration tests (loader-schema + complete-kwargs +
LM end-to-end) pass against the live dataset.

## Vendored verifier source

| | |
|---|---|
| Upstream repo | <https://github.com/allenai/open-instruct> |
| Upstream path | `open_instruct/IFEvalG/` |
| Pinned commit | **`ebca9f7d4921042d7d8a65369e513d6566f0f3ba`** (2026-04-08) |
| Files vendored | `instructions.py`, `instructions_registry.py`, `instructions_util.py` |
| Modifications | exactly two relative-import rewrites; everything else byte-identical |
| License | Apache 2.0; provenance in `src/ifbench_verifiers/PROVENANCE.md` |
| nltk data | `punkt` + `punkt_tab` English-only, pre-downloaded to `src/ifbench_verifiers/data/nltk_data/`; nltk.data.path patched in `src/ifbench_verifiers/__init__.py` so the verifier path is fully offline |

## Revision summary

The five revisions on the approved plan are all implemented:

| # | revision | implementation |
|---|---|---|
| 1 | kwargs verbatim; never default `None` → `{}` | `src/ifbench_feedback.py::_verify_one` raises if an arg-requiring id is missing kwargs; `src/ifbench_data.py::_kwargs_complete` pre-filters at load time |
| 2 | drop self-referential wiring test #6; add kwargs-completeness test | `tests/test_chunk12.py::TestLoaderSchema::test_every_carved_row_has_complete_kwargs` |
| 3 | loader produces both `instruction_id_list` and `kwargs_list`; schema test | already in Chunk 11; `TestLoaderSchema::test_every_carved_example_has_parallel_arrays` codifies the invariant |
| 4 | uniform renderer for all 54 ids; no per-family customization | `src/ifbench_feedback.py::_render_constraint` |
| 5 | pre-download + pin nltk data; spot-check prompt-constraint alignment | nltk data pinned offline (see above); spot-check below |

## Arg-requiring partition (Chunk-12 revision #1, correctness hinge)

The set of instruction_ids whose verifiers `random-generate` missing
kwargs was determined empirically by scanning the live
`IF_multi_constraints_upto5` dataset: for every id in the dataset,
either every occurrence carries a populated kwargs dict, or every
occurrence carries `None` / an all-None placeholder dict. The dataset
is consistent. The partition:

| count | family | description |
|---|---|---|
| **35** | `ARG_REQUIRING_VERIFIER_IDS` | always populated; feedback must pass kwargs verbatim |
| **19** | `KNOWN_VERIFIER_IDS - ARG_REQUIRING_VERIFIER_IDS` | always `None` or all-None placeholder; verifier takes no args |

Both sets are frozen in `src/ifbench_verifiers/__init__.py` and asserted
to add up to 54 (the catalog total).

## Loader audit at live numbers

```
n_rows_raw:                              95,373
n_above_floor (>=3 constraints):         48,463
n_excluded_for_uncovered_verifier:            0
n_excluded_for_incomplete_kwargs:             0
n_usable (carved-pool source):           48,463
carve sizes:    d_feedback=150, accept_batch=20, d_pareto=75, test=300
```

Verified live by `tests/test_chunk11.py::TestIFMultiConstraintsLive` and
`tests/test_chunk12.py::TestLoaderSchema`.

## Feedback wording

One uniform renderer for all 54 instruction IDs (revision #4). Each
constraint renders as `instruction_id (key=value, ...)`; the
`instruction_id` is already semantic and `gpt-4.1-mini` parses literal
dicts fine. Example feedback string from a 2/3-satisfied case:

```
Score 0.6667 (2/3 constraints satisfied).
Satisfied (2): punctuation:no_comma, change_case:english_capital.
Violated (1):
  - keywords:existence (keywords=['octopus'])
```

The format is fixed across all 54 ids, so the feedback-quality variable
the user flagged ("per-family formatting variance would introduce a
feedback-quality variable that is not the band") is removed by
construction.

## Prompt-constraint spot-check (revision #5, second half)

For the first 5 d_feedback rows under `seed_splits=0`, every verifier-
checked constraint has a corresponding NL clause in the prompt text.
Example (row 0 prompt verbatim):

> "Write a song inspired by a storyteller's tale that consists of
> exactly three verses. Each verse should be two paragraphs long.
> Highlight two sections in each verse that capture the essence of the
> story. The song must include the words "whisper" at least four times
> and "journey" at least three times. The song should conclude with the
> line: "And thus began the endless song." **The last word of your
> response should be the word tension. There should be 2 paragraphs.
> Paragraphs and only paragraphs are separated with each other by two
> line breaks. Enclose every word in your response within square
> brackets.**"

Verifier checks on row 0:
- `last_word:last_word_answer` with `{'last_word': 'tension'}`
  ✓ stated explicitly: "The last word of your response should be the word tension."
- `paragraphs:paragraphs2`
  ✓ stated: "There should be 2 paragraphs..."
- `detectable_format:square_brackets`
  ✓ stated: "Enclose every word in your response within square brackets."

Across the 5-row spot-check, **no verifier-checked constraint was
absent from its prompt text**.

**Substrate observation worth noting (NOT a defect, but flagged for the
operator):** the prompts contain MORE NL constraints than the verifier
registry checks. In the row above, "exactly three verses", "Each verse
should be two paragraphs long", "Highlight two sections", "whisper at
least four times", "journey at least three times", and "conclude with
'And thus began the endless song.'" are all NL-stated constraints
the IFEvalG registry does **not** check. The model can satisfy or
violate them independently of its score. This is consistent with how
`IF_multi_constraints_upto5` was assembled: constraints were COMPOSED
by appending NL phrases to a base prompt, with only a subset entering
the verifier's `instruction_id_list`. The scoring axis is the verifier
subset; the visible NL surface is broader. This makes the substrate a
*precise-instruction-following* task where the model is allowed to
ignore some NL constraints without losing points, but must satisfy
every constraint the verifier checks. The IFBench paper's framing is
consistent with this design.

## Files added in this chunk

| file | role |
|---|---|
| `src/ifbench_verifiers/__init__.py` | nltk path patch, INSTRUCTION_DICT re-export, KNOWN_VERIFIER_IDS, ARG_REQUIRING_VERIFIER_IDS |
| `src/ifbench_verifiers/instructions.py` | vendored verbatim (one relative-import rewrite) |
| `src/ifbench_verifiers/instructions_registry.py` | vendored verbatim (one relative-import rewrite) |
| `src/ifbench_verifiers/instructions_util.py` | vendored verbatim |
| `src/ifbench_verifiers/LICENSE` | Apache 2.0 text |
| `src/ifbench_verifiers/PROVENANCE.md` | source URL, pinned commit, modification log |
| `src/ifbench_verifiers/data/nltk_data/tokenizers/punkt/english.pickle` | tokenizer data (English-only) |
| `src/ifbench_verifiers/data/nltk_data/tokenizers/punkt/PY3/english.pickle` | tokenizer data (English-only, PY3 variant) |
| `src/ifbench_verifiers/data/nltk_data/tokenizers/punkt_tab/english/...` | new-style tokenizer tables (English-only) |
| `src/ifbench_program.py` | `IFBenchAnswer` signature + `IFBenchProgram` one-module class + `build_program` + `load_seed_instruction` |
| `src/ifbench_feedback.py` | `score_and_feedback`, `metric_fn`, `_verify_one`, `_render_feedback`, `_render_constraint` |
| `prompts/seeds/ifbench_answer.md` | minimal seed prompt (2 sentences) |
| `tests/test_chunk12.py` | 14 offline + 3 integration tests |
| `src/ifbench_data.py` (modified) | `_kwargs_complete` filter; `n_excluded_for_incomplete_kwargs` field added to `IFBenchUsable` |
| `tests/test_chunk11.py` (modified) | updated synthetic fixture for the new field; live test now asserts `n_usable` accounts for all exclusion buckets |
| `requirements.txt` (modified) | added `nltk`, `langdetect`, `immutabledict`, `absl-py` |

## What this chunk does NOT do (deferred)

- Chunk 13 will score the base system on D_feedback at temp 0.6, build
  the difficulty table, and run the continuity diagnostic.
- Chunk 14 wires this program + feedback + verifier into `run_gepa.run`
  and launches the matrix.

No Chunk 13 launch. No CLAUDE.md flip.
