# Vendored verifier provenance

## Source

- Upstream repo: <https://github.com/allenai/open-instruct>
- Upstream path: `open_instruct/IFEvalG/`
- Pinned commit: **`ebca9f7d4921042d7d8a65369e513d6566f0f3ba`** (2026-04-08)
- Upstream URL at pin: <https://github.com/allenai/open-instruct/tree/ebca9f7d4921042d7d8a65369e513d6566f0f3ba/open_instruct/IFEvalG>

## Files

| file | upstream | modification |
|---|---|---|
| `instructions.py` | `open_instruct/IFEvalG/instructions.py` | **one mechanical edit**: `from open_instruct.IFEvalG import instructions_util` → `from . import instructions_util` |
| `instructions_registry.py` | `open_instruct/IFEvalG/instructions_registry.py` | **one mechanical edit**: `from open_instruct.IFEvalG import instructions` → `from . import instructions` |
| `instructions_util.py` | `open_instruct/IFEvalG/instructions_util.py` | **byte-identical** to upstream |
| `data/nltk_data/tokenizers/punkt/` | downloaded from NLTK at vendoring time | pruned to English-only (`english.pickle`, `PY3/english.pickle`, `README`); other languages removed |
| `data/nltk_data/tokenizers/punkt_tab/english/` | downloaded from NLTK at vendoring time | pruned to English-only; other 18 languages removed |

The two import rewrites are the **only** code changes. They make
`instructions.py` and `instructions_registry.py` work as relative
imports inside this vendored package; without them, importing the
package would attempt to resolve `open_instruct.IFEvalG`, which is not
installed in this repo by design.

## License

Apache 2.0. The IFEval code in this package is © 2024 The Google
Research Authors. The IFBench-Train constraint extensions and the
IFEvalG aggregation are © AllenAI. Both are licensed under the Apache
License, Version 2.0. See `LICENSE` in this directory.

## Why vendoring (not a pip dep)

`open-instruct` is a large training repo (multi-gigabyte deps in its
optional extras) and pinning it as a Python dependency would bring in
significantly more than the small verifier subset we need. Vendoring
the three IFEvalG files plus the punkt tokenizer keeps the verifier
path deterministic, offline, and reproducible across the matrix runs.

## Modification log

| date | action | notes |
|---|---|---|
| 2026-06-02 | initial vendor at pinned commit `ebca9f7d4921042d7d8a65369e513d6566f0f3ba` | three files copied; two mechanical relative-import rewrites; nltk punkt + punkt_tab pruned to English-only |

Future modifications must be documented in this table.
