"""HotpotQA loading and disjoint split assignment, distractor substrate.

The §9 fallback path: the hosted ColBERTv2 endpoint is chronically unreachable,
so we use the `distractor` config of HotpotQA, which provides 10 paragraphs per
example (the 2 gold supporting paragraphs plus 8 distractors). The program in
`src/program.py` summarizes over those paragraphs directly, no retrieval call.

Splits are sliced off a single deterministic shuffle of the HotpotQA dev set
keyed by `seed`. Same seed -> same ids in each split, across runs.

Returned tuple is (d_feedback, accept_batch, d_pareto, test) in that order.
Sizes come from `config['splits']` (loaded from config/experiment.yaml by
default). Each example is a dspy.Example carrying:

  Inputs (passed to MultiHopQA.forward):
    - question:          the HotpotQA question string
    - context_titles:    list[str], the 10 paragraph titles
    - context_paragraphs:list[str], the 10 paragraph texts (sentences joined)
    - supporting_facts:  {title: list[str], sent_id: list[int]}, the gold
      payload. Carried as a forward kwarg so the program can annotate which
      provided paragraphs are gold vs distractor in its output. No predictor
      sees this field, so the reflection LM does not see gold labels.

  Labels (not inputs):
    - id:    the stable HotpotQA id
    - answer: the gold answer string
    - type, level: HotpotQA metadata (bridge/comparison, easy/medium/hard)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import dspy
import yaml
from datasets import load_dataset

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO / "config" / "experiment.yaml"

SplitTuple = tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example], list[dspy.Example]]
SPLIT_ORDER = ("d_feedback", "accept_batch", "d_pareto", "test")

INPUT_FIELDS = ("question", "context_titles", "context_paragraphs", "supporting_facts")


def _row_to_example(row: dict[str, Any]) -> dspy.Example:
    sf_titles = list(row["supporting_facts"]["title"])
    sf_sent_ids = list(row["supporting_facts"]["sent_id"])
    ctx_titles = list(row["context"]["title"])
    ctx_paragraphs = [" ".join(sents) for sents in row["context"]["sentences"]]

    return dspy.Example(
        question=row["question"],
        answer=row["answer"],
        id=row["id"],
        type=row["type"],
        level=row["level"],
        supporting_facts={"title": sf_titles, "sent_id": sf_sent_ids},
        context_titles=ctx_titles,
        context_paragraphs=ctx_paragraphs,
    ).with_inputs(*INPUT_FIELDS)


def _normalize_split_name(name: str) -> str:
    return "validation" if name == "dev" else name


def _load_default_config() -> dict[str, Any]:
    return yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text())


def load_splits(
    seed: int | None = None,
    config: dict[str, Any] | None = None,
    hf_cache_dir: str | Path | None = None,
) -> SplitTuple:
    """Return (d_feedback, accept_batch, d_pareto, test) sliced from HotpotQA dev.

    If `seed` is None, falls back to config['splits']['seed_splits'].
    If `config` is None, loads config/experiment.yaml from the repo root.
    """
    if config is None:
        config = _load_default_config()
    sp = config["splits"]
    if seed is None:
        seed = int(sp.get("seed_splits", 0))
    sizes = {key: int(sp[key]) for key in SPLIT_ORDER}
    total = sum(sizes.values())

    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "hotpotqa/hotpot_qa")
    if dataset_name == "hotpotqa":
        dataset_name = "hotpotqa/hotpot_qa"
    split_name = _normalize_split_name(dataset_cfg.get("split", "validation"))

    ds = load_dataset(
        dataset_name,
        "distractor",
        split=split_name,
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
    )
    if len(ds) < total:
        raise ValueError(f"{dataset_name}/{split_name} has {len(ds)} examples; need {total}")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    chosen = indices[:total]

    cursor = 0
    split_lists: list[list[dspy.Example]] = []
    for key in SPLIT_ORDER:
        n = sizes[key]
        split_lists.append([_row_to_example(ds[i]) for i in chosen[cursor : cursor + n]])
        cursor += n

    _assert_disjoint(split_lists)
    return tuple(split_lists)  # type: ignore[return-value]


def _assert_disjoint(splits: list[list[dspy.Example]]) -> None:
    seen: dict[str, int] = {}
    for split_idx, items in enumerate(splits):
        for ex in items:
            hid = ex["id"]
            if hid in seen:
                raise AssertionError(
                    f"HotpotQA id {hid!r} appears in both "
                    f"{SPLIT_ORDER[seen[hid]]!r} and {SPLIT_ORDER[split_idx]!r}"
                )
            seen[hid] = split_idx
