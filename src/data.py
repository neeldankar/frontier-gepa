"""HotpotQA loading and disjoint split assignment.

Splits are sliced off a single deterministic shuffle of the HotpotQA dev set
keyed by `seed`. Same seed -> same ids in each split, across runs.

Returned tuple is (d_feedback, accept_batch, d_pareto, test) in that order.
Sizes come from `config['splits']` (loaded from config/experiment.yaml by
default). Each example is a dspy.Example carrying:
  - id: the stable HotpotQA id
  - question, answer, type, level
  - supporting_facts: {title: list[str], sent_id: list[int]}, the gold
    supporting-fact payload that the Chunk-3 feedback function consumes
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


def _row_to_example(row: dict[str, Any]) -> dspy.Example:
    sf = row["supporting_facts"]
    supporting_facts = {
        "title": list(sf["title"]),
        "sent_id": list(sf["sent_id"]),
    }
    return dspy.Example(
        question=row["question"],
        answer=row["answer"],
        id=row["id"],
        type=row["type"],
        level=row["level"],
        supporting_facts=supporting_facts,
    ).with_inputs("question")


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
