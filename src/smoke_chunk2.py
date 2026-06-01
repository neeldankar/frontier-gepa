"""Chunk 2 smoke test, §9 distractor substrate.

Structural part (no LM):
  - splits load with the right sizes,
  - ids across the four splits are disjoint,
  - two calls with the same seed return identical id lists per split,
  - program initializes, named_predictors() lists the three expected names
    (summarize1, summarize2, final_answer),
  - each predictor's signature carries its seed instruction text.

Rollout part (only if TASK_MODEL is set in the env):
  - configure the task LM,
  - run the program on one example from d_feedback (no retrieval call;
    the example carries the 10 distractor paragraphs directly),
  - print predicted answer, summaries, per-hop titles, and gold-in-context info.

Run as: .venv/bin/python -m src.smoke_chunk2
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import SPLIT_ORDER, load_splits  # noqa: E402
from src.program import COMPONENT_NAMES, build_program, load_seed_instructions  # noqa: E402


def _load_config() -> dict:
    return yaml.safe_load((REPO / "config" / "experiment.yaml").read_text())


def structural_checks(config: dict):
    splits = load_splits(config=config)
    sizes = {name: len(s) for name, s in zip(SPLIT_ORDER, splits)}
    expected = {k: config["splits"][k] for k in SPLIT_ORDER}
    assert sizes == expected, f"split size mismatch: got {sizes}, want {expected}"
    print(f"splits sizes OK: {sizes}")

    seen: set[str] = set()
    for name, items in zip(SPLIT_ORDER, splits):
        for ex in items:
            assert ex["id"] not in seen, f"overlap in {name}: {ex['id']}"
            seen.add(ex["id"])
    assert len(seen) == sum(sizes.values())
    print(f"disjoint OK: {len(seen)} unique HotpotQA ids across {sum(sizes.values())} examples")

    splits2 = load_splits(config=config)
    for name, a, b in zip(SPLIT_ORDER, splits, splits2):
        ids_a = [ex["id"] for ex in a]
        ids_b = [ex["id"] for ex in b]
        assert ids_a == ids_b, f"reproducibility broken in {name}"
    print("reproducibility OK (same seed -> identical id sequence per split)")

    seeds = load_seed_instructions(REPO / "prompts" / "seeds")
    assert set(seeds.keys()) == set(COMPONENT_NAMES), f"seed keys: {set(seeds.keys())}"
    program = build_program()
    pred_names = [name for name, _ in program.named_predictors()]
    assert set(pred_names) == set(COMPONENT_NAMES), f"named_predictors mismatch: {pred_names}"
    print(f"named_predictors(): {pred_names}")
    for name in COMPONENT_NAMES:
        loaded = program.get_module_instruction(name)
        assert loaded == seeds[name], (
            f"seed instruction not wired into {name}: got {loaded[:80]!r}"
        )
    print("seed instructions wired into all three signatures")
    return splits


def lm_rollout(config: dict, splits) -> None:
    import dspy

    task_model = os.getenv("TASK_MODEL")
    api_key = os.getenv(config["task_model"]["api_key_env"])
    base_url = os.getenv(config["task_model"]["base_url_env"]) or None

    lm_kwargs: dict = dict(
        model=task_model,
        temperature=float(config["task_model"]["temperature"]),
        top_p=float(config["task_model"]["top_p"]),
        max_tokens=int(config["task_model"]["max_tokens"]),
    )
    if api_key:
        lm_kwargs["api_key"] = api_key
    if base_url:
        lm_kwargs["api_base"] = base_url
    # top_k is omitted here; not a standard chat-completions kwarg. The
    # Chunk-4/6 runner will pass it via litellm extra_body when needed.

    lm = dspy.LM(**lm_kwargs)
    dspy.settings.configure(lm=lm)

    program = build_program()
    d_feedback = splits[0]
    example = d_feedback[0]
    print()
    print(f"=== Rollout on {example['id']} ===")
    print(f"  Q:      {example['question']}")
    print(f"  Gold A: {example['answer']}")
    print(f"  Gold supporting titles: {example['supporting_facts']['title']}")
    print(f"  Context (10 titles): {example['context_titles']}")

    out = program(**dict(example.inputs()))
    print()
    print(f"  Predicted A: {out.answer}")
    print(f"  summary_1: {out.summary_1!r}")
    print(f"  summary_2: {out.summary_2!r}")
    print(f"  Hop1 titles: {out.hop1_titles}")
    print(f"  Hop2 titles: {out.hop2_titles}")
    print(f"  Hop1 gold-titles: {out.hop1_gold_titles}")
    print(f"  Hop2 gold-titles: {out.hop2_gold_titles}")


def main() -> int:
    load_dotenv(REPO / ".env")
    config = _load_config()

    print("--- Structural checks ---")
    splits = structural_checks(config)

    if os.getenv("TASK_MODEL"):
        print()
        print("--- LM rollout (TASK_MODEL is set) ---")
        lm_rollout(config, splits)
    else:
        print()
        print("TASK_MODEL not set; skipping LM rollout.")
        print("To run the rollout: copy .env.example to .env, fill in TASK_MODEL")
        print("and its API key, then rerun: .venv/bin/python -m src.smoke_chunk2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
