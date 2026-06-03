"""mu_f for the IFBench substrate (BUILD_PLAN.md §7 Chunk 12).

`score_and_feedback(example, prediction)` scores a single carved row's
response against ALL of its constraints via the vendored official
verifiers (`src.ifbench_verifiers.INSTRUCTION_DICT`) and returns the
fraction satisfied plus a uniform reflective feedback string naming the
violated constraints.

Correctness contract (Chunk-12 revision #1):
  - Kwargs are passed VERBATIM to `build_description`. We never default
    None to `{}`, because the upstream IFEval verifiers silently
    random-generate missing args when called without their configured
    kwargs, which would corrupt the score for the 35 arg-requiring
    instruction IDs.
  - For no-arg constraints (the 19 IDs in
    `KNOWN_VERIFIER_IDS - ARG_REQUIRING_VERIFIER_IDS`), we call
    `build_description()` with no args.
  - For arg-requiring constraints, we assert the dataset's kwargs are a
    populated dict and call `build_description(**kw)`.
  - The carved-pool loader (`src/ifbench_data.py::_kwargs_complete`)
    pre-filters rows that violate this contract, so this assertion
    should never fire in practice; it is a defense-in-depth invariant.

Feedback format (Chunk-12 revision #4):
  Uniform across ALL 54 instruction IDs (no per-family customization).
  Each violation line is `instruction_id (key=value, ...)`; sorting and
  formatting are deterministic so the reflection LM sees a stable shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.ifbench_verifiers import (
    ARG_REQUIRING_VERIFIER_IDS,
    INSTRUCTION_DICT,
)


def _verify_one(instruction_id: str, kwargs_entry: Any, response: str) -> bool:
    """Run one verifier against one response. Raises on a contract
    violation (arg-requiring id with missing kwargs) rather than
    silently random-generating, per Chunk-12 revision #1."""
    verifier_cls = INSTRUCTION_DICT[instruction_id]
    verifier = verifier_cls(instruction_id)
    if instruction_id in ARG_REQUIRING_VERIFIER_IDS:
        if not isinstance(kwargs_entry, Mapping) or not any(
            v is not None for v in kwargs_entry.values()
        ):
            raise ValueError(
                f"contract violation: {instruction_id!r} is arg-requiring but "
                f"the dataset entry has kwargs={kwargs_entry!r}; calling "
                "build_description() without args would silently random-"
                "generate the missing args and corrupt the score."
            )
        # Pass the dataset's kwargs verbatim. Some args have value None
        # (placeholder slots); the verifier expects a kwargs dict where
        # the relevant args are populated and irrelevant ones may be
        # None. Pass through unchanged.
        verifier.build_description(**dict(kwargs_entry))
    else:
        # No-arg constraints: build_description() with no kwargs is
        # correct -- there is nothing to random-generate.
        verifier.build_description()
    return bool(verifier.check_following(response))


def _render_constraint(instruction_id: str, kwargs_entry: Any) -> str:
    """One uniform renderer for all 54 instruction IDs (Chunk-12
    revision #4). The instruction_id is already semantic; the kwargs
    are printed as sorted readable key=value pairs."""
    if isinstance(kwargs_entry, Mapping):
        items = [(k, v) for k, v in kwargs_entry.items() if v is not None]
        if items:
            kv = ", ".join(f"{k}={v!r}" for k, v in sorted(items, key=lambda kv: kv[0]))
            return f"{instruction_id} ({kv})"
    return instruction_id


def _render_feedback(score: float, satisfied: list[dict], violations: list[dict]) -> str:
    n_sat = len(satisfied)
    n_vio = len(violations)
    n = n_sat + n_vio
    lines: list[str] = []
    lines.append(f"Score {score:.4f} ({n_sat}/{n} constraints satisfied).")
    if satisfied:
        sat_names = [_render_constraint(s["instruction_id"], s["kwargs"]) for s in satisfied]
        lines.append(f"Satisfied ({n_sat}): {', '.join(sat_names)}.")
    else:
        lines.append("Satisfied (0): none.")
    if violations:
        lines.append(f"Violated ({n_vio}):")
        for v in violations:
            lines.append(f"  - {_render_constraint(v['instruction_id'], v['kwargs'])}")
    else:
        lines.append("Violated (0): none.")
    return "\n".join(lines)


def score_and_feedback(example: Any, prediction: Any) -> dict[str, Any]:
    """mu_f for IFBench. Returns
        {score: float in [0,1], feedback: str, violations: list, satisfied: list}.
    Mirrors the Chunk-3 `compute_feedback` shape so the Chunk-14
    DspyAdapter wiring is parallel."""
    response = str(_get(prediction, "response", "") or "")
    iids = list(_get(example, "instruction_id_list", []) or [])
    kwargs_list = list(_get(example, "kwargs_list", []) or [])
    if len(iids) != len(kwargs_list):
        raise ValueError(
            f"length mismatch: instruction_id_list={len(iids)} "
            f"vs kwargs_list={len(kwargs_list)}"
        )
    if not iids:
        return {"score": 0.0, "feedback": "no constraints", "violations": [], "satisfied": []}

    satisfied: list[dict] = []
    violations: list[dict] = []
    for iid, kw in zip(iids, kwargs_list):
        ok = _verify_one(iid, kw, response)
        rec = {"instruction_id": iid, "kwargs": dict(kw) if isinstance(kw, Mapping) else kw}
        if ok:
            satisfied.append(rec)
        else:
            violations.append(rec)

    score = len(satisfied) / len(iids)
    return {
        "score": float(score),
        "feedback": _render_feedback(score, satisfied, violations),
        "violations": violations,
        "satisfied": satisfied,
    }


def metric_fn(example: Any, prediction: Any, trace: Any = None) -> float:
    """Module-level metric for gepa's D_pareto evaluation: just the
    fraction satisfied."""
    return float(score_and_feedback(example, prediction)["score"])


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Generic accessor for dspy.Example / dspy.Prediction / dict."""
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    try:
        return obj[key]
    except (KeyError, TypeError):
        return default
