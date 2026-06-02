"""mu_f for the HotpotQA distractor substrate.

`compute_feedback(example, prediction)` returns the trajectory-level feedback
string that GEPA's reflection LM sees, plus the F1 score the engine uses for
the per-instance D_pareto entry. The wording of the string is fixed by
`prompts/feedback_template.md`; this module implements the detection logic
(slot computation and branch selection).

Substrate is distractor: both gold supporting paragraphs are always in the
provided context, so the failure axis is summarization-vs-answering, not
retrieval. The first-half / last-half bucket split is arbitrary; the bucket
fields exist only to point the reflection LM at which `summarize` stage saw
each gold paragraph.

Yes/no and comparison answers route to a neutral diagnosis for the partial
and zero-F1 branches because token-overlap and substring signals are not
reliable for those types (per the template's Notes section).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping
from typing import Any

# ---- HotpotQA F1 --------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, gold: str) -> dict[str, Any]:
    """Token-level F1 in [0,1] plus the overlap / missing / extra token lists
    (multiset semantics, mirrored from the HotpotQA reference implementation).
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return {"f1": 1.0, "overlap": [], "missing": [], "extra": []}

    pred_c = Counter(pred_tokens)
    gold_c = Counter(gold_tokens)
    common = pred_c & gold_c
    num_same = sum(common.values())
    overlap = sorted(common.elements())
    missing = sorted((gold_c - pred_c).elements())
    extra = sorted((pred_c - gold_c).elements())

    if num_same == 0:
        return {"f1": 0.0, "overlap": overlap, "missing": missing, "extra": extra}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"f1": f1, "overlap": overlap, "missing": missing, "extra": extra}


# ---- Branch selection ---------------------------------------------------

YES_NO = {"yes", "no"}


def is_yn_or_comparison(gold_answer: str, example_type: str | None) -> bool:
    if normalize_answer(gold_answer) in YES_NO:
        return True
    return (example_type or "").strip().lower() == "comparison"


def missing_in_summaries(missing_tokens: list[str], summary_1: str, summary_2: str) -> bool:
    """True iff every missing gold token appears in the normalized token set of
    summary_1 concatenated with summary_2. Empty `missing_tokens` defaults to
    True (vacuous: there is nothing to localize to summarization)."""
    if not missing_tokens:
        return True
    summary_token_set = set(normalize_answer(summary_1 + " " + summary_2).split())
    return all(t in summary_token_set for t in missing_tokens)


def select_branch(
    f1: float,
    answer_in_summaries: bool,
    missing_is_empty: bool,
    missing_in_summaries_flag: bool,
    gold_answer: str,
    example_type: str | None,
) -> str:
    """Returns one of:
      'exact', 'partial_verbose', 'partial_answering', 'partial_summarization',
      'answering', 'summarization', 'neutral'.

    F1 == 1.0 always routes to 'exact' regardless of answer type (no caveat
    needed when the answer matches). Otherwise yes/no and comparison override
    branches 2-4 with 'neutral' per the template's Notes section. Branch 2
    (partial) has three sub-cases:
      2c. missing_tokens empty (prediction is a superset of gold; F1 < 1
          because of extra tokens) -> partial_verbose. Checked BEFORE the
          2a/2b split.
      2a. all missing tokens in summaries -> partial_answering.
      2b. otherwise -> partial_summarization.
    """
    if f1 >= 1.0:
        return "exact"
    if is_yn_or_comparison(gold_answer, example_type):
        return "neutral"
    if f1 == 0.0:
        return "answering" if answer_in_summaries else "summarization"
    if missing_is_empty:
        return "partial_verbose"
    return "partial_answering" if missing_in_summaries_flag else "partial_summarization"


# ---- Helpers ------------------------------------------------------------


def _render_list(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _yn(b: bool) -> str:
    return "yes" if b else "no"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """dspy.Example and dspy.Prediction support both .attr and ['key']; fall
    back to either form so tests can pass plain dicts as well."""
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    try:
        return obj[key]
    except (KeyError, TypeError):
        return default


# ---- Diagnosis wording (fixed by the template) --------------------------


def _diagnosis_exact() -> str:
    return "The answer matches the gold. Nothing to correct."


def _partial_preamble(overlap: list[str], missing: list[str], extra: list[str]) -> str:
    return (
        f"The answer is partially correct. It overlaps the gold on "
        f"{_render_list(overlap)} but does not match exactly "
        f"(missing from the answer: {_render_list(missing)}; "
        f"extra in the answer: {_render_list(extra)}). "
    )


def _diagnosis_partial_answering(overlap: list[str], missing: list[str], extra: list[str]) -> str:
    return _partial_preamble(overlap, missing, extra) + (
        "The missing token(s) appear in the summaries, so the failure is in "
        "the final answering step, not in summarization."
    )


def _diagnosis_partial_summarization(overlap: list[str], missing: list[str], extra: list[str]) -> str:
    return _partial_preamble(overlap, missing, extra) + (
        "The missing token(s) did not survive into the summaries, so the "
        "failure is in summarization, not in the final answering step."
    )


def _diagnosis_partial_verbose(overlap: list[str], extra: list[str]) -> str:
    return (
        f"The answer is partially correct. It contains the full gold answer "
        f"({_render_list(overlap)}) but adds extra tokens "
        f"({_render_list(extra)}) that lower its precision. The supporting "
        "content was present, so the fix is a more concise final answer, "
        "not summarization."
    )


def _diagnosis_answering() -> str:
    return (
        "The gold answer appears in the summaries, but the final answer did "
        "not use it. The failure is in the final answering step, not in "
        "reading the paragraphs."
    )


def _diagnosis_summarization(hop1_gold: list[str], hop2_gold: list[str]) -> str:
    return (
        f"Both gold paragraphs were provided (summarize-1 bucket: "
        f"{_render_list(hop1_gold)}; summarize-2 bucket: "
        f"{_render_list(hop2_gold)}), but the gold answer did not survive "
        "into the summaries. A summarization stage dropped the supporting "
        "fact or was pulled toward a distractor paragraph. The failure is "
        "in summarization, not in what was provided."
    )


def _diagnosis_neutral() -> str:
    return (
        "The supporting paragraphs were provided, so this yes/no or "
        "comparison question failed in reasoning over them, not in what "
        "was available."
    )


# ---- Main entry ---------------------------------------------------------


def compute_feedback(example: Any, prediction: Any) -> dict[str, Any]:
    """Compute the F1 score and the trajectory-level feedback string.

    Returns a dict with:
      - score:               float in [0,1], the HotpotQA F1
      - feedback:            the trajectory-level string (template wording)
      - branch:              the diagnosis branch that fired (for logging)
      - slots:               raw slot values (for tests/logging)
    """
    pred_answer = str(_get(prediction, "answer", "") or "")
    gold_answer = str(_get(example, "answer", "") or "")
    example_type = _get(example, "type", "") or ""

    supporting_facts = _get(example, "supporting_facts", {}) or {}
    gold_titles = list(_get(supporting_facts, "title", []) or [])

    hop1_gold_titles = list(_get(prediction, "hop1_gold_titles", []) or [])
    hop2_gold_titles = list(_get(prediction, "hop2_gold_titles", []) or [])

    summary_1 = str(_get(prediction, "summary_1", "") or "")
    summary_2 = str(_get(prediction, "summary_2", "") or "")

    f1_info = f1_score(pred_answer, gold_answer)
    f1 = float(f1_info["f1"])

    combined = (summary_1 + "\n" + summary_2).lower()
    answer_in_summaries = bool(gold_answer.strip()) and gold_answer.strip().lower() in combined
    missing_is_empty = not f1_info["missing"]
    missing_in_summaries_flag = missing_in_summaries(f1_info["missing"], summary_1, summary_2)

    branch = select_branch(
        f1,
        answer_in_summaries,
        missing_is_empty,
        missing_in_summaries_flag,
        gold_answer,
        example_type,
    )

    if branch == "exact":
        diagnosis = _diagnosis_exact()
    elif branch == "partial_verbose":
        diagnosis = _diagnosis_partial_verbose(f1_info["overlap"], f1_info["extra"])
    elif branch == "partial_answering":
        diagnosis = _diagnosis_partial_answering(
            f1_info["overlap"], f1_info["missing"], f1_info["extra"]
        )
    elif branch == "partial_summarization":
        diagnosis = _diagnosis_partial_summarization(
            f1_info["overlap"], f1_info["missing"], f1_info["extra"]
        )
    elif branch == "answering":
        diagnosis = _diagnosis_answering()
    elif branch == "summarization":
        diagnosis = _diagnosis_summarization(hop1_gold_titles, hop2_gold_titles)
    elif branch == "neutral":
        diagnosis = _diagnosis_neutral()
    else:
        raise AssertionError(f"unknown branch {branch!r}")

    # Line-4 ("summary signal") varies by branch per the 2026-06-01 spec change:
    # partial_answering/partial_summarization use the missing-tokens-in-summaries
    # flag; partial_verbose and neutral omit the line entirely; the rest use
    # the original answer-string substring.
    if branch in ("partial_answering", "partial_summarization"):
        summary_signal_line = (
            f"Missing token(s) present in the summaries: {_yn(missing_in_summaries_flag)}."
        )
    elif branch in ("partial_verbose", "neutral"):
        summary_signal_line = None
    else:
        summary_signal_line = f"Gold answer present in the summaries: {_yn(answer_in_summaries)}."

    lines = [
        f'F1 {f1:.2f}. Predicted answer: "{pred_answer}". Gold answer: "{gold_answer}".',
        f"Gold supporting paragraphs (always provided): {_render_list(gold_titles)}.",
        (
            f"In summarize-1's paragraphs: {_render_list(hop1_gold_titles)}. "
            f"In summarize-2's paragraphs: {_render_list(hop2_gold_titles)}."
        ),
    ]
    if summary_signal_line is not None:
        lines.append(summary_signal_line)
    lines.append(f"Diagnosis: {diagnosis}")
    feedback_string = "\n".join(lines)

    slots = {
        "f1": f1,
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
        "gold_titles": gold_titles,
        "hop1_gold_titles": hop1_gold_titles,
        "hop2_gold_titles": hop2_gold_titles,
        "answer_in_summaries": answer_in_summaries,
        "missing_in_summaries": missing_in_summaries_flag,
        "example_type": example_type,
        "overlap_tokens": f1_info["overlap"],
        "missing_tokens": f1_info["missing"],
        "extra_tokens": f1_info["extra"],
    }
    return {
        "score": f1,
        "feedback": feedback_string,
        "branch": branch,
        "slots": slots,
    }


def mu_f(example: Any, prediction: Any) -> tuple[float, str]:
    """Spec-shape entry: returns (score, feedback_string). Wraps compute_feedback."""
    out = compute_feedback(example, prediction)
    return float(out["score"]), str(out["feedback"])


def metric_fn(example: Any, prediction: Any, trace: Any = None) -> float:
    """Module-level metric for gepa's D_pareto evaluation: just the F1."""
    pred_answer = str(_get(prediction, "answer", "") or "")
    gold_answer = str(_get(example, "answer", "") or "")
    return float(f1_score(pred_answer, gold_answer)["f1"])
