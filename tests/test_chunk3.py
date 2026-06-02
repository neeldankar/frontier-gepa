"""Chunk 3 tests: mu_f slots, branch selection, and rendered feedback strings.

All tests are offline (no API calls). Each case constructs a synthetic
(dspy.Example, dspy.Prediction) pair that targets one of the five diagnosis
branches:

  - exact:         F1 == 1.0
  - partial:       0 < F1 < 1, span-answer
  - answering:     F1 == 0 and gold answer is a substring of summary_1+summary_2
  - summarization: F1 == 0 and gold answer is NOT in the summaries
  - neutral:       yes/no or comparison answer with F1 < 1

A console-printing test (`test_print_three_rendered_strings`) emits three
representative rendered strings for human review.
"""

from __future__ import annotations

import dspy
import pytest

from src.feedback import (
    compute_feedback,
    f1_score,
    is_yn_or_comparison,
    metric_fn,
    missing_in_summaries,
    mu_f,
    normalize_answer,
    select_branch,
)


# ---- Example / Prediction builders --------------------------------------


def _example(
    *,
    question: str = "Who won?",
    answer: str = "Alice",
    type_: str = "bridge",
    supporting_titles: list[str] | None = None,
) -> dspy.Example:
    return dspy.Example(
        question=question,
        answer=answer,
        id="synthetic",
        type=type_,
        level="medium",
        supporting_facts={"title": supporting_titles or ["Alice", "Bob"], "sent_id": [0, 0]},
    )


def _prediction(
    *,
    answer: str,
    summary_1: str = "",
    summary_2: str = "",
    hop1_gold_titles: list[str] | None = None,
    hop2_gold_titles: list[str] | None = None,
) -> dspy.Prediction:
    return dspy.Prediction(
        answer=answer,
        summary_1=summary_1,
        summary_2=summary_2,
        hop1_titles=[],
        hop2_titles=[],
        hop1_paragraphs=[],
        hop2_paragraphs=[],
        hop1_gold_titles=hop1_gold_titles or [],
        hop2_gold_titles=hop2_gold_titles or [],
    )


# ---- Low-level helpers --------------------------------------------------


def test_normalize_answer_strips_articles_and_punct():
    assert normalize_answer("The Great Gatsby!") == "great gatsby"
    assert normalize_answer("an APPLE.") == "apple"
    assert normalize_answer("  October 1922  ") == "october 1922"


def test_f1_exact():
    out = f1_score("October 1922", "October 1922")
    assert out["f1"] == 1.0
    assert out["overlap"] == ["1922", "october"]
    assert out["missing"] == []
    assert out["extra"] == []


def test_f1_partial_subset_of_gold():
    # pred "1922" vs gold "October 1922" -> 1 overlap, 1 missing, 0 extra
    out = f1_score("1922", "October 1922")
    assert out["overlap"] == ["1922"]
    assert out["missing"] == ["october"]
    assert out["extra"] == []
    # precision = 1/1 = 1, recall = 1/2 = 0.5, F1 = 0.667
    assert 0.66 < out["f1"] < 0.67


def test_f1_zero_no_overlap():
    out = f1_score("Lenin", "October 1922")
    assert out["f1"] == 0.0
    assert out["overlap"] == []
    assert out["missing"] == ["1922", "october"]
    assert out["extra"] == ["lenin"]


def test_yn_caveat_detection():
    assert is_yn_or_comparison("yes", "bridge")
    assert is_yn_or_comparison("No", "bridge")
    assert is_yn_or_comparison("anything", "comparison")
    assert not is_yn_or_comparison("October 1922", "bridge")


def test_branch_selector_table():
    # Signature: (f1, answer_in_summaries, missing_is_empty, missing_in_summaries, gold, type).
    assert select_branch(1.0, False, False, True, "anything", "bridge") == "exact"
    # partial verbose: missing empty wins over the missing-in-summaries split.
    assert select_branch(0.5, True, True, True, "October 1922", "bridge") == "partial_verbose"
    assert select_branch(0.5, False, True, False, "October 1922", "bridge") == "partial_verbose"
    # partial splits on missing_in_summaries when missing is non-empty.
    assert select_branch(0.5, False, False, True, "October 1922", "bridge") == "partial_answering"
    assert select_branch(0.5, False, False, False, "October 1922", "bridge") == "partial_summarization"
    # F1==0 splits on answer_in_summaries (missing_is_empty / flag irrelevant).
    assert select_branch(0.0, True, False, False, "October 1922", "bridge") == "answering"
    assert select_branch(0.0, False, False, False, "October 1922", "bridge") == "summarization"
    # yes/no overrides partial/answering/summarization but not exact.
    assert select_branch(1.0, True, False, True, "yes", "bridge") == "exact"
    assert select_branch(0.5, False, True, True, "yes", "bridge") == "neutral"
    assert select_branch(0.0, True, False, True, "no", "bridge") == "neutral"
    assert select_branch(0.0, False, False, False, "no", "bridge") == "neutral"
    # comparison override.
    assert select_branch(0.5, False, False, True, "Alice", "comparison") == "neutral"


def test_missing_in_summaries_helper():
    # Missing "october" present in summaries -> True.
    assert missing_in_summaries(["october"], "Ended in October 1922.", "Bolsheviks won.")
    # Missing "october" not present in summaries -> False.
    assert not missing_in_summaries(["october"], "Ended in 1922.", "Bolsheviks won.")
    # All missing tokens must be in summaries (strict, not any).
    assert missing_in_summaries(["october", "1922"], "October 1922 ended it.", "")
    assert not missing_in_summaries(["october", "1922"], "October ended it.", "")
    # Empty missing list defaults to True (nothing to localize).
    assert missing_in_summaries([], "anything", "")


# ---- End-to-end branch coverage -----------------------------------------


def test_branch_exact_renders():
    ex = _example(answer="October 1922", supporting_titles=["Socialist Revolutionary Party", "Russian Civil War"])
    pred = _prediction(answer="October 1922", summary_1="...", summary_2="...")
    out = compute_feedback(ex, pred)
    assert out["branch"] == "exact"
    assert out["score"] == 1.0
    assert "matches the gold" in out["feedback"]
    assert out["slots"]["overlap_tokens"] == ["1922", "october"]


def test_branch_partial_summarization_renders():
    """Missing token 'october' is NOT in the summaries -> partial_summarization."""
    ex = _example(answer="October 1922", supporting_titles=["Socialist Revolutionary Party", "Russian Civil War"])
    pred = _prediction(
        answer="1922",
        summary_1="The Russian Civil War ended in 1922.",
        summary_2="Kerensky was defeated by 1922.",
        hop1_gold_titles=["Socialist Revolutionary Party"],
        hop2_gold_titles=["Russian Civil War"],
    )
    out = compute_feedback(ex, pred)
    assert out["branch"] == "partial_summarization"
    assert 0 < out["score"] < 1
    assert "partially correct" in out["feedback"]
    assert "overlaps the gold on 1922" in out["feedback"]
    assert "missing from the answer: october" in out["feedback"]
    assert "extra in the answer: none" in out["feedback"]
    assert "did not survive into the summaries" in out["feedback"]
    assert "Missing token(s) present in the summaries: no." in out["feedback"]
    # Line 4 no longer mentions the gold-answer substring signal for partial.
    assert "Gold answer present in the summaries" not in out["feedback"]
    assert out["slots"]["missing_in_summaries"] is False


def test_branch_partial_answering_renders():
    """Missing token 'october' IS in the summaries -> partial_answering."""
    ex = _example(answer="October 1922", supporting_titles=["Socialist Revolutionary Party", "Russian Civil War"])
    pred = _prediction(
        answer="1922",
        summary_1="The Russian Civil War ended in October 1922.",
        summary_2="Bolsheviks consolidated control by the end of 1922.",
        hop1_gold_titles=["Socialist Revolutionary Party"],
        hop2_gold_titles=["Russian Civil War"],
    )
    out = compute_feedback(ex, pred)
    assert out["branch"] == "partial_answering"
    assert 0 < out["score"] < 1
    assert "partially correct" in out["feedback"]
    assert "missing from the answer: october" in out["feedback"]
    assert "appear in the summaries, so the failure is in the final answering step" in out["feedback"]
    assert "Missing token(s) present in the summaries: yes." in out["feedback"]
    assert out["slots"]["missing_in_summaries"] is True


def test_branch_partial_verbose_renders():
    """Prediction is a superset of gold (missing_tokens empty, extras non-empty)
    -> partial_verbose. The missing-in-summaries line is dropped (4-line string)."""
    ex = _example(answer="1922", supporting_titles=["Russian Civil War", "Soviet Union"])
    pred = _prediction(
        answer="1922 AD",
        summary_1="The civil war ended in 1922.",
        summary_2="The Soviet Union was formed.",
        hop1_gold_titles=["Russian Civil War"],
        hop2_gold_titles=["Soviet Union"],
    )
    out = compute_feedback(ex, pred)
    assert out["branch"] == "partial_verbose"
    assert 0 < out["score"] < 1
    assert "contains the full gold answer (1922)" in out["feedback"]
    assert "adds extra tokens (ad)" in out["feedback"]
    assert "more concise final answer" in out["feedback"]
    # Verbose branch drops the summary-signal line entirely.
    assert "Missing token(s) present in the summaries" not in out["feedback"]
    assert "Gold answer present in the summaries" not in out["feedback"]
    # Four-line string.
    assert out["feedback"].count("\n") == 3
    # Underlying slots: missing empty.
    assert out["slots"]["missing_tokens"] == []


def test_branch_answering_renders():
    ex = _example(answer="October 1922")
    pred = _prediction(
        answer="Lenin",
        summary_1="The civil war ended in October 1922 with Bolshevik victory.",
        summary_2="Kerensky was forced into exile.",
        hop1_gold_titles=["Russian Civil War"],
        hop2_gold_titles=["Socialist Revolutionary Party"],
    )
    out = compute_feedback(ex, pred)
    assert out["branch"] == "answering"
    assert out["score"] == 0.0
    assert "gold answer appears in the summaries" in out["feedback"]
    assert out["slots"]["answer_in_summaries"] is True


def test_branch_summarization_renders():
    ex = _example(answer="October 1922", supporting_titles=["Socialist Revolutionary Party", "Russian Civil War"])
    pred = _prediction(
        answer="Lenin",
        summary_1="Kerensky led a moderate-socialist faction during the revolution.",
        summary_2="The Bolsheviks took power after the October Revolution and a civil war followed.",
        hop1_gold_titles=["Socialist Revolutionary Party"],
        hop2_gold_titles=["Russian Civil War"],
    )
    out = compute_feedback(ex, pred)
    assert out["branch"] == "summarization"
    assert out["score"] == 0.0
    assert "did not survive into the summaries" in out["feedback"]
    assert "summarize-1 bucket: Socialist Revolutionary Party" in out["feedback"]
    assert "summarize-2 bucket: Russian Civil War" in out["feedback"]
    assert out["slots"]["answer_in_summaries"] is False


def test_branch_neutral_yes_no_renders():
    ex = _example(answer="yes", type_="bridge")
    pred = _prediction(
        answer="no",
        summary_1="The two films share a director.",
        summary_2="One was released first.",
    )
    out = compute_feedback(ex, pred)
    assert out["branch"] == "neutral"
    assert "yes/no or comparison question" in out["feedback"]
    # Neutral branch drops the summary-signal line entirely.
    assert "Gold answer present in the summaries" not in out["feedback"]
    assert "Missing token(s) present in the summaries" not in out["feedback"]
    # The neutral string is four lines, not five.
    assert out["feedback"].count("\n") == 3


def test_branch_neutral_comparison_renders():
    ex = _example(answer="Alice", type_="comparison", supporting_titles=["Alice", "Bob"])
    pred = _prediction(
        answer="Bob",
        summary_1="Alice was born in 1980.",
        summary_2="Bob was born in 1985.",
        hop1_gold_titles=["Alice"],
        hop2_gold_titles=["Bob"],
    )
    out = compute_feedback(ex, pred)
    assert out["branch"] == "neutral"
    assert "yes/no or comparison" in out["feedback"]
    assert "Gold answer present in the summaries" not in out["feedback"]
    assert "Missing token(s) present in the summaries" not in out["feedback"]


# ---- Spec-shape entry / metric ------------------------------------------


def test_mu_f_returns_score_and_string_tuple():
    ex = _example(answer="October 1922")
    pred = _prediction(answer="October 1922", summary_1="", summary_2="")
    score, feedback = mu_f(ex, pred)
    assert score == 1.0
    assert isinstance(feedback, str)
    assert feedback.startswith("F1 1.00.")


def test_metric_fn_returns_f1_only():
    ex = _example(answer="October 1922")
    pred = _prediction(answer="1922", summary_1="", summary_2="")
    assert 0.6 < metric_fn(ex, pred) < 0.7


# ---- Rendered strings printed for human review --------------------------


@pytest.mark.parametrize(
    "label,builder",
    [
        (
            "partial_summarization (missing token NOT in summaries)",
            lambda: (
                _example(
                    answer="October 1922",
                    supporting_titles=["Socialist Revolutionary Party", "Russian Civil War"],
                ),
                _prediction(
                    answer="1922",
                    summary_1="The Russian Civil War ended in 1922.",
                    summary_2="Kerensky was defeated by 1922.",
                    hop1_gold_titles=["Socialist Revolutionary Party"],
                    hop2_gold_titles=["Russian Civil War"],
                ),
            ),
        ),
        (
            "partial_answering (missing token IS in summaries)",
            lambda: (
                _example(
                    answer="October 1922",
                    supporting_titles=["Socialist Revolutionary Party", "Russian Civil War"],
                ),
                _prediction(
                    answer="1922",
                    summary_1="The Russian Civil War ended in October 1922.",
                    summary_2="Bolsheviks consolidated control by the end of 1922.",
                    hop1_gold_titles=["Socialist Revolutionary Party"],
                    hop2_gold_titles=["Russian Civil War"],
                ),
            ),
        ),
        (
            "partial_verbose (prediction is a superset of gold)",
            lambda: (
                _example(answer="1922", supporting_titles=["Russian Civil War", "Soviet Union"]),
                _prediction(
                    answer="1922 AD",
                    summary_1="The civil war ended in 1922.",
                    summary_2="The Soviet Union was formed.",
                    hop1_gold_titles=["Russian Civil War"],
                    hop2_gold_titles=["Soviet Union"],
                ),
            ),
        ),
        (
            "neutral (yes/no with wrong answer)",
            lambda: (
                _example(answer="yes", type_="bridge"),
                _prediction(
                    answer="no",
                    summary_1="The two films share a director.",
                    summary_2="One was released first.",
                ),
            ),
        ),
    ],
)
def test_print_three_rendered_strings(label, builder, capsys):
    ex, pred = builder()
    out = compute_feedback(ex, pred)
    print()
    print(f"--- branch={out['branch']!r} | {label} ---")
    print(out["feedback"])
    captured = capsys.readouterr().out
    # cheap sanity check the print succeeded
    assert "Diagnosis:" in captured
