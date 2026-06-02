# Feedback string template for mu_f (HotpotQA distractor substrate, 3 modules)

`mu_f(example, trace) -> (score, feedback_string)`. `score` is answer F1. This
file fixes the wording of `feedback_string`. Detection (computing the slots and
selecting the branch) is yours; the wording here is fixed.

Substrate note: this is the distractor substrate. Both gold supporting paragraphs
are always present in the provided context, so there is NO retrieval gap to report.
The program reads the 10 provided paragraphs in two buckets (summarize1 over the
first, summarize2 over the second plus summary_1), then final_answer. Feedback
therefore localizes failure to summarization vs answering, not retrieval.

Design rule: the string states specific facts about this one instance. It does not
give generic advice. Turning facts into instruction edits is the reflection
meta-prompt's job. Keep it factual.

## Emitted string

```
F1 {f1:.2f}. Predicted answer: "{pred_answer}". Gold answer: "{gold_answer}".
Gold supporting paragraphs (always provided): {gold_titles}.
In summarize-1's paragraphs: {hop1_gold_titles}. In summarize-2's paragraphs: {hop2_gold_titles}.
{summary_signal_line}
Diagnosis: {diagnosis}
```

`{summary_signal_line}` varies by branch:
- **Partial branch (0 < F1 < 1):**
  - Sub-case 2c (verbose, `missing_tokens` is empty): omitted entirely
    (the string is four lines).
  - Sub-cases 2a / 2b: `Missing token(s) present in the summaries: {missing_in_summaries}.`
- **Neutral branch (yes/no or comparison with F1 < 1):** omitted entirely
  (the string is four lines, not five).
- **All other branches (exact, answering, summarization):**
  `Gold answer present in the summaries: {answer_in_summaries}.`

## Slots (your computation)

- `f1`, `pred_answer`, `gold_answer`.
- `gold_titles`: the gold supporting-paragraph titles (2, always in context).
- `hop1_gold_titles` / `hop2_gold_titles`: which gold paragraphs fell into each
  summarize stage's bucket. Render "none" if none.
- `answer_in_summaries`: "yes" or "no", substring check of `gold_answer` against
  summary_1 concatenated with summary_2. Used by branches 1, 3, 4.
- `missing_in_summaries`: "yes" or "no", whether the missing gold tokens (gold
  tokens not in the predicted answer, under the F1 normalization) **all** appear
  in the normalized token set of summary_1 concatenated with summary_2. Used
  only by sub-cases 2a/2b. When `missing_tokens` is empty (sub-case 2c), the
  slot is vacuous and the line is dropped entirely from the emitted string.
- `overlap_tokens` / `missing_tokens` / `extra_tokens`: token diff from the F1
  computation, under the F1 normalization (lowercase, strip punctuation, drop
  articles, collapse whitespace). Render each as a comma-separated list, or
  "none" if empty.
- `diagnosis`: one sentence, selected per the branches below.

## Diagnosis branches (you select, wording fixed)

1. F1 == 1.0:
   `The answer matches the gold. Nothing to correct.`

2. 0 < F1 < 1 (partial). Sub-branches, checked in order:

   2c. `missing_tokens` is empty (prediction is a superset of gold; F1 < 1
   only because of extra tokens):
   `The answer is partially correct. It contains the full gold answer ({overlap_tokens}) but adds extra tokens ({extra_tokens}) that lower its precision. The supporting content was present, so the fix is a more concise final answer, not summarization.`

   2a. else, `missing_in_summaries` == yes:
   `The answer is partially correct. It overlaps the gold on {overlap_tokens} but does not match exactly (missing from the answer: {missing_tokens}; extra in the answer: {extra_tokens}). The missing token(s) appear in the summaries, so the failure is in the final answering step, not in summarization.`

   2b. else, `missing_in_summaries` == no:
   `The answer is partially correct. It overlaps the gold on {overlap_tokens} but does not match exactly (missing from the answer: {missing_tokens}; extra in the answer: {extra_tokens}). The missing token(s) did not survive into the summaries, so the failure is in summarization, not in the final answering step.`

3. F1 == 0, `answer_in_summaries` == yes:
   `The gold answer appears in the summaries, but the final answer did not use it. The failure is in the final answering step, not in reading the paragraphs.`

4. F1 == 0, `answer_in_summaries` == no:
   `Both gold paragraphs were provided (summarize-1 bucket: {hop1_gold_titles}; summarize-2 bucket: {hop2_gold_titles}), but the gold answer did not survive into the summaries. A summarization stage dropped the supporting fact or was pulled toward a distractor paragraph. The failure is in summarization, not in what was provided.`

## Notes

- Branches 2a/2b use token-overlap and a token-set check against the summaries;
  branches 3 and 4 use the answer-string substring check against the summaries.
  Both signal families are span-answer oriented. For yes/no and comparison
  answers (`gold_answer` in {"yes","no"} or HotpotQA `type` == "comparison"),
  do not report token overlap or any substring claim; use a neutral diagnosis:
  `The supporting paragraphs were provided, so this yes/no or comparison
  question failed in reasoning over them, not in what was available.` Omit the
  summary-signal line entirely for the neutral branch. Record which branch
  fired in the logs.
- There is intentionally no retrieval-gap branch. In distractor both golds are
  always present, so a "missing document" diagnosis would always be false. The
  whole diagnostic axis here is summarization-and-answering.
- The first-half/last-half bucket split is arbitrary (gold positions in the
  provided 10 are not meaningful), so the feedback leans on the
  summarization-vs-answering localization rather than the bucket structure. The
  bucket fields are included only to point the reflection LM at which
  summarize stage saw the gold.
- Keep the whole string to roughly four-to-five short lines so three
  concatenated examples stay readable to the reflection LM.
