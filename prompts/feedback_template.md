# Feedback string template for mu_f (retrieval HotpotQA)

`mu_f(example, trace) -> (score, feedback_string)`. `score` is answer F1. This file
fixes the wording of `feedback_string`. Detection (computing the slots and selecting
the diagnosis branch) is yours; the wording here is fixed.

Design rule: the string states specific facts about this one instance. It does not
give generic advice. Turning facts into instruction edits is the reflection
meta-prompt's job. Keep it factual.

This is a single trajectory-level string. It is shown for whichever module
round-robin selects this iteration, so it covers the whole trajectory; the module's
own inputs and outputs supply the local context.

## Emitted string

```
F1 {f1:.2f}. Predicted answer: "{pred_answer}". Gold answer: "{gold_answer}".
Gold supporting documents: {gold_titles}.
Hop 1 (original question as the query) retrieved: {hop1_titles}. Gold documents still missing after hop 1: {missing_after_hop1}.
Hop 2 query: "{hop2_query}". Retrieved: {hop2_titles}. Gold documents still missing after hop 2: {missing_after_hop2}.
Gold answer string present in the gathered context: {answer_in_context}.
Diagnosis: {diagnosis}
```

## Slots (your computation)

- `f1`: answer F1 in [0,1].
- `pred_answer`, `gold_answer`: the predicted and gold answer strings.
- `gold_titles`: titles of the gold supporting documents.
- `hop1_titles`, `hop2_titles`: titles retrieved at each hop. Cap each to the top 3 to keep the string legible.
- `hop2_query`: the query string create_query_hop2 produced.
- `missing_after_hop1`, `missing_after_hop2`: gold titles not yet retrieved after each hop. Render "none" if all found.
- `answer_in_context`: "yes" or "no", from the substring check of `gold_answer` against the retrieved passages and the summaries.
- `diagnosis`: one sentence, selected per the branches below.

## Diagnosis branches (you select, wording fixed)

1. `missing_after_hop2` is non-empty:
   `A required document ({missing_after_hop2}) was never retrieved. The hop-2 query did not surface it, so the information needed to answer was never gathered. This is a retrieval and query-formation failure.`

2. all gold docs retrieved, span answer, `answer_in_context` == yes, F1 < 1.0:
   `All gold documents were retrieved and the gold answer is present in the gathered context, but the final answer is wrong. The answer was available and not used. The failure is in the final answering step, not in retrieval or summarization.`

3. all gold docs retrieved, span answer, `answer_in_context` == no, F1 < 1.0:
   `All gold documents were retrieved, but the gold answer is not in the gathered context. The supporting fact was lost during summarization, or the answer must be composed from facts that are present. The failure is downstream of retrieval.`

4. all gold docs retrieved, yes/no or comparison answer, F1 < 1.0:
   `All gold documents were retrieved, so this yes/no or comparison question failed in reasoning over the retrieved facts, not in retrieval.`

5. F1 == 1.0:
   `All gold documents were retrieved and the answer matches. Nothing to correct.`

## Notes

- Branches 2 vs 3 use `answer_in_context`, which is a span-answer signal. It is not
  reliable for yes/no or comparison answers, which is why those route to branch 4
  instead. Detect the answer type from `gold_answer in {"yes","no"}` or the HotpotQA
  `type` field, and record in the logs which branch fired.
- Optional bridge-entity field (Chunk 3): if you can detect whether `hop2_query`
  references the bridge entity, insert a line `Hop-2 query references the bridge entity: {yes/no}.`
  before the Diagnosis line. If you cannot extract the bridge entity reliably, omit
  the line. Branch 1 already carries the "wrong target" signal via the query versus
  the missing title, so this field is a bonus, not a dependency.
- Keep the whole string to roughly six short lines so three concatenated examples
  stay readable to the reflection LM.
