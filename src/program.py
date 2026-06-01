"""Three-module multi-hop DSPy program over HotpotQA distractor paragraphs.

§9 fallback: the hosted ColBERTv2 endpoint is chronically unreachable, so the
program now consumes the 10 distractor-config paragraphs (2 gold + 8 distractor)
that HotpotQA provides per example, instead of running retrieval. The
create_query_hop2 module is dropped; the experimental design (band-sampled
reflection minibatches over an N-iteration loop) is unchanged for the remaining
three modules.

The three predictors are named to match the candidate dict keys gepa's
dspy_adapter expects (it iterates `program.named_predictors()` and updates each
predictor's signature via `signature.with_instructions(candidate[name])`, see
`gepa/adapters/dspy_adapter/dspy_adapter.py:198-200`).

Component names: summarize1, summarize2, final_answer.
Seed instruction texts live in `prompts/seeds/<name>.md` and are loaded into
the predictor signatures at program-init time. `prompts/seeds/create_query_hop2.md`
is left in place but is unused in distractor mode.

Per-instance flow:
  hop1_paragraphs, hop2_paragraphs = split(context_paragraphs, by first/last half)
  summary_1 = summarize1(question, hop1_paragraphs as wiki17-formatted blocks)
  summary_2 = summarize2(question, summary_1, hop2_paragraphs as blocks)
  answer    = final_answer(question, summary_1, summary_2)

Gold-vs-distractor annotation: forward receives `supporting_facts` as a kwarg
(declared input on the Example) and uses it only to compute per-hop gold-title
lists in the output Prediction. No predictor sees supporting_facts, so the
reflection LM never receives gold labels via the dspy trace.
"""

from __future__ import annotations

from pathlib import Path

import dspy


class Summarize1(dspy.Signature):
    question: str = dspy.InputField()
    passages: str = dspy.InputField(desc="Paragraphs from the first half of the provided context.")
    summary: str = dspy.OutputField()


class Summarize2(dspy.Signature):
    question: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Summary from the first pass over the context.")
    passages: str = dspy.InputField(desc="Paragraphs from the second half of the provided context.")
    summary: str = dspy.OutputField()


class FinalAnswer(dspy.Signature):
    question: str = dspy.InputField()
    summary_1: str = dspy.InputField()
    summary_2: str = dspy.InputField()
    answer: str = dspy.OutputField()


COMPONENT_NAMES: tuple[str, str, str] = ("summarize1", "summarize2", "final_answer")


def load_seed_instructions(prompts_dir: Path | str) -> dict[str, str]:
    prompts_dir = Path(prompts_dir)
    instructions: dict[str, str] = {}
    for name in COMPONENT_NAMES:
        path = prompts_dir / f"{name}.md"
        instructions[name] = path.read_text().strip()
    return instructions


def _format_block(title: str, body: str) -> str:
    """Render a paragraph in the wiki17 ColBERTv2 format the original prompts
    were designed against: leading title in quotes, then ' | ', then the body."""
    return f'"{title}" | {body}'


class MultiHopQA(dspy.Module):
    def __init__(
        self,
        seed_instructions: dict[str, str] | None = None,
    ):
        super().__init__()
        self.summarize1 = dspy.Predict(Summarize1)
        self.summarize2 = dspy.Predict(Summarize2)
        self.final_answer = dspy.Predict(FinalAnswer)
        if seed_instructions is not None:
            self.apply_instructions(seed_instructions)

    def apply_instructions(self, instructions: dict[str, str]) -> None:
        for name, pred in self.named_predictors():
            if name in instructions:
                pred.signature = pred.signature.with_instructions(instructions[name])

    def get_module_instruction(self, name: str) -> str:
        for pred_name, pred in self.named_predictors():
            if pred_name == name:
                return pred.signature.instructions
        raise KeyError(f"No predictor named {name!r}; have {[n for n, _ in self.named_predictors()]}")

    def forward(
        self,
        question: str,
        context_titles: list[str],
        context_paragraphs: list[str],
        supporting_facts: dict,
    ) -> dspy.Prediction:
        assert len(context_titles) == len(context_paragraphs), (
            f"context_titles ({len(context_titles)}) != context_paragraphs "
            f"({len(context_paragraphs)})"
        )
        n = len(context_paragraphs)
        mid = n // 2
        hop1_titles = list(context_titles[:mid])
        hop2_titles = list(context_titles[mid:])
        hop1_paragraphs = list(context_paragraphs[:mid])
        hop2_paragraphs = list(context_paragraphs[mid:])

        hop1_blocks = [_format_block(t, b) for t, b in zip(hop1_titles, hop1_paragraphs)]
        hop2_blocks = [_format_block(t, b) for t, b in zip(hop2_titles, hop2_paragraphs)]

        summary_1 = self.summarize1(
            question=question, passages="\n\n".join(hop1_blocks)
        ).summary
        summary_2 = self.summarize2(
            question=question,
            summary_1=summary_1,
            passages="\n\n".join(hop2_blocks),
        ).summary
        answer = self.final_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        gold_set = set(supporting_facts.get("title", []))
        hop1_gold_titles = [t for t in hop1_titles if t in gold_set]
        hop2_gold_titles = [t for t in hop2_titles if t in gold_set]

        return dspy.Prediction(
            answer=answer,
            summary_1=summary_1,
            summary_2=summary_2,
            hop1_titles=hop1_titles,
            hop2_titles=hop2_titles,
            hop1_paragraphs=hop1_paragraphs,
            hop2_paragraphs=hop2_paragraphs,
            hop1_gold_titles=hop1_gold_titles,
            hop2_gold_titles=hop2_gold_titles,
        )


def build_program(seeds_dir: Path | str | None = None) -> MultiHopQA:
    if seeds_dir is None:
        seeds_dir = Path(__file__).resolve().parents[1] / "prompts" / "seeds"
    return MultiHopQA(seed_instructions=load_seed_instructions(seeds_dir))
