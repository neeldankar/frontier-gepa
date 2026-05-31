"""Four-module multi-hop DSPy program (paper Appendix L: HoVer-MultiHop with
the last hop replaced by a final answerer).

The four predictors are named to match the candidate dict keys gepa's
dspy_adapter expects (it iterates `program.named_predictors()` and updates
each predictor's signature via `signature.with_instructions(candidate[name])`,
see `gepa/adapters/dspy_adapter/dspy_adapter.py:198-200`).

Component names: summarize1, create_query_hop2, summarize2, final_answer.
Seed instruction texts live in `prompts/seeds/<name>.md` and are loaded into
the predictor signatures at program-init time.

Per-instance flow:
  hop1_passages = retrieve(question)
  summary_1     = summarize1(question, hop1_passages)
  hop2_query    = create_query_hop2(question, summary_1)
  hop2_passages = retrieve(hop2_query)
  summary_2     = summarize2(question, summary_1, hop2_passages)
  answer        = final_answer(question, summary_1, summary_2)
"""

from __future__ import annotations

from pathlib import Path

import dspy

from src.retrieval import parse_titles


class Summarize1(dspy.Signature):
    question: str = dspy.InputField()
    passages: str = dspy.InputField(desc="Passages retrieved with the original question.")
    summary: str = dspy.OutputField()


class CreateQueryHop2(dspy.Signature):
    question: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Summary of information gathered so far.")
    query: str = dspy.OutputField()


class Summarize2(dspy.Signature):
    question: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Summary from the first hop.")
    passages: str = dspy.InputField(desc="Additional passages retrieved in the second hop.")
    summary: str = dspy.OutputField()


class FinalAnswer(dspy.Signature):
    question: str = dspy.InputField()
    summary_1: str = dspy.InputField()
    summary_2: str = dspy.InputField()
    answer: str = dspy.OutputField()


COMPONENT_NAMES: tuple[str, str, str, str] = (
    "summarize1",
    "create_query_hop2",
    "summarize2",
    "final_answer",
)


def load_seed_instructions(prompts_dir: Path | str) -> dict[str, str]:
    prompts_dir = Path(prompts_dir)
    instructions: dict[str, str] = {}
    for name in COMPONENT_NAMES:
        path = prompts_dir / f"{name}.md"
        instructions[name] = path.read_text().strip()
    return instructions


class MultiHopQA(dspy.Module):
    def __init__(
        self,
        k_retrieve: int = 5,
        seed_instructions: dict[str, str] | None = None,
    ):
        super().__init__()
        self.k_retrieve = k_retrieve
        self.retrieve = dspy.Retrieve(k=k_retrieve)
        self.summarize1 = dspy.Predict(Summarize1)
        self.create_query_hop2 = dspy.Predict(CreateQueryHop2)
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

    def forward(self, question: str) -> dspy.Prediction:
        hop1_passages = self.retrieve(question).passages
        summary_1 = self.summarize1(
            question=question, passages="\n\n".join(hop1_passages)
        ).summary

        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_passages = self.retrieve(hop2_query).passages
        summary_2 = self.summarize2(
            question=question,
            summary_1=summary_1,
            passages="\n\n".join(hop2_passages),
        ).summary

        answer = self.final_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(
            answer=answer,
            hop1_passages=hop1_passages,
            hop2_passages=hop2_passages,
            hop1_titles=parse_titles(hop1_passages),
            hop2_titles=parse_titles(hop2_passages),
            hop2_query=hop2_query,
            summary_1=summary_1,
            summary_2=summary_2,
        )


def build_program(
    k_retrieve: int = 5,
    seeds_dir: Path | str | None = None,
) -> MultiHopQA:
    if seeds_dir is None:
        seeds_dir = Path(__file__).resolve().parents[1] / "prompts" / "seeds"
    return MultiHopQA(
        k_retrieve=k_retrieve,
        seed_instructions=load_seed_instructions(seeds_dir),
    )
