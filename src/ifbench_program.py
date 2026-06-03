"""One-module DSPy program for Experiment 2 (IFBench).

The model sees ONLY the natural-language prompt (the user message
content, which already embeds the constraints inline). It does NOT see
the structured `instruction_id_list` / `kwargs_list`; those flow to the
feedback function (`src/ifbench_feedback.py`) but never to the
predictor. This matches BUILD_PLAN §7 Chunk 12 ("a single predict that
takes the natural-language prompt... and produces a response. No
retrieval, no multi-hop").

The single optimizable component is named `answer`. GEPA will mutate
this one predictor's instruction text across iterations; the seed
prompt lives in `prompts/seeds/ifbench_answer.md` and is kept minimal
so the optimizer has room to add specificity.

Parallel to Chunk 2's `src/program.py` (which is HotpotQA-3-module
shaped). Reused at Chunk 14 matrix-launch time.
"""

from __future__ import annotations

from pathlib import Path

import dspy


COMPONENT_NAMES: tuple[str, ...] = ("answer",)


class IFBenchAnswer(dspy.Signature):
    prompt: str = dspy.InputField(
        desc="The instruction-following prompt; instructions are embedded in NL."
    )
    response: str = dspy.OutputField(
        desc="The model's response, expected to satisfy every instruction in the prompt."
    )


def load_seed_instruction(prompts_dir: Path | str | None = None) -> str:
    """Read the seed instruction text. Mirrors the Chunk-2 pattern."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parents[1] / "prompts" / "seeds"
    return Path(prompts_dir, "ifbench_answer.md").read_text().strip()


class IFBenchProgram(dspy.Module):
    """One-module program: a single predict over the prompt -> response."""

    def __init__(self, seed_instruction: str | None = None):
        super().__init__()
        self.answer = dspy.Predict(IFBenchAnswer)
        if seed_instruction is not None:
            self.apply_instructions({"answer": seed_instruction})

    def apply_instructions(self, instructions: dict[str, str]) -> None:
        for name, pred in self.named_predictors():
            if name in instructions:
                pred.signature = pred.signature.with_instructions(instructions[name])

    def get_module_instruction(self, name: str) -> str:
        for pred_name, pred in self.named_predictors():
            if pred_name == name:
                return pred.signature.instructions
        raise KeyError(
            f"No predictor named {name!r}; have "
            f"{[n for n, _ in self.named_predictors()]}"
        )

    def forward(self, prompt: str, **_: object) -> dspy.Prediction:
        # **_ swallows any other inputs the carved Example carries
        # (instruction_id_list, kwargs_list); the predictor must NOT see
        # them.
        out = self.answer(prompt=prompt).response
        return dspy.Prediction(response=out)


def build_program(prompts_dir: Path | str | None = None) -> IFBenchProgram:
    return IFBenchProgram(seed_instruction=load_seed_instruction(prompts_dir))
