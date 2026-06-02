"""Token-level cost accounting from dspy.LM histories.

Reads the `.history` list on a `dspy.LM` (and an optional reflection LM),
sums prompt and completion tokens, and multiplies by per-1M-token rates.
Cache-hit entries (where the LM returned a stored completion without an
API call) typically have `cost = 0` and zero tokens in their usage block,
so they correctly contribute $0 to the running total.

The per-1M rates are pinned as constants here. They are estimates as of
June 2026 and should be verified against the provider's current pricing
page before trusting absolute dollar figures. RELATIVE costs across cells
are reliable independent of the constants (only the conversion factor
changes).

Together AI Qwen2.5-7B-Instruct-Turbo: $0.20 / 1M blended (in == out).
OpenAI gpt-4.1-mini: ~$0.40 / 1M input, $1.60 / 1M output (rough
estimate; this rate matters less since reflection calls are infrequent
compared to task-LM calls -- one reflection call per N=44 iteration vs.
~50 task-LM calls per iteration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TASK_LM_USD_PER_M_IN = 0.20
TASK_LM_USD_PER_M_OUT = 0.20
REFL_LM_USD_PER_M_IN = 0.40
REFL_LM_USD_PER_M_OUT = 1.60


@dataclass
class CostReport:
    task_tokens_in: int = 0
    task_tokens_out: int = 0
    refl_tokens_in: int = 0
    refl_tokens_out: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "CostReport") -> "CostReport":
        return CostReport(
            task_tokens_in=self.task_tokens_in + other.task_tokens_in,
            task_tokens_out=self.task_tokens_out + other.task_tokens_out,
            refl_tokens_in=self.refl_tokens_in + other.refl_tokens_in,
            refl_tokens_out=self.refl_tokens_out + other.refl_tokens_out,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_tokens_in": self.task_tokens_in,
            "task_tokens_out": self.task_tokens_out,
            "refl_tokens_in": self.refl_tokens_in,
            "refl_tokens_out": self.refl_tokens_out,
            "cost_usd": round(self.cost_usd, 6),
        }


def _usage(entry: Any) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from a history entry.

    dspy.LM entries are dicts with a nested 'response' field; tokens live
    at response.usage.{prompt_tokens, completion_tokens}. Cache hits often
    skip the usage block entirely, so we return 0 in that case.
    """
    if not isinstance(entry, dict):
        return 0, 0

    # Try common shapes:
    for path in (("response", "usage"), ("usage",), ("response", "model_dump", "usage")):
        cur: Any = entry
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, dict):
            pt = cur.get("prompt_tokens", cur.get("input_tokens", 0)) or 0
            ct = cur.get("completion_tokens", cur.get("output_tokens", 0)) or 0
            return int(pt), int(ct)

    # Fall back: try the OpenAI/litellm chat-completion shape inside response.
    resp = entry.get("response")
    if isinstance(resp, dict):
        usage = resp.get("usage")
        if isinstance(usage, dict):
            return int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)

    return 0, 0


def _tokens(lm: Any, start: int = 0) -> tuple[int, int]:
    if lm is None:
        return 0, 0
    history = getattr(lm, "history", None)
    if not history:
        return 0, 0
    pt_total = 0
    ct_total = 0
    for entry in history[start:]:
        pt, ct = _usage(entry)
        pt_total += pt
        ct_total += ct
    return pt_total, ct_total


def cost_from_lms(
    task_lm: Any,
    refl_lm: Any,
    task_baseline: int = 0,
    refl_baseline: int = 0,
) -> CostReport:
    """Compute a CostReport from the entries in the two LMs' histories
    starting at the given baselines."""
    task_in, task_out = _tokens(task_lm, task_baseline)
    refl_in, refl_out = _tokens(refl_lm, refl_baseline)
    cost = (
        task_in * TASK_LM_USD_PER_M_IN / 1_000_000
        + task_out * TASK_LM_USD_PER_M_OUT / 1_000_000
        + refl_in * REFL_LM_USD_PER_M_IN / 1_000_000
        + refl_out * REFL_LM_USD_PER_M_OUT / 1_000_000
    )
    return CostReport(
        task_tokens_in=task_in,
        task_tokens_out=task_out,
        refl_tokens_in=refl_in,
        refl_tokens_out=refl_out,
        cost_usd=cost,
    )
