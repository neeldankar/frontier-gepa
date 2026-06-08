"""Exp-3 curriculum runner: run one (arm, seed) cell and emit accept metrics.

Wires the curriculum schedule (src/schedule.py) into the GEPA reflection loop
while keeping the Exp-1 decoupled acceptance machinery
(src/decoupled_proposer.py): every arm reflects on its schedule-selected
minibatch and accepts on strict improvement over the cached parent on the
FROZEN shared A=20 batch (hard constraints #3, #4).

Arms (the four Exp-3 arms; all decoupled, no coupled reference here):
  - random        : native GEPA sampling (EpochShuffledBatchSampler); not binned.
  - easy_to_hard  : CurriculumBatchSampler over the schedule.
  - hard_to_easy  : "
  - static_medium : "

Concurrency: each (arm, seed) cell writes ONLY to its own paths
  - run_dir   = <out_root>/<arm>/seed<seed>/      (gepa state, per_iter.jsonl)
  - summary   = <out_root>/<arm>/seed<seed>.json
and holds no shared mutable or global state (its own rng, adapter, proposer,
log callback, and records list). Chunk 5 can run cells in parallel without
contention. This module does not implement the parallelism itself.

The engine construction mirrors src/run_gepa.run() (the Exp-1 runner) but uses
the curriculum sampler and an Exp-3 per-iteration logging callback. run_gepa is
left untouched so the Exp-1 / 1b cells still reproduce byte-for-byte.

``run_cell`` takes an already-built ``adapter`` so it can be exercised fully
offline with a stub adapter (Chunk 3 gate: a mocked dry-run with no model
calls). ``build_and_run`` builds the real HotpotQA adapter + LMs and calls
``run_cell`` (used from Chunk 4 onward).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from gepa.core.callbacks import GEPACallback
from gepa.core.engine import GEPAEngine
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.strategies.batch_sampler import BatchSampler, EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import MaxCandidateProposalsStopper

from src.bins import Bins, load_bins
from src.decoupled_proposer import DecoupledReflectiveMutationProposer
from src.schedule import ARMS as SCHEDULE_ARMS
from src.schedule import bin_for_iteration, sample_minibatch

_REPO = Path(__file__).resolve().parents[1]

# Exp-3 locked params (CLAUDE_CODE_BUILD_PROMPT_exp3_curriculum.md, "Locked
# params"). T=40 is the Exp-3 iteration budget (distinct from the Exp-1
# experiment.yaml stopping.n=44, which is NOT used here).
EXP3_ARMS: tuple[str, ...] = ("random", "easy_to_hard", "hard_to_easy", "static_medium")
BINNED_ARMS: frozenset[str] = frozenset(SCHEDULE_ARMS) - {"random"}
DEFAULT_T: int = 40
DEFAULT_B: int = 3
DEFAULT_OUT_ROOT = _REPO / "experiments" / "exp3_curriculum"


class _NoopLogger:
    def log(self, *args: Any, **kwargs: Any) -> None:
        pass


# ---- Curriculum batch sampler (binned arms only) ----


class CurriculumBatchSampler(BatchSampler):
    """A gepa BatchSampler that draws the reflection minibatch from the
    schedule-selected bin at the current iteration (``state.i``, 0-based).

    Only the three binned arms use this; the ``random`` arm uses the native
    EpochShuffledBatchSampler instead (Chunk 3 spec: schedule drives all arms
    except random).
    """

    def __init__(
        self,
        arm: str,
        T: int,
        rng: random.Random,
        bins: Bins,
        b: int = DEFAULT_B,
    ):
        if arm == "random":
            raise ValueError(
                "CurriculumBatchSampler: the 'random' arm must use native "
                "EpochShuffledBatchSampler, not the curriculum sampler."
            )
        if arm not in BINNED_ARMS:
            raise ValueError(
                f"CurriculumBatchSampler: unknown binned arm {arm!r}; expected {sorted(BINNED_ARMS)}"
            )
        self.arm = arm
        self.T = T
        self.rng = rng
        self.bins = bins
        self.b = b

    def next_minibatch_ids(self, loader, state) -> list[int]:
        # state.i is the 0-based iteration index inside the loop (engine
        # increments it at the top of each iteration before propose()).
        all_ids = list(loader.all_ids())
        return sample_minibatch(
            self.arm, state.i, self.T, self.rng, self.bins, all_ids, self.b
        )


# ---- Per-iteration accept logging ----


class CurriculumLogCallback(GEPACallback):
    """Append one record per iteration with the Exp-3 schema and accumulate
    them for the run summary. Writes to a per-cell JSONL file (no shared file)."""

    def __init__(self, arm: str, T: int, accept_batch_size: int, per_iter_path: Path | str):
        self.arm = arm
        self.T = T
        self.accept_batch_size = accept_batch_size
        self.per_iter_path = Path(per_iter_path)
        self.records: list[dict[str, Any]] = []
        # Truncate so a resumed/rerun cell starts clean for this file.
        self.per_iter_path.write_text("")

    def on_iteration_end(self, event) -> None:
        state = event["state"]
        accepted = bool(event["proposal_accepted"])
        trace = state.full_program_trace[-1] if state.full_program_trace else {}
        it = int(trace.get("i", state.i))

        minibatch_ids = trace.get("subsample_ids")
        if minibatch_ids is not None:
            minibatch_ids = [int(x) for x in minibatch_ids]

        parent_sum = trace.get("accept_batch_parent_score")
        cand_sum = trace.get("accept_batch_new_score")
        A = self.accept_batch_size
        parent_a20 = (float(parent_sum) / A) if parent_sum is not None else None
        candidate_a20 = (float(cand_sum) / A) if cand_sum is not None else None

        record = {
            "iter": it,
            "bin_sampled": bin_for_iteration(self.arm, it, self.T),
            "minibatch_ids": minibatch_ids,
            "parent_a20": parent_a20,
            "candidate_a20": candidate_a20,
            "accepted": accepted,
        }
        self.records.append(record)
        with open(self.per_iter_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ---- Core: run one cell (adapter is injected; fully offline-testable) ----


def run_cell(
    arm: str,
    seed: int,
    *,
    d_feedback: Sequence[Any],
    accept_batch: Sequence[Any],
    d_pareto: Sequence[Any],
    test: Sequence[Any],
    bins: Bins,
    adapter: Any,
    seed_candidate: dict[str, str],
    run_dir: Path | str,
    summary_path: Path | str,
    T: int = DEFAULT_T,
    b: int = DEFAULT_B,
    reflection_lm: Optional[Callable] = None,
    raise_on_exception: bool = True,
    logger: Any | None = None,
    extra_callbacks: Optional[list[GEPACallback]] = None,
    state_capture: Optional[dict] = None,
) -> dict[str, Any]:
    """Run one (arm, seed) cell with an injected adapter; write + return the
    Exp-3 run summary.

    The summary schema:
        {arm, seed, T, first_accept_iter, cumulative_accepts, final_test_f1,
         best_candidate_idx, iterations: [per-iter records]}
    Each per-iter record:
        {iter, bin_sampled, minibatch_ids, parent_a20, candidate_a20, accepted}
    """
    if arm not in EXP3_ARMS:
        raise ValueError(f"run_cell: unknown arm {arm!r}; expected {EXP3_ARMS}")
    if T <= 0:
        raise ValueError(f"run_cell: T must be positive, got {T}")
    if seed is None or seed < 0:
        raise ValueError(f"run_cell: seed must be a non-negative int, got {seed!r}")
    if arm in BINNED_ARMS and len(bins.scores) != len(d_feedback):
        raise ValueError(
            f"run_cell: bins has {len(bins.scores)} ids but d_feedback has "
            f"{len(d_feedback)}; bin ids must index D_feedback."
        )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_meta.json").write_text(
        json.dumps({"arm": arm, "seed": seed, "T": T, "b": b, "A": len(accept_batch)}, indent=2)
    )

    if logger is None:
        logger = _NoopLogger()

    # Seed everything stochastic in the engine from the run seed (matched
    # across arms): sampler, candidate selector, module rotation.
    rng = random.Random(seed)

    if arm == "random":
        batch_sampler: BatchSampler = EpochShuffledBatchSampler(minibatch_size=b, rng=rng)
    else:
        batch_sampler = CurriculumBatchSampler(arm=arm, T=T, rng=rng, bins=bins, b=b)

    candidate_selector = ParetoCandidateSelector(rng=rng)
    module_selector = RoundRobinReflectionComponentSelector()
    experiment_tracker = create_experiment_tracker(
        use_wandb=False, wandb_api_key=None, wandb_init_kwargs=None,
        use_mlflow=False, mlflow_tracking_uri=None, mlflow_experiment_name=None,
    )

    proposer = DecoupledReflectiveMutationProposer(
        logger=logger,
        trainset=list(d_feedback),
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector,
        batch_sampler=batch_sampler,
        perfect_score=1.0,
        skip_perfect_score=True,
        experiment_tracker=experiment_tracker,
        reflection_lm=reflection_lm,
        reflection_prompt_template=None,
        custom_candidate_proposer=None,
        callbacks=None,
        accept_batch=list(accept_batch),
        accept_batch_ids=list(range(len(accept_batch))),
    )

    log_callback = CurriculumLogCallback(
        arm=arm, T=T, accept_batch_size=len(accept_batch),
        per_iter_path=run_dir / "per_iter.jsonl",
    )
    callbacks: list[GEPACallback] = [log_callback]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    engine = GEPAEngine(
        adapter=adapter,
        run_dir=str(run_dir),
        valset=list(d_pareto),
        seed_candidate=seed_candidate,
        perfect_score=1.0,
        seed=seed,
        reflective_proposer=proposer,
        merge_proposer=None,  # no merge in this experiment
        frontier_type="instance",
        logger=logger,
        experiment_tracker=experiment_tracker,
        callbacks=callbacks,
        track_best_outputs=False,
        display_progress_bar=False,
        raise_on_exception=raise_on_exception,
        stop_callback=MaxCandidateProposalsStopper(max_proposals=T),
        val_evaluation_policy=FullEvaluationPolicy(),
        use_cloudpickle=False,
        evaluation_cache=None,
    )

    import time as _time

    _t_engine = _time.perf_counter()
    with experiment_tracker:
        state = engine.run()
    engine_run_seconds = _time.perf_counter() - _t_engine

    if state_capture is not None:
        state_capture["state"] = state
        state_capture["engine_run_seconds"] = engine_run_seconds

    # ---- Final held-out test F1 (single eval of the best candidate by
    #      D_pareto aggregate; GEPA Algorithm 1 line 21) ----
    val_scores = list(state.program_full_scores_val_set)
    best_idx = max(range(len(val_scores)), key=lambda k: val_scores[k]) if val_scores else 0
    best_candidate = state.program_candidates[best_idx]
    _t_test = _time.perf_counter()
    test_eval = adapter.evaluate(list(test), best_candidate, capture_traces=False)
    test_eval_seconds = _time.perf_counter() - _t_test
    final_test_f1 = (
        sum(test_eval.scores) / len(test_eval.scores) if test_eval.scores else None
    )
    if state_capture is not None:
        state_capture["test_eval_seconds"] = test_eval_seconds

    # ---- Summary ----
    records = log_callback.records
    accepted_iters = [r["iter"] for r in records if r["accepted"]]
    summary = {
        "arm": arm,
        "seed": seed,
        "T": T,
        "first_accept_iter": (min(accepted_iters) if accepted_iters else None),
        "cumulative_accepts": len(accepted_iters),
        "final_test_f1": final_test_f1,
        "best_candidate_idx": int(best_idx),
        "iterations": records,
    }
    _atomic_write_json(summary_path, summary)
    return summary


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


# ---- Real wrapper: build the HotpotQA adapter + LMs, then run_cell ----


def build_and_run(
    arm: str,
    seed: int,
    *,
    out_root: Path | str = DEFAULT_OUT_ROOT,
    T: int = DEFAULT_T,
    config: Mapping[str, Any] | None = None,
    substrate: Any | None = None,
    task_lm_config: Any | None = None,
    reflection_lm_config: Any | None = None,
    raise_on_exception: bool = True,
    extra_callbacks: Optional[list[GEPACallback]] = None,
    state_capture: Optional[dict] = None,
) -> dict[str, Any]:
    """Build the real HotpotQA program/adapter/LMs and run one (arm, seed) cell.

    Splits use the constant ``seed_splits`` from config (NOT the run seed), so
    the A=20 batch and all splits are identical across every arm and seed
    (hard constraints #4, #7). The run ``seed`` only seeds the engine rng.
    """
    import dspy
    import yaml

    from src.data import load_splits
    from src.retry import retryable
    from src.run_gepa import (
        LMConfig,
        Substrate,
        _PatchedDspyAdapter,
        _RetryingAdapter,
        _default_hotpot_substrate,
        _make_feedback_map,
        load_lm_configs_from_env,
    )

    if arm not in EXP3_ARMS:
        raise ValueError(f"build_and_run: unknown arm {arm!r}; expected {EXP3_ARMS}")
    if config is None:
        config = yaml.safe_load((_REPO / "config" / "experiment.yaml").read_text())
    if substrate is None:
        substrate = _default_hotpot_substrate()
    if task_lm_config is None or reflection_lm_config is None:
        _task, _refl = load_lm_configs_from_env(config)
        task_lm_config = task_lm_config or _task
        reflection_lm_config = reflection_lm_config or _refl

    d_feedback, accept_batch, d_pareto, test = load_splits(config=config)
    bins = load_bins()

    task_lm = task_lm_config.to_lm()
    dspy.settings.configure(lm=task_lm)
    reflection_lm_obj = reflection_lm_config.to_lm()

    @retryable(max_attempts=5, base_delay=1.0, max_delay=30.0, label="reflection_lm")
    def reflection_lm_callable(x):
        if isinstance(x, str):
            return reflection_lm_obj(prompt=x)
        return reflection_lm_obj(messages=x)

    program = substrate.build_program()
    base_adapter = _PatchedDspyAdapter(
        student_module=program,
        metric_fn=substrate.metric_fn,
        feedback_map=_make_feedback_map(substrate),
        failure_score=0.0,
        num_threads=substrate.num_threads,
        add_format_failure_as_feedback=True,
        rng=random.Random(seed),
        reflection_lm=reflection_lm_callable,
        warn_on_score_mismatch=False,
    )
    adapter = _RetryingAdapter(base_adapter)

    seed_candidate = {
        name: program.get_module_instruction(name) for name in substrate.component_names
    }

    out_dir = Path(out_root) / arm
    run_dir = out_dir / f"seed{seed}"
    summary_path = out_dir / f"seed{seed}.json"

    return run_cell(
        arm,
        seed,
        d_feedback=d_feedback,
        accept_batch=accept_batch,
        d_pareto=d_pareto,
        test=test,
        bins=bins,
        adapter=adapter,
        seed_candidate=seed_candidate,
        run_dir=run_dir,
        summary_path=summary_path,
        T=T,
        b=int(config["minibatch"]["b"]),
        reflection_lm=reflection_lm_callable,
        raise_on_exception=raise_on_exception,
        extra_callbacks=extra_callbacks,
        state_capture=state_capture,
    )
