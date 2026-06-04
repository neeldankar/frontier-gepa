"""Engine-direct GEPA runner (no gepa.optimize() wrapper).

Constructs ``gepa.core.engine.GEPAEngine`` directly so we can:
  - inject our ``DecoupledReflectiveMutationProposer`` (the §4 acceptance
    decoupling -- ``gepa.optimize()`` does not expose a proposer override),
  - inject our ``BandBatchSampler`` per arm (this one IS exposed by
    ``optimize()`` but we get it via the engine path for free),
  - snapshot a single shared ``random.Random`` per-iteration to disk for
    bit-exact resume (ARCHITECTURE.md §5 risk 7 / the operator requirement
    added 2026-06-01).

The "loop" itself is untouched -- engine.run() is upstream code verbatim.
We only replace the construction in api.py:182-409.

Arm dispatch:
  - "random" / "static_easy" / "static_frontier" / "static_hard":
        BandBatchSampler + DecoupledReflectiveMutationProposer.
  - "vanilla_coupled_gepa":
        Stock EpochShuffledBatchSampler + stock ReflectiveMutationProposer.
        Acceptance is on the b=3 minibatch (engine default, unchanged).

Fail-fast: ``_validate_invariants`` raises before any LM call if the splits,
A-batch, difficulty table, or arm name disagree with the locked config.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import dspy
import yaml

from gepa.adapters.dspy_adapter.dspy_adapter import DspyAdapter, ScoreWithFeedback
from gepa import EvaluationBatch
from gepa.core.callbacks import GEPACallback, IterationEndEvent
from gepa.core.data_loader import ensure_loader
from gepa.core.engine import GEPAEngine
from gepa.core.state import GEPAState
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import Logger, LoggerProtocol, StdOutLogger
from gepa.proposer.reflective_mutation.reflective_mutation import (
    ReflectiveMutationProposer,
)
from gepa.strategies.batch_sampler import BatchSampler, EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import MaxCandidateProposalsStopper
from gepa.utils.stop_condition import StopperProtocol

from src.band_sampler import (
    DEFAULT_MINIBATCH_SIZE,
    DEFAULT_MIX,
    BandBatchSampler,
    TargetBand,
)
from src.decoupled_proposer import DecoupledReflectiveMutationProposer
from src.difficulty import DifficultyTable
from src.retry import retryable

_REPO = Path(__file__).resolve().parents[1]

ARM_NAMES: tuple[str, ...] = (
    "random",
    "static_easy",
    "static_frontier",
    "static_hard",
    "vanilla_coupled_gepa",
)
DECOUPLED_ARMS: frozenset[str] = frozenset(
    {"random", "static_easy", "static_frontier", "static_hard"}
)
ARM_TO_TARGET_BAND: Mapping[str, TargetBand] = {
    "random": "random",
    "static_easy": "easy",
    "static_frontier": "mid",
    "static_hard": "hard",
}


@dataclass
class LMConfig:
    """Subset of the experiment.yaml task_model / reflection_model blocks
    needed to instantiate ``dspy.LM``. The runner reads keys from the env
    using the names recorded in the config (so secrets stay out of process
    arguments and out of logs)."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 16384
    num_retries: int = 5  # dspy.LM's native retry on transient errors

    def to_lm(self) -> dspy.LM:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            num_retries=self.num_retries,
        )
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return dspy.LM(**kwargs)


def load_lm_configs_from_env(
    config: Mapping[str, Any] | None = None,
) -> tuple[LMConfig, LMConfig]:
    """Read task and reflection LM configs from env vars named in config[
    'task_model'] / config['reflection_model']."""
    if config is None:
        config = yaml.safe_load((_REPO / "config" / "experiment.yaml").read_text())
    task = config["task_model"]
    refl = config["reflection_model"]

    task_model = os.environ.get(task["name_env"])
    if not task_model:
        raise RuntimeError(
            f"environment variable {task['name_env']!r} is not set; "
            f"task model unresolved."
        )
    reflection_model = os.environ.get(refl["name_env"])
    if not reflection_model:
        raise RuntimeError(
            f"environment variable {refl['name_env']!r} is not set; "
            f"reflection model unresolved."
        )

    return (
        LMConfig(
            model=task_model,
            api_key=os.environ.get(task["api_key_env"]) or None,
            api_base=os.environ.get(task["base_url_env"]) or None,
            temperature=float(task["temperature"]),
            top_p=float(task["top_p"]),
            max_tokens=int(task["max_tokens"]),
        ),
        LMConfig(
            model=reflection_model,
            api_key=os.environ.get(refl["api_key_env"]) or None,
        ),
    )


# ---- Substrate: the bag of substrate-specific callables. ----
#
# Chunk-14 wiring: the runner used to import directly from src.feedback +
# src.program (HotpotQA-shaped). To wire IFBench (Experiment 2) without
# changing the HotpotQA path bit-for-bit, the substrate-specific bits are
# packaged into a `Substrate` and threaded through `run()`. The default
# substrate is HotpotQA, so existing call sites are unchanged.
#
# A Substrate carries:
#   - build_program:  callable returning a dspy.Module with predictors
#                     whose names match `component_names`.
#   - metric_fn:      module-level metric the adapter uses for evaluation
#                     (returns float in [0,1]).
#   - feedback_fn:    (example, prediction) -> dict with keys 'score' and
#                     'feedback'. Used by the per-predictor reflective
#                     callback (DspyAdapter.feedback_map).
#   - component_names: the tuple of optimisable predictor names; mirrors
#                     `seed_candidate` keys.


@dataclass
class Substrate:
    name: str
    build_program: Callable[[], dspy.Module]
    metric_fn: Callable[..., float]
    feedback_fn: Callable[..., Mapping[str, Any]]
    component_names: tuple[str, ...]
    # `num_threads` is the dspy.Evaluate concurrency for adapter.evaluate
    # calls and for the temp-0 drift rescoring. None = serial (preserves
    # the HotpotQA Experiment-1 / 1b behaviour byte-for-byte). IFBench
    # sets this to 16 because per-call generations are ~12s and the
    # serial pilot demonstrated that the loop is wall-time bound, not
    # rate-limit bound (no throttle/backoff observed under serial).
    num_threads: int | None = None


def _default_hotpot_substrate() -> Substrate:
    """The Experiment-1 / 1b HotpotQA substrate. Imported lazily so the
    HotpotQA modules don't have to load for an IFBench-only call site.

    Threading: stays serial (num_threads=None) to preserve byte-identical
    HotpotQA reproduction; the existing matrices already ran and have
    cell_summary.json's the orchestrator will skip-resume from."""
    from src.feedback import compute_feedback, metric_fn as hotpot_metric_fn
    from src.program import COMPONENT_NAMES as HOTPOT_COMPONENT_NAMES
    from src.program import build_program as hotpot_build_program

    return Substrate(
        name="hotpot",
        build_program=hotpot_build_program,
        metric_fn=hotpot_metric_fn,
        feedback_fn=compute_feedback,
        component_names=HOTPOT_COMPONENT_NAMES,
        num_threads=None,
    )


def _make_feedback_map(substrate: Substrate) -> dict[str, Callable]:
    """Same trajectory-level feedback for every predictor (per the
    feedback-template spec: 'a single trajectory-level string ... shown for
    whichever module round-robin selects this iteration')."""

    def feedback_fn(
        predictor_output: dict,
        predictor_inputs: dict,
        module_inputs: Any,
        module_outputs: Any,
        captured_trace: Any,
    ) -> ScoreWithFeedback:
        out = substrate.feedback_fn(module_inputs, module_outputs)
        return ScoreWithFeedback(score=float(out["score"]), feedback=str(out["feedback"]))

    return {name: feedback_fn for name in substrate.component_names}


# ---- Validation (fail-fast before any spend) ----


def _validate_invariants(
    config: Mapping[str, Any],
    arm: str,
    seed: int,
    splits: Sequence[Sequence[Any]],
    difficulty_table: DifficultyTable | None,
) -> None:
    if arm not in ARM_NAMES:
        raise ValueError(f"arm must be one of {ARM_NAMES}, got {arm!r}")
    if seed is None or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    sp = config["splits"]
    n_feedback = int(sp["d_feedback"])
    n_accept = int(sp["accept_batch"])
    n_pareto = int(sp["d_pareto"])
    n_test = int(sp["test"])

    if len(splits) != 4:
        raise ValueError(f"splits must be (d_feedback, accept_batch, d_pareto, test); got {len(splits)} lists")
    d_feedback, accept_batch, d_pareto, test = splits
    if len(d_feedback) != n_feedback:
        raise ValueError(
            f"d_feedback size {len(d_feedback)} != config {n_feedback}"
        )
    if len(accept_batch) != n_accept:
        raise ValueError(
            f"accept_batch size {len(accept_batch)} != config {n_accept}"
        )
    if len(d_pareto) != n_pareto:
        raise ValueError(
            f"d_pareto size {len(d_pareto)} != config {n_pareto}"
        )
    if len(test) != n_test:
        raise ValueError(f"test size {len(test)} != config {n_test}")
    # The "fixed accept batch 20" §15 rule is enforced at the
    # config-experiment.yaml level (see pytest tests/test_chunk2.py
    # test_config_locked_params); the runner only checks that runtime sizes
    # match the passed config so it can run against shrunk configs for the
    # tiny-slice smoke.

    # Disjointness sanity check (data.py already checks, but this protects
    # against caller bugs that bypassed load_splits).
    seen: set[str] = set()
    for split_name, items in zip(
        ("d_feedback", "accept_batch", "d_pareto", "test"), splits
    ):
        for ex in items:
            hid = ex["id"]
            if hid in seen:
                raise ValueError(
                    f"split overlap: id {hid!r} appears in {split_name!r} and another split"
                )
            seen.add(hid)

    # Stopper / minibatch invariants
    if int(config["minibatch"]["b"]) != DEFAULT_MINIBATCH_SIZE:
        raise ValueError(
            f"minibatch.b is locked at {DEFAULT_MINIBATCH_SIZE}; config has {config['minibatch']['b']}"
        )
    if config["stopping"]["rule"] != "iterations":
        raise ValueError(
            f"stopping.rule must be 'iterations'; got {config['stopping']['rule']!r}"
        )
    if config["acceptance"]["rule"] != "strict_improvement":
        raise ValueError(
            f"acceptance.rule must be 'strict_improvement'; got {config['acceptance']['rule']!r}"
        )
    n = int(config["stopping"]["n"])
    if n <= 0:
        raise ValueError(f"stopping.n must be positive; got {n}")

    # Difficulty table check (only for non-random band arms)
    if arm in DECOUPLED_ARMS and arm != "random":
        if difficulty_table is None:
            raise ValueError(f"arm={arm!r} requires a difficulty_table")
        if difficulty_table.n != n_feedback:
            raise ValueError(
                f"difficulty_table.n {difficulty_table.n} != d_feedback size {n_feedback}"
            )


# ---- RNG persistence helpers (per-iteration snapshot) ----

_RNG_FILE = "shared_rng.pkl"


def _snapshot_rng(rng: random.Random, run_dir: Path) -> None:
    state = rng.getstate()
    path = run_dir / _RNG_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def _restore_rng(rng: random.Random, run_dir: Path) -> bool:
    path = run_dir / _RNG_FILE
    if not path.exists():
        return False
    with open(path, "rb") as f:
        rng.setstate(pickle.load(f))
    return True


class _RNGSnapshotCallback(GEPACallback):
    """Flush the shared RNG state to disk at the end of every iteration so
    a kill-and-resume restarts from a bit-exact RNG position."""

    def __init__(self, rng: random.Random, run_dir: Path):
        self.rng = rng
        self.run_dir = run_dir

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        _snapshot_rng(self.rng, self.run_dir)


# ---- The runner ----


def run(
    arm: str,
    seed: int,
    splits: Sequence[Sequence[Any]],
    difficulty_table: DifficultyTable | None,
    run_dir: Path | str,
    *,
    task_lm_config: LMConfig,
    reflection_lm_config: LMConfig,
    config: Mapping[str, Any] | None = None,
    logger: LoggerProtocol | None = None,
    raise_on_exception: bool = True,
    lm_capture: dict | None = None,
    mix: tuple[float, float, float] = DEFAULT_MIX,
    substrate: Substrate | None = None,
) -> GEPAState:
    """Run one (arm, seed) cell. Returns the final GEPAState.

    The run is resumable: pass the same ``run_dir`` to continue from the last
    on-disk checkpoint. ``gepa_state.bin`` is written by the engine at the
    start of each iteration; the shared RNG is snapshotted by
    ``_RNGSnapshotCallback`` at the end of each iteration so resume is
    bit-exact.

    Operational note: this function is the only place where API spend
    happens in Chunk 4 (and Chunk 5 base scoring will call adapter.evaluate
    directly without this function). All retry logic lives below; the engine
    never sees a raw transient error.
    """
    if config is None:
        config = yaml.safe_load((_REPO / "config" / "experiment.yaml").read_text())
    if substrate is None:
        substrate = _default_hotpot_substrate()

    _validate_invariants(config, arm, seed, splits, difficulty_table)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist arm/seed metadata for after-the-fact debugging.
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "seed": seed,
                "config_locked": {
                    "splits": config["splits"],
                    "minibatch": config["minibatch"],
                    "stopping": config["stopping"],
                    "acceptance": config["acceptance"],
                },
            },
            indent=2,
        )
    )

    d_feedback, accept_batch, d_pareto, _test = splits

    # ---- Logger ----
    if logger is None:
        logger = Logger(str(run_dir / "run_log.txt"))

    # ---- LM setup, with retry on transient errors ----
    # dspy.LM uses litellm internally and exposes num_retries (set in to_lm()).
    # We wrap on top with @retryable for transient classes litellm misses
    # (e.g., httpx.ReadError on a flaky connection).
    task_lm = task_lm_config.to_lm()
    dspy.settings.configure(lm=task_lm)
    reflection_lm_obj = reflection_lm_config.to_lm()

    # Expose the LMs to the caller for cost tracking; the orchestrator reads
    # .history off these instances after the run completes.
    if lm_capture is not None:
        lm_capture["task_lm"] = task_lm
        lm_capture["refl_lm"] = reflection_lm_obj

    @retryable(max_attempts=5, base_delay=1.0, max_delay=30.0, label="reflection_lm")
    def reflection_lm_callable(x: str | list[dict[str, str]]) -> list[str]:
        # dspy.LM is callable with (prompt=) or (messages=) and returns a
        # list[str] of completions. DspyAdapter.stripped_lm_call iterates
        # over this list, so the return must be a list, NOT a single string.
        if isinstance(x, str):
            return reflection_lm_obj(prompt=x)
        return reflection_lm_obj(messages=x)

    # ---- Adapter: dspy program + module-level F1 + per-predictor feedback ----
    program = substrate.build_program()
    base_adapter = _PatchedDspyAdapter(
        student_module=program,
        metric_fn=substrate.metric_fn,
        feedback_map=_make_feedback_map(substrate),
        failure_score=0.0,
        # Per-substrate parallelism. HotpotQA stays serial (None) so the
        # Experiment-1 / 1b cells reproduce byte-identically; IFBench sets
        # 16 because per-call generations are output-heavy and the serial
        # baseline was wall-time bound.
        num_threads=substrate.num_threads,
        add_format_failure_as_feedback=True,
        rng=random.Random(seed),  # internal rng used for trace-instance selection
        reflection_lm=reflection_lm_callable,
        warn_on_score_mismatch=False,  # we are intentionally returning module-level F1
    )
    adapter = _RetryingAdapter(base_adapter)

    # ---- Shared RNG for candidate_selector + batch_sampler + (gepa's
    #      internal merge_proposer, if we ever enable merge).
    rng = random.Random(seed)
    resumed = _restore_rng(rng, run_dir)

    # ---- Batch sampler + proposer per arm ----
    n_train = len(d_feedback)
    if arm in DECOUPLED_ARMS:
        target_band = ARM_TO_TARGET_BAND[arm]
        batch_sampler: BatchSampler = BandBatchSampler(
            target_band=target_band,
            rng=rng,
            difficulty_table=(difficulty_table if target_band != "random" else None),
            b=int(config["minibatch"]["b"]),
            mix=mix,
        )
        proposer_cls = DecoupledReflectiveMutationProposer
        # Build proposer below with the A-batch kwargs.
        proposer_extra_kwargs = {
            "accept_batch": list(accept_batch),
            "accept_batch_ids": list(range(len(accept_batch))),
        }
    elif arm == "vanilla_coupled_gepa":
        batch_sampler = EpochShuffledBatchSampler(
            minibatch_size=int(config["minibatch"]["b"]), rng=rng
        )
        proposer_cls = ReflectiveMutationProposer
        proposer_extra_kwargs = {}
    else:
        raise AssertionError(f"unreachable: arm {arm!r}")

    candidate_selector = ParetoCandidateSelector(rng=rng)
    module_selector = RoundRobinReflectionComponentSelector()

    experiment_tracker = create_experiment_tracker(
        use_wandb=False, wandb_api_key=None, wandb_init_kwargs=None,
        use_mlflow=False, mlflow_tracking_uri=None, mlflow_experiment_name=None,
    )

    proposer = proposer_cls(
        logger=logger,
        trainset=list(d_feedback),
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector,
        batch_sampler=batch_sampler,
        perfect_score=1.0,
        skip_perfect_score=True,
        experiment_tracker=experiment_tracker,
        reflection_lm=reflection_lm_callable,
        reflection_prompt_template=None,
        custom_candidate_proposer=None,
        callbacks=None,  # we attach the snapshot callback to the engine
        **proposer_extra_kwargs,
    )

    # ---- Engine ----
    seed_candidate = {
        name: program.get_module_instruction(name) for name in substrate.component_names
    }
    stop_callback: StopperProtocol = MaxCandidateProposalsStopper(
        max_proposals=int(config["stopping"]["n"])
    )
    rng_callback = _RNGSnapshotCallback(rng, run_dir)

    engine = GEPAEngine(
        adapter=adapter,
        run_dir=str(run_dir),
        valset=list(d_pareto),
        seed_candidate=seed_candidate,
        perfect_score=1.0,
        seed=seed,
        reflective_proposer=proposer,
        merge_proposer=None,  # §15 says no merge in this experiment
        frontier_type="instance",
        logger=logger,
        experiment_tracker=experiment_tracker,
        callbacks=[rng_callback],
        track_best_outputs=False,
        display_progress_bar=False,
        raise_on_exception=raise_on_exception,
        stop_callback=stop_callback,
        val_evaluation_policy=FullEvaluationPolicy(),
        use_cloudpickle=False,
        evaluation_cache=None,  # not needed; the engine's per-iter logic
                                # already short-circuits where it matters
    )

    if resumed:
        logger.log(f"Resumed run_dir {str(run_dir)} from on-disk RNG + state.")
    else:
        logger.log(f"Fresh run: arm={arm!r}, seed={seed}, run_dir={str(run_dir)}.")

    with experiment_tracker:
        if isinstance(logger, Logger):
            with logger:
                state = engine.run()
        else:
            state = engine.run()

    # Final snapshot (in case the last iteration's callback didn't fire, e.g.
    # an exception path).
    _snapshot_rng(rng, run_dir)
    return state


# ---- DspyAdapter patched for dspy 3.2.x ----


class _PatchedDspyAdapter(DspyAdapter):
    """Override evaluate(capture_traces=False) to use the dspy 3.2 Evaluate
    signature. Upstream gepa 0.1.1 passes ``return_outputs=True`` and
    ``return_all_scores=True``, both removed in dspy 3.2. The data those args
    used to expose is now always returned via ``EvaluationResult.results``,
    so the only change is dropping the deprecated kwargs and pulling the
    same fields from ``res.results`` directly.
    """

    def evaluate(self, batch, candidate, capture_traces=False):
        # The capture_traces=True branch in upstream uses bootstrap_trace_data
        # which does NOT pass return_outputs, so it works unchanged.
        if capture_traces:
            return super().evaluate(batch, candidate, capture_traces=True)

        program = self.build_program(candidate)
        callback_metadata = (
            {"metric_key": "eval_full"}
            if self.reflection_minibatch_size is None
            or len(batch) > self.reflection_minibatch_size
            else {"disable_logging": True}
        )

        evaluator = dspy.Evaluate(
            devset=batch,
            metric=self.metric_fn,
            num_threads=self.num_threads,
            failure_score=self.failure_score,
            provide_traceback=True,
            max_errors=len(batch) * 100,
            callback_metadata=callback_metadata,
        )
        res = evaluator(program)
        # res.results is list[tuple[Example, Prediction, score]]; align order
        # with the input batch since dspy.Evaluate preserves devset order.
        outputs = [r[1] for r in res.results]
        raw_scores = [r[2] for r in res.results]

        scores: list[float] = []
        subscores: list[dict[str, float]] = []
        for raw_score in raw_scores:
            score_val, subscore_dict = self._extract_score_and_subscores(raw_score)
            if score_val is None:
                score_val = self.failure_score
            scores.append(score_val)
            subscores.append(subscore_dict)
        has_subscores = any(subscores)
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=None,
            objective_scores=subscores if has_subscores else None,
        )


# ---- Retry-wrapping adapter ----


class _RetryingAdapter:
    """Forwards everything to an inner ``GEPAAdapter`` but retries
    ``evaluate(...)`` on transient errors. Used so the engine and the
    decoupled proposer both get retry transparently without touching their
    code."""

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @retryable(max_attempts=5, base_delay=1.0, max_delay=30.0, label="adapter.evaluate")
    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        return self._inner.evaluate(batch, candidate, capture_traces=capture_traces)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> Any:
        return self._inner.make_reflective_dataset(
            candidate, eval_batch, components_to_update
        )
