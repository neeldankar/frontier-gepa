"""IFBench substrate wiring for the Chunk-14 matrix.

Bundles the Experiment-2 program, metric, feedback function, and
component names into the `Substrate` shape `src/run_gepa.py::run`
consumes, plus a `load_ifbench_splits` helper that returns the
four-pool tuple in the same shape `src/data.py::load_splits` returns
for HotpotQA.

This module is the only place the orchestrator imports IFBench-specific
code from. The HotpotQA path is byte-identical to before -- it uses
`src.run_gepa._default_hotpot_substrate()` and `src.data.load_splits`.

The expected frozen-table SHA-256 is the value Chunk 13 wrote to
`results/ifbench/difficulty_table.sha256`. The orchestrator MUST hash-
verify before any cell runs (BUILD_PLAN §7 Chunk 14, mirroring Chunk 9's
HotpotQA invariant).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dspy

from src.run_gepa import Substrate

_REPO = Path(__file__).resolve().parents[1]

# Frozen-table provenance constants from Chunk 13.
IFBENCH_DIFFICULTY_PATH: Path = (
    _REPO / "results" / "ifbench" / "difficulty_table.json"
)
IFBENCH_DIFFICULTY_HASH_PATH: Path = (
    _REPO / "results" / "ifbench" / "difficulty_table.sha256"
)
# The SHA-256 baked at Chunk 13. Source of truth: the on-disk
# .sha256 file. Duplicated here as a constant so a tampering of the
# JSON without updating the .sha256 file still trips the assert.
EXPECTED_IFBENCH_DIFFICULTY_HASH: str = (
    "e7878d44450e7e67fa58d58b1664899da79ebf3d7298ba3d384a7b558c649365"
)

# Carved-pool sizes Chunk 11 (redo) committed for IFBench. These match
# the Experiment-1 layout but are repeated here so the IFBench loader
# does not silently depend on `config["splits"]`. The orchestrator's
# `_validate_invariants` still cross-checks against the YAML.
IFBENCH_CARVE_SEED: int = 0


def load_ifbench_splits(
    config: Mapping[str, Any] | None = None,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """Carve disjoint IFBench pools at `config['splits']['seed_splits']`
    (default 0). Returns the (d_feedback, accept_batch, d_pareto, test)
    4-tuple in the shape `src/data.py::load_splits` returns.

    Importing `carve_ifbench_splits` lazily so the heavy `datasets` /
    HF stack is not loaded unless an IFBench call site actually runs.
    """
    from src.ifbench_data import carve_ifbench_splits

    if config is None:
        import yaml  # local import; HotpotQA path doesn't need yaml here.
        config = yaml.safe_load((_REPO / "config" / "experiment.yaml").read_text())
    seed = int(config["splits"]["seed_splits"])
    carve = carve_ifbench_splits(seed=seed)
    return tuple(list(s) for s in carve.splits)  # type: ignore[return-value]


IFBENCH_NUM_THREADS: int = 16
"""dspy.Evaluate concurrency for IFBench evals (adapter.evaluate and the
temp-0 drift rescoring). Set to 16 because IFBench's per-call task-model
generations are ~12s and the initial serial pilot showed no rate-limit
backoff -- the loop was wall-time bound, not throttle bound. The matrix
launch will re-confirm by greping the pilot stderr."""


def ifbench_substrate() -> Substrate:
    """The Chunk-14 IFBench substrate. Lazy imports keep HotpotQA-only
    call sites from pulling the vendored IFEval verifiers."""
    from src.ifbench_feedback import metric_fn as ifbench_metric_fn
    from src.ifbench_feedback import score_and_feedback
    from src.ifbench_program import COMPONENT_NAMES as IFBENCH_COMPONENT_NAMES
    from src.ifbench_program import build_program as ifbench_build_program

    return Substrate(
        name="ifbench",
        build_program=ifbench_build_program,
        metric_fn=ifbench_metric_fn,
        feedback_fn=score_and_feedback,
        component_names=IFBENCH_COMPONENT_NAMES,
        num_threads=IFBENCH_NUM_THREADS,
    )


def verify_ifbench_difficulty_table_hash() -> str:
    """Assert the on-disk IFBench difficulty table matches the frozen
    Chunk-13 SHA-256. Returns the verified hash. Raises if the file is
    missing or the hash drifts.

    This MUST be called before any IFBench cell launches. The frozen
    table is the experiment's binding; a silent re-score between
    chunks would change the bands and invalidate cross-cell
    comparison."""
    from src.difficulty import difficulty_table_sha256

    if not IFBENCH_DIFFICULTY_PATH.exists():
        raise RuntimeError(
            f"IFBench difficulty table not found at {IFBENCH_DIFFICULTY_PATH}. "
            "Chunk 13 must complete first."
        )
    actual = difficulty_table_sha256(IFBENCH_DIFFICULTY_PATH)
    if actual != EXPECTED_IFBENCH_DIFFICULTY_HASH:
        raise RuntimeError(
            f"IFBench difficulty table hash mismatch.\n"
            f"  expected (Chunk 13 frozen): {EXPECTED_IFBENCH_DIFFICULTY_HASH}\n"
            f"  actual (on disk):           {actual}\n"
            f"  path: {IFBENCH_DIFFICULTY_PATH}\n"
            "Refusing to launch: the table is the experiment's binding."
        )
    # Cross-check the side-car .sha256 file written by Chunk 13.
    if IFBENCH_DIFFICULTY_HASH_PATH.exists():
        sidecar = IFBENCH_DIFFICULTY_HASH_PATH.read_text().strip()
        if sidecar != actual:
            raise RuntimeError(
                f"IFBench difficulty table side-car hash mismatch.\n"
                f"  .json hash:   {actual}\n"
                f"  .sha256 file: {sidecar}\n"
                "The two have diverged; investigate before proceeding."
            )
    return actual
