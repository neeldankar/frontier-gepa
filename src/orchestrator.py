"""Chunk 6 matrix orchestrator: pilot, cost gate, 15-cell run, summary.

Phases (run in order via the `all` subcommand):

  1. Pilot: static_frontier seed=0 at full pool sizes (D_feedback=150,
     A=20, D_pareto=75) but a short horizon (N=4). End-to-end including
     a held-out 300-example test eval. Measures real per-iteration cost.

  2. Cost gate: project the full 15x44 matrix at the pilot's per-iter
     rate (frontier is the upper end for acceptance, so it's a
     conservative bound across all arms), plus 15 test evals at the
     pilot's test-eval cost. STOP if projected matrix cost >= $100;
     continue only if it stays below.

  3. Matrix: 15 cells serial = 5 arms x 3 seeds. Each cell:
        - resumes from results/logs/<arm>_<seed>/ if present;
        - runs run_gepa.run() to N=44;
        - evaluates the best-by-D_pareto candidate on the 300-example
          held-out test set;
        - writes cell_summary.json (arm, seed, evals, candidates,
          best F1, test F1, wall time, cost).
     After each cell, the orchestrator adds that cell's cost to a
     running cumulative; if cumulative >= $150, the kill switch fires
     and no further cells are launched.

  4. Summary at the end: which cells completed, total spend, total wall
     time, any errors for resume.

Resumability:
  - The Chunk-4 GEPAEngine already checkpoints state to disk at the
    start of each iteration (results/logs/<arm>_<seed>/gepa_state.bin)
    and the orchestrator's shared RNG snapshot to shared_rng.pkl. A
    resumed cell picks up at the next iteration. The orchestrator skips
    cells where cell_summary.json already exists.
  - Test eval is idempotent: if results/logs/<arm>_<seed>/test_eval.json
    already exists, it is reused without re-running.

CLI:
  - .venv/bin/python -m src.orchestrator pilot      # phases 1-2 only
  - .venv/bin/python -m src.orchestrator matrix     # phases 3-4 only
  - .venv/bin/python -m src.orchestrator all        # phases 1-4
  - .venv/bin/python -m src.orchestrator cell --arm X --seed Y
        (internal: run a single cell; used for resume / debugging)

Logging: everything also prints to stdout. For overnight runs use
`nohup ... > results/logs/orchestrator.log 2>&1 &` so the process
survives this session.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random as _random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.band_sampler import DEFAULT_MIX, PURE_ON_BAND_MIX  # noqa: E402

HOTPOT_DIFFICULTY_PATH = REPO / "results" / "difficulty_table.json"
IFBENCH_DIFFICULTY_PATH = REPO / "results" / "ifbench" / "difficulty_table.json"

# Variants supported by this orchestrator. The current variant is set by
# main() (via the --variant CLI flag) before any phase function runs, so the
# module-level LOGS_ROOT / ARMS / CURRENT_MIX / VARIANT_NAME are read
# correctly by the existing _run_single_cell / run_matrix code paths.
#
#   default:      Experiment 1 (BUILD_PLAN.md context). 5 arms x 3 seeds
#                 at 70/15/15 into results/logs/.
#   hotpot_100:   Experiment 1b (BUILD_PLAN.md §7 Chunk 9). 3 static arms
#                 at 100/0/0 into results/logs_hotpot_100/. Reuses the
#                 frozen Experiment-1 difficulty table.
#   ifbench_100:  Experiment 2 (BUILD_PLAN.md §7 Chunk 14). 5 arms x 3
#                 seeds at 100/0/0, N=80, IFBench substrate (one-module
#                 program + IFEval verifier feedback), into
#                 results/logs_ifbench/. Hash-verifies the frozen Chunk-13
#                 IFBench difficulty table; runs the D4 temp-0 drift
#                 instrumentation after each cell.
VARIANTS: dict[str, dict[str, Any]] = {
    "default": {
        "logs_root": REPO / "results" / "logs",
        "arms": (
            "random",
            "static_easy",
            "static_frontier",
            "static_hard",
            "vanilla_coupled_gepa",
        ),
        "mix": DEFAULT_MIX,
        "n_iter": 44,
        "sampler_name": "70/15/15",
        "substrate": "hotpot",
        "difficulty_path": HOTPOT_DIFFICULTY_PATH,
        "drift_enabled": False,
    },
    "hotpot_100": {
        "logs_root": REPO / "results" / "logs_hotpot_100",
        "arms": ("static_easy", "static_frontier", "static_hard"),
        "mix": PURE_ON_BAND_MIX,
        "n_iter": 44,
        "sampler_name": "100/0/0",
        "substrate": "hotpot",
        "difficulty_path": HOTPOT_DIFFICULTY_PATH,
        "drift_enabled": False,
    },
    "ifbench_100": {
        "logs_root": REPO / "results" / "logs_ifbench",
        "arms": (
            "random",
            "static_easy",
            "static_frontier",
            "static_hard",
            "vanilla_coupled_gepa",
        ),
        "mix": PURE_ON_BAND_MIX,
        "n_iter": 80,
        "sampler_name": "100/0/0",
        "substrate": "ifbench",
        "difficulty_path": IFBENCH_DIFFICULTY_PATH,
        "drift_enabled": True,
    },
}

# Module-level current-variant state. Defaults to the Experiment-1
# layout so a bare `python -m src.orchestrator pilot` still reproduces
# the Experiment-1 entry point.
LOGS_ROOT: Path = VARIANTS["default"]["logs_root"]
ARMS: tuple[str, ...] = VARIANTS["default"]["arms"]
CURRENT_MIX: tuple[float, float, float] = VARIANTS["default"]["mix"]
CURRENT_VARIANT_NAME: str = "default"
SAMPLER_NAME: str = VARIANTS["default"]["sampler_name"]
CURRENT_DIFFICULTY_PATH: Path = VARIANTS["default"]["difficulty_path"]
CURRENT_SUBSTRATE_NAME: str = VARIANTS["default"]["substrate"]
CURRENT_DRIFT_ENABLED: bool = bool(VARIANTS["default"]["drift_enabled"])
CURRENT_N_ITER: int = int(VARIANTS["default"]["n_iter"])

SEEDS: tuple[int, ...] = (0, 1, 2)

PILOT_ARM = "static_frontier"
PILOT_SEED = 0
PILOT_N_ITER = 4

PROJECTION_GATE_USD = 100.0
HARD_KILL_USD = 150.0


def _set_variant(name: str) -> None:
    """Swap the module-level variant state. Must be called BEFORE any
    phase function runs.

    For variants whose `substrate` is 'ifbench', also assert the frozen
    Chunk-13 IFBench table hash. The HotpotQA variants don't hash-check
    here (Chunk-9 invariant is hash-checked per cell as before)."""
    global LOGS_ROOT, ARMS, CURRENT_MIX, CURRENT_VARIANT_NAME, SAMPLER_NAME
    global CURRENT_DIFFICULTY_PATH, CURRENT_SUBSTRATE_NAME, CURRENT_DRIFT_ENABLED
    global CURRENT_N_ITER
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; supported: {list(VARIANTS)}")
    v = VARIANTS[name]
    LOGS_ROOT = v["logs_root"]
    ARMS = v["arms"]
    CURRENT_MIX = v["mix"]
    CURRENT_VARIANT_NAME = name
    SAMPLER_NAME = v["sampler_name"]
    CURRENT_DIFFICULTY_PATH = v["difficulty_path"]
    CURRENT_SUBSTRATE_NAME = v["substrate"]
    CURRENT_DRIFT_ENABLED = bool(v["drift_enabled"])
    CURRENT_N_ITER = int(v["n_iter"])

    if CURRENT_SUBSTRATE_NAME == "ifbench":
        from src.ifbench_substrate import verify_ifbench_difficulty_table_hash
        h = verify_ifbench_difficulty_table_hash()
        print(f"  IFBench difficulty table hash-verified: {h}")


def _difficulty_table_hash() -> str:
    """SHA-256 of the current variant's frozen difficulty table file.
    Used in cell provenance."""
    return hashlib.sha256(CURRENT_DIFFICULTY_PATH.read_bytes()).hexdigest()


def _load_substrate_for_current_variant():
    """Resolve the variant's `substrate` string into the `Substrate`
    dataclass instance that `run_gepa.run()` expects."""
    if CURRENT_SUBSTRATE_NAME == "hotpot":
        from src.run_gepa import _default_hotpot_substrate
        return _default_hotpot_substrate()
    if CURRENT_SUBSTRATE_NAME == "ifbench":
        from src.ifbench_substrate import ifbench_substrate
        return ifbench_substrate()
    raise ValueError(f"unknown substrate name: {CURRENT_SUBSTRATE_NAME!r}")


def _load_splits_for_current_variant(config):
    """Carve splits for the current variant's substrate."""
    if CURRENT_SUBSTRATE_NAME == "hotpot":
        from src.data import load_splits
        return load_splits(config=config)
    if CURRENT_SUBSTRATE_NAME == "ifbench":
        from src.ifbench_substrate import load_ifbench_splits
        return load_ifbench_splits(config=config)
    raise ValueError(f"unknown substrate name: {CURRENT_SUBSTRATE_NAME!r}")


# ============================================================================
# Subprocess-callable single-cell runner
# ============================================================================


def _run_single_cell(arm: str, seed: int, n_iter_override: int | None = None) -> dict:
    """Run one (arm, seed) cell end-to-end. Writes cell_summary.json into
    its run_dir. Returns the summary dict."""
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")

    import dspy
    import yaml

    from src.cost_tracker import cost_from_lms
    from src.difficulty import DifficultyTable
    from src.run_gepa import (
        load_lm_configs_from_env,
        run,
    )

    config = yaml.safe_load((REPO / "config" / "experiment.yaml").read_text())
    if n_iter_override is None and CURRENT_N_ITER != int(config["stopping"]["n"]):
        # Variant overrides config's N (e.g. ifbench_100 sets N=80, config has 44).
        config = copy.deepcopy(config)
        config["stopping"]["n"] = CURRENT_N_ITER
    if n_iter_override is not None:
        config = copy.deepcopy(config)
        config["stopping"]["n"] = int(n_iter_override)

    splits = _load_splits_for_current_variant(config)
    substrate = _load_substrate_for_current_variant()
    needs_difficulty = arm in ("static_easy", "static_frontier", "static_hard")
    difficulty_table = (
        DifficultyTable.load(CURRENT_DIFFICULTY_PATH) if needs_difficulty else None
    )

    task_lm_config, refl_lm_config = load_lm_configs_from_env(config=config)
    run_dir = LOGS_ROOT / f"{arm}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already finished (cell_summary.json present).
    summary_path = run_dir / "cell_summary.json"
    if summary_path.exists():
        print(f"[skip] {arm}_{seed}: cell_summary.json already exists")
        return json.loads(summary_path.read_text())

    # Save resolved config + git head for provenance.
    git_head = _git_head_safe()
    provenance = {
        "arm": arm,
        "seed": seed,
        "git_commit": git_head,
        "config_resolved": config,
        "n_iter_target": int(config["stopping"]["n"]),
        "variant": CURRENT_VARIANT_NAME,
        "substrate": CURRENT_SUBSTRATE_NAME,
        "sampler": SAMPLER_NAME,           # e.g. "70/15/15" or "100/0/0"
        "sampler_mix": list(CURRENT_MIX),  # numeric weights
        "difficulty_table_hash": _difficulty_table_hash() if needs_difficulty else None,
        "difficulty_table_path": str(CURRENT_DIFFICULTY_PATH.relative_to(REPO)),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if CURRENT_SUBSTRATE_NAME == "ifbench":
        from src.ifbench_data import (
            CONSTRAINT_COUNT_FLOOR,
            IFBENCH_DATASET_ID,
            TARGET_SIZES,
        )
        provenance["dataset"] = IFBENCH_DATASET_ID
        provenance["constraint_count_floor"] = CONSTRAINT_COUNT_FLOOR
        provenance["carved_sizes"] = dict(TARGET_SIZES)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    lm_capture: dict[str, Any] = {}
    t0 = time.time()
    try:
        state = run(
            arm=arm,
            seed=seed,
            splits=splits,
            difficulty_table=difficulty_table,
            run_dir=run_dir,
            task_lm_config=task_lm_config,
            reflection_lm_config=refl_lm_config,
            config=config,
            raise_on_exception=True,
            lm_capture=lm_capture,
            mix=CURRENT_MIX,
            substrate=substrate,
        )
    except Exception as exc:
        wall = time.time() - t0
        err = {
            "arm": arm, "seed": seed,
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "wall_time_s": wall,
            "git_commit": git_head,
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (run_dir / "cell_error.json").write_text(json.dumps(err, indent=2))
        raise

    wall = time.time() - t0
    task_lm = lm_capture.get("task_lm")
    refl_lm = lm_capture.get("refl_lm")

    # Cost from LM histories (real spend; cache hits are $0).
    cost_so_far = cost_from_lms(task_lm, refl_lm)

    # Held-out test evaluation
    test_eval_path = run_dir / "test_eval.json"
    if test_eval_path.exists():
        test_summary = json.loads(test_eval_path.read_text())
        test_cost_report = None
    else:
        # Baseline histories so we can attribute test-eval cost separately.
        task_baseline = len(task_lm.history) if task_lm and task_lm.history else 0
        refl_baseline = len(refl_lm.history) if refl_lm and refl_lm.history else 0
        test_summary = _eval_best_on_test(
            state, splits[3], task_lm_config, refl_lm_config, substrate
        )
        test_eval_path.write_text(json.dumps(test_summary, indent=2))
        # Note: _eval_best_on_test built its OWN LM instances; we
        # cannot diff our task_lm/refl_lm baselines to capture test cost.
        # We report total cost from the run LMs PLUS an estimate of the
        # test cost from the in-function LMs (returned via test_summary).
        test_cost_report = test_summary.get("cost_report")

    # D4 drift instrumentation (only for variants that enable it).
    drift_summary: dict[str, Any] | None = None
    if CURRENT_DRIFT_ENABLED:
        drift_path = run_dir / "drift.json"
        if drift_path.exists():
            drift_summary = json.loads(drift_path.read_text())
        else:
            from src.ifbench_drift import compute_drift_for_cell
            # Final best candidate = the one the test eval just scored.
            best_idx = test_summary["best_idx"]
            final_candidate = state.program_candidates[best_idx]
            # Seed/base candidate: by gepa convention, program_candidates[0]
            # is the seed.
            base_candidate = state.program_candidates[0]
            drift_summary = compute_drift_for_cell(
                base_seed_candidate=base_candidate,
                final_candidate=final_candidate,
                d_feedback=list(splits[0]),
                task_lm_config=task_lm_config,
                substrate=substrate,
                drift_cache_dir=LOGS_ROOT / "_drift",
            )
            drift_path.write_text(json.dumps(drift_summary, indent=2))

    # Final cumulative cost from the run LMs
    final_cost = cost_from_lms(task_lm, refl_lm)
    if test_cost_report:
        final_cost.task_tokens_in += int(test_cost_report.get("task_tokens_in", 0))
        final_cost.task_tokens_out += int(test_cost_report.get("task_tokens_out", 0))
        final_cost.refl_tokens_in += int(test_cost_report.get("refl_tokens_in", 0))
        final_cost.refl_tokens_out += int(test_cost_report.get("refl_tokens_out", 0))
        final_cost.cost_usd += float(test_cost_report.get("cost_usd", 0.0))

    # Pull a few per-iteration signals out of the trace for Chunk 7.
    accepted_in_iters: list[int] = []
    for it_idx, entry in enumerate(state.full_program_trace, start=1):
        if entry.get("new_program_idx") is not None:
            accepted_in_iters.append(it_idx)

    summary = {
        "arm": arm,
        "seed": seed,
        "n_iter_target": int(config["stopping"]["n"]),
        "state_i_final": int(state.i),
        "total_num_evals": int(state.total_num_evals),
        "n_candidates": len(state.program_candidates),
        "accepted_in_iters": accepted_in_iters,
        "best_idx": test_summary["best_idx"],
        "best_val_f1": test_summary["best_val_f1"],
        "avg_test_f1": test_summary["avg_test_f1"],
        "n_test": test_summary["n_test"],
        "wall_time_s": round(wall, 2),
        "cost": final_cost.as_dict(),
        "git_commit": git_head,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variant": CURRENT_VARIANT_NAME,
        "substrate": CURRENT_SUBSTRATE_NAME,
    }
    if drift_summary is not None:
        # Inline the compact drift fields into the cell summary; the full
        # per-id score arrays stay in drift.json.
        summary["drift"] = {
            "spearman_base_vs_final_temp0": drift_summary["spearman_base_vs_final_temp0"],
            "frontier_leaving_rate": drift_summary["frontier_leaving"]["frontier_leaving_rate"],
            "n_base_frontier": drift_summary["frontier_leaving"]["n_base_frontier"],
            "n_final_frontier": drift_summary["frontier_leaving"]["n_final_frontier"],
            "n_left_frontier": drift_summary["frontier_leaving"]["n_left_frontier"],
            "base_binning": drift_summary["frontier_leaving"]["base_binning"],
            "final_binning": drift_summary["frontier_leaving"]["final_binning"],
            "base_temp0_reused": drift_summary["base_temp0_reused"],
        }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _eval_best_on_test(
    state: Any,
    test_examples: list,
    task_lm_config: Any,
    reflection_lm_config: Any,
    substrate: Any = None,
) -> dict:
    """Evaluate the best-by-D_pareto-avg candidate on the held-out test set.

    Returns a dict with best_idx, best_val_f1, avg_test_f1, n_test,
    per_instance_f1, and a cost_report for the test eval itself.

    `substrate` is the Chunk-14 injectable: defaults to the HotpotQA
    substrate when omitted, so Experiment 1 / 1b call sites are
    byte-identical.
    """
    import dspy
    from src.cost_tracker import cost_from_lms
    from src.run_gepa import (
        _default_hotpot_substrate,
        _make_feedback_map,
        _PatchedDspyAdapter,
        _RetryingAdapter,
        retryable,
    )

    if substrate is None:
        substrate = _default_hotpot_substrate()

    # Pick the program with the highest D_pareto average F1.
    scores = state.program_full_scores_val_set
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_cand = state.program_candidates[best_idx]
    best_val_f1 = float(scores[best_idx])

    task_lm = task_lm_config.to_lm()
    refl_lm_obj = reflection_lm_config.to_lm()
    dspy.settings.configure(lm=task_lm)
    program = substrate.build_program()

    @retryable(max_attempts=5, base_delay=1.0, max_delay=30.0, label="test_eval_refl_lm")
    def refl_callable(x):
        if isinstance(x, str):
            return refl_lm_obj(prompt=x)
        return refl_lm_obj(messages=x)

    base_adapter = _PatchedDspyAdapter(
        student_module=program,
        metric_fn=substrate.metric_fn,
        feedback_map=_make_feedback_map(substrate),
        failure_score=0.0,
        # Per-substrate parallelism (see run_gepa.Substrate docstring).
        num_threads=substrate.num_threads,
        add_format_failure_as_feedback=True,
        rng=_random.Random(0),
        reflection_lm=refl_callable,
        warn_on_score_mismatch=False,
    )
    adapter = _RetryingAdapter(base_adapter)

    task_base = len(task_lm.history) if task_lm.history else 0
    refl_base = len(refl_lm_obj.history) if refl_lm_obj.history else 0

    eval_out = adapter.evaluate(list(test_examples), best_cand, capture_traces=False)
    avg_f1 = sum(eval_out.scores) / max(len(eval_out.scores), 1)

    cost = cost_from_lms(task_lm, refl_lm_obj, task_base, refl_base)
    return {
        "best_idx": int(best_idx),
        "best_val_f1": best_val_f1,
        "avg_test_f1": float(avg_f1),
        "n_test": len(eval_out.scores),
        "per_instance_f1": [float(s) for s in eval_out.scores],
        "cost_report": cost.as_dict(),
    }


def _git_head_safe() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# ============================================================================
# Pilot + projection
# ============================================================================


def run_pilot(pilot_n_iter: int = PILOT_N_ITER) -> dict:
    print()
    print("=" * 72)
    print(
        f"PHASE 1: PILOT  (static_frontier seed=0, N={pilot_n_iter}, "
        f"full pool sizes, variant={CURRENT_VARIANT_NAME!r})"
    )
    print("=" * 72)
    print()
    pilot_dir = LOGS_ROOT / f"{PILOT_ARM}_{PILOT_SEED}_pilot"
    summary = _run_pilot_into_dir(pilot_dir, pilot_n_iter=pilot_n_iter)
    print()
    print(f"Pilot wall time:       {summary['wall_time_s']:.1f}s")
    print(f"Pilot total_num_evals: {summary['total_num_evals']}")
    print(f"Pilot iters completed: {summary['state_i_final'] + 1} (target {summary['n_iter_target']})")
    print(f"Pilot cost (USD):      ${summary['cost']['cost_usd']:.4f}")
    if "drift" in summary:
        d = summary["drift"]
        print(f"Pilot drift Spearman:  {d['spearman_base_vs_final_temp0']:.4f}")
        print(
            f"Pilot frontier-leaving rate: {d['frontier_leaving_rate']:.4f}  "
            f"(base_frontier={d['n_base_frontier']}, "
            f"final_frontier={d['n_final_frontier']}, "
            f"left={d['n_left_frontier']})"
        )
    return summary


def _run_pilot_into_dir(pilot_dir: Path, pilot_n_iter: int = PILOT_N_ITER) -> dict:
    """Run a pilot cell into a sandboxed directory so it doesn't corrupt the
    eventual matrix cell's run_dir. Pilot N may be overridden by the caller
    (Chunk 14 uses 5)."""
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")

    import dspy
    import yaml

    from src.cost_tracker import cost_from_lms
    from src.difficulty import DifficultyTable
    from src.run_gepa import load_lm_configs_from_env, run

    config = yaml.safe_load((REPO / "config" / "experiment.yaml").read_text())
    config = copy.deepcopy(config)
    config["stopping"]["n"] = int(pilot_n_iter)

    splits = _load_splits_for_current_variant(config)
    substrate = _load_substrate_for_current_variant()
    difficulty_table = DifficultyTable.load(CURRENT_DIFFICULTY_PATH)
    task_lm_config, refl_lm_config = load_lm_configs_from_env(config=config)

    pilot_dir.mkdir(parents=True, exist_ok=True)
    lm_capture: dict[str, Any] = {}
    t0 = time.time()
    state = run(
        arm=PILOT_ARM,
        seed=PILOT_SEED,
        splits=splits,
        difficulty_table=difficulty_table,
        run_dir=pilot_dir,
        task_lm_config=task_lm_config,
        reflection_lm_config=refl_lm_config,
        config=config,
        raise_on_exception=True,
        lm_capture=lm_capture,
        mix=CURRENT_MIX,
        substrate=substrate,
    )
    wall = time.time() - t0
    task_lm = lm_capture.get("task_lm")
    refl_lm = lm_capture.get("refl_lm")
    cost_iter = cost_from_lms(task_lm, refl_lm)

    # Held-out test eval
    test_summary = _eval_best_on_test(
        state, splits[3], task_lm_config, refl_lm_config, substrate
    )
    (pilot_dir / "test_eval.json").write_text(json.dumps(test_summary, indent=2))
    test_cost_dict = test_summary["cost_report"]

    # D4 drift instrumentation (if the variant enables it).
    drift_summary: dict[str, Any] | None = None
    drift_cost = 0.0
    drift_wall = 0.0
    if CURRENT_DRIFT_ENABLED:
        from src.ifbench_drift import compute_drift_for_cell
        best_idx = test_summary["best_idx"]
        final_candidate = state.program_candidates[best_idx]
        base_candidate = state.program_candidates[0]
        t_drift = time.time()
        drift_summary = compute_drift_for_cell(
            base_seed_candidate=base_candidate,
            final_candidate=final_candidate,
            d_feedback=list(splits[0]),
            task_lm_config=task_lm_config,
            substrate=substrate,
            drift_cache_dir=pilot_dir / "_drift",
        )
        drift_wall = time.time() - t_drift
        (pilot_dir / "drift.json").write_text(json.dumps(drift_summary, indent=2))

    total_cost = cost_iter.cost_usd + float(test_cost_dict["cost_usd"]) + drift_cost
    summary = {
        "arm": PILOT_ARM,
        "seed": PILOT_SEED,
        "n_iter_target": int(pilot_n_iter),
        "state_i_final": int(state.i),
        "total_num_evals": int(state.total_num_evals),
        "wall_time_s": round(wall, 2),
        "test_eval_wall_time_s": round(test_summary.get("wall_time_s", 0.0), 2)
            if "wall_time_s" in test_summary else None,
        "drift_wall_time_s": round(drift_wall, 2) if CURRENT_DRIFT_ENABLED else None,
        "cost": {
            "iter_cost_usd": round(cost_iter.cost_usd, 6),
            "test_cost_usd": round(float(test_cost_dict["cost_usd"]), 6),
            "cost_usd": round(total_cost, 6),
        },
        "iter_cost_per_iter_usd": round(cost_iter.cost_usd / max(pilot_n_iter, 1), 6),
        "test_eval": {
            "best_idx": test_summary["best_idx"],
            "best_val_f1": test_summary["best_val_f1"],
            "avg_test_f1": test_summary["avg_test_f1"],
            "n_test": test_summary["n_test"],
        },
        "variant": CURRENT_VARIANT_NAME,
        "substrate": CURRENT_SUBSTRATE_NAME,
        "matrix_n_iter": CURRENT_N_ITER,
        "matrix_n_cells": len(ARMS) * len(SEEDS),
    }
    if drift_summary is not None:
        summary["drift"] = {
            "spearman_base_vs_final_temp0": drift_summary["spearman_base_vs_final_temp0"],
            "frontier_leaving_rate": drift_summary["frontier_leaving"]["frontier_leaving_rate"],
            "n_base_frontier": drift_summary["frontier_leaving"]["n_base_frontier"],
            "n_final_frontier": drift_summary["frontier_leaving"]["n_final_frontier"],
            "n_left_frontier": drift_summary["frontier_leaving"]["n_left_frontier"],
            "base_binning": drift_summary["frontier_leaving"]["base_binning"],
            "final_binning": drift_summary["frontier_leaving"]["final_binning"],
            "wall_s_base": drift_summary["wall_s_base"],
            "wall_s_final": drift_summary["wall_s_final"],
        }
    (pilot_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def project_matrix_cost(pilot: dict) -> dict:
    per_iter = float(pilot["iter_cost_per_iter_usd"])
    test_cost = float(pilot["cost"]["test_cost_usd"])
    n_cells = len(ARMS) * len(SEEDS)
    n_iter = CURRENT_N_ITER
    projected_iter = n_cells * n_iter * per_iter
    projected_test = n_cells * test_cost
    projected_total = projected_iter + projected_test
    out = {
        "per_iter_usd": per_iter,
        "test_cost_usd": test_cost,
        "n_cells": n_cells,
        "n_iter": n_iter,
        "projected_iter_total_usd": projected_iter,
        "projected_test_total_usd": projected_test,
        "projected_total_usd": projected_total,
    }
    # Drift-enabled variants project the temp-0 base + per-cell final
    # passes as a separate line item (one base pass shared, N cells of
    # final passes). Cost per pass is taken from the pilot's drift
    # final pass: it touched 150 examples at temp 0 just like the
    # matrix cells will.
    if "drift" in pilot:
        # Estimate per-pass cost as (pilot.drift_wall / pilot.wall) * pilot.iter_cost,
        # taking the pilot's mean per-call cost as a proxy. The
        # measured pilot cost reflects the ACTUAL temp-0 pass cost in
        # the lm_capture, but is collapsed into iter_cost_usd above
        # because both share the same task_lm instance. We expose the
        # walltime-derived projection separately and let the operator
        # judge.
        out["matrix_n_drift_passes"] = n_cells + 1  # +1 for shared base
    return out


def check_gate(projection: dict) -> bool:
    print()
    print("=" * 72)
    print("PHASE 2: COST GATE  (projection)")
    print("=" * 72)
    print()
    print(f"  per-iter cost (pilot):        ${projection['per_iter_usd']:.4f}")
    print(f"  test-eval cost (300 examples): ${projection['test_cost_usd']:.4f}")
    print(f"  cells x iters: {projection['n_cells']} x {projection['n_iter']}")
    print(f"  projected matrix iter total:  ${projection['projected_iter_total_usd']:.2f}")
    print(f"  projected matrix test total:  ${projection['projected_test_total_usd']:.2f}")
    print(f"  PROJECTED TOTAL:               ${projection['projected_total_usd']:.2f}")
    print(f"  gate (auto-launch if under):   ${PROJECTION_GATE_USD:.2f}")
    if projection["projected_total_usd"] >= PROJECTION_GATE_USD:
        print()
        print(f"  GATE FAILED: projected ${projection['projected_total_usd']:.2f} "
              f">= ${PROJECTION_GATE_USD:.2f}. Stopping.")
        return False
    print()
    print(f"  GATE PASSED: under ${PROJECTION_GATE_USD:.2f}. Continuing to matrix.")
    return True


# ============================================================================
# Matrix
# ============================================================================


def run_matrix() -> dict:
    print()
    print("=" * 72)
    print("PHASE 3: MATRIX  (15 cells = 5 arms x 3 seeds, serial)")
    print("=" * 72)
    print()

    cells = [(arm, seed) for arm in ARMS for seed in SEEDS]
    cumulative_cost = 0.0
    cumulative_wall = 0.0
    done: list[dict] = []
    errored: list[tuple[str, int, str]] = []
    skipped: list[tuple[str, int]] = []

    for arm, seed in cells:
        cell_label = f"{arm}_seed{seed}"
        # Skip if completed
        run_dir = LOGS_ROOT / f"{arm}_{seed}"
        if (run_dir / "cell_summary.json").exists():
            print(f"[skip] {cell_label}: cell_summary.json already exists")
            existing = json.loads((run_dir / "cell_summary.json").read_text())
            done.append(existing)
            cumulative_cost += float(existing["cost"]["cost_usd"])
            cumulative_wall += float(existing["wall_time_s"])
            skipped.append((arm, seed))
            continue

        # Kill switch check BEFORE launching
        if cumulative_cost >= HARD_KILL_USD:
            print()
            print(f"!!! KILL SWITCH: cumulative ${cumulative_cost:.2f} >= "
                  f"${HARD_KILL_USD:.2f}. Not launching {cell_label}.")
            break

        print()
        print("-" * 72)
        print(f"Launching {cell_label}  (cumulative so far: ${cumulative_cost:.2f})")
        print("-" * 72)
        try:
            cell_summary = _run_single_cell(arm, seed)
        except Exception as exc:
            print(f"[error] {cell_label}: {type(exc).__name__}: {exc}")
            errored.append((arm, seed, f"{type(exc).__name__}: {exc}"))
            continue
        done.append(cell_summary)
        cumulative_cost += float(cell_summary["cost"]["cost_usd"])
        cumulative_wall += float(cell_summary["wall_time_s"])

    summary = {
        "n_completed": len(done),
        "n_total": len(cells),
        "n_skipped_already_done": len(skipped),
        "errored": errored,
        "cumulative_cost_usd": round(cumulative_cost, 4),
        "cumulative_wall_time_s": round(cumulative_wall, 2),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (LOGS_ROOT / "matrix_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 72)
    print("PHASE 4: SUMMARY")
    print("=" * 72)
    print(f"  completed: {len(done)}/{len(cells)}  (skipped-already-done: {len(skipped)})")
    print(f"  errored:   {len(errored)}")
    print(f"  cumulative cost: ${cumulative_cost:.2f}")
    print(f"  cumulative wall: {cumulative_wall / 60:.1f} min")
    if errored:
        print(f"  cells with errors (re-run to resume):")
        for arm, seed, msg in errored:
            print(f"    {arm}_seed{seed}: {msg}")
    print()
    return summary


# ============================================================================
# Main / CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk 6 / 9 matrix orchestrator")
    parser.add_argument("phase", choices=("pilot", "matrix", "all", "cell"))
    parser.add_argument(
        "--variant",
        type=str,
        default="default",
        choices=sorted(VARIANTS),
        help="Which variant to run. 'default' = Experiment 1 (Chunk 6, "
             "70/15/15, 5 arms, results/logs/). 'hotpot_100' = Experiment "
             "1b (Chunk 9, 100/0/0, 3 static arms, results/logs_hotpot_100/).",
    )
    parser.add_argument("--arm", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-iter", type=int, default=None,
                        help="Override config N (for the `cell` subcommand)")
    parser.add_argument(
        "--pilot-n-iter",
        type=int,
        default=PILOT_N_ITER,
        help="Pilot iteration count override (default 4). Chunk-14 uses 5.",
    )
    args = parser.parse_args()

    _set_variant(args.variant)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"variant: {CURRENT_VARIANT_NAME!r} "
        f"(sampler={SAMPLER_NAME}, arms={list(ARMS)}, "
        f"logs_root={LOGS_ROOT.relative_to(REPO)})"
    )

    if args.phase == "cell":
        if args.arm is None or args.seed is None:
            print("ERROR: --arm and --seed required for `cell`")
            return 1
        _run_single_cell(args.arm, args.seed, n_iter_override=args.n_iter)
        return 0

    if args.phase in ("pilot", "all"):
        pilot = run_pilot(pilot_n_iter=int(args.pilot_n_iter))
        projection = project_matrix_cost(pilot)
        gate_passed = check_gate(projection)
        if args.phase == "pilot":
            return 0 if gate_passed else 2
        if not gate_passed:
            return 2

    if args.phase in ("matrix", "all"):
        summary = run_matrix()
        # rc reflects whether anything errored
        return 0 if not summary["errored"] else 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
