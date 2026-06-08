# ARCHITECTURE.md — gepa source recon and the Chunk 4 hook plan

Recon target: `gepa==0.0.21` (the version `uv pip install` resolved on 2026-05-30, installed at
`.venv/lib/python3.10/site-packages/gepa/`). All file:line citations below are into that tree.

## 0. Correction to the handoff

`dspy.GEPA` does not exist in `dspy==2.6.27`. The entire GEPA implementation lives in the
standalone `gepa` package. The DSPy integration is via `gepa.adapters.dspy_adapter`. Wherever
the handoff says "dspy.GEPA," substitute `gepa.optimize()` or, where we need finer control
(Chunk 4), `gepa.core.engine.GEPAEngine` constructed directly.

## 1. Source map (what lives where)

| Concern from the handoff | Class / function | File |
|---|---|---|
| Top-level entry point | `optimize()` | `gepa/api.py` |
| Main optimization loop | `GEPAEngine.run()` | `gepa/core/engine.py` |
| Reflection minibatch sampling (Algorithm 1 line 9) | `BatchSampler` Protocol; `EpochShuffledBatchSampler` | `gepa/strategies/batch_sampler.py` |
| Reflective mutation step (eval parent + propose + eval new) | `ReflectiveMutationProposer.propose()` | `gepa/proposer/reflective_mutation/reflective_mutation.py` |
| Acceptance test (Algorithm 1 lines 13-14) | inline in `GEPAEngine.run()` | `gepa/core/engine.py:538-560` |
| D_pareto re-scoring on accept (Algorithm 1 lines 16-18) | `_run_full_eval_and_add()` -> `_evaluate_on_valset()` | `gepa/core/engine.py:146,125` |
| Per-instance D_pareto scores live here | `state.prog_candidate_val_subscores: list[dict[DataId, float]]` | `gepa/core/state.py:196` |
| Per-candidate aggregate score | `state.program_full_scores_val_set: list[float]` | referenced from `ScoreThresholdStopper`, populated inside `update_state_with_new_program` |
| Pareto candidate selection (Algorithm 2) | `ParetoCandidateSelector` | `gepa/strategies/candidate_selector.py` |
| Round-robin module selection | `RoundRobinReflectionComponentSelector` | `gepa/strategies/component_selector.py` |
| Stopping rule abstraction | `StopperProtocol`; `MaxCandidateProposalsStopper` | `gepa/utils/stop_condition.py` |
| Iteration counter | `state.i` (-1 initial; incremented at top of each loop iter, `engine.py:440`) | `gepa/core/state.py:236` |
| Adapter contract | `GEPAAdapter` Protocol with `evaluate(...)`, `make_reflective_dataset(...)` | `gepa/core/adapter.py` |
| Data loading | `DataLoader` Protocol; lists get wrapped in `ListDataLoader` (DataId = int = list index) | `gepa/core/data_loader.py` |

## 2. The two override questions, answered

### Q1: Can the D_feedback reflection sampler be overridden via a hook or subclass?

**Yes, trivially.** `BatchSampler` is a single-method `Protocol`:

```python
# gepa/strategies/batch_sampler.py:13-14
class BatchSampler(Protocol[DataId, DataInst]):
    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]: ...
```

`optimize()` accepts a `BatchSampler` instance via the `batch_sampler=` parameter
(`api.py:55`). Implementing our own class that ignores the default epoch-shuffle logic and
returns a 70/15/15 draw from the precomputed tercile map is a ~30-line file. **No subclass
needed**, just implement the Protocol.

### Q2: Can the accept test be split off the minibatch onto the fixed batch A?

**Yes, but not through a callback — through a `ReflectiveMutationProposer` subclass plus
bypassing `gepa.optimize()` to construct `GEPAEngine` directly.**

The accept test is hardcoded in the engine:

```python
# gepa/core/engine.py:539-541
old_sum = sum(proposal.subsample_scores_before or [])
new_sum = sum(proposal.subsample_scores_after or [])
if new_sum <= old_sum:
    ... # reject
```

There is no callback hook for this decision. Callbacks (`on_candidate_rejected`,
`on_candidate_accepted`) only observe; they cannot veto. So we cannot keep the engine
unchanged AND change what acceptance is computed on, unless we change what scores the
proposer hands the engine.

But the engine only inspects `proposal.subsample_scores_before` and
`proposal.subsample_scores_after`. These are populated by
`ReflectiveMutationProposer.propose()` from the size-3 minibatch (`reflective_mutation.py:382-383`).
**If we subclass the proposer and overwrite these two fields with the parent's and new
candidate's scores on the fixed batch A, the engine will compute strict improvement on A
without modification.** The accept rule is already `>` (strict, not `>=`), so it matches
the spec out of the box.

One snag: `gepa.optimize()` instantiates `ReflectiveMutationProposer` directly
(`api.py:354-368`) and does not expose a way to inject a custom proposer. To use our
subclass we must skip `optimize()` and construct `GEPAEngine` ourselves with the same
plumbing `api.py` does — adapter, reflection_lm, candidate_selector, etc. This is
"bypass the convenience entry point," not "fork the loop." The loop body in
`engine.run()` is untouched.

**Verdict: subclass route works for both. No fork required.**

## 3. Stop condition

`MaxCandidateProposalsStopper(max_proposals=N)` already exists
(`gepa/utils/stop_condition.py:176-190`) and is exactly the N-iteration stop the handoff
specifies. It returns True when `state.i >= N - 1` (the counter increments at the start
of each iteration), giving exactly N attempted iterations. Use `N = 44`.

Important: an "iteration" here counts every loop body, including iterations where the
proposer returns `None` (no proposal, e.g. all minibatch scores already perfect). That
matches the spec's "fix iteration count, not budget" framing — fix the number of
attempts, not the number of accepts. Merge will be disabled (`use_merge=False`), so
every iteration is a reflective attempt.

## 4. Chunk 4 hook plan (concrete)

### 4.1 `src/band_sampler.py` — band-aware reflection sampler

Implements `gepa.strategies.batch_sampler.BatchSampler`. Constructor takes:
- `terciles: dict[DataId, Literal["easy","mid","hard"]]` — the frozen difficulty map
  computed once in Chunk 5 over D_feedback (DataId = int = D_feedback list index).
- `target_band: Literal["easy","mid","hard","random"]` — which band this arm targets.
- `mix: tuple[float,float,float] = (0.70, 0.15, 0.15)` — target / off1 / off2.
- `b: int = 3` — minibatch size.
- `rng: random.Random` — seeded once per (arm, seed) pair.

`next_minibatch_ids(loader, state)`:
- If `target_band == "random"`, return `rng.sample(loader.all_ids(), b)`.
- Else: build the three id pools from `terciles`. Draw `round(0.70*b)`, `round(0.15*b)`,
  `round(0.15*b)` from each respectively (with `b=3` that resolves to 2/1/0 or similar
  small-integer rounding; lock the exact rounding rule deterministically before Chunk 4).

Notes:
- `state.i` is irrelevant for this sampler. Each call draws fresh; we are not doing
  epoch shuffling.
- Sampling with replacement vs. without: keep within-call draws **without replacement**
  but allow across-call repeats (the band has only ~33 ids and the run does 44 iters x 3
  draws = 132 reflection-instance touches, so we will revisit instances).

### 4.2 `src/decoupled_proposer.py` — A-decoupled reflective mutation

Subclass `gepa.proposer.reflective_mutation.reflective_mutation.ReflectiveMutationProposer`.
Override `propose(self, state) -> CandidateProposal | None`.

The override mostly delegates to the existing flow but replaces what goes into
`subsample_scores_before` and `subsample_scores_after`:

1. Reuse the parent class's flow for steps 1-3 (select candidate; sample minibatch via
   `self.batch_sampler`; eval parent on minibatch *with traces*; skip-perfect check;
   build reflective dataset; propose new texts). The minibatch eval is the source of
   the reflection signal and is needed regardless.
2. After `new_candidate` is constructed:
   - Look up `accept_score_cache[curr_prog_id]`. If missing, evaluate the parent on the
     fixed accept batch A via `self.adapter.evaluate(self.accept_batch, curr_prog, capture_traces=False)`,
     cache the per-example scores, and call `state.increment_evals(len(A))`.
   - Evaluate the new candidate on A: `self.adapter.evaluate(self.accept_batch, new_candidate, capture_traces=False)`.
     Call `state.increment_evals(len(A))`.
3. Return `CandidateProposal(candidate=new_candidate, parent_program_ids=[curr_prog_id], subsample_indices=self.accept_batch_ids, subsample_scores_before=parent_A_scores, subsample_scores_after=new_A_scores, tag="reflective_mutation_decoupled")`.

The engine's accept test then runs on A. On accept, the engine does the standard D_pareto
re-eval via `_run_full_eval_and_add()`, unchanged.

Constructor adds two args beyond the parent:
- `accept_batch: list[DataInst]` — the 20 fixed examples.
- `accept_batch_ids: list[DataId]` — for the `CandidateProposal.subsample_indices` field
  (informational, not consumed by the accept test).

Internal state:
- `self.accept_score_cache: dict[int, list[float]]` keyed by parent `program_idx`.

### 4.3 `src/run_gepa.py` — engine-direct runner

Replicates the construction in `gepa/api.py:182-409` (~70 lines) minus the
`optimize()` wrapper. Per-arm differences:

- **`random`, `static_easy`, `static_frontier`, `static_hard`**: use `BandBatchSampler`
  with the right `target_band`. Use `DecoupledReflectiveMutationProposer`. Acceptance is
  on A, on D_pareto re-eval on accept (engine default).
- **`vanilla_coupled_gepa` (reference)**: use stock `EpochShuffledBatchSampler` and stock
  `ReflectiveMutationProposer`. Acceptance is on the b=3 minibatch (engine's default
  behavior, unmodified).

All arms: `use_merge=False`, `candidate_selection_strategy="pareto"`,
`module_selector="round_robin"`, `frontier_type="instance"`, `skip_perfect_score=True`,
`stop_callbacks=[MaxCandidateProposalsStopper(44)]`, `valset=D_pareto`, `trainset=D_feedback`
(both as plain lists; `ensure_loader` wraps them into `ListDataLoader` so `DataId = int`).

Pair seeds across arms: use the same `seed` value for all arms in a given replicate so
`ParetoCandidateSelector`'s RNG, the round-robin module schedule, and the seed-candidate
init are identical across arms; trajectories diverge only after the sampling differs.

### 4.4 Why this is not a fork

We construct `GEPAEngine` directly with our own `ReflectiveMutationProposer` subclass and
our own `BatchSampler`. The engine code (`run()` loop, accept-test if-statement, D_pareto
re-eval, Pareto front maintenance, callback firing, state save/load) is the upstream code
verbatim. The "loop" is one method (`engine.run()`, 700-line file but `~250` lines of
loop body). We touch zero lines of it.

## 5. Risks and gotchas for Chunk 4

1. **Metric-call accounting**: every `adapter.evaluate(...)` we call from the subclassed
   proposer must be matched by `state.increment_evals(len(batch))`, or else the budget
   stopper and on-budget-updated callback will under-count. (We are using
   `MaxCandidateProposalsStopper`, not `MaxMetricCallsStopper`, so this is a
   reporting/logging concern, not a correctness one for the stop rule.)

2. **Cache for parent's A-score**: keyed by `program_idx` (int). It must not be keyed by
   candidate-text hash, because `ParetoCandidateSelector` may pick the same text via two
   different program indices in theory. Using `program_idx` matches how the engine
   identifies the "parent program."

3. **`_run_full_eval_and_add` already re-evaluates the accepted candidate on the valset**
   (`engine.py:153`). That is our D_pareto re-eval; do not double-do it from the
   proposer.

4. **`skip_perfect_score=True` behavior**: if all 3 minibatch scores are at
   `perfect_score`, the proposer returns `None` without proposing
   (`reflective_mutation.py:238-255`). With `perfect_score=1.0` and HotpotQA F1 in
   [0,1], a frontier-band minibatch will almost never hit all-1.0, so this fires rarely.
   But for the `static_easy` arm it can fire more often. Acceptable: the iteration still
   counts toward `N=44` per `MaxCandidateProposalsStopper`. Confirm in the smoke test.

5. **`reflection_minibatch_size`** in `optimize()` is only consulted when
   `batch_sampler="epoch_shuffled"` (the string). When we pass a custom `BatchSampler`
   instance, the size is whatever the sampler returns. Set `b=3` inside
   `BandBatchSampler` and inside the `EpochShuffledBatchSampler` we use for the
   reference arm.

6. **`raise_on_exception=True`** (default in `optimize()`). Our `run_gepa.py` should
   keep this on during dev so transient errors are not silently swallowed; the runner
   in Chunk 6 may flip it off for the matrix.

7. **State persistence**: `GEPAEngine` writes `gepa_state.bin` etc. to `run_dir`. The
   `accept_score_cache` in our subclassed proposer is **not** in `state.__dict__`, so
   it is not persisted. For a 44-iteration run this is fine (caches rebuild from
   `program_candidates` re-evaluation on resume), but if Chunk 6 wants resumability,
   the cache should be reconstructable from a re-eval of all known parents on A at
   resume time, or move it into a custom field on `state`.

## 6. Decision: GO on subclass + engine-direct construction

Both override questions answered Yes. No engine fork. Chunk 4's work surface is:
- one `BatchSampler` implementation (~30 LOC),
- one `ReflectiveMutationProposer` subclass overriding `propose()` (~80 LOC),
- one engine-direct runner (~120 LOC, mostly mechanical replication of `api.py`).

Cost ledger from the handoff (per-iter ≈ 3 + 20 + 0.3*75 = 45.5 rollouts) holds, since:
- the b=3 minibatch eval still happens (reflection signal),
- the parent's A eval is paid once per parent (amortizes to ~0 per iter for a long-lived
  parent),
- the new candidate's A eval is paid every iter (20),
- D_pareto eval is paid only on accept (≈ 0.3 * 75 = 22.5).
