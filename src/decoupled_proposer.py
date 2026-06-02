"""A-batch acceptance for GEPA's reflective mutation.

Subclasses ``gepa.proposer.reflective_mutation.reflective_mutation.ReflectiveMutationProposer``
and overrides ``propose()`` to swap the parent-on-minibatch / new-on-minibatch
scores (stock GEPA's acceptance signal) for parent-on-A / new-on-A scores
(this experiment's decoupled acceptance per Section 4 of the handoff).

The engine's accept test is hardcoded at ``gepa/core/engine.py:539-541`` as a
strict `>` on ``sum(proposal.subsample_scores_after) > sum(...before)``. We
get the decoupling for free by stuffing A-batch scores into those two fields;
the engine's accept logic compares strict `>` on A automatically. The accept
rule is already `>` (strict, not `>=`) so it matches the spec verbatim.

See ARCHITECTURE.md §2 Q2 and §4.2.

§5 risk handling:
  1. metric_call accounting: every adapter.evaluate(...) call is matched by
     state.increment_evals(len(batch)). Both the parent-on-A eval (cached) and
     the new-on-A eval are accounted; the parent-on-minibatch eval is
     accounted by parent (we still call adapter.evaluate for it).
  2. Cache for parent A-score: keyed by curr_prog_id (int), NOT by
     candidate-text hash. Matches gepa's notion of "the parent program."
  3. Engine's _run_full_eval_and_add already does the D_pareto re-eval; we
     do not duplicate that here.
  4. skip_perfect_score: preserved verbatim from parent; an all-perfect
     minibatch returns None (the iteration still counts toward N).
  5. b is whatever the BandBatchSampler returns (3).
  6. raise_on_exception: untouched; engine's behavior is unchanged.
  7. accept_score_cache rebuild on resume: lazy. The cache is empty at start
     and populates on first selection of each parent. State.program_candidates
     survives resume; the first iteration after resume that picks a known
     parent will pay the 20-LM-call re-evaluation once and cache it.

Implementation note: we deliberately reimplement parent.propose() rather than
calling super().propose() and post-processing. super() would do a third
adapter.evaluate (new candidate on the b=3 minibatch) whose scores we'd
overwrite anyway, wasting ~3 LM calls per iteration. The duplication is
mechanical and clearly tagged with "parent steps 1-3 verbatim" comments.
"""

from __future__ import annotations

import traceback
from collections.abc import Sequence
from typing import Any

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.callbacks import (
    CandidateSelectedEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    MinibatchSampledEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.reflective_mutation import (
    ReflectiveMutationProposer,
)


class DecoupledReflectiveMutationProposer(ReflectiveMutationProposer):
    """Acceptance is on the fixed batch A, not on the reflection minibatch.

    `accept_batch` is the list of A=20 DataInst examples; `accept_batch_ids`
    is the parallel list of ids (informational; carried in subsample_indices
    for logging only -- the engine's accept test does not consume the ids).
    """

    def __init__(
        self,
        *args: Any,
        accept_batch: list[DataInst],
        accept_batch_ids: list[DataId],
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if not accept_batch:
            raise ValueError("accept_batch must be non-empty.")
        if len(accept_batch) != len(accept_batch_ids):
            raise ValueError(
                f"accept_batch (n={len(accept_batch)}) and accept_batch_ids "
                f"(n={len(accept_batch_ids)}) must have the same length."
            )
        self.accept_batch: list[DataInst] = list(accept_batch)
        self.accept_batch_ids: list[DataId] = list(accept_batch_ids)
        # Per §5 risk 2: keyed by program index (int), not candidate-text hash.
        self.accept_score_cache: dict[int, list[float]] = {}

    # ---- Public propose() ----

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        i = state.i + 1

        # ====== Parent steps 1-3 verbatim, with our band sampler ======
        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            f"Iteration {i}: Selected program {curr_prog_id} score: "
            f"{state.program_full_scores_val_set[curr_prog_id]}"
        )

        notify_callbacks(
            self.callbacks,
            "on_candidate_selected",
            CandidateSelectedEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                candidate=curr_prog,
                score=state.program_full_scores_val_set[curr_prog_id],
            ),
        )

        self.experiment_tracker.log_metrics(
            {
                "iteration": i,
                "selected_program_candidate": curr_prog_id,
                "total_metric_calls": state.total_num_evals,
            },
            step=i,
        )

        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = self.trainset.fetch(subsample_ids)

        notify_callbacks(
            self.callbacks,
            "on_minibatch_sampled",
            MinibatchSampledEvent(
                iteration=i,
                minibatch_ids=subsample_ids,
                trainset_size=len(self.trainset),
            ),
        )

        # Parent on minibatch (with traces, for reflection signal). The scores
        # from this eval are NOT used for acceptance in our override -- they
        # are only the input to the reflective dataset.
        curr_parent_ids = [
            p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None
        ]
        is_seed_candidate = curr_prog_id == 0
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                batch_size=len(minibatch),
                capture_traces=True,
                parent_ids=curr_parent_ids,
                inputs=minibatch,
                is_seed_candidate=is_seed_candidate,
            ),
        )
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        state.increment_evals(len(subsample_ids))  # §5 risk 1
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                scores=eval_curr.scores,
                has_trajectories=bool(eval_curr.trajectories),
                parent_ids=curr_parent_ids,
                outputs=eval_curr.outputs,
                trajectories=eval_curr.trajectories,
                objective_scores=eval_curr.objective_scores,
                is_seed_candidate=is_seed_candidate,
            ),
        )

        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    reason="no_trajectories",
                    scores=eval_curr.scores,
                    is_seed_candidate=is_seed_candidate,
                ),
            )
            return None

        if (
            self.skip_perfect_score
            and self.perfect_score is not None
            and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)
        ):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    reason="all_scores_perfect",
                    scores=eval_curr.scores,
                    is_seed_candidate=is_seed_candidate,
                ),
            )
            return None

        # Module selection (round-robin).
        predictor_names_to_update = self.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
        )

        # Build reflective dataset and propose new texts.
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(
                curr_prog, eval_curr, predictor_names_to_update
            )
            reflective_dataset_concrete: dict[str, list[dict[str, Any]]] = {
                k: [dict(item) for item in v] for k, v in reflective_dataset.items()
            }
            notify_callbacks(
                self.callbacks,
                "on_reflective_dataset_built",
                ReflectiveDatasetBuiltEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    components=predictor_names_to_update,
                    dataset=reflective_dataset_concrete,
                ),
            )
            notify_callbacks(
                self.callbacks,
                "on_proposal_start",
                ProposalStartEvent(
                    iteration=i,
                    parent_candidate=curr_prog,
                    components=predictor_names_to_update,
                    reflective_dataset=reflective_dataset_concrete,
                ),
            )
            new_texts = self.propose_new_texts(
                curr_prog, reflective_dataset, predictor_names_to_update
            )
            notify_callbacks(
                self.callbacks,
                "on_proposal_end",
                ProposalEndEvent(iteration=i, new_instructions=new_texts),
            )
            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            self.logger.log(traceback.format_exc())
            return None

        # ====== Step 4 (decoupled): acceptance on A, not on the minibatch ======
        new_candidate = curr_prog.copy()
        for pname, text in new_texts.items():
            assert pname in new_candidate, f"{pname} missing in candidate"
            new_candidate[pname] = text

        parent_a_scores = self._get_or_compute_parent_a_scores(state, curr_prog_id, curr_prog, i)
        new_a_scores = self._evaluate_on_accept_batch(state, new_candidate, i, candidate_idx=None)

        new_sum = sum(new_a_scores)
        old_sum = sum(parent_a_scores)
        state.full_program_trace[-1]["accept_batch_parent_score"] = old_sum
        state.full_program_trace[-1]["accept_batch_new_score"] = new_sum
        state.full_program_trace[-1]["new_subsample_scores"] = new_a_scores  # for engine compat
        self.experiment_tracker.log_metrics(
            {
                "accept_batch/parent": old_sum,
                "accept_batch/new": new_sum,
                "total_metric_calls": state.total_num_evals,
            },
            step=i,
        )

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=list(self.accept_batch_ids),
            subsample_scores_before=list(parent_a_scores),
            subsample_scores_after=list(new_a_scores),
            tag="reflective_mutation_decoupled",
        )

    # ---- A-batch helpers ----

    def _get_or_compute_parent_a_scores(
        self,
        state: GEPAState,
        program_idx: int,
        program: dict[str, str],
        iteration: int,
    ) -> list[float]:
        cached = self.accept_score_cache.get(program_idx)
        if cached is not None:
            return cached
        scores = self._evaluate_on_accept_batch(
            state, program, iteration, candidate_idx=program_idx
        )
        self.accept_score_cache[program_idx] = scores
        return scores

    def _evaluate_on_accept_batch(
        self,
        state: GEPAState,
        program: dict[str, str],
        iteration: int,
        candidate_idx: int | None,
    ) -> list[float]:
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=iteration,
                candidate_idx=candidate_idx,
                batch_size=len(self.accept_batch),
                capture_traces=False,
                parent_ids=[],
                inputs=self.accept_batch,
                is_seed_candidate=(candidate_idx == 0),
            ),
        )
        eval_out = self.adapter.evaluate(self.accept_batch, program, capture_traces=False)
        state.increment_evals(len(self.accept_batch))  # §5 risk 1
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=iteration,
                candidate_idx=candidate_idx,
                scores=eval_out.scores,
                has_trajectories=False,
                parent_ids=[],
                outputs=eval_out.outputs,
                trajectories=None,
                objective_scores=eval_out.objective_scores,
                is_seed_candidate=(candidate_idx == 0),
            ),
        )
        return list(eval_out.scores)
