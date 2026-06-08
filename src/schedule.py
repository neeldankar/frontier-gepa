"""Curriculum schedule for Exp 3: which difficulty bin feeds the reflection
minibatch at each iteration, and how the minibatch is drawn.

The schedule is a STATIC function of the iteration index only (hard constraint
#6): never adaptive, never tracking a "current learner". GEPA is a Pareto
population with stochastic parent selection, so there is no single learner to
track.

Arms (CLAUDE_CODE_BUILD_PROMPT_exp3_curriculum.md, "Arms"):
  - random        : ignore bins; uniform over all D_feedback ids (the honest
                    baseline). In the real run this arm uses native GEPA
                    sampling; ``bin_for_iteration`` returns None for it.
  - easy_to_hard  : phase 1 -> easy, phase 2 -> medium, phase 3 -> hard.
  - hard_to_easy  : phase 1 -> hard, phase 2 -> medium, phase 3 -> easy.
  - static_medium : every phase -> medium.

Phase schedule (default, "Schedule" section): phase boundaries at T/3 and
2T/3. For T=40 this is iters 0-13 (phase 1), 14-26 (phase 2), 27-39 (phase 3).

The phase mapping is kept as data (``_PHASE_SCHEDULES``) behind a thin
``bin_for_iteration`` so a soft-mixing variant can be swapped in later without
touching the runner: the runner only calls ``bin_for_iteration`` /
``sample_minibatch``.

ids are integer D_feedback list indices (gepa DataId; see src/bins.py).
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Sequence

from src.bins import Bins, BinName

logger = logging.getLogger(__name__)

Arm = str
ARMS: tuple[Arm, ...] = ("random", "easy_to_hard", "hard_to_easy", "static_medium")

# Per-arm (phase0, phase1, phase2) bin assignment. ``random`` is special-cased
# to None (ignore bins) and is not in this table.
_PHASE_SCHEDULES: dict[Arm, tuple[BinName, BinName, BinName]] = {
    "easy_to_hard": ("easy", "medium", "hard"),
    "hard_to_easy": ("hard", "medium", "easy"),
    "static_medium": ("medium", "medium", "medium"),
}

N_PHASES = 3


def phase_for_iteration(iteration: int, T: int) -> int:
    """Return the phase index (0, 1, or 2) for ``iteration`` under a budget of
    ``T`` iterations, with boundaries at T/3 and 2T/3.

    Uses float boundaries with strict less-than so that, e.g., T=40 gives
    iters 0-13 -> 0, 14-26 -> 1, 27-39 -> 2.
    """
    if T <= 0:
        raise ValueError(f"phase_for_iteration: T must be positive, got {T}")
    if not (0 <= iteration < T):
        raise ValueError(
            f"phase_for_iteration: iteration {iteration} out of range [0, {T})"
        )
    if iteration < T / 3:
        return 0
    if iteration < 2 * T / 3:
        return 1
    return 2


def bin_for_iteration(arm: Arm, iteration: int, T: int) -> Optional[BinName]:
    """Return the difficulty bin the reflection minibatch should be drawn from
    at ``iteration``, or None meaning "uniform over all ids" (the ``random``
    arm).

    Pure function of (arm, iteration, T). Raises on an unknown arm or an
    out-of-range iteration so a wiring bug surfaces here.
    """
    if arm not in ARMS:
        raise ValueError(f"bin_for_iteration: unknown arm {arm!r}; expected {ARMS}")
    if arm == "random":
        # Validate range even for random so callers get consistent bounds checks.
        phase_for_iteration(iteration, T)
        return None
    phase = phase_for_iteration(iteration, T)
    return _PHASE_SCHEDULES[arm][phase]


def sample_minibatch(
    arm: Arm,
    iteration: int,
    T: int,
    rng: random.Random,
    bins: Bins,
    all_ids: Sequence[int],
    b: int = 3,
) -> list[int]:
    """Draw a size-``b`` reflection minibatch of D_feedback ids for ``arm`` at
    ``iteration``.

    - ``random`` arm: uniform WITHOUT replacement over ``all_ids`` (b distinct
      ids, matching native GEPA minibatch semantics).
    - binned arms: uniform WITH replacement over the scheduled bin's members
      (the medium bin is small, so replacement is required). If the scheduled
      bin has fewer than ``b`` distinct members, still sample with replacement
      and log a warning; never silently fall back to another bin
      (hard constraint, "Bin definitions").

    All randomness comes from the passed ``rng`` (seeded); no global random
    state is touched.
    """
    if b <= 0:
        raise ValueError(f"sample_minibatch: b must be positive, got {b}")

    bin_name = bin_for_iteration(arm, iteration, T)

    if bin_name is None:
        pool = list(all_ids)
        if len(pool) == 0:
            raise ValueError("sample_minibatch: all_ids is empty")
        if len(pool) < b:
            raise ValueError(
                f"sample_minibatch: all_ids has {len(pool)} < b={b} ids; "
                "cannot draw a distinct random minibatch"
            )
        # Without replacement: b distinct ids.
        return rng.sample(pool, b)

    pool = bins.get_bin_members(bin_name)
    if len(pool) == 0:
        raise ValueError(
            f"sample_minibatch: scheduled bin {bin_name!r} is empty at "
            f"iter={iteration} (arm={arm}); cannot draw"
        )
    n_distinct = len(set(pool))
    if n_distinct < b:
        logger.warning(
            "sample_minibatch: scheduled bin %r has %d < b=%d distinct members "
            "(arm=%s, iter=%d); sampling with replacement.",
            bin_name,
            n_distinct,
            b,
            arm,
            iteration,
        )
    # With replacement: index uniformly each draw.
    return [pool[rng.randrange(len(pool))] for _ in range(b)]
