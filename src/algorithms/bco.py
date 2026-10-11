"""
Bee Colony Optimisation for RCPSP.

Variant chosen
--------------
Artificial Bee Colony (ABC) adapted to random-key RCPSP representations,
following the structure described in:

  • Karaboga & Basturk (2007) — canonical ABC.
  • Sonmez (2013) — RCPSP bee-algorithm comparison (ScienceDirect).
  • Küçükoğlu et al. (2026) — random-key ABC with serial SGS (Springer).

Engineering decisions vs. paper requirements are documented in
docs/ALGORITHM_NOTES.md.

Representation : random-key vector — one float ∈ [0, 1] per activity.
Decoding       : common SSGS via src.evaluation.Evaluator
Objective      : minimise makespan.

ABC roles
---------
Employed bees  — search neighbourhood of each food source (candidate).
Onlooker bees  — preferentially exploit better food sources (makespan-biased).
Scout bees     — abandon exhausted sources; replace with random exploration.
"""

import random
import time
from typing import List, Tuple

from src.evaluation import Evaluator
from src.model import AlgorithmResult, Project, ScheduleEntry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _random_source(n: int, rng: random.Random) -> List[float]:
    return [rng.random() for _ in range(n)]


def _perturb_source(
    source: List[float],
    partner: List[float],
    rng: random.Random,
) -> List[float]:
    """
    ABC-style perturbation:
        v_i = x_i + φ · (x_i − partner_i)
    where φ ∈ [−1, 1].  Clamp result to [0, 1].
    """
    n = len(source)
    idx = rng.randrange(n)
    phi = rng.uniform(-1.0, 1.0)
    v = source[:]
    v[idx] = max(0.0, min(1.0, v[idx] + phi * (v[idx] - partner[idx])))
    return v


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def solve(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
    colony_size: int = 20,
    limit: int = 15,
) -> AlgorithmResult:
    """
    Run the ABC-style Bee Colony Optimisation on *project*.

    Parameters
    ----------
    evaluation_budget : int
        Maximum number of SSGS evaluations.
    seed : int
        Controls all randomness.
    colony_size : int
        Number of food sources (employed bees = colony_size).
        Onlooker bees also = colony_size per cycle.
    limit : int
        Maximum trials before a food source is abandoned (scout trigger).
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()

    sorted_ids = sorted(project.activities.keys())
    n = len(sorted_ids)
    evaluations = 0
    convergence_history: List[Tuple[int, int]] = []

    # ---- Initialise food sources ----
    sources: List[List[float]] = [_random_source(n, rng) for _ in range(colony_size)]
    fitnesses: List[int] = []
    schedules: List[List[ScheduleEntry]] = []
    trials: List[int] = [0] * colony_size  # trial counters

    for src in sources:
        _, ms, sched = Evaluator.evaluate(project, src)
        fitnesses.append(ms)
        schedules.append(sched)
        evaluations += 1

    best_idx = min(range(colony_size), key=lambda i: fitnesses[i])
    best_ms = fitnesses[best_idx]
    best_vec = sources[best_idx][:]
    best_sched = schedules[best_idx]
    convergence_history.append((evaluations, best_ms))

    # ---- Main cycle ----
    while evaluations < evaluation_budget:

        # -- Employed bees phase --
        for i in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            # Pick a random partner (different from i)
            partner_idx = rng.choice([j for j in range(colony_size) if j != i])
            candidate = _perturb_source(sources[i], sources[partner_idx], rng)

            _, cand_ms, cand_sched = Evaluator.evaluate(project, candidate)
            evaluations += 1

            if cand_ms <= fitnesses[i]:  # greedy acceptance
                sources[i] = candidate
                fitnesses[i] = cand_ms
                schedules[i] = cand_sched
                trials[i] = 0
            else:
                trials[i] += 1

            if fitnesses[i] < best_ms:
                best_ms = fitnesses[i]
                best_vec = sources[i][:]
                best_sched = schedules[i]
                convergence_history.append((evaluations, best_ms))

        # -- Onlooker bees phase --
        # Selection probability inversely proportional to makespan
        max_ms = max(fitnesses) if fitnesses else 1
        weights = [(max_ms - f + 1) for f in fitnesses]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        def weighted_choice() -> int:
            r = rng.random()
            cumulative = 0.0
            for idx, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    return idx
            return colony_size - 1

        for _ in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            i = weighted_choice()
            partner_idx = rng.choice([j for j in range(colony_size) if j != i])
            candidate = _perturb_source(sources[i], sources[partner_idx], rng)

            _, cand_ms, cand_sched = Evaluator.evaluate(project, candidate)
            evaluations += 1

            if cand_ms <= fitnesses[i]:
                sources[i] = candidate
                fitnesses[i] = cand_ms
                schedules[i] = cand_sched
                trials[i] = 0
            else:
                trials[i] += 1

            if fitnesses[i] < best_ms:
                best_ms = fitnesses[i]
                best_vec = sources[i][:]
                best_sched = schedules[i]
                convergence_history.append((evaluations, best_ms))

        # -- Scout bees phase --
        for i in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            if trials[i] >= limit:
                sources[i] = _random_source(n, rng)
                _, ms, sched = Evaluator.evaluate(project, sources[i])
                fitnesses[i] = ms
                schedules[i] = sched
                trials[i] = 0
                evaluations += 1

                if ms < best_ms:
                    best_ms = ms
                    best_vec = sources[i][:]
                    best_sched = sched
                    convergence_history.append((evaluations, best_ms))

    runtime = time.perf_counter() - start_ts

    return AlgorithmResult(
        algorithm="BCO",
        best_priority_vector=best_vec,
        schedule=best_sched,
        makespan=best_ms,
        runtime=runtime,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )
