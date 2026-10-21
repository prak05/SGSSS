"""
Simulated Annealing for RCPSP.

Representation : random-key vector — one float ∈ [0, 1] per activity.
Decoding       : common SSGS via src.evaluation.Evaluator
Objective      : minimise makespan.

Reference studied (ideas only, no code copied):
  https://github.com/mbenahmed1/ScheduleMe
"""

import math
import random
import time
from typing import List, Tuple

from src.evaluation import Evaluator
from src.model import AlgorithmResult, Project, ScheduleEntry


# ---------------------------------------------------------------------------
# Neighbourhood operators
# ---------------------------------------------------------------------------

def _swap(vec: List[float], rng: random.Random) -> List[float]:
    """Swap two randomly chosen keys."""
    n = len(vec)
    if n < 2:
        return vec[:]
    i, j = rng.sample(range(n), 2)
    v = vec[:]
    v[i], v[j] = v[j], v[i]
    return v


def _perturb(vec: List[float], rng: random.Random, sigma: float = 0.1) -> List[float]:
    """Add Gaussian noise to one gene; clamp to [0, 1]."""
    v = vec[:]
    i = rng.randrange(len(v))
    v[i] = max(0.0, min(1.0, v[i] + rng.gauss(0, sigma)))
    return v


def _neighbour(vec: List[float], rng: random.Random) -> List[float]:
    """Randomly apply one neighbourhood operator."""
    op = rng.choice([_swap, _perturb])
    return op(vec, rng)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def solve(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.01,
    initial_seed: Optional[List[float]] = None,
) -> AlgorithmResult:
    """
    Run Simulated Annealing on *project*.

    Parameters
    ----------
    evaluation_budget : int
        Maximum number of SSGS evaluations.
    seed : int
        Controls all randomness.
    initial_temp : float
        Starting temperature T₀.
    cooling_rate : float
        Geometric cooling factor α  (T ← α·T after each step).
    min_temp : float
        Algorithm stops when T falls below this value.
    initial_seed : Optional[List[float]]
        GNN-predicted priority vector for warm-starting SA.
        When provided, SA begins from this solution instead of random,
        eliminating burn-in and focusing cooling on GNN-quality regions.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()

    sorted_ids = sorted(project.activities.keys())
    n = len(sorted_ids)
    evaluations = 0
    convergence_history: List[Tuple[int, int]] = []

    # Initial solution
    if initial_seed is not None and len(initial_seed) == n:
        current = initial_seed[:]
    else:
        current = [rng.random() for _ in range(n)]
    _, current_ms, current_sched = Evaluator.evaluate(project, current)
    evaluations += 1

    best = current[:]
    best_ms = current_ms
    best_sched = current_sched
    convergence_history.append((evaluations, best_ms))

    temp = initial_temp

    while evaluations < evaluation_budget and temp > min_temp:
        candidate = _neighbour(current, rng)
        _, cand_ms, cand_sched = Evaluator.evaluate(project, candidate)
        evaluations += 1

        delta = cand_ms - current_ms  # positive = worse

        if delta < 0:
            # Improvement: always accept
            current = candidate
            current_ms = cand_ms
            current_sched = cand_sched
        else:
            # Worse solution: accept with Boltzmann probability
            prob = math.exp(-delta / temp)
            if rng.random() < prob:
                current = candidate
                current_ms = cand_ms
                current_sched = cand_sched

        if current_ms < best_ms:
            best = current[:]
            best_ms = current_ms
            best_sched = current_sched
            convergence_history.append((evaluations, best_ms))

        temp *= cooling_rate

    runtime = time.perf_counter() - start_ts

    return AlgorithmResult(
        algorithm="SA",
        best_priority_vector=best,
        schedule=best_sched,
        makespan=best_ms,
        runtime=runtime,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )
