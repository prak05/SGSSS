"""
================================================================================
src/algorithms/sa.py — Simulated Annealing (SA) Solver
================================================================================
Trajectory-based metaheuristic inspired by thermal annealing:
  • Geometric cooling schedule: T ← T × α  (default α = 0.995)
  • Boltzmann acceptance: P(accept worse) = exp(−Δ / T)
  • Neighbourhood: 50 % pairwise gene swap, 50 % Gaussian perturbation
  • Terminates when budget exhausted or T < T_min

The SA outputs a random-key priority vector decoded by the shared SSGS.
It NEVER contains independent scheduling or resource-feasibility logic.

References
──────────
[3][9][12] Mitsou & Koulinas (2026); Kirkpatrick et al. (1983).
"""

from __future__ import annotations

import math
import random
import time
from typing import List

from src.model import AlgorithmResult, Project
from src.evaluation import Evaluator


def solve_sa(
    project: Project,
    evaluation_budget: int = 5_000,
    seed: int = 42,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.01,
) -> AlgorithmResult:
    """
    Simulated Annealing solver for RCPSP.

    Args:
        project:          RCPSP instance.
        evaluation_budget: Maximum SSGS calls (default 5 000).
        seed:             Random seed for reproducibility.
        initial_temp:     Starting temperature T₀.
        cooling_rate:     Geometric cooling factor α (0 < α < 1).
        min_temp:         Temperature floor; stop cooling below this.

    Returns:
        An ``AlgorithmResult`` with algorithm = ``"SA"``.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history: List[tuple] = []

    # ==============================================================================
    # SECTION SUMMARY: Initial Solution Generation & Baseline Evaluation
    # What is in this section: Creates an initial random-key priority vector,
    # evaluates its project makespan using SSGS, and initializes annealing temperature.
    # ==============================================================================
    curr = [rng.random() for _ in range(n)]
    curr_sched, curr_ms = Evaluator.evaluate(project, curr)
    evaluations += 1

    best_vec, best_ms, best_sched = curr[:], curr_ms, curr_sched
    convergence_history.append((evaluations, best_ms))
    temp = initial_temp

    # ==============================================================================
    # SECTION SUMMARY: Annealing Loop & Metropolis Acceptance
    # What is in this section: Iteratively mutates candidate priority vectors using
    # pairwise gene swaps or Gaussian perturbations. Accepts improvements unconditionally
    # and accepts worse moves probabilistically via the Metropolis criterion (exp(-Δ/T)).
    # Cools temperature geometrically (T = T * cooling_rate).
    # ==============================================================================
    while evaluations < evaluation_budget and temp > min_temp:
        cand = curr[:]

        # Neighbourhood move
        if rng.random() < 0.5 and n >= 2:
            # Pairwise swap of two random-key genes
            i, j = rng.sample(range(n), 2)
            cand[i], cand[j] = cand[j], cand[i]
        else:
            # Gaussian perturbation on one gene
            i = rng.randrange(n)
            cand[i] = max(0.0, min(1.0, cand[i] + rng.gauss(0, 0.1)))

        cand_sched, cand_ms = Evaluator.evaluate(project, cand)
        evaluations += 1
        delta = cand_ms - curr_ms

        # Metropolis acceptance
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            curr, curr_ms, curr_sched = cand, cand_ms, cand_sched

        # Update global best
        if curr_ms < best_ms:
            best_vec, best_ms, best_sched = curr[:], curr_ms, curr_sched
            convergence_history.append((evaluations, best_ms))

        temp *= cooling_rate

    # ==============================================================================
    # SECTION SUMMARY: Result Packaging
    # What is in this section: Packages the best found schedule, makespan, runtime,
    # evaluation count, and convergence history into an AlgorithmResult object.
    # ==============================================================================
    return AlgorithmResult(
        algorithm="SA",
        best_priority_vector=best_vec,
        schedule=best_sched,
        makespan=best_ms,
        runtime=time.perf_counter() - start_ts,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )

