"""
================================================================================
src/algorithms/ga.py — Genetic Algorithm (GA) Solver
================================================================================
Population-based evolutionary search with:
  • Random-key chromosome representation (floats ∈ [0, 1])
  • Tournament selection (configurable k)
  • Uniform blend crossover (50 % gene swap probability)
  • Creep mutation (random-key reset with probability mutation_rate)
  • Elitism (top elite_count individuals carry over unchanged)

The GA outputs a random-key priority vector decoded by the shared SSGS.
It NEVER contains independent scheduling or resource-feasibility logic.

References
──────────
[3][9][12] Mitsou & Koulinas (2026).
"""

from __future__ import annotations

import random
import time
from typing import List, Optional

from src.model import AlgorithmResult, Project
from src.evaluation import Evaluator


def solve_ga(
    project: Project,
    evaluation_budget: int = 5_000,
    seed: int = 42,
    population_size: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    elite_count: int = 2,
    tournament_k: int = 3,
) -> AlgorithmResult:
    """
    Genetic Algorithm solver for RCPSP.

    Args:
        project:          RCPSP instance.
        evaluation_budget: Maximum SSGS calls (default 5 000).
        seed:             Random seed for reproducibility.
        population_size:  Number of individuals per generation.
        crossover_rate:   Probability of performing crossover.
        mutation_rate:    Per-gene mutation probability.
        elite_count:      Number of best individuals preserved per generation.
        tournament_k:     Tournament selection pool size.

    Returns:
        An ``AlgorithmResult`` with algorithm = ``"GA"``.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history: List[tuple] = []

    # ==============================================================================
    # SECTION SUMMARY: Population Initialization & Evaluation
    # What is in this section: Creates initial generation of random-key vectors,
    # evaluates each via SSGS, and records the initial global best makespan.
    # ==============================================================================
    pop: List[List[float]] = [
        [rng.random() for _ in range(n)] for _ in range(population_size)
    ]
    fits: List[int] = []
    scheds = []
    for ind in pop:
        sched, ms = Evaluator.evaluate(project, ind)
        fits.append(ms)
        scheds.append(sched)
        evaluations += 1

    # Track global best
    best_idx = min(range(population_size), key=lambda i: fits[i])
    best_ms = fits[best_idx]
    best_vec = pop[best_idx][:]
    best_sched = scheds[best_idx]
    convergence_history.append((evaluations, best_ms))

    # ==============================================================================
    # SECTION SUMMARY: Generational Evolution Loop
    # What is in this section: Preserves elite individuals, conducts tournament
    # selection, applies uniform crossover and creep mutation, and decodes offspring.
    # ==============================================================================
    while evaluations < evaluation_budget:
        ranked = sorted(range(len(pop)), key=lambda i: fits[i])

        # Elitism
        new_pop = [pop[i][:] for i in ranked[:elite_count]]
        new_fits = [fits[i] for i in ranked[:elite_count]]
        new_scheds = [scheds[i] for i in ranked[:elite_count]]

        # Breed offspring
        while len(new_pop) < population_size and evaluations < evaluation_budget:
            # Tournament selection
            s1 = rng.sample(range(len(pop)), min(tournament_k, len(pop)))
            p1 = pop[min(s1, key=lambda i: fits[i])][:]
            s2 = rng.sample(range(len(pop)), min(tournament_k, len(pop)))
            p2 = pop[min(s2, key=lambda i: fits[i])][:]

            # Uniform crossover
            if rng.random() < crossover_rate:
                c1 = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(n)]
                c2 = [p2[i] if rng.random() < 0.5 else p1[i] for i in range(n)]
            else:
                c1, c2 = p1[:], p2[:]

            # Creep mutation
            c1 = [rng.random() if rng.random() < mutation_rate else g for g in c1]
            c2 = [rng.random() if rng.random() < mutation_rate else g for g in c2]

            for child in (c1, c2):
                if len(new_pop) >= population_size or evaluations >= evaluation_budget:
                    break
                sched, ms = Evaluator.evaluate(project, child)
                evaluations += 1
                new_pop.append(child)
                new_fits.append(ms)
                new_scheds.append(sched)

                if ms < best_ms:
                    best_ms, best_vec, best_sched = ms, child[:], sched
                    convergence_history.append((evaluations, best_ms))

        pop, fits, scheds = new_pop, new_fits, new_scheds

    return AlgorithmResult(
        algorithm="GA",
        best_priority_vector=best_vec,
        schedule=best_sched,
        makespan=best_ms,
        runtime=time.perf_counter() - start_ts,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )
