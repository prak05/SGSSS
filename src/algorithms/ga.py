"""
Genetic Algorithm for RCPSP.

Representation : random-key chromosome — one float ∈ [0, 1] per activity.
Decoding       : common SSGS via src.evaluation.Evaluator
Objective      : minimise makespan.

References studied (ideas only, no code copied):
  https://github.com/derhendrik/RCPSP_GA
  https://github.com/juliusf/Genetic-SRCPSP
"""

import random
import time
from typing import List, Optional, Tuple

from src.evaluation import Evaluator
from src.model import AlgorithmResult, Project, ScheduleEntry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _random_individual(n: int, rng: random.Random) -> List[float]:
    return [rng.random() for _ in range(n)]


def _evaluate(project: Project, individual: List[float]) -> Tuple[bool, int, List[ScheduleEntry]]:
    """Thin wrapper so the GA never calls SSGS directly."""
    return Evaluator.evaluate(project, individual)


def _tournament_select(
    population: List[List[float]],
    fitnesses: List[int],
    k: int,
    rng: random.Random,
) -> List[float]:
    """Return the best individual from a random sample of k."""
    sample = rng.sample(range(len(population)), k)
    best = min(sample, key=lambda i: fitnesses[i])
    return population[best][:]


def _blend_crossover(
    p1: List[float],
    p2: List[float],
    rng: random.Random,
) -> Tuple[List[float], List[float]]:
    """
    Uniform blend crossover: each gene independently taken from either parent.
    Preserves [0, 1] bounds by construction.
    """
    n = len(p1)
    c1 = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(n)]
    c2 = [p2[i] if rng.random() < 0.5 else p1[i] for i in range(n)]
    return c1, c2


def _mutate(individual: List[float], rate: float, rng: random.Random) -> List[float]:
    """Creep mutation: replace each gene with a new uniform [0,1] value at rate `rate`."""
    return [
        rng.random() if rng.random() < rate else g
        for g in individual
    ]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def solve(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
    population_size: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    elite_count: int = 2,
    tournament_k: int = 3,
    initial_seed: Optional[List[float]] = None,
) -> AlgorithmResult:
    """
    Run a random-key Genetic Algorithm on *project*.

    Parameters
    ----------
    evaluation_budget : int
        Maximum number of SSGS evaluations.
    seed : int
        Controls all randomness for reproducibility.
    population_size : int
        Number of chromosomes per generation.
    crossover_rate : float
        Probability that two parents produce children via crossover.
    mutation_rate : float
        Per-gene probability of random reset.
    elite_count : int
        Number of best individuals carried unchanged to next generation.
    tournament_k : int
        Tournament size for parent selection.
    initial_seed : Optional[List[float]]
        GNN-predicted priority vector for warm-starting the population.
        When provided, ~20%% of the initial population is seeded with
        Gaussian perturbations around this vector.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()

    sorted_ids = sorted(project.activities.keys())
    n = len(sorted_ids)

    evaluations = 0
    convergence_history: List[Tuple[int, int]] = []

    # ---- Initialise population ----
    population: List[List[float]] = []

    if initial_seed is not None and len(initial_seed) == n:
        # GNN-seeded: ~20% of population from GNN priorities
        population.append(initial_seed[:])  # Exact GNN seed
        n_seeded = max(1, population_size // 5)
        for _ in range(n_seeded):
            perturbed = [
                max(0.0, min(1.0, g + rng.gauss(0, 0.1)))
                for g in initial_seed
            ]
            population.append(perturbed)
        # Fill rest randomly
        for _ in range(population_size - 1 - n_seeded):
            population.append(_random_individual(n, rng))
    else:
        population = [_random_individual(n, rng) for _ in range(population_size)]
    fitnesses: List[int] = []
    schedules: List[List[ScheduleEntry]] = []

    for ind in population:
        _, ms, sched = _evaluate(project, ind)
        fitnesses.append(ms)
        schedules.append(sched)
        evaluations += 1

    best_idx = min(range(population_size), key=lambda i: fitnesses[i])
    best_makespan = fitnesses[best_idx]
    best_vector = population[best_idx][:]
    best_schedule = schedules[best_idx]
    convergence_history.append((evaluations, best_makespan))

    # ---- Generational loop ----
    while evaluations < evaluation_budget:
        # Sort by fitness for elitism
        ranked = sorted(range(population_size), key=lambda i: fitnesses[i])
        elites = [population[i][:] for i in ranked[:elite_count]]
        elite_fits = [fitnesses[i] for i in ranked[:elite_count]]
        elite_scheds = [schedules[i] for i in ranked[:elite_count]]

        new_pop: List[List[float]] = elites[:]
        new_fits: List[int] = elite_fits[:]
        new_scheds: List[List[ScheduleEntry]] = elite_scheds[:]

        while len(new_pop) < population_size and evaluations < evaluation_budget:
            p1 = _tournament_select(population, fitnesses, tournament_k, rng)
            p2 = _tournament_select(population, fitnesses, tournament_k, rng)

            if rng.random() < crossover_rate:
                c1, c2 = _blend_crossover(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]

            c1 = _mutate(c1, mutation_rate, rng)
            c2 = _mutate(c2, mutation_rate, rng)

            for child in (c1, c2):
                if len(new_pop) >= population_size or evaluations >= evaluation_budget:
                    break
                _, ms, sched = _evaluate(project, child)
                evaluations += 1
                new_pop.append(child)
                new_fits.append(ms)
                new_scheds.append(sched)

                if ms < best_makespan:
                    best_makespan = ms
                    best_vector = child[:]
                    best_schedule = sched
                    convergence_history.append((evaluations, best_makespan))

        population = new_pop
        fitnesses = new_fits
        schedules = new_scheds

    runtime = time.perf_counter() - start_ts

    return AlgorithmResult(
        algorithm="GA",
        best_priority_vector=best_vector,
        schedule=best_schedule,
        makespan=best_makespan,
        runtime=runtime,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )
