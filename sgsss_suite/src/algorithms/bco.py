"""
================================================================================
src/algorithms/bco.py — Bee Colony Optimization (BCO / ABC) Solver
================================================================================
Swarm-intelligence metaheuristic modelling honeybee foraging (Karaboga, 2005):

  Phase 1 — **Employed bees**: local search around each food source via
            differential variation  v_ij = x_ij + φ·(x_ij − x_kj).
  Phase 2 — **Onlooker bees**: roulette-wheel selection biased towards
            better (lower-makespan) food sources, then local search.
  Phase 3 — **Scout bees**: abandon food sources that have not improved
            for ``limit`` consecutive trials; reinitialise randomly.

The BCO outputs random-key priority vectors decoded by the shared SSGS.
It NEVER contains independent scheduling or resource-feasibility logic.

References
──────────
[3][9][12] Mitsou & Koulinas (2026); Karaboga (2005).
"""

from __future__ import annotations

import random
import time
from typing import List

from src.model import AlgorithmResult, Project
from src.evaluation import Evaluator


def solve_bco(
    project: Project,
    evaluation_budget: int = 5_000,
    seed: int = 42,
    colony_size: int = 20,
    limit: int = 15,
) -> AlgorithmResult:
    """
    Bee Colony Optimization (ABC-style) solver for RCPSP.

    Args:
        project:          RCPSP instance.
        evaluation_budget: Maximum SSGS calls (default 5 000).
        seed:             Random seed for reproducibility.
        colony_size:      Number of food sources (= employed bees).
        limit:            Abandonment threshold — max stagnation trials before
                          a scout bee reinitialises the food source.

    Returns:
        An ``AlgorithmResult`` with algorithm = ``"BCO"``.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history: List[tuple] = []

    # ==============================================================================
    # SECTION SUMMARY: Food Source Population Initialization
    # What is in this section: Creates an initial population of food sources (random-key
    # vectors for employed bees), decodes them with SSGS, tracks initial makespans,
    # and records the initial best solution.
    # ==============================================================================
    sources = [[rng.random() for _ in range(n)] for _ in range(colony_size)]
    fits: List[int] = []
    scheds = []
    trials = [0] * colony_size

    for src in sources:
        sched, ms = Evaluator.evaluate(project, src)
        fits.append(ms)
        scheds.append(sched)
        evaluations += 1

    best_idx = min(range(colony_size), key=lambda i: fits[i])
    best_ms = fits[best_idx]
    best_vec = sources[best_idx][:]
    best_sched = scheds[best_idx]
    convergence_history.append((evaluations, best_ms))

    # ==============================================================================
    # SECTION SUMMARY: Artificial Bee Colony (ABC) Main Optimization Loop
    # What is in this section: Iterates through Employed Bee, Onlooker Bee, and Scout
    # Bee foraging phases until the evaluation budget is exhausted.
    # ==============================================================================
    while evaluations < evaluation_budget:

        # --------------------------------------------------------------------------
        # PHASE 1: Employed Bee Phase (Local Differential Neighborhood Search)
        # What is in this section: Each employed bee searches around its food source
        # using differential vectors with another randomly chosen bee partner.
        # --------------------------------------------------------------------------
        for i in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            partner = rng.choice([j for j in range(colony_size) if j != i])
            cand = sources[i][:]
            dim = rng.randrange(n)
            phi = rng.uniform(-1.0, 1.0)
            cand[dim] = max(
                0.0, min(1.0, cand[dim] + phi * (cand[dim] - sources[partner][dim]))
            )

            cand_sched, cand_ms = Evaluator.evaluate(project, cand)
            evaluations += 1

            if cand_ms <= fits[i]:
                sources[i], fits[i], scheds[i], trials[i] = (
                    cand, cand_ms, cand_sched, 0,
                )
            else:
                trials[i] += 1

            if fits[i] < best_ms:
                best_ms, best_vec, best_sched = fits[i], sources[i][:], scheds[i]
                convergence_history.append((evaluations, best_ms))

        # --------------------------------------------------------------------------
        # PHASE 2: Onlooker Bee Phase (Fitness-Proportional Selection)
        # What is in this section: Simulates dance area recruitment where onlooker bees
        # select food sources via roulette-wheel selection biased toward lower makespans.
        # --------------------------------------------------------------------------
        max_ms = max(fits) if fits else 1
        weights = [(max_ms - f + 1) for f in fits]
        tot = sum(weights) or 1.0
        probs = [w / tot for w in weights]

        for _ in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            # Roulette-wheel selection
            r = rng.random()
            cum = 0.0
            sel_i = colony_size - 1
            for idx_p, p in enumerate(probs):
                cum += p
                if r <= cum:
                    sel_i = idx_p
                    break

            partner = rng.choice([j for j in range(colony_size) if j != sel_i])
            cand = sources[sel_i][:]
            dim = rng.randrange(n)
            phi = rng.uniform(-1.0, 1.0)
            cand[dim] = max(
                0.0,
                min(1.0, cand[dim] + phi * (cand[dim] - sources[partner][dim])),
            )

            cand_sched, cand_ms = Evaluator.evaluate(project, cand)
            evaluations += 1

            if cand_ms <= fits[sel_i]:
                sources[sel_i], fits[sel_i], scheds[sel_i], trials[sel_i] = (
                    cand, cand_ms, cand_sched, 0,
                )
            else:
                trials[sel_i] += 1

            if fits[sel_i] < best_ms:
                best_ms, best_vec, best_sched = (
                    fits[sel_i], sources[sel_i][:], scheds[sel_i],
                )
                convergence_history.append((evaluations, best_ms))

        # --------------------------------------------------------------------------
        # PHASE 3: Scout Bee Phase (Stagnation Abandonment & Random Exploration)
        # What is in this section: Identifies food sources that exceeded the stagnation
        # limit without improvement and replaces them with brand new random vectors.
        # --------------------------------------------------------------------------
        for i in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            if trials[i] >= limit:
                sources[i] = [rng.random() for _ in range(n)]
                sched, ms = Evaluator.evaluate(project, sources[i])
                fits[i], scheds[i], trials[i] = ms, sched, 0
                evaluations += 1

                if ms < best_ms:
                    best_ms, best_vec, best_sched = ms, sources[i][:], sched
                    convergence_history.append((evaluations, best_ms))

    # ==============================================================================
    # SECTION SUMMARY: Result Compilation & Return
    # What is in this section: Packages the global optimal BCO schedule, makespan,
    # and convergence trace into an AlgorithmResult object.
    # ==============================================================================
    return AlgorithmResult(
        algorithm="BCO",
        best_priority_vector=best_vec,
        schedule=best_sched,
        makespan=best_ms,
        runtime=time.perf_counter() - start_ts,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )

