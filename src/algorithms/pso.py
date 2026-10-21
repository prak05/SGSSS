"""
Particle Swarm Optimization for RCPSP.

Representation : random-key vector — one float ∈ [0, 1] per activity.
Decoding       : common SSGS via src.evaluation.Evaluator
Objective      : minimise makespan.

PSO dynamics
------------
Each particle i has:
  - Position x_i ∈ [0, 1]^N  (random-key vector decoded via SSGS)
  - Velocity v_i ∈ [-v_max, v_max]^N
  - Personal best position p_i and personal best makespan f(p_i)

Global best:  g* = argmin_i f(p_i)

Velocity update (standard PSO with inertia weight):
    v_i(t+1) = w * v_i(t)
             + c1 * r1 * (p_i - x_i(t))       [cognitive component]
             + c2 * r2 * (g* - x_i(t))          [social component]

Position update:
    x_i(t+1) = clamp(x_i(t) + v_i(t+1), 0.0, 1.0)

Inertia weight decays linearly from w_max to w_min over the search
to transition from global exploration to local exploitation.

GNN Seeding Support
-------------------
When an initial_seed (priority vector from GNN) is provided:
  - The swarm leader (particle 0) is initialized at x_0 = seed.
  - 20% of the swarm is initialized as Gaussian perturbations around
    the seed, providing focused exploitation near GNN-predicted priorities.
  - Remaining particles explore randomly for diversity.

References studied (ideas only, no code copied):
  Kennedy & Eberhart (1995) — original PSO.
  Shi & Eberhart (1998) — inertia weight variant.
  Zhang et al. (2006) — discrete PSO for scheduling (random-key adaptation).
"""

import random
import time
from typing import List, Optional, Tuple

from src.evaluation import Evaluator
from src.model import AlgorithmResult, Project, ScheduleEntry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _random_position(n: int, rng: random.Random) -> List[float]:
    return [rng.random() for _ in range(n)]


def _random_velocity(n: int, rng: random.Random, v_max: float = 0.2) -> List[float]:
    return [rng.uniform(-v_max, v_max) for _ in range(n)]


def _perturb_seed(
    seed: List[float],
    rng: random.Random,
    sigma: float = 0.1,
) -> List[float]:
    """Create a Gaussian perturbation of the seed vector, clamped to [0, 1]."""
    return [_clamp(g + rng.gauss(0, sigma)) for g in seed]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def solve(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
    swarm_size: int = 30,
    w_max: float = 0.9,
    w_min: float = 0.4,
    c1: float = 1.5,
    c2: float = 1.5,
    v_max: float = 0.2,
    initial_seed: Optional[List[float]] = None,
) -> AlgorithmResult:
    """
    Run Particle Swarm Optimization on *project*.

    Parameters
    ----------
    evaluation_budget : int
        Maximum number of SSGS evaluations.
    seed : int
        Controls all randomness for reproducibility.
    swarm_size : int
        Number of particles in the swarm.
    w_max : float
        Starting inertia weight (exploration-heavy).
    w_min : float
        Final inertia weight (exploitation-heavy).
    c1 : float
        Cognitive acceleration coefficient (personal best attraction).
    c2 : float
        Social acceleration coefficient (global best attraction).
    v_max : float
        Maximum velocity component magnitude.
    initial_seed : Optional[List[float]]
        GNN-predicted priority vector for warm-starting the swarm.
        When provided, the leader particle starts at this position and
        ~20% of the swarm is seeded with Gaussian perturbations around it.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()

    sorted_ids = sorted(project.activities.keys())
    n = len(sorted_ids)
    evaluations = 0
    convergence_history: List[Tuple[int, int]] = []

    # Estimate max iterations for inertia decay
    max_iterations = max(1, evaluation_budget // swarm_size)

    # ---- Initialize swarm ----
    positions: List[List[float]] = []
    velocities: List[List[float]] = []

    if initial_seed is not None and len(initial_seed) == n:
        # GNN-seeded initialization
        # Particle 0: exact GNN seed
        positions.append(initial_seed[:])
        velocities.append([0.0] * n)  # Start with zero velocity at seed

        # ~20% of swarm: Gaussian perturbations around seed
        n_seeded = max(1, swarm_size // 5)
        for _ in range(n_seeded):
            positions.append(_perturb_seed(initial_seed, rng, sigma=0.1))
            velocities.append(_random_velocity(n, rng, v_max * 0.5))

        # Remaining: random exploration
        for _ in range(swarm_size - 1 - n_seeded):
            positions.append(_random_position(n, rng))
            velocities.append(_random_velocity(n, rng, v_max))
    else:
        # Fully random initialization
        for _ in range(swarm_size):
            positions.append(_random_position(n, rng))
            velocities.append(_random_velocity(n, rng, v_max))

    # ---- Evaluate initial positions ----
    fitnesses: List[int] = []
    schedules: List[List[ScheduleEntry]] = []

    # Personal bests
    p_best_pos: List[List[float]] = []
    p_best_fit: List[int] = []

    for i in range(swarm_size):
        if evaluations >= evaluation_budget:
            break
        _, ms, sched = Evaluator.evaluate(project, positions[i])
        evaluations += 1
        fitnesses.append(ms)
        schedules.append(sched)
        p_best_pos.append(positions[i][:])
        p_best_fit.append(ms)

    # Pad if budget ran out during initialization
    actual_swarm = len(fitnesses)
    if actual_swarm == 0:
        # Edge case: budget too small for even one evaluation
        return AlgorithmResult(
            algorithm="PSO",
            best_priority_vector=[0.5] * n,
            schedule=[],
            makespan=999999,
            runtime=time.perf_counter() - start_ts,
            evaluations=0,
            convergence_history=[],
        )

    # Global best
    g_best_idx = min(range(actual_swarm), key=lambda i: p_best_fit[i])
    g_best_pos = p_best_pos[g_best_idx][:]
    g_best_fit = p_best_fit[g_best_idx]
    g_best_sched = schedules[g_best_idx]
    convergence_history.append((evaluations, g_best_fit))

    # ---- Main PSO loop ----
    iteration = 0

    while evaluations < evaluation_budget:
        iteration += 1

        # Linear inertia weight decay
        w = w_max - (w_max - w_min) * min(1.0, iteration / max_iterations)

        for i in range(actual_swarm):
            if evaluations >= evaluation_budget:
                break

            # Update velocity
            for d in range(n):
                r1 = rng.random()
                r2 = rng.random()

                cognitive = c1 * r1 * (p_best_pos[i][d] - positions[i][d])
                social = c2 * r2 * (g_best_pos[d] - positions[i][d])

                velocities[i][d] = w * velocities[i][d] + cognitive + social

                # Clamp velocity
                velocities[i][d] = _clamp(velocities[i][d], -v_max, v_max)

            # Update position
            for d in range(n):
                positions[i][d] = _clamp(positions[i][d] + velocities[i][d])

            # Evaluate new position
            _, ms, sched = Evaluator.evaluate(project, positions[i])
            evaluations += 1
            fitnesses[i] = ms
            schedules[i] = sched

            # Update personal best
            if ms < p_best_fit[i]:
                p_best_fit[i] = ms
                p_best_pos[i] = positions[i][:]

                # Update global best
                if ms < g_best_fit:
                    g_best_fit = ms
                    g_best_pos = positions[i][:]
                    g_best_sched = sched
                    convergence_history.append((evaluations, g_best_fit))

    runtime = time.perf_counter() - start_ts

    return AlgorithmResult(
        algorithm="PSO",
        best_priority_vector=g_best_pos,
        schedule=g_best_sched,
        makespan=g_best_fit,
        runtime=runtime,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )
