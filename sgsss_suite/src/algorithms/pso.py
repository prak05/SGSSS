"""
================================================================================
src/algorithms/pso.py — Particle Swarm Optimization (MSP_PSO Integration)
================================================================================
Adapted from henryyjiang/MSP_PSO (GlobalBestPSO architecture with continuous
random-key priority-vector optimization for Resource-Constrained Project Scheduling).

Key Algorithmic Mechanics:
  • Continuous Search Space: Particles navigate continuous space x ∈ [0, 1]^D,
    where D is the number of project activities.
  • Swarm Dynamics (Eberhart & Kennedy, 1995; Jiang et al., MSP_PSO):
      v_i(t+1) = w(t) * v_i(t) + c1 * r1 * (pbest_i - x_i(t)) + c2 * r2 * (gbest - x_i(t))
      x_i(t+1) = clamp(x_i(t) + v_i(t+1), 0.0, 1.0)
    with default options: c1 = 0.5, c2 = 0.3, w_init = 0.8, w_min = 0.4.
  • Heuristic Seeding: Particle 0 initialized with Critical Path Method (CPM)
    topological priority weights (Minimum Slack / LST) to jumpstart search.
  • Basin-Hopping / Perturbation (from MSP_PSO):
    Applies stochastic perturbation to stagnant particles when swarm variance drops,
    preventing premature convergence into local resource-bottleneck traps.
  • Centralized SSGS Decoder:
    All particle positions are decoded into 100% active, precedence- and
    capacity-feasible schedules via SSGS.generate_schedule.
================================================================================
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Tuple

from src.evaluation import Evaluator
from src.model import AlgorithmResult, Project


def _compute_cpm_seed(project: Project) -> List[float]:
    """Compute normalized CPM priority vector to seed particle 0."""
    sorted_ids = sorted(project.activities.keys())
    n = len(sorted_ids)
    if n == 0:
        return []

    in_deg: Dict[int, int] = {i: len(a.predecessors) for i, a in project.activities.items()}
    ready = [i for i, deg in in_deg.items() if deg == 0]
    topo: List[int] = []
    while ready:
        curr = ready.pop(0)
        topo.append(curr)
        for succ in project.activities[curr].successors:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                ready.append(succ)

    eft: Dict[int, int] = {}
    for act_id in topo:
        act = project.activities[act_id]
        e_start = max((eft[p] for p in act.predecessors), default=0)
        eft[act_id] = e_start + act.duration

    horizon = max(eft.values(), default=1)
    lst: Dict[int, int] = {}
    for act_id in reversed(topo):
        act = project.activities[act_id]
        l_finish = min((lst[s] for s in act.successors), default=horizon)
        lst[act_id] = l_finish - act.duration

    max_lst = max(lst.values(), default=1) or 1
    return [1.0 - (lst[act_id] / max_lst) for act_id in sorted_ids]


def solve_pso(
    project: Project,
    evaluation_budget: int = 500,
    seed: int = 42,
    swarm_size: Optional[int] = None,
    c1: float = 0.5,
    c2: float = 0.3,
    w_start: float = 0.8,
    w_end: float = 0.4,
    v_max: float = 0.2,
    perturb_prob: float = 0.15,
    budget: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> AlgorithmResult:
    """
    Execute Particle Swarm Optimization on RCPSP instance based on MSP_PSO.

    Args:
        project: RCPSP project instance.
        evaluation_budget: Evaluation budget (number of SSGS schedule generations).
        seed: Random generator seed for strict reproducibility.
        swarm_size: Number of particles (defaults to adaptive based on budget).
        c1: Cognitive parameter (attraction to personal best).
        c2: Social parameter (attraction to global swarm best).
        w_start: Initial inertia weight.
        w_end: Final inertia weight (linearly decayed).
        v_max: Velocity limit per dimension.
        perturb_prob: Basin-hopping perturbation probability for stagnant particles.
        budget: Optional alias for evaluation_budget.
        random_seed: Optional alias for seed.

    Returns:
        AlgorithmResult containing optimal schedule, makespan, runtime, and convergence curve.
    """
    if budget is not None:
        evaluation_budget = budget
    if random_seed is not None:
        seed = random_seed
    budget = evaluation_budget

    start_time = time.perf_counter()
    rng = random.Random(seed)

    sorted_ids = sorted(project.activities.keys())
    dim = len(sorted_ids)

    if dim == 0 or budget <= 0:
        return AlgorithmResult(
            best_schedule=None,
            best_makespan=0,
            evaluations=0,
            runtime_seconds=time.perf_counter() - start_time,
            history=[],
            algorithm_name="Particle Swarm Optimization (MSP_PSO)",
        )

    # Adaptive swarm sizing: between 10 and 30 particles
    if swarm_size is None:
        swarm_size = max(10, min(30, max(5, budget // 25)))

    max_iters = max(1, budget // swarm_size)

    # 1. Initialize Swarm
    particles_x: List[List[float]] = []
    particles_v: List[List[float]] = []
    pbest_x: List[List[float]] = []
    pbest_val: List[int] = []

    # Seed particle 0 with CPM prior
    cpm_prior = _compute_cpm_seed(project)
    p0_pos = [max(0.0, min(1.0, cpm_prior[d] + rng.uniform(-0.05, 0.05))) for d in range(dim)]
    particles_x.append(p0_pos)
    particles_v.append([rng.uniform(-v_max / 2, v_max / 2) for _ in range(dim)])

    # Initialize remaining particles uniformly across [0, 1]^D
    for _ in range(1, swarm_size):
        particles_x.append([rng.random() for _ in range(dim)])
        particles_v.append([rng.uniform(-v_max, v_max) for _ in range(dim)])

    # 2. Evaluate Initial Swarm
    eval_count = 0
    history: List[Tuple[int, int]] = []
    best_schedule = None
    gbest_val = math.inf
    gbest_x: List[float] = list(particles_x[0])

    for i in range(swarm_size):
        if eval_count >= budget:
            break
        sched, ms = Evaluator.evaluate(project, particles_x[i])
        eval_count += 1
        pbest_x.append(list(particles_x[i]))
        pbest_val.append(ms)

        if ms < gbest_val:
            gbest_val = ms
            best_schedule = sched
            gbest_x = list(particles_x[i])

        history.append((eval_count, int(gbest_val)))

    # 3. Main Swarm Optimization Loop
    iteration = 0
    while eval_count < budget and iteration < max_iters:
        iteration += 1
        # Inertia weight linear decay (from MSP_PSO / Shi & Eberhart)
        frac = iteration / max(1, max_iters)
        w = w_start - frac * (w_start - w_end)

        for i in range(swarm_size):
            if eval_count >= budget:
                break

            # Velocity & Position update
            for d in range(dim):
                r1 = rng.random()
                r2 = rng.random()
                cog = c1 * r1 * (pbest_x[i][d] - particles_x[i][d])
                soc = c2 * r2 * (gbest_x[d] - particles_x[i][d])

                v_new = w * particles_v[i][d] + cog + soc
                # Clamp velocity
                v_new = max(-v_max, min(v_max, v_new))
                particles_v[i][d] = v_new

                # Position update & boundary clamping
                x_new = particles_x[i][d] + v_new
                if x_new < 0.0:
                    x_new = 0.0
                    particles_v[i][d] = -0.5 * particles_v[i][d]  # soft wall bounce
                elif x_new > 1.0:
                    x_new = 1.0
                    particles_v[i][d] = -0.5 * particles_v[i][d]
                particles_x[i][d] = x_new

            # Basin-hopping perturbation: perturb stagnant particle
            if rng.random() < perturb_prob and pbest_val[i] > gbest_val:
                p_dim = rng.randint(0, dim - 1)
                particles_x[i][p_dim] = max(0.0, min(1.0, particles_x[i][p_dim] + rng.gauss(0.0, 0.15)))

            # Evaluate candidate
            sched, ms = Evaluator.evaluate(project, particles_x[i])
            eval_count += 1

            # Update personal best
            if ms < pbest_val[i]:
                pbest_val[i] = ms
                pbest_x[i] = list(particles_x[i])

                # Update global best
                if ms < gbest_val:
                    gbest_val = ms
                    best_schedule = sched
                    gbest_x = list(particles_x[i])

            history.append((eval_count, int(gbest_val)))

    runtime = time.perf_counter() - start_time
    return AlgorithmResult(
        algorithm="PSO",
        best_priority_vector=gbest_x,
        schedule=best_schedule,
        makespan=int(gbest_val) if best_schedule else 0,
        runtime=runtime,
        evaluations=eval_count,
        convergence_history=history,
    )
