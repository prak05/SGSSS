"""
================================================================================
src/algorithms/__init__.py — Central Algorithm Registry & Dispatcher
================================================================================
Defines the unified metaheuristic & neural scheduling suite:
  1. GA    — Genetic Algorithm (Evolutionary search, uniform blend crossover)
  2. SA    — Simulated Annealing (Metropolis criterion, geometric cooling)
  3. BCO   — Bee Colony Optimization (Artificial Bee Colony with scouting)
  4. PRH   — Priority Rule Heuristics (Multi-heuristic portfolio: LST, LFT, MSLK, etc.)
  5. PSO   — Particle Swarm Optimization (Adapted from henryyjiang/MSP_PSO)
  6. TBGAT — Topology-Aware Bidirectional Graph Attention Network (Adapted from zcaicaros/TBGAT)

All candidate priority vectors are decoded exclusively by src.ssgs.SSGS
to guarantee 100% active, precedence- and resource-feasible schedules.
"""

from __future__ import annotations

from typing import Callable, Dict

from src.model import AlgorithmResult, Project
from src.algorithms.ga import solve_ga
from src.algorithms.sa import solve_sa
from src.algorithms.bco import solve_bco
from src.algorithms.prh import solve_prh
from src.algorithms.pso import solve_pso
from src.algorithms.tbgat import solve_tbgat

# ==============================================================================
# SECTION SUMMARY: Algorithm Registry & Permitted Solvers
# What is in this section: Defines the canonical mapping of algorithm names to their
# respective solver functions (GA, SA, BCO, PRH, PSO, TBGAT).
# ==============================================================================
SOLVERS: Dict[str, Callable[..., AlgorithmResult]] = {
    "GA": solve_ga,
    "SA": solve_sa,
    "BCO": solve_bco,
    "PRH": solve_prh,
    "PSO": solve_pso,
    "TBGAT": solve_tbgat,
}

ALGORITHM_NAMES = list(SOLVERS.keys())


# ==============================================================================
# SECTION SUMMARY: Single Solver Dispatcher
# What is in this section: Validates solver requests and dispatches the specified
# algorithm solver with budget and seed parameters.
# ==============================================================================
def run_algorithm(
    name: str,
    project: Project,
    evaluation_budget: int = 500,
    seed: int = 42,
    **kwargs,
) -> AlgorithmResult:
    """
    Execute a single named solver.

    Args:
        name:              Algorithm identifier (one of GA, SA, BCO, PRH, PSO, TBGAT).
        project:           RCPSP instance.
        evaluation_budget: Max SSGS calls.
        seed:              Random seed for reproducibility.
        **kwargs:          Additional algorithm-specific parameters.

    Returns:
        The resulting AlgorithmResult.

    Raises:
        ValueError: If name is unknown.
    """
    upper_name = name.upper()
    if upper_name not in SOLVERS:
        raise ValueError(
            f"Unknown algorithm '{name}'. Permitted algorithms: {ALGORITHM_NAMES}"
        )
    solver_fn = SOLVERS[upper_name]
    return solver_fn(project, evaluation_budget=evaluation_budget, seed=seed, **kwargs)


# ==============================================================================
# SECTION SUMMARY: Tournament Multi-Algorithm Runner
# What is in this section: Sequentially executes all solvers on a project instance
# under identical evaluation budgets and random seeds.
# ==============================================================================
def run_all_algorithms(
    project: Project,
    evaluation_budget: int = 500,
    seed: int = 42,
) -> Dict[str, AlgorithmResult]:
    """
    Execute all solvers sequentially on a project instance under identical budgets.
    """
    results: Dict[str, AlgorithmResult] = {}
    for name in ALGORITHM_NAMES:
        results[name] = run_algorithm(name, project, evaluation_budget=evaluation_budget, seed=seed)
    return results


# ==============================================================================
# SECTION SUMMARY: Public Module Exports
# What is in this section: Exports solvers, registries, and runner helper functions.
# ==============================================================================
__all__ = [
    "solve_ga",
    "solve_sa",
    "solve_bco",
    "solve_prh",
    "solve_pso",
    "solve_tbgat",
    "SOLVERS",
    "ALGORITHM_NAMES",
    "run_algorithm",
    "run_all_algorithms",
]
