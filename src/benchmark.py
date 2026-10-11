"""
Benchmark runner for the AI-RCPSP project.

Runs all FIVE optimisation algorithms:
    GA  SA  BCO  PRH  TS

on a single project instance, with configurable seeds and evaluation
budget, and returns structured result records.

Design rules
------------
- Every algorithm receives the same Project object.
- Every algorithm receives the same evaluation_budget.
- Seeds are applied per-run, not globally.
- No global random state is modified.
- Results are plain dataclasses — no UI, no ML logic here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.evaluation import Evaluator
from src.model import AlgorithmResult, Project

from src.algorithms import ga, sa, bco, prh, tabu


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRecord:
    """Single algorithm result from one benchmark run."""
    instance_id:       str
    algorithm:         str
    seed:              int
    makespan:          int
    runtime:           float
    evaluations:       int
    feasible:          bool
    best_priority_vector: List[float]
    convergence_history: List[tuple] = field(default_factory=list)
    notes:             str = ""

    @property
    def label(self) -> str:
        return f"{self.instance_id}/{self.algorithm}/seed={self.seed}"


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

# Maps short names to solve callables.
# Each callable must accept (project, evaluation_budget, seed, **kwargs).
ALGORITHM_REGISTRY: Dict[str, Any] = {
    "GA":  ga.solve,
    "SA":  sa.solve,
    "BCO": bco.solve,
    "PRH": prh.solve,
    "TS":  tabu.solve,
}

ALGORITHM_NAMES = list(ALGORITHM_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_single(
    project: Project,
    algorithm: str,
    seed: int,
    evaluation_budget: int,
    instance_id: str = "unnamed",
    **algo_kwargs,
) -> BenchmarkRecord:
    """
    Run one algorithm on one project instance.

    Parameters
    ----------
    project          : Project to solve.
    algorithm        : Short name — one of GA, SA, BCO, PRH, TS.
    seed             : Random seed passed to the algorithm.
    evaluation_budget: Maximum SSGS evaluations.
    instance_id      : Label for the result record.
    **algo_kwargs    : Extra algorithm-specific parameters (optional).

    Returns
    -------
    BenchmarkRecord
    """
    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: {ALGORITHM_NAMES}"
        )

    solve_fn = ALGORITHM_REGISTRY[algorithm]
    result: AlgorithmResult = solve_fn(
        project,
        evaluation_budget=evaluation_budget,
        seed=seed,
        **algo_kwargs,
    )

    # Validate the returned schedule
    feasible = Evaluator.check_constraints(project, result.schedule)

    # Verify makespan consistency
    if result.schedule:
        actual_ms = max(e.finish_time for e in result.schedule)
        notes = ""
        if actual_ms != result.makespan:
            notes = (
                f"WARNING: reported makespan {result.makespan} differs from "
                f"schedule finish time {actual_ms}."
            )
    else:
        notes = "WARNING: empty schedule returned."
        feasible = False

    return BenchmarkRecord(
        instance_id=instance_id,
        algorithm=result.algorithm,
        seed=seed,
        makespan=result.makespan,
        runtime=result.runtime,
        evaluations=result.evaluations,
        feasible=feasible,
        best_priority_vector=result.best_priority_vector,
        convergence_history=result.convergence_history,
        notes=notes,
    )


def run_all_algorithms(
    project: Project,
    seeds: List[int],
    evaluation_budget: int = 5000,
    instance_id: str = "unnamed",
    ga_kwargs:  Optional[Dict] = None,
    sa_kwargs:  Optional[Dict] = None,
    bco_kwargs: Optional[Dict] = None,
    prh_kwargs: Optional[Dict] = None,
    ts_kwargs:  Optional[Dict] = None,
) -> List[BenchmarkRecord]:
    """
    Run all five algorithms for each seed and return a flat list of records.

    Parameters
    ----------
    project          : Project to solve.
    seeds            : List of integer seeds (one run per algorithm per seed).
    evaluation_budget: Same budget applied to every algorithm and seed.
    instance_id      : Label attached to all result records.
    *_kwargs         : Optional per-algorithm extra parameters.

    Returns
    -------
    List[BenchmarkRecord] — length = 5 × len(seeds)
    """
    ga_kwargs  = ga_kwargs  or {}
    sa_kwargs  = sa_kwargs  or {}
    bco_kwargs = bco_kwargs or {}
    prh_kwargs = prh_kwargs or {}
    ts_kwargs  = ts_kwargs  or {}

    algo_kwargs_map = {
        "GA":  ga_kwargs,
        "SA":  sa_kwargs,
        "BCO": bco_kwargs,
        "PRH": prh_kwargs,
        "TS":  ts_kwargs,
    }

    records: List[BenchmarkRecord] = []

    for seed in seeds:
        for algo_name in ALGORITHM_NAMES:
            record = run_single(
                project=project,
                algorithm=algo_name,
                seed=seed,
                evaluation_budget=evaluation_budget,
                instance_id=instance_id,
                **algo_kwargs_map[algo_name],
            )
            records.append(record)

    return records
