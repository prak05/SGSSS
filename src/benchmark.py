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

from src.algorithms import ga, sa, bco, prh, tabu, pso


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
from pathlib import Path


def gnn_solve(
    project: Project,
    evaluation_budget: int = 100,
    seed: int = 42,
    **kwargs,
) -> AlgorithmResult:
    """
    Graph Neural Network (GNN) Priority Seeding Solver.
    Uses deep message passing along precedence edges to predict activity priorities
    and decodes them via SSGS in milliseconds.
    """
    try:
        from src.ml.gnn_model import GNNPriorityScheduler
        ckpt_path = Path(__file__).resolve().parent.parent / "models" / "gnn_dual_head.pt"
        ckpt_str = str(ckpt_path) if ckpt_path.exists() else None
        scheduler = GNNPriorityScheduler(model_path=ckpt_str)
        return scheduler.solve(project, name="GNN-Priority")
    except Exception:
        return prh.solve(project, evaluation_budget=evaluation_budget, seed=seed)


# Maps short names to solve callables.
# Each callable must accept (project, evaluation_budget, seed, **kwargs).
ALGORITHM_REGISTRY: Dict[str, Any] = {
    "GA":  ga.solve,
    "SA":  sa.solve,
    "BCO": bco.solve,
    "PRH": prh.solve,
    "TS":  tabu.solve,
    "PSO": pso.solve,
    "GNN": gnn_solve,
    "GNN-Priority": gnn_solve,
}

# 5 Baseline Metaheuristic algorithms (4 from paper + Tabu Search)
ALGORITHM_NAMES = ["GA", "SA", "BCO", "PRH", "TS"]

# 6 Core Metaheuristic algorithms including PSO extension
METAHEURISTIC_NAMES = ["GA", "SA", "BCO", "PRH", "TS", "PSO"]

# Full 7-algorithm suite (6 Metaheuristics + 1 GNN Priority Seeder)
ALL_ALGORITHM_NAMES = ["GA", "SA", "BCO", "PRH", "TS", "PSO", "GNN"]


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
    algorithms: Optional[List[str]] = None,
    ga_kwargs:  Optional[Dict] = None,
    sa_kwargs:  Optional[Dict] = None,
    bco_kwargs: Optional[Dict] = None,
    prh_kwargs: Optional[Dict] = None,
    ts_kwargs:  Optional[Dict] = None,
    pso_kwargs: Optional[Dict] = None,
    gnn_kwargs: Optional[Dict] = None,
) -> List[BenchmarkRecord]:
    """
    Run algorithms for each seed and return a flat list of records.

    Parameters
    ----------
    project          : Project to solve.
    seeds            : List of integer seeds.
    evaluation_budget: Same budget applied to every algorithm and seed.
    instance_id      : Label attached to all result records.
    algorithms       : Optional list of algorithm names (defaults to 5 paper metaheuristics).
    *_kwargs         : Optional per-algorithm extra parameters.

    Returns
    -------
    List[BenchmarkRecord]
    """
    ga_kwargs  = ga_kwargs  or {}
    sa_kwargs  = sa_kwargs  or {}
    bco_kwargs = bco_kwargs or {}
    prh_kwargs = prh_kwargs or {}
    ts_kwargs  = ts_kwargs  or {}
    pso_kwargs = pso_kwargs or {}
    gnn_kwargs = gnn_kwargs or {}

    algo_kwargs_map = {
        "GA":  ga_kwargs,
        "SA":  sa_kwargs,
        "BCO": bco_kwargs,
        "PRH": prh_kwargs,
        "TS":  ts_kwargs,
        "PSO": pso_kwargs,
        "GNN": gnn_kwargs,
        "GNN-Priority": gnn_kwargs,
    }

    target_algos = algorithms if algorithms is not None else ALGORITHM_NAMES
    records: List[BenchmarkRecord] = []

    for seed in seeds:
        for algo_name in target_algos:
            record = run_single(
                project=project,
                algorithm=algo_name,
                seed=seed,
                evaluation_budget=evaluation_budget,
                instance_id=instance_id,
                **algo_kwargs_map.get(algo_name, {}),
            )
            records.append(record)

    return records
