"""
================================================================================
src/algorithms/prh.py — Priority Rule Heuristics (PRH) Solver
================================================================================
Deterministic and randomized dispatching heuristics based on network topology
and Critical Path Method (CPM) calculations:
  • EST (Earliest Start Time)
  • EFT (Earliest Finish Time)
  • LST (Latest Start Time)
  • LFT (Latest Finish Time)
  • MSLK (Minimum Slack)
  • SPT (Shortest Processing Time)
  • LPT (Longest Processing Time)
  • MIS (Most Immediate Successors)
  • MTS (Most Total Successors)
  • GRPW (Greatest Rank Positional Weight)

Followed by biased/random sampling to consume remaining evaluation budget.
All candidate priority vectors are decoded via the centralized SSGS decoder.
PSO is strictly forbidden.

References
──────────
[3][9][12] Mitsou & Koulinas (2026); Kolisch (1996).
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Set, Tuple

from src.model import AlgorithmResult, Project
from src.evaluation import Evaluator


# ==============================================================================
# SECTION SUMMARY: Critical Path Method (CPM) Forward & Backward Pass
# What is in this section: Computes topological ordering, forward pass for Earliest
# Start/Finish Times (EST/EFT), and backward pass for Latest Start/Finish Times (LST/LFT).
# ==============================================================================
def _compute_cpm_bounds(project: Project) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Compute EST, EFT, LST, LFT across topological ordering."""
    # Kahn's topological sort
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

    # Forward pass
    est: Dict[int, int] = {}
    eft: Dict[int, int] = {}
    for act_id in topo:
        act = project.activities[act_id]
        e_start = max((eft[p] for p in act.predecessors), default=0)
        est[act_id] = e_start
        eft[act_id] = e_start + act.duration

    horizon = max(eft.values(), default=0)

    # Backward pass
    lft: Dict[int, int] = {}
    lst: Dict[int, int] = {}
    for act_id in reversed(topo):
        act = project.activities[act_id]
        l_finish = min((lst[s] for s in act.successors), default=horizon)
        lft[act_id] = l_finish
        lst[act_id] = l_finish - act.duration

    return est, eft, lst, lft


def solve_prh(
    project: Project,
    evaluation_budget: int = 5_000,
    seed: int = 42,
) -> AlgorithmResult:
    """
    Priority Rule Heuristics solver for RCPSP.

    Args:
        project:          RCPSP instance.
        evaluation_budget: Maximum SSGS calls.
        seed:             Random seed for sampling phase.

    Returns:
        An ``AlgorithmResult`` with algorithm = ``"PRH"``.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    evaluations = 0
    convergence_history: List[Tuple[int, int]] = []

    sorted_ids = sorted(project.activities.keys())
    n = len(sorted_ids)

    # ==============================================================================
    # SECTION SUMMARY: Network Metrics & Successor Positional Weights
    # What is in this section: Computes CPM bounds, total descendant sets (MTS),
    # and Greatest Rank Positional Weights (GRPW) for all project activities.
    # ==============================================================================
    est, eft, lst, lft = _compute_cpm_bounds(project)
    horizon = max(eft.values(), default=1)

    all_successors: Dict[int, Set[int]] = {}
    for act_id in reversed(sorted_ids):
        succs = set(project.activities[act_id].successors)
        for s in list(succs):
            succs.update(all_successors.get(s, set()))
        all_successors[act_id] = succs

    grpw: Dict[int, float] = {}
    for act_id in sorted_ids:
        cone = {act_id}.union(all_successors.get(act_id, set()))
        grpw[act_id] = sum(project.activities[s].duration for s in cone)

    # ==============================================================================
    # SECTION SUMMARY: Priority Rule Vector Generation & Deterministic Evaluation
    # What is in this section: Builds normalized priority vectors from 10 classical
    # heuristics (EST, EFT, LST, LFT, MSLK, SPT, LPT, MIS, MTS, GRPW) and evaluates
    # each through the centralized SSGS decoder.
    # ==============================================================================
    rules: Dict[str, Dict[int, float]] = {
        "EST": {a: float(horizon + 1 - est[a]) for a in sorted_ids},
        "EFT": {a: float(horizon + 1 - eft[a]) for a in sorted_ids},
        "LST": {a: float(horizon + 1 - lst[a]) for a in sorted_ids},
        "LFT": {a: float(horizon + 1 - lft[a]) for a in sorted_ids},
        "MSLK": {a: float(horizon + 1 - (lst[a] - est[a])) for a in sorted_ids},
        "SPT": {a: float(horizon + 1 - project.activities[a].duration) for a in sorted_ids},
        "LPT": {a: float(project.activities[a].duration + 1) for a in sorted_ids},
        "MIS": {a: float(len(project.activities[a].successors) + 1) for a in sorted_ids},
        "MTS": {a: float(len(all_successors.get(a, set())) + 1) for a in sorted_ids},
        "GRPW": {a: float(grpw[a] + 1) for a in sorted_ids},
    }

    best_ms: int = 999_999_999
    best_vec: List[float] = [rng.random() for _ in range(n)]
    best_sched = []

    for rule_name, prio_dict in rules.items():
        if evaluations >= evaluation_budget:
            break
        max_p = max(prio_dict.values()) or 1.0
        min_p = min(prio_dict.values())
        span = (max_p - min_p) if (max_p - min_p) > 1e-9 else 1.0
        # Normalize to [0.05, 0.95]
        vec = [0.05 + 0.9 * ((prio_dict[a] - min_p) / span) for a in sorted_ids]

        sched, ms = Evaluator.evaluate(project, vec)
        evaluations += 1

        if ms < best_ms:
            best_ms = ms
            best_vec = vec[:]
            best_sched = sched
            convergence_history.append((evaluations, best_ms))

    # ==============================================================================
    # SECTION SUMMARY: Biased Perturbation & Random Sampling Search Loop
    # What is in this section: Consumes remaining evaluation budget through a hybrid
    # of Gaussian perturbations around the best heuristic priority vector and random sampling.
    # ==============================================================================
    top_base_vec = best_vec[:]
    while evaluations < evaluation_budget:
        if rng.random() < 0.6:
            # Perturb around best priority rule vector
            cand = [max(0.0, min(1.0, g + rng.gauss(0, 0.15))) for g in top_base_vec]
        else:
            cand = [rng.random() for _ in range(n)]

        sched, ms = Evaluator.evaluate(project, cand)
        evaluations += 1

        if ms < best_ms:
            best_ms = ms
            best_vec = cand[:]
            best_sched = sched
            top_base_vec = cand[:]
            convergence_history.append((evaluations, best_ms))

    # ==============================================================================
    # SECTION SUMMARY: Result Compilation & Return
    # What is in this section: Packages the best found schedule, makespan, runtime,
    # and convergence history into an AlgorithmResult object.
    # ==============================================================================
    return AlgorithmResult(
        algorithm="PRH",
        best_priority_vector=best_vec,
        schedule=best_sched,
        makespan=best_ms,
        runtime=time.perf_counter() - start_ts,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )

