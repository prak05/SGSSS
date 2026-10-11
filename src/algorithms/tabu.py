"""
Tabu Search for RCPSP.

Representation : random-key vector — one float ∈ [0, 1] per activity.
Decoding       : common SSGS via src.evaluation.Evaluator
Objective      : minimise makespan.

Tabu mechanism
--------------
A move is identified by (index, rounded_old_value).  After accepting a
move the pair is added to the tabu list for `tenure` iterations.
Aspiration criterion: a tabu move is accepted if its makespan strictly
beats the global best.

References studied (ideas only, no code copied):
  https://github.com/ThomasK33/rcpsp.rs
  https://github.com/CTU-IIG/RCPSPCpu
"""

import random
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

from src.evaluation import Evaluator
from src.model import AlgorithmResult, Project, ScheduleEntry


# ---------------------------------------------------------------------------
# Neighbourhood generation
# ---------------------------------------------------------------------------

def _generate_neighbourhood(
    current: List[float],
    rng: random.Random,
    n_neighbours: int = 10,
) -> List[Tuple[List[float], Tuple[int, int]]]:
    """
    Generate candidate (neighbour_vector, move_id) pairs.

    A move is a swap of two positions.  move_id = (min(i,j), max(i,j)).
    Swap is chosen because it preserves the [0,1] range exactly and keeps
    the neighbourhood well-defined for tabu tracking.
    """
    n = len(current)
    neighbours = []
    seen_moves = set()
    attempts = 0

    while len(neighbours) < n_neighbours and attempts < n_neighbours * 5:
        attempts += 1
        i, j = sorted(rng.sample(range(n), 2))
        move_id = (i, j)
        if move_id in seen_moves:
            continue
        seen_moves.add(move_id)
        v = current[:]
        v[i], v[j] = v[j], v[i]
        neighbours.append((v, move_id))

    return neighbours


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def solve(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
    tenure: int = 7,
    n_neighbours: int = 10,
) -> AlgorithmResult:
    """
    Run Tabu Search on *project*.

    Parameters
    ----------
    evaluation_budget : int
        Maximum number of SSGS evaluations.
    seed : int
        Controls all randomness for reproducibility.
    tenure : int
        Number of iterations a move stays tabu.
    n_neighbours : int
        Neighbourhood size per iteration.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()

    sorted_ids = sorted(project.activities.keys())
    n = len(sorted_ids)
    evaluations = 0
    convergence_history: List[Tuple[int, int]] = []

    # Initial solution
    current = [rng.random() for _ in range(n)]
    _, current_ms, current_sched = Evaluator.evaluate(project, current)
    evaluations += 1

    best = current[:]
    best_ms = current_ms
    best_sched = current_sched
    convergence_history.append((evaluations, best_ms))

    tabu_list: Deque[Tuple[int, int]] = deque()
    tabu_set = set()

    while evaluations < evaluation_budget:
        neighbours = _generate_neighbourhood(current, rng, n_neighbours)

        best_cand: Optional[List[float]] = None
        best_cand_ms: Optional[int] = None
        best_cand_sched: Optional[List[ScheduleEntry]] = None
        best_cand_move: Optional[Tuple[int, int]] = None

        for vec, move_id in neighbours:
            if evaluations >= evaluation_budget:
                break

            _, ms, sched = Evaluator.evaluate(project, vec)
            evaluations += 1

            is_tabu = move_id in tabu_set
            aspiration = ms < best_ms  # accept tabu move if global best beaten

            if is_tabu and not aspiration:
                continue

            if best_cand_ms is None or ms < best_cand_ms:
                best_cand = vec
                best_cand_ms = ms
                best_cand_sched = sched
                best_cand_move = move_id

        if best_cand is None:
            # All neighbours were tabu and none triggered aspiration;
            # accept the first non-tabu candidate evaluated regardless of quality.
            for vec, move_id in neighbours:
                if move_id not in tabu_set:
                    best_cand = vec
                    best_cand_ms = None
                    best_cand_sched = None
                    best_cand_move = move_id
                    break

            if best_cand is None:
                # Entire neighbourhood is tabu — restart from a random point
                current = [rng.random() for _ in range(n)]
                if evaluations < evaluation_budget:
                    _, current_ms, current_sched = Evaluator.evaluate(project, current)
                    evaluations += 1
                tabu_list.clear()
                tabu_set.clear()
                continue

        # Move to best candidate
        current = best_cand
        if best_cand_ms is not None:
            current_ms = best_cand_ms
            current_sched = best_cand_sched or []

        # Update tabu list
        if best_cand_move is not None:
            tabu_list.append(best_cand_move)
            tabu_set.add(best_cand_move)
            if len(tabu_list) > tenure:
                evicted = tabu_list.popleft()
                tabu_set.discard(evicted)

        # Update global best
        if best_cand_ms is not None and best_cand_ms < best_ms:
            best_ms = best_cand_ms
            best = best_cand[:]
            best_sched = best_cand_sched or []
            convergence_history.append((evaluations, best_ms))

    runtime = time.perf_counter() - start_ts

    return AlgorithmResult(
        algorithm="TS",
        best_priority_vector=best,
        schedule=best_sched,
        makespan=best_ms,
        runtime=runtime,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )
