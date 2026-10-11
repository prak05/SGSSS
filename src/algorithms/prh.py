"""
Priority Rule-based Heuristic for RCPSP.

Each priority rule assigns a numerical priority to every activity.
The priority vector is decoded through the common SSGS.

Rules implemented
-----------------
EST  — Earliest Start Time  (lower EST → higher priority)
LST  — Latest Start Time    (lower LST → higher priority)
EFT  — Earliest Finish Time (lower EFT → higher priority)
LFT  — Latest Finish Time   (lower LFT → higher priority)
RAND — Random priorities     (for stochastic baseline)

The algorithm evaluates every deterministic rule once and evaluates
RAND up to remaining_budget times, then returns the best result found.

References studied (ideas only, no code copied):
  https://github.com/sydney-machine-learning/MEHH_RCPSP
  UGent RCPSP research — https://www.projectmanagement.ugent.be

Representation : priority dict {act_id: float}  decoded by SSGS
Decoding       : common SSGS via src.evaluation.Evaluator
Objective      : minimise makespan.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

from src.evaluation import Evaluator
from src.model import AlgorithmResult, Project, ScheduleEntry


# ---------------------------------------------------------------------------
# Critical-path utilities (for LST / LFT computation)
# ---------------------------------------------------------------------------

def _compute_est(project: Project) -> Dict[int, int]:
    """Earliest Start Time for each activity via forward pass."""
    sorted_ids = sorted(project.activities.keys())
    est: Dict[int, int] = {}
    for a_id in sorted_ids:
        act = project.activities[a_id]
        if not act.predecessors:
            est[a_id] = 0
        else:
            est[a_id] = max(
                est[p] + project.activities[p].duration
                for p in act.predecessors
            )
    return est


def _compute_eft(project: Project, est: Dict[int, int]) -> Dict[int, int]:
    return {a_id: est[a_id] + project.activities[a_id].duration for a_id in est}


def _compute_lft(project: Project, eft: Dict[int, int]) -> Dict[int, int]:
    """
    Latest Finish Time via backward pass.
    Uses the maximum EFT (project horizon) as the deadline.
    """
    horizon = max(eft.values())
    sorted_ids = sorted(project.activities.keys(), reverse=True)
    lft: Dict[int, int] = {}
    for a_id in sorted_ids:
        act = project.activities[a_id]
        if not act.successors:
            lft[a_id] = horizon
        else:
            lft[a_id] = min(
                lft[s] - project.activities[s].duration
                for s in act.successors
            )
    return lft


def _compute_lst(project: Project, lft: Dict[int, int]) -> Dict[int, int]:
    return {a_id: lft[a_id] - project.activities[a_id].duration for a_id in lft}


# ---------------------------------------------------------------------------
# Rule-to-priority-vector converters
# ---------------------------------------------------------------------------

def _priority_from_rule(
    rule: str,
    project: Project,
    rng: Optional[random.Random] = None,
) -> Dict[int, float]:
    """
    Convert a priority rule name into a {act_id: float} dict suitable
    for the common SSGS.  Higher float = scheduled earlier.
    """
    est = _compute_est(project)
    eft = _compute_eft(project, est)
    lft = _compute_lft(project, eft)
    lst = _compute_lst(project, lft)

    act_ids = list(project.activities.keys())

    if rule == "EST":
        # Lower EST → higher priority → negate and shift to positive
        max_v = max(est.values()) + 1
        return {a: (max_v - est[a]) for a in act_ids}

    elif rule == "EFT":
        max_v = max(eft.values()) + 1
        return {a: (max_v - eft[a]) for a in act_ids}

    elif rule == "LST":
        max_v = max(lst.values()) + 1
        return {a: (max_v - lst[a]) for a in act_ids}

    elif rule == "LFT":
        max_v = max(lft.values()) + 1
        return {a: (max_v - lft[a]) for a in act_ids}

    elif rule == "RAND":
        if rng is None:
            raise ValueError("rng must be provided for RAND rule")
        return {a: rng.random() for a in act_ids}

    else:
        raise ValueError(f"Unknown priority rule: {rule}")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

DETERMINISTIC_RULES = ["EST", "EFT", "LST", "LFT"]
RANDOM_RULE = "RAND"
ALL_RULES = DETERMINISTIC_RULES + [RANDOM_RULE]


def solve(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
) -> AlgorithmResult:
    """
    Evaluate all deterministic priority rules and use remaining budget
    on randomised RAND evaluations.  Return the best result found.

    Engineering note
    ----------------
    Deterministic rules (EST, EFT, LST, LFT) each produce exactly one
    schedule; they consume 4 evaluations.  The remaining budget is
    allocated to RAND trials.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()

    evaluations = 0
    convergence_history: List[Tuple[int, int]] = []

    best_ms: Optional[int] = None
    best_vec: Optional[List[float]] = None
    best_sched: Optional[List[ScheduleEntry]] = None
    best_rule: str = ""

    # ---- Deterministic rules ----
    for rule in DETERMINISTIC_RULES:
        if evaluations >= evaluation_budget:
            break
        prio = _priority_from_rule(rule, project)
        # Normalise to [0, 1] for a consistent representation
        max_p = max(prio.values()) or 1.0
        normed = {a: v / max_p for a, v in prio.items()}

        _, ms, sched = Evaluator.evaluate(project, normed)
        evaluations += 1

        # Convert dict to sorted list for storage
        sorted_ids = sorted(project.activities.keys())
        vec = [normed[a] for a in sorted_ids]

        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_vec = vec
            best_sched = sched
            best_rule = rule
            convergence_history.append((evaluations, best_ms))

    # ---- Random rule — consume remaining budget ----
    while evaluations < evaluation_budget:
        prio = _priority_from_rule(RANDOM_RULE, project, rng=rng)
        sorted_ids = sorted(project.activities.keys())
        vec = [prio[a] for a in sorted_ids]

        _, ms, sched = Evaluator.evaluate(project, vec)
        evaluations += 1

        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_vec = vec
            best_sched = sched
            best_rule = RANDOM_RULE
            convergence_history.append((evaluations, best_ms))

    runtime = time.perf_counter() - start_ts

    return AlgorithmResult(
        algorithm=f"PRH[{best_rule}]",
        best_priority_vector=best_vec or [],
        schedule=best_sched or [],
        makespan=best_ms or -1,
        runtime=runtime,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )
