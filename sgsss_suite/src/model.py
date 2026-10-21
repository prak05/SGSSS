"""
================================================================================
src/model.py — Core Data Classes for AI-RCPSP
================================================================================
Defines the canonical data structures used by EVERY module in the project:
  • Resource        — renewable resource with fixed per-period capacity
  • Activity        — task node in the precedence DAG
  • Project         — full RCPSP instance (activities + resources)
  • ScheduleEntry   — one row of a decoded schedule
  • AlgorithmResult — standardised solver output (all 5 algorithms return this)
  • determine_winner — multi-tier tie-breaking tournament resolver

Design notes
────────────
* Dataclasses are *immutable* in intent — solvers MUST NOT mutate a Project
  after construction.
* All IDs are **1-based integers** to match PSPLIB conventions.
* `resource_requirements` maps resource string IDs (e.g. "R1") to integer
  demand per time step while the activity is active.

References
──────────
[1] Mitsou & Koulinas (2026), AI-Enhanced Metaheuristics for RCPSP.
[12] System Architecture Specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# SECTION SUMMARY: Renewable Resource Model
# What is in this section: Defines the Resource dataclass representing renewable
# project assets (workers, equipment) with constant capacity available each time step.
# ==============================================================================

@dataclass
class Resource:
    """
    A renewable resource available every time period at constant capacity.

    Examples: workers, machines, equipment, tools.
    Attributes:
        id:       Unique string tag, e.g. "R1", "R2".
        capacity: Maximum units available at every discrete time step.
    """
    id: str
    capacity: int

    @property
    def name(self) -> str:
        return str(self.id)


# ==============================================================================
# SECTION SUMMARY: Activity / Task Model
# What is in this section: Defines the Activity dataclass representing individual
# project tasks in the Directed Acyclic Graph (DAG) with duration, predecessor,
# successor links, and per-step resource consumption requirements.
# ==============================================================================

@dataclass
class Activity:
    """
    A project task in the precedence DAG.

    In PSPLIB convention:
      * Activity 1       → dummy source (duration = 0, demands = 0).
      * Activity N + 2   → dummy sink   (duration = 0, demands = 0).

    Attributes:
        id:                    1-based integer activity identifier.
        duration:              Processing time in discrete time steps (>= 0).
        predecessors:          IDs of tasks that MUST finish before this starts.
        successors:            IDs of tasks that CANNOT start until this finishes.
        resource_requirements: {resource_id: units_per_time_step} while active.
    """
    id: int
    duration: int
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    resource_requirements: Dict[str, int] = field(default_factory=dict)

    @property
    def demands(self) -> Dict[str, int]:
        return self.resource_requirements


# ==============================================================================
# SECTION SUMMARY: Project Container Model
# What is in this section: Defines the Project dataclass containing the entire
# problem instance (all activities, renewable resources, and parsed metadata).
# ==============================================================================

@dataclass
class Project:
    """
    Complete RCPSP problem instance: a set of activities + renewable resources.

    Attributes:
        activities: {activity_id: Activity} mapping.
        resources:  {resource_id: Resource} mapping.
        metadata:   Arbitrary key-value pairs parsed from the instance file.
        file_seed:  Random generator seed extracted from the file header.
        file_budget: Evaluation budget extracted from / inferred for this file.
    """
    activities: Dict[int, Activity] = field(default_factory=dict)
    resources: Dict[str, Resource] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_seed: Optional[int] = None
    file_budget: Optional[int] = None

    @property
    def num_activities(self) -> int:
        """Total activity count including dummy source/sink nodes."""
        return len(self.activities)

    @property
    def num_resources(self) -> int:
        """Number of distinct renewable resource types."""
        return len(self.resources)

    @property
    def renewable_capacities(self) -> Dict[str, int]:
        """Dictionary of resource capacities {resource_id: capacity}."""
        return {r_id: r.capacity for r_id, r in self.resources.items()}

    @property
    def real_activities(self) -> List[Activity]:
        """List of real (non-dummy) project tasks requiring processing time."""
        return [
            act for act in self.activities.values()
            if act.duration > 0 or (act.predecessors and act.successors)
        ]

    @property
    def cpm_lower_bound(self) -> int:
        """Critical Path Method (CPM) theoretical minimum makespan with infinite resources."""
        return compute_cpm_lower_bound(self)

    @property
    def total_work_content(self) -> int:
        """Total work content (sum of durations of all activities)."""
        return sum(act.duration for act in self.activities.values())


# ==============================================================================
# SECTION SUMMARY: Directed Acyclic Graph (DAG) Validation & CPM Calculation
# What is in this section: Kahn's topological sort algorithm to verify acyclicity
# and Critical Path Method (CPM) forward pass for lower-bound duration estimation.
# ==============================================================================

def validate_dag(project: Project) -> List[int]:
    """
    Validate that the project precedence graph is acyclic and return a topological order.

    Uses Kahn's algorithm:
      1. Compute in-degree for each node.
      2. Enqueue all nodes with in-degree 0.
      3. Dequeue, decrement successors, enqueue when in-degree reaches 0.
      4. If all nodes visited -> valid DAG; else -> cycle detected.

    Raises:
        ValueError: if a precedence cycle is detected.
    """
    in_degree: Dict[int, int] = {
        a: len(act.predecessors)
        for a, act in project.activities.items()
    }
    queue = deque(a for a, deg in in_degree.items() if deg == 0)
    topo: List[int] = []

    while queue:
        curr = queue.popleft()
        topo.append(curr)
        for s in project.activities[curr].successors:
            if s in in_degree:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)

    if len(topo) != len(project.activities):
        raise ValueError("Cycle detected in project precedence graph — not a valid DAG.")
    return topo


def compute_cpm_lower_bound(project: Project) -> int:
    """
    Compute Critical Path Method (CPM) theoretical minimum makespan assuming infinite resources.
    Performs a topological forward pass: EST[i] = max(EFT[p] for p in Pred(i)).
    """
    if not project.activities:
        return 0
    try:
        topo_order = validate_dag(project)
    except Exception:
        return 0

    eft: Dict[int, int] = {}
    for a_id in topo_order:
        act = project.activities[a_id]
        est_val = max((eft[p] for p in act.predecessors), default=0)
        eft[a_id] = est_val + act.duration

    return max(eft.values(), default=0)


# ==============================================================================
# SECTION SUMMARY: Schedule Entry Record
# What is in this section: Defines the ScheduleEntry dataclass representing an
# activity assigned to a concrete execution window [start_time, finish_time].
# ==============================================================================

@dataclass
class ScheduleEntry:
    """
    A single scheduled activity with assigned times and resource allocation.

    finish_time == start_time + duration always holds.

    Attributes:
        activity_id:    Task integer identifier.
        start_time:     Earliest feasible start step (>= 0).
        finish_time:    Completion step.
        duration:       Processing duration.
        resource_usage: Resource consumption {resource_id: units} per step.
    """
    activity_id: int
    start_time: int
    finish_time: int
    duration: int
    resource_usage: Dict[str, int]


# ==============================================================================
# SECTION SUMMARY: Solver Algorithm Result Container
# What is in this section: Defines the AlgorithmResult dataclass returned by every
# solver (makespan, runtime, evaluations, schedule, and convergence trajectory).
# ==============================================================================

@dataclass
class AlgorithmResult:
    """
    Standardised output returned by every solver (GA, SA, BCO, PRH, TS).

    Attributes:
        algorithm:            Solver name string ("GA", "SA", etc.).
        best_priority_vector: Best random-key vector found.
        schedule:             Complete feasible schedule.
        makespan:             Cmax of the best schedule found.
        runtime:              Wall-clock seconds elapsed.
        evaluations:          Number of SSGS decoder calls consumed.
        convergence_history:  [(eval_count, best_makespan), ...].
    """
    algorithm: str
    best_priority_vector: List[float]
    schedule: List[ScheduleEntry]
    makespan: int
    runtime: float
    evaluations: int
    convergence_history: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def best_eval(self) -> int:
        """Evaluation count when the best makespan was first discovered."""
        if self.convergence_history:
            return self.convergence_history[-1][0]
        return self.evaluations

    @property
    def best_schedule(self) -> List[ScheduleEntry]:
        return self.schedule

    @property
    def history(self) -> List[Tuple[int, int]]:
        return self.convergence_history

    @property
    def best_makespan(self) -> int:
        return self.makespan

    @property
    def runtime_seconds(self) -> float:
        return self.runtime

    @property
    def algorithm_name(self) -> str:
        return self.algorithm


# ==============================================================================
# SECTION SUMMARY: Single Tournament Winner Resolution
# What is in this section: Defines determine_winner(), which applies multi-tier
# lexicographical tie-breaking so that exactly one algorithm wins without ties.
# ==============================================================================

def determine_winner(results: Dict[str, AlgorithmResult]) -> Tuple[str, AlgorithmResult]:
    """
    Deterministically select EXACTLY ONE SINGLE WINNER algorithm across all results.

    Resolves ties via formal lexicographical ranking:
      1. Makespan (lowest wins).
      2. Fewest SSGS evaluations to first find that best makespan (faster convergence).
      3. Fastest computation runtime (wall-clock seconds).
      4. Tie-breaking preference: TS > PRH > GA > SA > BCO.

    Returns:
        (winner_name, winner_result)
    """
    if not results:
        raise ValueError("Cannot determine winner from empty results dictionary.")

    priority_order = {"TS": 0, "PRH": 1, "GA": 2, "SA": 3, "BCO": 4}

    def rank_key(item: Tuple[str, AlgorithmResult]):
        name, res = item
        best_eval = res.convergence_history[-1][0] if res.convergence_history else res.evaluations
        pref = priority_order.get(name.upper(), 99)
        return (res.makespan, best_eval, res.runtime, pref)

    winner_name, winner_res = min(results.items(), key=rank_key)
    return winner_name, winner_res
