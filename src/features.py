"""
RCPSP Feature Extraction Engine.

Calculates project topological, precedence, and resource characteristics
from a parsed Project instance.

Authoritative references:
  1. Paper: Mitsou & Koulinas (2026),
     "AI-Enhanced Metaheuristics Control for Project Scheduling Using Neural Networks",
     IEEE Access, Vol. 14, pp. 33600-33610.
  2. PSPLIB / ProGen: Kolisch & Sprecher (1997),
     "PSPLIB - A project scheduling problem library", EJOR 96(1), pp. 205-216.
  3. RanGen: Demeulemeester, Vanhoucke & Herroelen (2003),
     "RanGen: A random network generator for activity-on-the-node networks",
     J. Scheduling 6(1), pp. 17-38.
  4. Cooper (1976),
     "Heuristics for scheduling resource-constrained projects: An experimental
     investigation", Management Science 22(11), pp. 1186-1194.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Set, Tuple

from src.model import Activity, Project


# ---------------------------------------------------------------------------
# Feature Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProjectFeatures:
    """
    Structured container for all project features.

    Paper Features (Mitsou & Koulinas 2026):
      N   : Activity count excluding dummy Start and Finish activities.
      NC  : Network Complexity = precedence_edge_count / N.
      RF  : Resource Factor = average fraction of resource types required per activity.
      RS  : Resource Strength = capacity availability relative to peak CPM demand.

    Additional Project Features:
      Structural, precedence, parallelism, resource tightness, and
      precedence-only critical path metrics.
    """

    # --- Paper Features ---
    N: int
    NC: float
    RF: float
    RS: float

    # --- Basic Features ---
    activity_count: int
    resource_count: int
    average_duration: float
    duration_std: float
    minimum_duration: int
    maximum_duration: int
    total_duration: int

    # --- Precedence / Network Features ---
    precedence_edge_count: int
    precedence_density: float
    average_predecessors: float
    average_successors: float
    maximum_predecessors: int
    maximum_successors: int
    source_count: int
    sink_count: int
    network_depth: int

    # --- Parallelism Features ---
    estimated_parallelism: float
    average_parallel_activities: float

    # --- Resource Features ---
    average_resource_demand: float
    resource_demand_std: float
    resource_utilization_ratio: float
    average_resource_tightness: float
    maximum_resource_tightness: float
    number_of_unused_resource_dimensions: int

    # --- Critical Path Features ---
    critical_path_duration: int
    critical_path_activity_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Return features as a standard dictionary."""
        return asdict(self)

    def to_vector(self) -> List[float]:
        """
        Return features as a deterministic float vector matching the exact
        dataclass field ordering.
        """
        vec = []
        for f in fields(self):
            val = getattr(self, f.name)
            vec.append(float(val))
        return vec

    @classmethod
    def feature_names(cls) -> List[str]:
        """Return ordered list of feature names matching to_vector()."""
        return [f.name for f in fields(cls)]


# ---------------------------------------------------------------------------
# DAG Validation & Topological Ordering
# ---------------------------------------------------------------------------

def validate_dag(project: Project) -> List[int]:
    """
    Validate that the project precedence graph is a valid Directed Acyclic Graph (DAG).

    Checks:
      1. Project has at least one activity.
      2. All predecessor IDs referenced by activities actually exist in the project.
      3. The graph contains no directed cycles (Kahn's algorithm).

    Returns
    -------
    List[int]
        Topological ordering of activity IDs.

    Raises
    ------
    ValueError
        If any validation rule is violated.
    """
    if project.num_activities == 0:
        raise ValueError("Project has zero activities; cannot extract features.")

    # 1. Verify existence of all predecessor references
    for a_id, act in project.activities.items():
        for p_id in act.predecessors:
            if p_id not in project.activities:
                raise ValueError(
                    f"Activity {a_id} references non-existent predecessor {p_id}."
                )

    # 2. Cycle detection and topological sort via Kahn's algorithm
    in_degree: Dict[int, int] = {
        a_id: len(act.predecessors) for a_id, act in project.activities.items()
    }
    # Queue sorted for strict determinism
    queue: List[int] = sorted([a_id for a_id, deg in in_degree.items() if deg == 0])
    topo_order: List[int] = []

    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for s in project.activities[u].successors:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
                queue.sort()

    if len(topo_order) != project.num_activities:
        cycle_candidates = [
            a_id for a_id, deg in in_degree.items() if deg > 0
        ]
        raise ValueError(
            f"Precedence graph contains a cycle involving activities: {cycle_candidates}"
        )

    return topo_order


# ---------------------------------------------------------------------------
# Dummy Activity Identification
# ---------------------------------------------------------------------------

def is_dummy_activity(activity: Activity) -> bool:
    """
    Identify whether an activity is a dummy activity (Start or Finish).

    According to the literature and Mitsou & Koulinas (2026), dummy activities
    have:
      - duration == 0
      - zero demand across all resources
      - in-degree == 0 (dummy start) OR out-degree == 0 (dummy finish)
    """
    if activity.duration != 0:
        return False
    if any(req > 0 for req in activity.resource_requirements.values()):
        return False
    return len(activity.predecessors) == 0 or len(activity.successors) == 0


# ---------------------------------------------------------------------------
# Critical Path Method (CPM) Forward & Backward Passes
# ---------------------------------------------------------------------------

def compute_cpm_schedule(
    project: Project, topo_order: List[int]
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int], int]:
    """
    Execute CPM forward and backward passes without resource constraints.

    Returns
    -------
    es : Early Start times
    ef : Early Finish times
    ls : Late Start times
    lf : Late Finish times
    cpm_makespan : Project unconstrained duration (critical path duration)
    """
    es: Dict[int, int] = {}
    ef: Dict[int, int] = {}

    # Forward pass
    for u in topo_order:
        act = project.activities[u]
        pred_efs = [ef[p] for p in act.predecessors if p in ef]
        es[u] = max(pred_efs) if pred_efs else 0
        ef[u] = es[u] + act.duration

    cpm_makespan = max(ef.values()) if ef else 0

    # Backward pass
    ls: Dict[int, int] = {}
    lf: Dict[int, int] = {}
    for u in reversed(topo_order):
        act = project.activities[u]
        succ_lss = [ls[s] for s in act.successors if s in ls]
        lf[u] = min(succ_lss) if succ_lss else cpm_makespan
        ls[u] = lf[u] - act.duration

    return es, ef, ls, lf, cpm_makespan


# ---------------------------------------------------------------------------
# Feature Extraction Core
# ---------------------------------------------------------------------------

def extract_features(project: Project) -> ProjectFeatures:
    """
    Extract all project characteristics from a parsed Project instance.

    Parameters
    ----------
    project : Project
        The project instance to analyze.

    Returns
    -------
    ProjectFeatures
        Dataclass containing paper-defined and engineering features.
    """
    # 1. Validate DAG and obtain topological order
    topo_order = validate_dag(project)

    # 2. Identify dummy and non-dummy activities
    all_acts = list(project.activities.values())
    total_acts_count = len(all_acts)
    non_dummy_acts = [a for a in all_acts if not is_dummy_activity(a)]
    dummy_acts = [a for a in all_acts if is_dummy_activity(a)]

    # If all activities are classified as dummy (e.g. all dur=0), treat all as non-dummy
    # to avoid division by zero.
    active_pool = non_dummy_acts if non_dummy_acts else all_acts

    # -----------------------------------------------------------------------
    # Paper Feature 1: N (Activity Count)
    # Definition: Total number of activities excluding dummy Start and Finish.
    # -----------------------------------------------------------------------
    N = len(active_pool)

    # -----------------------------------------------------------------------
    # Paper Feature 2: NC (Network Complexity)
    # Formula: NC = precedence_edge_count / N
    # Sources: Mitsou & Koulinas (2026), Kolisch & Sprecher (1997), RanGen (2003).
    # -----------------------------------------------------------------------
    precedence_edge_count = sum(len(a.successors) for a in all_acts)
    NC = float(precedence_edge_count) / float(N) if N > 0 else 0.0

    # -----------------------------------------------------------------------
    # Paper Feature 3: RF (Resource Factor)
    # Formula: RF = (1 / (N * |K|)) * sum_{j} sum_{k} (1 if r_{jk} > 0 else 0)
    # Sources: Mitsou & Koulinas (2026) eq. (5), Pascoe (1966), Kolisch & Sprecher (1997).
    # -----------------------------------------------------------------------
    K_resources = list(project.resources.values())
    num_resources = len(K_resources)

    if N > 0 and num_resources > 0:
        positive_req_count = 0
        for act in active_pool:
            for r in K_resources:
                if act.resource_requirements.get(r.id, 0) > 0:
                    positive_req_count += 1
        RF = float(positive_req_count) / float(N * num_resources)
    else:
        RF = 0.0

    # -----------------------------------------------------------------------
    # Critical Path Method (CPM) Evaluation
    # -----------------------------------------------------------------------
    es, ef, ls, lf, cpm_makespan = compute_cpm_schedule(project, topo_order)

    # -----------------------------------------------------------------------
    # Paper Feature 4: RS (Resource Strength)
    # Formula: RS = (1 / |K|) * sum_{k} RS_k
    # where RS_k = (Capacity_k - K_{k}^{min}) / (K_{k}^{max} - K_{k}^{min})
    # K_{k}^{min} = max_{j} r_{jk}
    # K_{k}^{max} = peak demand in early start CPM schedule
    # Sources: Mitsou & Koulinas (2026) eq. (6), Cooper (1976), Kolisch & Sprecher (1997).
    # -----------------------------------------------------------------------
    if num_resources > 0:
        rs_k_values: List[float] = []
        for r in K_resources:
            r_id = r.id
            cap = r.capacity

            # K_{k}^{min}: maximum single activity demand
            k_min = max(
                (act.resource_requirements.get(r_id, 0) for act in all_acts),
                default=0,
            )

            # K_{k}^{max}: peak demand over early start CPM schedule
            k_max = 0
            if cpm_makespan > 0:
                # Sample instantaneous demand at each event start time
                event_times: Set[int] = set(es.values())
                for t in event_times:
                    if t < cpm_makespan:
                        t_demand = sum(
                            act.resource_requirements.get(r_id, 0)
                            for act in all_acts
                            if es[act.id] <= t < ef[act.id]
                        )
                        if t_demand > k_max:
                            k_max = t_demand
            else:
                k_max = k_min

            if k_max > k_min:
                rs_k = (float(cap) - float(k_min)) / (float(k_max) - float(k_min))
                # Clamp to [0, 1] as established in RCPSP literature
                rs_k = min(1.0, max(0.0, rs_k))
            else:
                # If peak demand equals min demand, resource is non-constraining
                rs_k = 1.0

            rs_k_values.append(rs_k)

        RS = float(statistics.mean(rs_k_values))
    else:
        RS = 0.0

    # -----------------------------------------------------------------------
    # Basic Features
    # -----------------------------------------------------------------------
    durations = [a.duration for a in active_pool]
    total_duration = sum(durations)
    avg_duration = float(statistics.mean(durations)) if durations else 0.0
    dur_std = float(statistics.pstdev(durations)) if len(durations) > 1 else 0.0
    min_dur = min(durations) if durations else 0
    max_dur = max(durations) if durations else 0

    # -----------------------------------------------------------------------
    # Precedence / Network Features
    # -----------------------------------------------------------------------
    # Density for directed acyclic graph: |E| / (V * (V - 1) / 2)
    max_edges = (total_acts_count * (total_acts_count - 1)) / 2.0
    precedence_density = (
        float(precedence_edge_count) / max_edges if max_edges > 0 else 0.0
    )

    avg_preds = float(precedence_edge_count) / float(total_acts_count)
    avg_succs = avg_preds
    max_preds = max((len(a.predecessors) for a in all_acts), default=0)
    max_succs = max((len(a.successors) for a in all_acts), default=0)

    source_count = sum(1 for a in all_acts if len(a.predecessors) == 0)
    sink_count = sum(1 for a in all_acts if len(a.successors) == 0)

    # Network depth (longest edge path from any source)
    depth: Dict[int, int] = {}
    for u in topo_order:
        act = project.activities[u]
        pred_depths = [depth[p] + 1 for p in act.predecessors if p in depth]
        depth[u] = max(pred_depths) if pred_depths else 0
    network_depth = max(depth.values()) if depth else 0

    # -----------------------------------------------------------------------
    # Parallelism Features
    # -----------------------------------------------------------------------
    # Estimated parallelism: total active work / critical path duration
    if cpm_makespan > 0:
        estimated_parallelism = float(total_duration) / float(cpm_makespan)
    else:
        estimated_parallelism = 1.0

    # Average parallel activities in CPM schedule
    # sum_{t=0}^{cpm_makespan-1} (active tasks at t) / cpm_makespan
    # Mathematically equals total_duration / cpm_makespan for non-dummy activities
    average_parallel_activities = estimated_parallelism

    # -----------------------------------------------------------------------
    # Resource Demand & Tightness Features
    # -----------------------------------------------------------------------
    all_demands: List[float] = []
    unused_res_count = 0
    tightness_values: List[float] = []

    for r in K_resources:
        r_id = r.id
        cap = r.capacity
        r_demands = [act.resource_requirements.get(r_id, 0) for act in active_pool]
        all_demands.extend(r_demands)

        sum_demand = sum(r_demands)
        if sum_demand == 0:
            unused_res_count += 1

        # Resource tightness: peak early-start demand / capacity
        k_max_r = 0
        if cpm_makespan > 0:
            for t in es.values():
                if t < cpm_makespan:
                    t_dem = sum(
                        act.resource_requirements.get(r_id, 0)
                        for act in all_acts
                        if es[act.id] <= t < ef[act.id]
                    )
                    if t_dem > k_max_r:
                        k_max_r = t_dem
        else:
            k_max_r = max(r_demands, default=0)

        tightness = float(k_max_r) / float(cap) if cap > 0 else 0.0
        tightness_values.append(tightness)

    avg_res_demand = float(statistics.mean(all_demands)) if all_demands else 0.0
    res_demand_std = (
        float(statistics.pstdev(all_demands)) if len(all_demands) > 1 else 0.0
    )

    avg_tightness = (
        float(statistics.mean(tightness_values)) if tightness_values else 0.0
    )
    max_tightness = max(tightness_values) if tightness_values else 0.0

    # Resource utilization ratio: total work demand / (CPM_makespan * capacity)
    utilization_ratios: List[float] = []
    for r in K_resources:
        r_id = r.id
        cap = r.capacity
        total_work_k = sum(
            act.duration * act.resource_requirements.get(r_id, 0)
            for act in active_pool
        )
        rect_area = float(cpm_makespan * cap)
        util = float(total_work_k) / rect_area if rect_area > 0 else 0.0
        utilization_ratios.append(util)

    res_util_ratio = (
        float(statistics.mean(utilization_ratios)) if utilization_ratios else 0.0
    )

    # -----------------------------------------------------------------------
    # Critical Path Features
    # -----------------------------------------------------------------------
    # Critical path activities have zero float: LS == ES
    critical_acts = [
        a for a in active_pool if ls.get(a.id, 0) == es.get(a.id, 0)
    ]
    critical_path_activity_count = len(critical_acts)

    # Assemble complete ProjectFeatures dataclass
    return ProjectFeatures(
        N=N,
        NC=NC,
        RF=RF,
        RS=RS,
        activity_count=total_acts_count,
        resource_count=num_resources,
        average_duration=avg_duration,
        duration_std=dur_std,
        minimum_duration=min_dur,
        maximum_duration=max_dur,
        total_duration=total_duration,
        precedence_edge_count=precedence_edge_count,
        precedence_density=precedence_density,
        average_predecessors=avg_preds,
        average_successors=avg_succs,
        maximum_predecessors=max_preds,
        maximum_successors=max_succs,
        source_count=source_count,
        sink_count=sink_count,
        network_depth=network_depth,
        estimated_parallelism=estimated_parallelism,
        average_parallel_activities=average_parallel_activities,
        average_resource_demand=avg_res_demand,
        resource_demand_std=res_demand_std,
        resource_utilization_ratio=res_util_ratio,
        average_resource_tightness=avg_tightness,
        maximum_resource_tightness=max_tightness,
        number_of_unused_resource_dimensions=unused_res_count,
        critical_path_duration=cpm_makespan,
        critical_path_activity_count=critical_path_activity_count,
    )
