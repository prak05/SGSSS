"""
================================================================================
AI-RCPSP Complete All-in-One Framework (Single-File Reference Implementation)
================================================================================
File: rcpsp_all_in_one.py
Description: A self-contained, 100% production-ready, single-file implementation
             of the Resource-Constrained Project Scheduling Problem (RCPSP)
             optimization suite.

FOR TEACHER / VIVA EXAM PRESENTATION:
- Section 1 (Line ~60):   Data Models (Resource, Activity, Project, ScheduleEntry, AlgorithmResult)
- Section 2 (Line ~135):  SSGS Decoder (Serial Schedule Generation Scheme)
- Section 3 (Line ~255):  30-Dimension Feature Extraction Engine & Math (N, NC, RF, RS, CPM)
- Section 4 (Line ~530):  Multi-Format File Parser (.sm PSPLIB, .rcp Paterson, .json)
- Section 5 (Line ~775):  Evaluator & Dual-Layer Constraint Validator
- Section 6 (Line ~780):  5 Metaheuristic Solvers (GA, SA, BCO, Tabu Search, Priority Rules)
- Section 7 (Line ~1195): Benchmark Execution & Statistical Comparison Engine
- Section 8 (Line ~1330): Command Line Interface (CLI) Execution Controller

Usage as Script:
    python rcpsp_all_in_one.py data/j10_sample.sm

Usage as Module:
    from rcpsp_all_in_one import Parser, extract_features, run_benchmark, compare_algorithms
    project = Parser.parse("data/j10_sample.sm")
    features = extract_features(project)
    records = run_benchmark(project, seeds=[42], evaluation_budget=500)
    report = compare_algorithms(records)
    print(report.to_text())
================================================================================
"""

from __future__ import annotations

import json
import math
import pickle
import random
import re
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

# Optional numpy / sklearn imports for ML inference
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ==============================================================================
# SECTION 1: DATA MODELS
# ==============================================================================

@dataclass
class Resource:
    """
    Represents a renewable resource (e.g., workers, machines, equipment).
    - id: Resource identifier (e.g. 'R1', 'R2')
    - capacity: Maximum available units per time period (constant)
    """
    id: str
    capacity: int


@dataclass
class Activity:
    """
    Represents a project activity/task.
    - id: Activity ID (1 to N+2, where 1 is dummy start, N+2 is dummy sink)
    - duration: Execution time in time steps
    - predecessors: List of activity IDs that MUST complete before this task starts
    - successors: List of activity IDs that CANNOT start until this task finishes
    - resource_requirements: Dict mapping resource ID -> units required per time step
    """
    id: int
    duration: int
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    resource_requirements: Dict[str, int] = field(default_factory=dict)


@dataclass
class Project:
    """
    Container storing all activities and renewable resources for an RCPSP instance.
    """
    activities: Dict[int, Activity] = field(default_factory=dict)
    resources: Dict[str, Resource] = field(default_factory=dict)

    @property
    def num_activities(self) -> int:
        return len(self.activities)

    @property
    def num_resources(self) -> int:
        return len(self.resources)


@dataclass
class ScheduleEntry:
    """
    Result of decoding an activity: contains assigned start/finish times.
    - activity_id: Task identifier
    - start_time: Scheduled start time step
    - finish_time: Scheduled completion time step (start_time + duration)
    - duration: Task duration
    - resource_usage: Active resource consumption per time unit
    """
    activity_id: int
    start_time: int
    finish_time: int
    duration: int
    resource_usage: Dict[str, int]


@dataclass
class AlgorithmResult:
    """
    Standard output structure returned by all 5 optimization solvers.
    - algorithm: Name of solver ('GA', 'SA', 'BCO', 'TS', 'PRH')
    - best_priority_vector: Optimized random-key priorities
    - schedule: List of ScheduleEntry objects for all activities
    - makespan: Total completion time of project (S_{N+2})
    - runtime: Execution time in seconds
    - evaluations: Number of SSGS calls made
    - convergence_history: List of (eval_count, best_makespan) tuples for plotting
    """
    algorithm: str
    best_priority_vector: List[float]
    schedule: List[ScheduleEntry]
    makespan: int
    runtime: float
    evaluations: int
    convergence_history: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class BenchmarkRecord:
    """Result record from a single benchmark execution run."""
    instance_id: str
    algorithm: str
    seed: int
    makespan: int
    runtime: float
    evaluations: int
    feasible: bool
    best_priority_vector: List[float]
    convergence_history: List[Tuple[int, int]] = field(default_factory=list)
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.instance_id}/{self.algorithm}/seed={self.seed}"


# ==============================================================================
# SECTION 2: SERIAL SCHEDULE GENERATION SCHEME (SSGS) DECODER
# ==============================================================================

class SSGS:
    """
    Serial Schedule Generation Scheme (SSGS).
    The core deterministic decoder that converts continuous Random-Key Priority Vectors
    into 100% precedence- and resource-feasible project schedules.

    TEACHER EXPLANATION:
    1. Maintains an 'eligible' list of activities whose predecessors are all finished.
    2. Selects the eligible task with highest priority value.
    3. Schedules task at earliest start time t >= max(finish(predecessors)) where
       resource capacity is available for duration [t, t + duration).
    """

    @staticmethod
    def _build_priority_dict(
        project: Project,
        priority_vector: Union[Dict[int, float], List[float]],
    ) -> Dict[int, float]:
        if isinstance(priority_vector, dict):
            return priority_vector
        sorted_ids = sorted(project.activities.keys())
        if len(priority_vector) != len(sorted_ids):
            raise ValueError(
                f"priority_vector length {len(priority_vector)} does not match "
                f"number of activities {len(sorted_ids)}"
            )
        return dict(zip(sorted_ids, priority_vector))

    @staticmethod
    def generate_schedule(
        project: Project,
        priority_vector: Union[Dict[int, float], List[float]],
    ) -> Tuple[List[ScheduleEntry], int]:
        """
        Decodes priority vector into a feasible schedule and computes total makespan.
        Returns (schedule_entries, makespan).
        """
        priority = SSGS._build_priority_dict(project, priority_vector)
        start_act = min(project.activities.keys())

        # Initialize resource availability timeline array
        horizon = sum(a.duration for a in project.activities.values()) + 100
        avail: Dict[str, List[int]] = {
            r_id: [res.capacity] * horizon for r_id, res in project.resources.items()
        }

        schedule_entries: List[ScheduleEntry] = []
        finish_times: Dict[int, int] = {start_act: 0}
        completed: set = {start_act}

        # Schedule dummy start activity at t=0
        schedule_entries.append(
            ScheduleEntry(
                activity_id=start_act,
                start_time=0,
                finish_time=0,
                duration=0,
                resource_usage={},
            )
        )

        uncompleted_preds = {
            a: sum(1 for p in act.predecessors if p != start_act)
            for a, act in project.activities.items()
            if a != start_act
        }
        eligible = [a for a, count in uncompleted_preds.items() if count == 0]

        while eligible:
            # Pick eligible activity with highest random-key priority
            eligible.sort(key=lambda a: priority.get(a, 0.0), reverse=True)
            selected_id = eligible.pop(0)
            act = project.activities[selected_id]

            # 1. Precedence Constraint: EST = max(finish_time of predecessors)
            est = max((finish_times[p] for p in act.predecessors), default=0)
            dur = act.duration
            reqs = [
                (r_id, req)
                for r_id, req in act.resource_requirements.items()
                if req > 0
            ]

            # 2. Resource Constraint: Find earliest start_time >= est with sufficient capacity
            start_time = est
            if dur > 0 and reqs:
                while True:
                    # Dynamically expand horizon if needed
                    if start_time + dur >= horizon:
                        add_h = horizon + dur + 500
                        for r_id in avail:
                            cap = project.resources[r_id].capacity
                            avail[r_id].extend([cap] * (add_h - len(avail[r_id])))
                        horizon = add_h

                    slot_ok = True
                    for r_id, req in reqs:
                        r_avail = avail[r_id]
                        for t in range(start_time, start_time + dur):
                            if r_avail[t] < req:
                                slot_ok = False
                                break
                        if not slot_ok:
                            break
                    if slot_ok:
                        break
                    start_time += 1

            finish_time = start_time + dur
            finish_times[selected_id] = finish_time

            # Deduct allocated resources from availability timeline
            if dur > 0 and reqs:
                for r_id, req in reqs:
                    r_avail = avail[r_id]
                    for t in range(start_time, finish_time):
                        r_avail[t] -= req

            schedule_entries.append(
                ScheduleEntry(
                    activity_id=selected_id,
                    start_time=start_time,
                    finish_time=finish_time,
                    duration=dur,
                    resource_usage=act.resource_requirements.copy(),
                )
            )

            completed.add(selected_id)
            for s in act.successors:
                if s in uncompleted_preds:
                    uncompleted_preds[s] -= 1
                    if uncompleted_preds[s] == 0:
                        eligible.append(s)

        makespan = max((e.finish_time for e in schedule_entries), default=0)
        return schedule_entries, makespan


# ==============================================================================
# SECTION 3: 30-DIMENSION FEATURE EXTRACTION & CRITICAL PATH METHOD (CPM)
# ==============================================================================

@dataclass
class ProjectFeatures:
    """Dataclass encapsulating 30 topological & resource feature dimensions."""
    N: float
    NC: float
    RF: float
    RS: float
    activity_count: int
    resource_count: int
    average_duration: float
    duration_std: float
    minimum_duration: int
    maximum_duration: int
    total_duration: int
    precedence_edge_count: int
    precedence_density: float
    average_predecessors: float
    average_successors: float
    maximum_predecessors: int
    maximum_successors: int
    source_count: int
    sink_count: int
    network_depth: int
    estimated_parallelism: float
    average_parallel_activities: float
    average_resource_demand: float
    resource_demand_std: float
    resource_utilization_ratio: float
    average_resource_tightness: float
    maximum_resource_tightness: float
    number_of_unused_resource_dimensions: int
    critical_path_duration: int
    critical_path_activity_count: int

    def to_dict(self) -> Dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


def validate_dag(project: Project) -> List[int]:
    """
    Validates that the project dependency graph is a valid Directed Acyclic Graph (DAG).
    Uses Kahn's Algorithm to return a valid Topological Sort order.
    Throws ValueError if a cycle is detected.
    """
    in_degree = {a: len(act.predecessors) for a, act in project.activities.items()}
    queue = deque([a for a, deg in in_degree.items() if deg == 0])
    topo_order = []

    while queue:
        curr = queue.popleft()
        topo_order.append(curr)
        for s in project.activities[curr].successors:
            if s in in_degree:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)

    if len(topo_order) != len(project.activities):
        raise ValueError("Cycle detected in project precedence graph! Not a valid DAG.")
    return topo_order


def extract_features(project: Project) -> ProjectFeatures:
    """
    Computes 30 topological features including CPM Critical Path, NC, RF, and RS.
    """
    topo_order = validate_dag(project)
    real_activities = [act for act in project.activities.values() if act.duration > 0 or (act.predecessors and act.successors)]
    N = float(len(real_activities))

    total_edges = sum(len(act.successors) for act in project.activities.values())
    NC = total_edges / N if N > 0 else 0.0

    # 1. Critical Path Method (CPM) Forward Pass
    est: Dict[int, int] = {}
    eft: Dict[int, int] = {}
    for a_id in topo_order:
        act = project.activities[a_id]
        est_val = max((eft[p] for p in act.predecessors), default=0)
        est[a_id] = est_val
        eft[a_id] = est_val + act.duration

    cp_duration = max(eft.values(), default=0)

    # CPM Backward Pass
    lft: Dict[int, int] = {}
    lst: Dict[int, int] = {}
    for a_id in reversed(topo_order):
        act = project.activities[a_id]
        if not act.successors:
            lft_val = cp_duration
        else:
            lft_val = min(lst[s] for s in act.successors)
        lft[a_id] = lft_val
        lst[a_id] = lft_val - act.duration

    total_slack = {a_id: lst[a_id] - est[a_id] for a_id in project.activities}
    cp_activities = [a_id for a_id, slack in total_slack.items() if slack == 0]

    # 2. Resource Strength (RS)
    resource_keys = sorted(project.resources.keys())
    K = len(resource_keys)

    if cp_duration > 0 and K > 0:
        cpm_timeline = {r_id: [0] * cp_duration for r_id in resource_keys}
        for a_id, act in project.activities.items():
            for r_id, req in act.resource_requirements.items():
                if r_id in cpm_timeline:
                    for t in range(est[a_id], eft[a_id]):
                        cpm_timeline[r_id][t] += req

        rs_list = []
        for r_id in resource_keys:
            r_cap = project.resources[r_id].capacity
            reqs = [act.resource_requirements.get(r_id, 0) for act in project.activities.values()]
            r_min = max(reqs) if reqs else 0
            r_max = max(cpm_timeline[r_id]) if cpm_timeline[r_id] else r_min
            if r_max == r_min:
                rs_k = 1.0
            else:
                rs_k = (r_cap - r_min) / (r_max - r_min)
            rs_list.append(max(0.0, min(1.0, rs_k)))
        RS = float(statistics.mean(rs_list)) if rs_list else 1.0
    else:
        RS = 1.0

    # 3. Resource Factor (RF)
    if N > 0 and K > 0:
        rf_count = sum(1 for act in real_activities for r_id in resource_keys if act.resource_requirements.get(r_id, 0) > 0)
        RF = float(rf_count) / float(N * K)
    else:
        RF = 0.0

    durations = [act.duration for act in project.activities.values()]
    avg_dur = float(statistics.mean(durations)) if durations else 0.0
    std_dur = float(statistics.pstdev(durations)) if len(durations) > 1 else 0.0

    pred_counts = [len(act.predecessors) for act in project.activities.values()]
    succ_counts = [len(act.successors) for act in project.activities.values()]

    sources = sum(1 for c in pred_counts if c == 0)
    sinks = sum(1 for c in succ_counts if c == 0)

    level: Dict[int, int] = {}
    for a_id in topo_order:
        act = project.activities[a_id]
        level[a_id] = max((level[p] + 1 for p in act.predecessors), default=1)
    depth = max(level.values(), default=0)

    pd = (2.0 * total_edges) / (project.num_activities * (project.num_activities - 1)) if project.num_activities > 1 else 0.0
    ep = sum(durations) / float(cp_duration) if cp_duration > 0 else 0.0

    rt_list = []
    unused_dims = 0
    all_demands = []

    for r_id in resource_keys:
        cap = project.resources[r_id].capacity
        total_req = sum(act.duration * act.resource_requirements.get(r_id, 0) for act in project.activities.values())
        if cap > 0 and cp_duration > 0:
            rt_list.append(total_req / float(cap * cp_duration))
        else:
            rt_list.append(0.0)
        if total_req == 0:
            unused_dims += 1

    for act in project.activities.values():
        for r_id in resource_keys:
            all_demands.append(act.resource_requirements.get(r_id, 0))

    avg_demand = float(statistics.mean(all_demands)) if all_demands else 0.0
    std_demand = float(statistics.pstdev(all_demands)) if len(all_demands) > 1 else 0.0

    return ProjectFeatures(
        N=N,
        NC=NC,
        RF=RF,
        RS=RS,
        activity_count=project.num_activities,
        resource_count=project.num_resources,
        average_duration=avg_dur,
        duration_std=std_dur,
        minimum_duration=min(durations, default=0),
        maximum_duration=max(durations, default=0),
        total_duration=sum(durations),
        precedence_edge_count=total_edges,
        precedence_density=pd,
        average_predecessors=float(statistics.mean(pred_counts)) if pred_counts else 0.0,
        average_successors=float(statistics.mean(succ_counts)) if succ_counts else 0.0,
        maximum_predecessors=max(pred_counts, default=0),
        maximum_successors=max(succ_counts, default=0),
        source_count=sources,
        sink_count=sinks,
        network_depth=depth,
        estimated_parallelism=ep,
        average_parallel_activities=ep,
        average_resource_demand=avg_demand,
        resource_demand_std=std_demand,
        resource_utilization_ratio=float(statistics.mean(rt_list)) if rt_list else 0.0,
        average_resource_tightness=float(statistics.mean(rt_list)) if rt_list else 0.0,
        maximum_resource_tightness=max(rt_list, default=0.0),
        number_of_unused_resource_dimensions=unused_dims,
        critical_path_duration=cp_duration,
        critical_path_activity_count=len(cp_activities),
    )


# ==============================================================================
# SECTION 4: EVALUATOR & DUAL-LAYER CONSTRAINT VALIDATOR
# ==============================================================================

class Evaluator:
    """
    Evaluates solutions and validates precedence and resource constraints.
    """

    @staticmethod
    def evaluate(project: Project, priority_vector: Union[Dict[int, float], List[float]]) -> Tuple[List[ScheduleEntry], int, List[ScheduleEntry]]:
        sched, ms = SSGS.generate_schedule(project, priority_vector)
        return sched, ms, sched

    @staticmethod
    def check_constraints(project: Project, schedule: List[ScheduleEntry]) -> bool:
        """
        Dual-Layer Constraint Validation:
        1. Precedence Validation: finish_time(p) <= start_time(act)
        2. Resource Validation: sum(r_ik at time t) <= capacity(k)
        """
        finish_map = {e.activity_id: e.finish_time for e in schedule}
        start_map = {e.activity_id: e.start_time for e in schedule}

        # Check Precedence Constraints
        for act_id, act in project.activities.items():
            if act_id in start_map:
                s_time = start_map[act_id]
                for p in act.predecessors:
                    if p in finish_map and finish_map[p] > s_time:
                        return False

        # Check Resource Capacity Constraints
        max_t = max((e.finish_time for e in schedule), default=0)
        for r_id, res in project.resources.items():
            usage = [0] * (max_t + 1)
            for e in schedule:
                req = project.activities[e.activity_id].resource_requirements.get(r_id, 0)
                for t in range(e.start_time, e.finish_time):
                    usage[t] += req
            if any(u > res.capacity for u in usage):
                return False

        return True


# ==============================================================================
# SECTION 5: FILE PARSER (.sm PSPLIB, .rcp Paterson, .json)
# ==============================================================================

class Parser:
    """Parser supporting PSPLIB .sm files, .rcp files, and JSON files."""

    @staticmethod
    def parse(file_path: Union[str, Path]) -> Project:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".sm":
            return Parser._parse_sm(str(path))
        elif ext == ".rcp":
            return Parser._parse_rcp(str(path))
        elif ext == ".json":
            return Parser._parse_json(str(path))
        else:
            raise ValueError(f"Unsupported format '{ext}'. Use .sm, .rcp, or .json")

    @staticmethod
    def _parse_sm(file_path: str) -> Project:
        try:
            import psplib
            raw = psplib.parse_psplib(file_path)
            project = Project()

            for i, res in enumerate(raw.resources):
                if res.renewable:
                    r_id = f"R{i + 1}"
                    project.resources[r_id] = Resource(id=r_id, capacity=res.capacity)

            renewable_indices = [i for i, r in enumerate(raw.resources) if r.renewable]
            renewable_r_ids = [f"R{i + 1}" for i in range(len(renewable_indices))]

            num_psp_acts = raw.num_activities
            predecessors: Dict[int, List[int]] = {(i + 1): [] for i in range(num_psp_acts)}

            for psp_idx, act in enumerate(raw.activities):
                job_nr = psp_idx + 1
                for succ_0idx in act.successors:
                    succ_job = succ_0idx + 1
                    if succ_job not in predecessors:
                        predecessors[succ_job] = []
                    predecessors[succ_job].append(job_nr)

            all_referenced_jobs = set()
            for act in raw.activities:
                for s in act.successors:
                    all_referenced_jobs.add(s + 1)
            present_jobs = set(range(1, num_psp_acts + 1))
            phantom_jobs = all_referenced_jobs - present_jobs

            if phantom_jobs:
                sink_job = max(phantom_jobs)
                if sink_job not in predecessors:
                    predecessors[sink_job] = []
                for psp_idx, act in enumerate(raw.activities):
                    for s in act.successors:
                        if s + 1 == sink_job:
                            src_job = psp_idx + 1
                            if src_job not in predecessors[sink_job]:
                                predecessors[sink_job].append(src_job)
            else:
                sink_job = num_psp_acts

            for psp_idx, psp_act in enumerate(raw.activities):
                job_nr = psp_idx + 1
                mode = psp_act.modes[0]
                reqs: Dict[str, int] = {}
                for r_idx, r_id in zip(renewable_indices, renewable_r_ids):
                    reqs[r_id] = mode.demands[r_idx] if r_idx < len(mode.demands) else 0

                project.activities[job_nr] = Activity(
                    id=job_nr,
                    duration=mode.duration,
                    predecessors=predecessors.get(job_nr, []),
                    resource_requirements=reqs,
                )

            if phantom_jobs:
                sink_reqs = {r_id: 0 for r_id in renewable_r_ids}
                project.activities[sink_job] = Activity(
                    id=sink_job,
                    duration=0,
                    predecessors=predecessors.get(sink_job, []),
                    resource_requirements=sink_reqs,
                )

            for act in project.activities.values():
                act.successors = []
            for a_id, act in project.activities.items():
                for p_id in act.predecessors:
                    if p_id in project.activities:
                        project.activities[p_id].successors.append(a_id)

            return project
        except Exception:
            return Parser._parse_sm_fallback(file_path)

    @staticmethod
    def _parse_sm_fallback(file_path: str) -> Project:
        project = Project()
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        mode = None
        num_res = 0

        for line in lines:
            line_str = line.strip()
            if line_str.startswith("- renewable"):
                parts = line_str.split(":")
                if len(parts) > 1:
                    num_res = int(parts[1].split()[0].strip())
            elif line_str.startswith("PRECEDENCE RELATIONS:"):
                mode = "PRECEDENCE"
                continue
            elif line_str.startswith("REQUESTS/DURATIONS:"):
                mode = "REQUESTS"
                continue
            elif line_str.startswith("RESOURCEAVAILABILITIES:"):
                mode = "RESOURCES"
                continue
            elif line_str.startswith("************************************************************************"):
                if mode == "RESOURCES":
                    mode = None

            if mode == "PRECEDENCE":
                parts = line_str.split()
                if not parts or not parts[0].isdigit():
                    continue
                job_nr = int(parts[0])
                num_succ = int(parts[2])
                succs = [int(x) for x in parts[3:3+num_succ]]
                if job_nr not in project.activities:
                    project.activities[job_nr] = Activity(id=job_nr, duration=0, predecessors=[], resource_requirements={})
                for suc in succs:
                    if suc not in project.activities:
                        project.activities[suc] = Activity(id=suc, duration=0, predecessors=[], resource_requirements={})
                    project.activities[suc].predecessors.append(job_nr)

            elif mode == "REQUESTS":
                parts = line_str.split()
                if not parts or not parts[0].isdigit():
                    continue
                job_nr = int(parts[0])
                duration = int(parts[2])
                reqs = [int(x) for x in parts[3:3+num_res]]

                if job_nr not in project.activities:
                    project.activities[job_nr] = Activity(id=job_nr, duration=duration, predecessors=[], resource_requirements={})
                else:
                    project.activities[job_nr].duration = duration

                for idx, r in enumerate(reqs):
                    project.activities[job_nr].resource_requirements[f"R{idx+1}"] = r

            elif mode == "RESOURCES":
                parts = line_str.split()
                if not parts or not parts[0].isdigit():
                    continue
                for idx, cap in enumerate(parts[:num_res]):
                    r_id = f"R{idx+1}"
                    project.resources[r_id] = Resource(id=r_id, capacity=int(cap))
                mode = None

        for act in project.activities.values():
            act.successors = []
        for a_id, act in project.activities.items():
            for p_id in act.predecessors:
                if p_id in project.activities:
                    project.activities[p_id].successors.append(a_id)

        return project

    @staticmethod
    def _parse_rcp(file_path: str) -> Project:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tokens = content.split()
        if not tokens:
            raise ValueError("Empty .rcp file")

        n_jobs = int(tokens[0])
        n_resources = int(tokens[1])
        idx = 2

        project = Project()
        for r in range(n_resources):
            r_id = f"R{r+1}"
            project.resources[r_id] = Resource(id=r_id, capacity=int(tokens[idx]))
            idx += 1

        for j in range(1, n_jobs + 1):
            dur = int(tokens[idx])
            idx += 1
            reqs = {}
            for r in range(n_resources):
                reqs[f"R{r+1}"] = int(tokens[idx])
                idx += 1
            n_succ = int(tokens[idx])
            idx += 1
            succs = []
            for _ in range(n_succ):
                succs.append(int(tokens[idx]))
                idx += 1

            project.activities[j] = Activity(
                id=j,
                duration=dur,
                resource_requirements=reqs,
                successors=succs,
            )

        for act_id, act in project.activities.items():
            for s in act.successors:
                if s in project.activities:
                    if act_id not in project.activities[s].predecessors:
                        project.activities[s].predecessors.append(act_id)

        return project

    @staticmethod
    def _parse_json(file_path: str) -> Project:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        project = Project()
        for r_id, cap in data.get("resources", {}).items():
            project.resources[r_id] = Resource(id=r_id, capacity=int(cap))

        for act_data in data.get("activities", []):
            act_id = int(act_data["id"])
            project.activities[act_id] = Activity(
                id=act_id,
                duration=int(act_data.get("duration", 0)),
                predecessors=[int(p) for p in act_data.get("predecessors", [])],
                successors=[int(s) for s in act_data.get("successors", [])],
                resource_requirements={k: int(v) for k, v in act_data.get("resource_requirements", {}).items()},
            )
        return project


# ==============================================================================
# SECTION 6: METAHEURISTIC SOLVERS (GA, SA, BCO, TS, PRH)
# ==============================================================================

def solve_ga(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
    population_size: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    elite_count: int = 2,
    tournament_k: int = 3,
) -> AlgorithmResult:
    """Genetic Algorithm solver using random-key representation."""
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history = []

    pop = [[rng.random() for _ in range(n)] for _ in range(population_size)]
    fits, scheds = [], []
    for ind in pop:
        _, ms, sched = Evaluator.evaluate(project, ind)
        fits.append(ms)
        scheds.append(sched)
        evaluations += 1

    best_idx = min(range(population_size), key=lambda i: fits[i])
    best_ms, best_vec, best_sched = fits[best_idx], pop[best_idx][:], scheds[best_idx]
    convergence_history.append((evaluations, best_ms))

    while evaluations < evaluation_budget:
        ranked = sorted(range(population_size), key=lambda i: fits[i])
        new_pop = [pop[i][:] for i in ranked[:elite_count]]
        new_fits = [fits[i] for i in ranked[:elite_count]]
        new_scheds = [scheds[i] for i in ranked[:elite_count]]

        while len(new_pop) < population_size and evaluations < evaluation_budget:
            # Tournament selection
            s1 = rng.sample(range(population_size), tournament_k)
            p1 = pop[min(s1, key=lambda i: fits[i])][:]
            s2 = rng.sample(range(population_size), tournament_k)
            p2 = pop[min(s2, key=lambda i: fits[i])][:]

            # Crossover
            if rng.random() < crossover_rate:
                c1 = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(n)]
                c2 = [p2[i] if rng.random() < 0.5 else p1[i] for i in range(n)]
            else:
                c1, c2 = p1[:], p2[:]

            # Mutation
            c1 = [rng.random() if rng.random() < mutation_rate else g for g in c1]
            c2 = [rng.random() if rng.random() < mutation_rate else g for g in c2]

            for child in (c1, c2):
                if len(new_pop) >= population_size or evaluations >= evaluation_budget:
                    break
                _, ms, sched = Evaluator.evaluate(project, child)
                evaluations += 1
                new_pop.append(child)
                new_fits.append(ms)
                new_scheds.append(sched)

                if ms < best_ms:
                    best_ms, best_vec, best_sched = ms, child[:], sched
                    convergence_history.append((evaluations, best_ms))

        pop, fits, scheds = new_pop, new_fits, new_scheds

    return AlgorithmResult(
        algorithm="GA",
        best_priority_vector=best_vec,
        schedule=best_sched,
        makespan=best_ms,
        runtime=time.perf_counter() - start_ts,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )


def solve_sa(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.01,
) -> AlgorithmResult:
    """Simulated Annealing solver with Metropolis acceptance criterion."""
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history = []

    curr = [rng.random() for _ in range(n)]
    _, curr_ms, curr_sched = Evaluator.evaluate(project, curr)
    evaluations += 1

    best_vec, best_ms, best_sched = curr[:], curr_ms, curr_sched
    convergence_history.append((evaluations, best_ms))
    temp = initial_temp

    while evaluations < evaluation_budget and temp > min_temp:
        cand = curr[:]
        if rng.random() < 0.5 and n >= 2:
            i, j = rng.sample(range(n), 2)
            cand[i], cand[j] = cand[j], cand[i]
        else:
            i = rng.randrange(n)
            cand[i] = max(0.0, min(1.0, cand[i] + rng.gauss(0, 0.1)))

        _, cand_ms, cand_sched = Evaluator.evaluate(project, cand)
        evaluations += 1
        delta = cand_ms - curr_ms

        if delta < 0 or rng.random() < math.exp(-delta / temp):
            curr, curr_ms, curr_sched = cand, cand_ms, cand_sched

        if curr_ms < best_ms:
            best_vec, best_ms, best_sched = curr[:], curr_ms, curr_sched
            convergence_history.append((evaluations, best_ms))

        temp *= cooling_rate

    return AlgorithmResult(
        algorithm="SA",
        best_priority_vector=best_vec,
        schedule=best_sched,
        makespan=best_ms,
        runtime=time.perf_counter() - start_ts,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )


def solve_bco(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
    colony_size: int = 20,
    limit: int = 15,
) -> AlgorithmResult:
    """Bee Colony Optimization (Artificial Bee Colony) solver."""
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history = []

    sources = [[rng.random() for _ in range(n)] for _ in range(colony_size)]
    fits, scheds, trials = [], [], [0] * colony_size

    for src in sources:
        _, ms, sched = Evaluator.evaluate(project, src)
        fits.append(ms)
        scheds.append(sched)
        evaluations += 1

    best_idx = min(range(colony_size), key=lambda i: fits[i])
    best_ms, best_vec, best_sched = fits[best_idx], sources[best_idx][:], scheds[best_idx]
    convergence_history.append((evaluations, best_ms))

    while evaluations < evaluation_budget:
        # Employed Bees phase
        for i in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            partner_idx = rng.choice([j for j in range(colony_size) if j != i])
            cand = sources[i][:]
            idx = rng.randrange(n)
            phi = rng.uniform(-1.0, 1.0)
            cand[idx] = max(0.0, min(1.0, cand[idx] + phi * (cand[idx] - sources[partner_idx][idx])))

            _, cand_ms, cand_sched = Evaluator.evaluate(project, cand)
            evaluations += 1

            if cand_ms <= fits[i]:
                sources[i], fits[i], scheds[i], trials[i] = cand, cand_ms, cand_sched, 0
            else:
                trials[i] += 1

            if fits[i] < best_ms:
                best_ms, best_vec, best_sched = fits[i], sources[i][:], scheds[i]
                convergence_history.append((evaluations, best_ms))

        # Onlooker Bees phase
        max_ms = max(fits) if fits else 1
        weights = [(max_ms - f + 1) for f in fits]
        tot = sum(weights) or 1.0
        probs = [w / tot for w in weights]

        for _ in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            r = rng.random()
            cum = 0.0
            sel_i = colony_size - 1
            for idx_p, p in enumerate(probs):
                cum += p
                if r <= cum:
                    sel_i = idx_p
                    break

            partner_idx = rng.choice([j for j in range(colony_size) if j != sel_i])
            cand = sources[sel_i][:]
            idx = rng.randrange(n)
            phi = rng.uniform(-1.0, 1.0)
            cand[idx] = max(0.0, min(1.0, cand[idx] + phi * (cand[idx] - sources[partner_idx][idx])))

            _, cand_ms, cand_sched = Evaluator.evaluate(project, cand)
            evaluations += 1

            if cand_ms <= fits[sel_i]:
                sources[sel_i], fits[sel_i], scheds[sel_i], trials[sel_i] = cand, cand_ms, cand_sched, 0
            else:
                trials[sel_i] += 1

            if fits[sel_i] < best_ms:
                best_ms, best_vec, best_sched = fits[sel_i], sources[sel_i][:], scheds[sel_i]
                convergence_history.append((evaluations, best_ms))

        # Scout Bees phase
        for i in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            if trials[i] >= limit:
                sources[i] = [rng.random() for _ in range(n)]
                _, ms, sched = Evaluator.evaluate(project, sources[i])
                fits[i], scheds[i], trials[i] = ms, sched, 0
                evaluations += 1

                if ms < best_ms:
                    best_ms, best_vec, best_sched = ms, sources[i][:], sched
                    convergence_history.append((evaluations, best_ms))

    return AlgorithmResult(
        algorithm="BCO",
        best_priority_vector=best_vec,
        schedule=best_sched,
        makespan=best_ms,
        runtime=time.perf_counter() - start_ts,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )


def solve_tabu(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
    tenure: int = 7,
    n_neighbours: int = 10,
) -> AlgorithmResult:
    """Tabu Search solver with short-term memory list and aspiration criterion."""
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history = []

    curr = [rng.random() for _ in range(n)]
    _, curr_ms, curr_sched = Evaluator.evaluate(project, curr)
    evaluations += 1

    best_vec, best_ms, best_sched = curr[:], curr_ms, curr_sched
    convergence_history.append((evaluations, best_ms))

    tabu_list: Deque[Tuple[int, int]] = deque()
    tabu_set = set()

    while evaluations < evaluation_budget:
        neighbours = []
        seen_moves = set()
        attempts = 0
        while len(neighbours) < n_neighbours and attempts < n_neighbours * 5 and n >= 2:
            attempts += 1
            i, j = sorted(rng.sample(range(n), 2))
            if (i, j) in seen_moves:
                continue
            seen_moves.add((i, j))
            v = curr[:]
            v[i], v[j] = v[j], v[i]
            neighbours.append((v, (i, j)))

        best_cand, best_cand_ms, best_cand_sched, best_cand_move = None, None, None, None

        for vec, move_id in neighbours:
            if evaluations >= evaluation_budget:
                break
            _, ms, sched = Evaluator.evaluate(project, vec)
            evaluations += 1

            is_tabu = move_id in tabu_set
            aspiration = ms < best_ms

            if is_tabu and not aspiration:
                continue

            if best_cand_ms is None or ms < best_cand_ms:
                best_cand, best_cand_ms, best_cand_sched, best_cand_move = vec, ms, sched, move_id

        if best_cand is None:
            curr = [rng.random() for _ in range(n)]
            if evaluations < evaluation_budget:
                _, curr_ms, curr_sched = Evaluator.evaluate(project, curr)
                evaluations += 1
            tabu_list.clear()
            tabu_set.clear()
            continue

        curr = best_cand
        if best_cand_ms is not None:
            curr_ms, curr_sched = best_cand_ms, best_cand_sched

        if best_cand_move is not None:
            tabu_list.append(best_cand_move)
            tabu_set.add(best_cand_move)
            if len(tabu_list) > tenure:
                tabu_set.discard(tabu_list.popleft())

        if best_cand_ms is not None and best_cand_ms < best_ms:
            best_ms, best_vec, best_sched = best_cand_ms, best_cand[:], best_cand_sched
            convergence_history.append((evaluations, best_ms))

    return AlgorithmResult(
        algorithm="TS",
        best_priority_vector=best_vec,
        schedule=best_sched,
        makespan=best_ms,
        runtime=time.perf_counter() - start_ts,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )


def solve_prh(
    project: Project,
    evaluation_budget: int = 5000,
    seed: int = 42,
) -> AlgorithmResult:
    """Priority Rule Heuristics solver (EST, EFT, LST, LFT, RAND)."""
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    evaluations = 0
    convergence_history = []

    topo_order = validate_dag(project)
    est = {}
    eft = {}
    for a_id in topo_order:
        act = project.activities[a_id]
        est_val = max((eft[p] for p in act.predecessors), default=0)
        est[a_id] = est_val
        eft[a_id] = est_val + act.duration

    horizon = max(eft.values(), default=0)
    lft, lst = {}, {}
    for a_id in reversed(topo_order):
        act = project.activities[a_id]
        lft_val = min((lst[s] for s in act.successors), default=horizon)
        lft[a_id] = lft_val
        lst[a_id] = lft_val - act.duration

    sorted_ids = sorted(project.activities.keys())
    rules = {
        "EST": {a: (horizon + 1 - est[a]) for a in sorted_ids},
        "EFT": {a: (horizon + 1 - eft[a]) for a in sorted_ids},
        "LST": {a: (horizon + 1 - lst[a]) for a in sorted_ids},
        "LFT": {a: (horizon + 1 - lft[a]) for a in sorted_ids},
    }

    best_ms, best_vec, best_sched, best_rule = None, None, None, ""

    for rule_name, prio_dict in rules.items():
        if evaluations >= evaluation_budget:
            break
        max_p = max(prio_dict.values()) or 1.0
        vec = [prio_dict[a] / max_p for a in sorted_ids]
        _, ms, sched = Evaluator.evaluate(project, vec)
        evaluations += 1

        if best_ms is None or ms < best_ms:
            best_ms, best_vec, best_sched, best_rule = ms, vec, sched, rule_name
            convergence_history.append((evaluations, best_ms))

    while evaluations < evaluation_budget:
        rand_vec = [rng.random() for _ in sorted_ids]
        _, ms, sched = Evaluator.evaluate(project, rand_vec)
        evaluations += 1

        if best_ms is None or ms < best_ms:
            best_ms, best_vec, best_sched, best_rule = ms, rand_vec, sched, "RAND"
            convergence_history.append((evaluations, best_ms))

    return AlgorithmResult(
        algorithm=f"PRH[{best_rule}]",
        best_priority_vector=best_vec or [],
        schedule=best_sched or [],
        makespan=best_ms or -1,
        runtime=time.perf_counter() - start_ts,
        evaluations=evaluations,
        convergence_history=convergence_history,
    )


# Solver Registry Map
SOLVER_REGISTRY = {
    "GA": solve_ga,
    "SA": solve_sa,
    "BCO": solve_bco,
    "TS": solve_tabu,
    "PRH": solve_prh,
}


# ==============================================================================
# SECTION 7: BENCHMARK RUNNER & COMPARISON REPORT
# ==============================================================================

@dataclass
class AlgorithmSummary:
    algorithm: str
    best_makespan: int
    mean_makespan: float
    worst_makespan: int
    std_makespan: float
    mean_runtime: float
    mean_evaluations: float
    n_runs: int
    n_feasible: int
    all_feasible: bool
    records: List[BenchmarkRecord] = field(default_factory=list)


@dataclass
class ComparisonReport:
    instance_id: str
    summaries: List[AlgorithmSummary]
    n_seeds: int
    budget: int
    winner: str
    winner_ms: int

    def to_text(self) -> str:
        lines = [
            f"=== Comparison Report: {self.instance_id} ===",
            f"Seeds: {self.n_seeds}  |  Evaluation budget: {self.budget} per run",
            "",
            f"{'Algorithm':<12} {'Best':>6} {'Mean':>7} {'Worst':>7} {'Std':>6} {'Runtime(s)':>11} {'Evals':>7} {'Feasible':>9}",
            "-" * 72,
        ]

        for s in sorted(self.summaries, key=lambda x: (x.best_makespan, x.mean_makespan)):
            marker = " ◄" if s.algorithm.split("[")[0] == self.winner.split("[")[0] or self.winner == "ALL" else ""
            lines.append(
                f"{s.algorithm:<12} {s.best_makespan:>6} {s.mean_makespan:>7.1f} "
                f"{s.worst_makespan:>7} {s.std_makespan:>6.2f} "
                f"{s.mean_runtime:>11.3f} {s.mean_evaluations:>7.0f} "
                f"{s.n_feasible}/{s.n_runs}{marker}"
            )

        lines += [
            "-" * 72,
            f"Best feasible solution found: {self.winner_ms} (by {self.winner})",
        ]
        return "\n".join(lines)


def run_benchmark(
    project: Project,
    seeds: List[int] = [42],
    evaluation_budget: int = 5000,
    instance_id: str = "instance",
) -> List[BenchmarkRecord]:
    """Execute all 5 solvers across multiple random seeds."""
    records = []
    for seed in seeds:
        for algo_name, solver_fn in SOLVER_REGISTRY.items():
            res = solver_fn(project, evaluation_budget=evaluation_budget, seed=seed)
            feasible = Evaluator.check_constraints(project, res.schedule)
            records.append(
                BenchmarkRecord(
                    instance_id=instance_id,
                    algorithm=res.algorithm,
                    seed=seed,
                    makespan=res.makespan,
                    runtime=res.runtime,
                    evaluations=res.evaluations,
                    feasible=feasible,
                    best_priority_vector=res.best_priority_vector,
                    convergence_history=res.convergence_history,
                )
            )
    return records


def compare_algorithms(records: List[BenchmarkRecord], instance_id: str = "instance", budget: int = 5000) -> ComparisonReport:
    """Build statistical comparison summary report from benchmark records."""
    by_algo: Dict[str, List[BenchmarkRecord]] = {}
    for r in records:
        base = r.algorithm.split("[")[0]
        by_algo.setdefault(base, []).append(r)

    summaries = []
    for base, recs in by_algo.items():
        feas = [r for r in recs if r.feasible]
        ms_list = [r.makespan for r in feas] if feas else [r.makespan for r in recs]
        runtimes = [r.runtime for r in recs]
        evals = [r.evaluations for r in recs]

        best_ms = min(ms_list)
        worst_ms = max(ms_list)
        mean_ms = float(statistics.mean(ms_list))
        std_ms = float(statistics.pstdev(ms_list)) if len(ms_list) > 1 else 0.0

        summaries.append(
            AlgorithmSummary(
                algorithm=recs[0].algorithm,
                best_makespan=best_ms,
                mean_makespan=mean_ms,
                worst_makespan=worst_ms,
                std_makespan=std_ms,
                mean_runtime=float(statistics.mean(runtimes)),
                mean_evaluations=float(statistics.mean(evals)),
                n_runs=len(recs),
                n_feasible=len(feas),
                all_feasible=(len(feas) == len(recs)),
                records=recs,
            )
        )

    best_overall = min(s.best_makespan for s in summaries)
    winners = [s for s in summaries if s.best_makespan == best_overall]
    if len(winners) == len(summaries):
        winner_name = "ALL"
    else:
        winners.sort(key=lambda s: (s.mean_makespan, s.mean_runtime))
        winner_name = winners[0].algorithm

    return ComparisonReport(
        instance_id=instance_id,
        summaries=summaries,
        n_seeds=len(set(r.seed for r in records)),
        budget=budget,
        winner=winner_name,
        winner_ms=best_overall,
    )


# ==============================================================================
# SECTION 8: COMMAND LINE INTERFACE (CLI) MAIN ENTRY POINT
# ==============================================================================

def main():
    import sys
    print("=" * 72)
    print(" AI-RCPSP Complete All-in-One Framework ")
    print("=" * 72)

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        possible = list(Path(".").rglob("*.sm")) + list(Path(".").rglob("*.rcp"))
        if possible:
            file_path = str(possible[0])
            print(f"No file argument provided. Auto-selecting instance: {file_path}")
        else:
            print("Usage: python rcpsp_all_in_one.py <path_to_instance.sm>")
            return

    print(f"\n[1/3] Parsing instance file: {file_path}")
    project = Parser.parse(file_path)
    print(f"      Loaded project: {project.num_activities} activities, {project.num_resources} resources.")

    print("\n[2/3] Extracting Project Features...")
    features = extract_features(project)
    print(f"      Paper Metrics -> N: {features.N}, NC: {features.NC:.2f}, RF: {features.RF:.2f}, RS: {features.RS:.2f}")
    print(f"      Critical Path Duration: {features.critical_path_duration}")

    print("\n[3/3] Executing Metaheuristic Benchmark Solvers (500 eval budget)...")
    records = run_benchmark(project, seeds=[42], evaluation_budget=500, instance_id=Path(file_path).stem)
    report = compare_algorithms(records, instance_id=Path(file_path).stem, budget=500)

    print("\n" + report.to_text())
    print("\n[SUCCESS] Completed all operations.")


if __name__ == "__main__":
    main()
