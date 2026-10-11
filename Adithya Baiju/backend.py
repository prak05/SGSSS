"""
================================================================================
AI-RCPSP CORE BACKEND ENGINE & OPTIMIZATION SUITE (ALL-IN-ONE BACKEND MODULE)
================================================================================
Author/Directory: Adithya Baiju
File: backend.py
Description: Self-contained backend engine for Resource-Constrained Project
             Scheduling (RCPSP). Contains:
             1. Core Data Models (Resource, Activity, Project, ScheduleEntry, AlgorithmResult)
             2. SSGS Decoder Engine (Serial Schedule Generation Scheme)
             3. 30-Dimension Feature Extraction & CPM Analysis (N, NC, RF, RS, CPM)
             4. Multi-Format Project File Parser (.sm, .rcp, .json)
             5. Evaluator & Dual-Layer Constraint Validator
             6. 5 Metaheuristic Optimization Solvers (GA, SA, BCO, Tabu Search, PRH)
             7. Statistical Benchmark & Comparison Engine
             8. Printable PDF Reference Report Generator (ReportLab engine)
             9. Command Line Interface (CLI) Execution Controller

Usage as CLI Script:
    python "Adithya Baiju/backend.py" data/j10_sample.sm

Usage as Module in ui.py:
    from backend import Parser, extract_features, Evaluator, SOLVER_REGISTRY, generate_pdf_report
================================================================================
"""

from __future__ import annotations

import io
import json
import math
import pickle
import random
import re
import statistics
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

# Matplotlib & ReportLab imports for PDF generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


# Optional numpy / sklearn imports for ML inference
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ==============================================================================
# SECTION 1: DATA MODELS
# ==============================================================================
# DEFINITION: Resource-Constrained Project Scheduling Problem (RCPSP)
# RCPSP is an NP-hard combinatorial optimization problem where a project consists
# of a set of activities with precedence relations and resource constraints.
# The goal is to find start times for all activities to minimize total project duration (makespan).

@dataclass
class Resource:
    """
    Represents a renewable resource (e.g., workers, machines, equipment, tools).
    
    DEFINITION - Renewable Resource:
    A resource whose capacity is renewed at every time unit (e.g., 5 manpower units available
    each hour). Unlike non-renewable resources (like budget or fuel), using a renewable resource
    in period t does NOT diminish its capacity in period t+1.
    
    Attributes:
        id: Unique string identifier for the resource (e.g., 'R1', 'R2').
        capacity: Maximum available units of this resource per time period (constant integer).
    """
    id: str        # Resource identifier tag (e.g. 'R1')
    capacity: int  # Fixed capacity available during every time step


@dataclass
class Activity:
    """
    Represents a project activity/task in the precedence graph (Directed Acyclic Graph - DAG).
    
    DEFINITION - Dummy Start & Sink Activities:
    In standard RCPSP benchmark formats (like PSPLIB):
    - Activity 1 is a "Dummy Start" task (duration = 0, resource demands = 0) representing project kickoff.
    - Activity N+2 is a "Dummy Sink" task (duration = 0, resource demands = 0) representing project completion.
    
    Attributes:
        id: Activity ID integer (1 to N+2).
        duration: Processing duration in integer time steps (duration >= 0).
        predecessors: List of activity IDs that MUST complete before this task can start (Finish-to-Start).
        successors: List of activity IDs that CANNOT start until this task finishes.
        resource_requirements: Dict mapping resource ID -> required units per time step while active.
    """
    id: int                                             # Activity integer identifier (1-indexed)
    duration: int                                       # Task execution time in discrete time steps
    predecessors: List[int] = field(default_factory=list) # List of predecessor activity IDs required before start
    successors: List[int] = field(default_factory=list)   # List of dependent successor activity IDs
    resource_requirements: Dict[str, int] = field(default_factory=dict) # Dict of {resource_id: units_required_per_step}


@dataclass
class Project:
    """
    Container storing all activities and renewable resources defining an RCPSP problem instance.
    
    Attributes:
        activities: Mapping of activity ID (int) -> Activity object instance.
        resources: Mapping of resource ID (str) -> Resource object instance.
    """
    activities: Dict[int, Activity] = field(default_factory=dict) # Map storing all activity nodes
    resources: Dict[str, Resource] = field(default_factory=dict)   # Map storing all renewable resource types

    @property
    def num_activities(self) -> int:
        """Returns total count of activities in the project (including dummy nodes)."""
        return len(self.activities) # Calculate total number of activity keys

    @property
    def num_resources(self) -> int:
        """Returns total count of distinct renewable resource types in the project."""
        return len(self.resources) # Calculate total number of resource keys


@dataclass
class ScheduleEntry:
    """
    Result of decoding an activity: contains assigned start/finish times and resource allocation.
    
    DEFINITION - Schedule Entry:
    A scheduled assignment for a single activity specifying exactly when it starts, when it finishes,
    and what resources it consumes during execution.
    
    Attributes:
        activity_id: Task integer identifier.
        start_time: Scheduled start time step S_i (S_i >= 0).
        finish_time: Scheduled completion time step F_i (F_i = S_i + duration).
        duration: Task processing duration in time steps.
        resource_usage: Active resource consumption dict per time step during task execution.
    """
    activity_id: int               # Activity identifier integer
    start_time: int                # Computed feasible start time step
    finish_time: int               # Computed finish time step (start_time + duration)
    duration: int                  # Activity execution duration
    resource_usage: Dict[str, int] # Resource requirements active during execution


@dataclass
class AlgorithmResult:
    """
    Standardized output structure returned by all 5 metaheuristic optimization solvers.
    
    DEFINITION - Makespan (C_max or S_{N+2}):
    The total duration of the project schedule, equal to the finish time of the final dummy sink activity.
    The primary optimization objective of standard RCPSP is minimizing makespan.
    
    DEFINITION - Random-Key Priority Vector:
    A continuous vector of floating-point numbers in [0, 1] assigned to activities. Higher numbers
    indicate higher scheduling priority when selecting eligible tasks in SSGS.
    
    Attributes:
        algorithm: Name string of solver ('GA', 'SA', 'BCO', 'TS', 'PRH').
        best_priority_vector: Optimized random-key priorities found by the solver.
        schedule: List of ScheduleEntry objects for all project activities.
        makespan: Total project completion time in time units.
        runtime: Solver execution time in seconds.
        evaluations: Total number of SSGS schedule decoder evaluations performed.
        convergence_history: List of (eval_count, best_makespan) tuples for plotting convergence curves.
    """
    algorithm: str                                                  # Name identifier of optimization solver
    best_priority_vector: List[float]                               # Best random-key priority vector found
    schedule: List[ScheduleEntry]                                   # Complete feasible project schedule
    makespan: int                                                   # Optimized total project completion time (makespan)
    runtime: float                                                  # Total computation time elapsed in seconds
    evaluations: int                                                # Total number of fitness function / SSGS evaluations
    convergence_history: List[Tuple[int, int]] = field(default_factory=list) # Trajectory of best makespan over evaluations


@dataclass
class BenchmarkRecord:
    """
    Result record representing a single benchmark run on a given instance with a specific random seed.
    
    Attributes:
        instance_id: Identifier name of the benchmark dataset/file.
        algorithm: Name of optimization algorithm.
        seed: Random seed used for reproducibility.
        makespan: Project completion time achieved.
        runtime: Execution time in seconds.
        evaluations: Number of schedule evaluations performed.
        feasible: Boolean flag indicating if both precedence and resource constraints were satisfied.
        best_priority_vector: Priority vector resulting in this schedule.
        convergence_history: Optimization trajectory history.
        notes: Optional extra metadata or notes.
    """
    instance_id: str                                                # Name/ID of problem dataset instance
    algorithm: str                                                  # Name of solver algorithm executed
    seed: int                                                       # Random seed integer used
    makespan: int                                                   # Resulting project makespan achieved
    runtime: float                                                  # Run duration in seconds
    evaluations: int                                                # Count of SSGS evaluations run
    feasible: bool                                                  # True if dual-layer constraint check passed
    best_priority_vector: List[float]                               # Priority vector corresponding to schedule
    convergence_history: List[Tuple[int, int]] = field(default_factory=list) # Trajectory of evaluation index vs makespan
    notes: str = ""                                                 # Optional auxiliary information string

    @property
    def label(self) -> str:
        """Constructs a readable identifier label string for the benchmark run."""
        return f"{self.instance_id}/{self.algorithm}/seed={self.seed}" # Format instance/solver/seed label string


# ==============================================================================
# SECTION 2: SERIAL SCHEDULE GENERATION SCHEME (SSGS) DECODER
# ==============================================================================
# DEFINITION - Serial Schedule Generation Scheme (SSGS):
# SSGS is a step-by-step heuristic construction procedure for RCPSP.
# It iterates stage by stage (1 to N). At each stage, it considers the set of "eligible"
# activities (tasks whose predecessors have all finished). It selects the eligible task with the
# highest priority rank, determines its Earliest Start Time (EST) based on predecessor finish times,
# and then shifts the start time forward to the earliest time slot where all required renewable resources
# have sufficient available capacity for the entire duration [start_time, start_time + duration).

class SSGS:
    """
    Serial Schedule Generation Scheme (SSGS).
    The core deterministic decoder that converts continuous Random-Key Priority Vectors
    into 100% precedence- and resource-feasible project schedules.
    """

    @staticmethod
    def _build_priority_dict(
        project: Project,
        priority_vector: Union[Dict[int, float], List[float]],
    ) -> Dict[int, float]:
        """
        Helper function to normalize priority_vector input into a dictionary mapping activity_id -> priority float.
        """
        if isinstance(priority_vector, dict): # If input is already a dictionary mapping activity ID to float
            return priority_vector            # Return dictionary directly
        
        sorted_ids = sorted(project.activities.keys()) # Retrieve sorted list of activity IDs (1, 2, ..., N+2)
        if len(priority_vector) != len(sorted_ids):     # Validate vector length against activity count
            raise ValueError(
                f"priority_vector length {len(priority_vector)} does not match "
                f"number of activities {len(sorted_ids)}"
            )
        return dict(zip(sorted_ids, priority_vector))   # Zip sorted activity IDs with priority floats into a dict

    @staticmethod
    def generate_schedule(
        project: Project,
        priority_vector: Union[Dict[int, float], List[float]],
    ) -> Tuple[List[ScheduleEntry], int]:
        """
        Decodes a continuous random-key priority vector into a 100% feasible schedule using SSGS.
        
        Returns:
            Tuple of (list_of_ScheduleEntry_objects, total_makespan_integer).
        """
        # Step 1: Normalize priority inputs into a lookup dict of {activity_id: priority_value}
        priority = SSGS._build_priority_dict(project, priority_vector)
        
        # Step 2: Identify the dummy start activity ID (usually activity 1)
        start_act = min(project.activities.keys())

        # Step 3: Compute an initial time horizon bound (sum of durations + buffer) for resource tracking arrays
        horizon = sum(a.duration for a in project.activities.values()) + 100
        
        # Step 4: Initialize resource availability matrix avail[resource_id][time_step] = max_capacity
        avail: Dict[str, List[int]] = {
            r_id: [res.capacity] * horizon for r_id, res in project.resources.items()
        }

        # Step 5: Data structures to track scheduling state
        schedule_entries: List[ScheduleEntry] = [] # List of completed ScheduleEntry output objects
        finish_times: Dict[int, int] = {start_act: 0} # Mapping of activity_id -> scheduled finish time
        completed: set = {start_act}                  # Set of scheduled/completed activity IDs

        # Step 6: Schedule the dummy start activity at time t = 0
        schedule_entries.append(
            ScheduleEntry(
                activity_id=start_act,
                start_time=0,
                finish_time=0,
                duration=0,
                resource_usage={},
            )
        )

        # Step 7: Track uncompleted predecessor counts for each remaining activity
        uncompleted_preds = {
            a: sum(1 for p in act.predecessors if p != start_act)
            for a, act in project.activities.items()
            if a != start_act
        }
        
        # Step 8: Build initial eligible list (activities with 0 remaining unscheduled predecessors)
        eligible = [a for a, count in uncompleted_preds.items() if count == 0]

        # Step 9: Main SSGS Decoding Loop (runs until all activities are scheduled)
        while eligible:
            # Sort eligible candidate tasks in descending order of their priority key
            eligible.sort(key=lambda a: priority.get(a, 0.0), reverse=True)
            
            # Select eligible activity with highest priority score
            selected_id = eligible.pop(0)
            act = project.activities[selected_id]

            # PRECEDENCE CONSTRAINT: Compute Earliest Start Time (EST)
            # Task cannot start until ALL its predecessors have finished: EST = max(finish_time(p))
            est = max((finish_times[p] for p in act.predecessors), default=0)
            dur = act.duration
            
            # Extract active non-zero resource requirements for this activity
            reqs = [
                (r_id, req)
                for r_id, req in act.resource_requirements.items()
                if req > 0
            ]

            # RESOURCE CONSTRAINT: Search for earliest start_time >= est where resources are available
            start_time = est
            if dur > 0 and reqs:
                while True:
                    # Dynamically expand resource availability timeline array if search exceeds current horizon
                    if start_time + dur >= horizon:
                        add_h = horizon + dur + 500 # Compute new enlarged horizon
                        for r_id in avail:
                            cap = project.resources[r_id].capacity
                            avail[r_id].extend([cap] * (add_h - len(avail[r_id]))) # Extend array with full capacity
                        horizon = add_h # Update horizon tracking variable

                    # Check if all required resources have sufficient capacity during interval [start_time, start_time + dur)
                    slot_ok = True
                    for r_id, req in reqs:
                        r_avail = avail[r_id] # Access timeline array for resource r_id
                        for t in range(start_time, start_time + dur):
                            if r_avail[t] < req: # Capacity shortfall at time t
                                slot_ok = False   # Slot is invalid
                                break
                        if not slot_ok:
                            break
                    if slot_ok:
                        break       # Found feasible start time slot! Exit search loop.
                    start_time += 1 # Slot unavailable due to resource conflict; try next time step

            # Compute finish time for selected activity
            finish_time = start_time + dur
            finish_times[selected_id] = finish_time # Store finish time in lookup map

            # Deduct allocated resource units from the availability timeline array across task duration
            if dur > 0 and reqs:
                for r_id, req in reqs:
                    r_avail = avail[r_id]
                    for t in range(start_time, finish_time):
                        r_avail[t] -= req # Consume 'req' units at time t

            # Record completed schedule entry
            schedule_entries.append(
                ScheduleEntry(
                    activity_id=selected_id,
                    start_time=start_time,
                    finish_time=finish_time,
                    duration=dur,
                    resource_usage=act.resource_requirements.copy(),
                )
            )

            # Mark activity as completed and update predecessor counts of its successors
            completed.add(selected_id)
            for s in act.successors:
                if s in uncompleted_preds:
                    uncompleted_preds[s] -= 1 # Decrement predecessor count
                    if uncompleted_preds[s] == 0:
                        eligible.append(s)    # All predecessors complete! Add successor to eligible set

        # Compute project makespan (maximum finish time among all scheduled entries)
        makespan = max((e.finish_time for e in schedule_entries), default=0)
        return schedule_entries, makespan


# ==============================================================================
# SECTION 3: 30-DIMENSION FEATURE EXTRACTION & CRITICAL PATH METHOD (CPM)
# ==============================================================================
# DEFINITION - Critical Path Method (CPM):
# CPM is an algorithm for scheduling project activities assuming INFINITE resource capacity.
# It performs a Forward Pass to compute Early Start Time (EST) and Early Finish Time (EFT),
# followed by a Backward Pass to compute Late Finish Time (LFT) and Late Start Time (LST).
# The difference (LST - EST) is "Total Slack". Activities with zero slack form the Critical Path.

# DEFINITION - Standard RCPSP Instance Complexity Metrics (Kolisch et al.):
# 1. Network Complexity (NC): Average number of precedence relations per non-dummy activity.
# 2. Resource Factor (RF): Average fraction of resource types requested by activities (0 = none, 1 = all).
# 3. Resource Strength (RS): Measures scarcity of resources (1 = unconstrained CPM schedule feasible, 0 = severe bottleneck).

@dataclass
class ProjectFeatures:
    """Dataclass encapsulating 30 topological & resource feature dimensions extracted from an RCPSP graph."""
    N: float                                        # Number of real non-dummy project activities
    NC: float                                       # Network Complexity (edges / N)
    RF: float                                       # Resource Factor (average proportion of resources requested per activity)
    RS: float                                       # Resource Strength (resource tightness normalized in [0, 1])
    activity_count: int                             # Total activity nodes in DAG (including dummy start/sink)
    resource_count: int                             # Total renewable resource types
    average_duration: float                         # Mean processing duration across activities
    duration_std: float                             # Standard deviation of activity durations
    minimum_duration: int                           # Minimum single task duration
    maximum_duration: int                           # Maximum single task duration
    total_duration: int                             # Sum of all activity durations combined
    precedence_edge_count: int                      # Total precedence dependency edges in graph
    precedence_density: float                       # Graph density ratio: 2*edges / (N * (N-1))
    average_predecessors: float                     # Mean predecessor count per node
    average_successors: float                       # Mean successor count per node
    maximum_predecessors: int                       # Max predecessors for any single node
    maximum_successors: int                         # Max successors for any single node
    source_count: int                               # Nodes with 0 predecessors (sources)
    sink_count: int                                 # Nodes with 0 successors (sinks)
    network_depth: int                              # Longest topological path depth (number of levels)
    estimated_parallelism: float                    # Total work duration divided by Critical Path duration
    average_parallel_activities: float              # Average activities that could run concurrently
    average_resource_demand: float                  # Mean resource units requested across all task-resource pairs
    resource_demand_std: float                      # Standard deviation of resource demands
    resource_utilization_ratio: float               # Mean total workload demand / (capacity * CPM duration)
    average_resource_tightness: float               # Mean tightness across resource dimensions
    maximum_resource_tightness: float               # Max tightness value among resource types
    number_of_unused_resource_dimensions: int       # Count of resource types with zero total demand
    critical_path_duration: int                     # CPM duration assuming unconstrained resources
    critical_path_activity_count: int               # Number of activities lying on the critical path (slack = 0)

    def to_dict(self) -> Dict[str, float]:
        """Converts feature dataclass fields to a plain Python string->float dictionary."""
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


def validate_dag(project: Project) -> List[int]:
    """
    Validates that the project dependency graph is a valid Directed Acyclic Graph (DAG).
    Uses Kahn's Algorithm to return a valid Topological Sort order.
    
    DEFINITION - Kahn's Algorithm for Topological Sort:
    1. Calculate in-degree (number of incoming predecessor edges) for every activity node.
    2. Enqueue all nodes with in-degree = 0 (sources).
    3. Dequeue a node, add it to topological order, and decrement in-degree of its successors.
    4. Enqueue successors whose in-degree becomes 0.
    5. If topological order length == total nodes, DAG is valid; else a CYCLE exists!
    """
    # Step 1: Compute in-degree (number of predecessors) for each activity node
    in_degree = {a: len(act.predecessors) for a, act in project.activities.items()}
    
    # Step 2: Queue all nodes with 0 predecessors (ready to start)
    queue = deque([a for a, deg in in_degree.items() if deg == 0])
    topo_order = []

    # Step 3: Process nodes in queue
    while queue:
        curr = queue.popleft()        # Extract node with zero remaining predecessors
        topo_order.append(curr)       # Append node to topological order
        for s in project.activities[curr].successors: # For each successor node
            if s in in_degree:
                in_degree[s] -= 1     # Decrement successor's remaining predecessor count
                if in_degree[s] == 0:
                    queue.append(s)   # Add successor to queue once all predecessors processed

    # Step 4: Verify DAG validity
    if len(topo_order) != len(project.activities):
        raise ValueError("Cycle detected in project precedence graph! Not a valid DAG.")
    return topo_order


def extract_features(project: Project) -> ProjectFeatures:
    """
    Computes 30 topological and resource feature dimensions including CPM Critical Path, NC, RF, and RS.
    """
    # Step 1: Obtain topological sort order and count real (non-dummy) activities
    topo_order = validate_dag(project)
    real_activities = [act for act in project.activities.values() if act.duration > 0 or (act.predecessors and act.successors)]
    N = float(len(real_activities))

    # Step 2: Compute Network Complexity (NC) = Total Precedence Edges / N
    total_edges = sum(len(act.successors) for act in project.activities.values())
    NC = total_edges / N if N > 0 else 0.0

    # Step 3: Critical Path Method (CPM) FORWARD PASS -> Compute EST and EFT
    est: Dict[int, int] = {} # Early Start Time map
    eft: Dict[int, int] = {} # Early Finish Time map
    for a_id in topo_order:
        act = project.activities[a_id]
        # EST is maximum EFT among all predecessor tasks
        est_val = max((eft[p] for p in act.predecessors), default=0)
        est[a_id] = est_val
        eft[a_id] = est_val + act.duration # EFT = EST + duration

    # Project Critical Path duration is the max EFT of any task (specifically the dummy sink)
    cp_duration = max(eft.values(), default=0)

    # Step 4: CPM BACKWARD PASS -> Compute LFT and LST
    lft: Dict[int, int] = {} # Late Finish Time map
    lst: Dict[int, int] = {} # Late Start Time map
    for a_id in reversed(topo_order):
        act = project.activities[a_id]
        if not act.successors:
            lft_val = cp_duration # Sink activity LFT equals project CP duration
        else:
            lft_val = min(lst[s] for s in act.successors) # LFT is min LST among successors
        lft[a_id] = lft_val
        lst[a_id] = lft_val - act.duration # LST = LFT - duration

    # Step 5: Calculate Total Float/Slack and Identify Critical Path Activities
    # Total Slack = LST - EST (time a task can be delayed without delaying project completion)
    total_slack = {a_id: lst[a_id] - est[a_id] for a_id in project.activities}
    cp_activities = [a_id for a_id, slack in total_slack.items() if slack == 0]

    # Step 6: Compute Resource Strength (RS)
    # RS_k = (Capacity_k - Min_Demand_k) / (Max_Demand_k_CPM - Min_Demand_k)
    resource_keys = sorted(project.resources.keys())
    K = len(resource_keys)

    if cp_duration > 0 and K > 0:
        # Build CPM peak resource usage timeline
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

    # Step 7: Compute Resource Factor (RF)
    # RF measures the average fraction of resource types requested by activities
    if N > 0 and K > 0:
        rf_count = sum(1 for act in real_activities for r_id in resource_keys if act.resource_requirements.get(r_id, 0) > 0)
        RF = float(rf_count) / float(N * K)
    else:
        RF = 0.0

    # Step 8: Compute activity duration statistics
    durations = [act.duration for act in project.activities.values()]
    avg_dur = float(statistics.mean(durations)) if durations else 0.0
    std_dur = float(statistics.pstdev(durations)) if len(durations) > 1 else 0.0

    # Predecessor/Successor degree metrics
    pred_counts = [len(act.predecessors) for act in project.activities.values()]
    succ_counts = [len(act.successors) for act in project.activities.values()]

    sources = sum(1 for c in pred_counts if c == 0) # Count of source nodes
    sinks = sum(1 for c in succ_counts if c == 0)   # Count of sink nodes

    # Calculate graph depth (longest path in terms of level index)
    level: Dict[int, int] = {}
    for a_id in topo_order:
        act = project.activities[a_id]
        level[a_id] = max((level[p] + 1 for p in act.predecessors), default=1)
    depth = max(level.values(), default=0)

    # Precedence Density and Estimated Parallelism
    pd = (2.0 * total_edges) / (project.num_activities * (project.num_activities - 1)) if project.num_activities > 1 else 0.0
    ep = sum(durations) / float(cp_duration) if cp_duration > 0 else 0.0

    # Step 9: Compute Resource Tightness metrics
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

    # Step 10: Assemble and return ProjectFeatures object instance
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
# DEFINITION - Dual-Layer Constraint Validation:
# An independent audit step that verifies whether a generated schedule satisfies BOTH standard RCPSP constraints:
# 1. Precedence Constraint Layer: For every task i and every predecessor p of i: FinishTime(p) <= StartTime(i).
# 2. Resource Capacity Constraint Layer: For every renewable resource k and every time step t:
#    The sum of demands of all tasks active at time t must NOT exceed capacity(k).

class Evaluator:
    """
    Evaluates optimization candidates and performs independent dual-layer constraint verification.
    """

    @staticmethod
    def evaluate(project: Project, priority_vector: Union[Dict[int, float], List[float]]) -> Tuple[List[ScheduleEntry], int, List[ScheduleEntry]]:
        """
        Evaluates a random-key priority vector by decoding it via SSGS.
        
        Returns:
            Tuple of (schedule_entries, makespan, schedule_entries).
        """
        sched, ms = SSGS.generate_schedule(project, priority_vector) # Invoke SSGS decoder
        return sched, ms, sched                                       # Return decoded schedule and makespan

    @staticmethod
    def check_constraints(project: Project, schedule: List[ScheduleEntry]) -> bool:
        """
        Dual-Layer Constraint Validation Engine:
        Independently audits the schedule against graph precedence rules and resource capacity limits.
        
        Returns:
            True if 100% FEASIBLE; False if ANY constraint violation is detected.
        """
        # Build quick lookup dictionaries for activity start and finish times
        finish_map = {e.activity_id: e.finish_time for e in schedule}
        start_map = {e.activity_id: e.start_time for e in schedule}

        # LAYER 1: Precedence Constraint Audit
        # Rule: Finish time of predecessor 'p' MUST be <= Start time of activity 'act_id'
        for act_id, act in project.activities.items():
            if act_id in start_map:
                s_time = start_map[act_id] # Start time of current task
                for p in act.predecessors:
                    # Violation: Predecessor finished AFTER current task started!
                    if p in finish_map and finish_map[p] > s_time:
                        return False # Precedence violation detected! Return False immediately.

        # LAYER 2: Resource Capacity Constraint Audit
        # Rule: Sum of resource requirements for all active tasks at time 't' MUST be <= resource capacity
        max_t = max((e.finish_time for e in schedule), default=0) # Find maximum time horizon
        for r_id, res in project.resources.items():
            usage = [0] * (max_t + 1) # Initialize usage tally array for resource r_id across all time steps
            for e in schedule:
                req = project.activities[e.activity_id].resource_requirements.get(r_id, 0)
                if req > 0:
                    for t in range(e.start_time, e.finish_time):
                        usage[t] += req # Accumulate active resource demand at time t
            # Violation: Active usage exceeds resource maximum capacity at any time step t
            if any(u > res.capacity for u in usage):
                return False # Resource overload detected! Return False immediately.

        return True # Both constraint layers passed successfully! Schedule is 100% FEASIBLE.


# ==============================================================================
# SECTION 5: FILE PARSER (.sm PSPLIB, .rcp Paterson, .json)
# ==============================================================================
# DEFINITION - Project File Formats:
# 1. PSPLIB (.sm): Standard benchmark format developed by Kolisch et al. (1995) for RCPSP instances (j30, j60, j120).
#    Contains sections: PRECEDENCE RELATIONS, REQUESTS/DURATIONS, RESOURCEAVAILABILITIES.
# 2. Paterson (.rcp): Classic file format by Paterson (1984). Lists activity count, resource count, capacities,
#    followed by line-by-line durations, resource demands, and successor lists.
# 3. JSON (.json): Structured JSON payload for web APIs and modern integrations.

class Parser:
    """Multi-format parser supporting PSPLIB .sm files, Paterson .rcp files, and custom JSON files."""

    @staticmethod
    def parse(file_path: Union[str, Path], display_name: Optional[str] = None) -> Project:
        """
        Main entry point for parsing project instance files into a Project object.
        Automatically detects file extension (.sm, .rcp, .json) and calls appropriate parser method.
        """
        path = Path(file_path)
        if not path.exists(): # Verify file exists on local filesystem
            raise FileNotFoundError(f"Project file not found: {file_path}")

        ext = path.suffix.lower() # Extract file extension in lowercase
        if ext == ".sm":
            return Parser._parse_sm(str(path))   # Parse PSPLIB .sm format
        elif ext == ".rcp":
            return Parser._parse_rcp(str(path))  # Parse Paterson .rcp format
        elif ext == ".json":
            return Parser._parse_json(str(path)) # Parse JSON format
        else:
            raise ValueError(f"Unsupported format '{ext}'. Use .sm, .rcp, or .json")

    @staticmethod
    def _parse_sm(file_path: str) -> Project:
        """Parses a PSPLIB .sm benchmark instance using the 'psplib' python package if available, else fallback."""
        try:
            import psplib # Try importing third-party psplib parser package
            raw = psplib.parse_psplib(file_path)
            project = Project()

            # Process renewable resource capacities
            for i, res in enumerate(raw.resources):
                if res.renewable:
                    r_id = f"R{i + 1}"
                    project.resources[r_id] = Resource(id=r_id, capacity=res.capacity)

            renewable_indices = [i for i, r in enumerate(raw.resources) if r.renewable]
            renewable_r_ids = [f"R{i + 1}" for i in range(len(renewable_indices))]

            num_psp_acts = raw.num_activities
            predecessors: Dict[int, List[int]] = {(i + 1): [] for i in range(num_psp_acts)}

            # Build predecessor mappings from successor lists
            for psp_idx, act in enumerate(raw.activities):
                job_nr = psp_idx + 1
                for succ_0idx in act.successors:
                    succ_job = succ_0idx + 1
                    if succ_job not in predecessors:
                        predecessors[succ_job] = []
                    predecessors[succ_job].append(job_nr)

            # Detect phantom/dummy sink nodes referenced in successors
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

            # Populate Activity objects with durations and resource demands
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

            # Add phantom dummy sink node if necessary
            if phantom_jobs:
                sink_reqs = {r_id: 0 for r_id in renewable_r_ids}
                project.activities[sink_job] = Activity(
                    id=sink_job,
                    duration=0,
                    predecessors=predecessors.get(sink_job, []),
                    resource_requirements=sink_reqs,
                )

            # Reconstruct successor lists for all activities
            for act in project.activities.values():
                act.successors = []
            for a_id, act in project.activities.items():
                for p_id in act.predecessors:
                    if p_id in project.activities:
                        project.activities[p_id].successors.append(a_id)

            return project
        except Exception:
            # Fall back to built-in pure Python regex/line parser if psplib is missing or errors
            return Parser._parse_sm_fallback(file_path)

    @staticmethod
    def _parse_sm_fallback(file_path: str) -> Project:
        """Pure-Python fallback parser for PSPLIB .sm files (reads line by line)."""
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
                    num_res = int(parts[1].split()[0].strip()) # Extract renewable resource count
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

            # Parse Precedence section
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

            # Parse Requests & Durations section
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

            # Parse Resource Capacities section
            elif mode == "RESOURCES":
                parts = line_str.split()
                if not parts or not parts[0].isdigit():
                    continue
                for idx, cap in enumerate(parts[:num_res]):
                    r_id = f"R{idx+1}"
                    project.resources[r_id] = Resource(id=r_id, capacity=int(cap))
                mode = None

        # Build successor lists from predecessor links
        for act in project.activities.values():
            act.successors = []
        for a_id, act in project.activities.items():
            for p_id in act.predecessors:
                if p_id in project.activities:
                    project.activities[p_id].successors.append(a_id)

        return project

    @staticmethod
    def _parse_rcp(file_path: str) -> Project:
        """Parses a Paterson .rcp format project benchmark file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tokens = content.split()
        if not tokens:
            raise ValueError("Empty .rcp file")

        n_jobs = int(tokens[0])      # Extract number of activities
        n_resources = int(tokens[1]) # Extract number of resource types
        idx = 2

        project = Project()
        # Parse resource capacities
        for r in range(n_resources):
            r_id = f"R{r+1}"
            project.resources[r_id] = Resource(id=r_id, capacity=int(tokens[idx]))
            idx += 1

        # Parse tasks line by line
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

        # Build predecessor links from successor lists
        for act_id, act in project.activities.items():
            for s in act.successors:
                if s in project.activities:
                    if act_id not in project.activities[s].predecessors:
                        project.activities[s].predecessors.append(act_id)

        return project

    @staticmethod
    def _parse_json(file_path: str) -> Project:
        """Parses a structured JSON representation of an RCPSP project instance."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        project = Project()
        # Parse resources block
        for r_id, cap in data.get("resources", {}).items():
            project.resources[r_id] = Resource(id=r_id, capacity=int(cap))

        # Parse activities block
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
# DEFINITION - Metaheuristics for RCPSP:
# High-level algorithmic frameworks designed to search for optimal or near-optimal solutions
# in large combinatorial search spaces. All 5 solvers use Random-Key Representation:
# Chromosomes are vectors of real numbers in [0, 1]. The decoder (SSGS) interprets these numbers
# as task priority ranks, guaranteeing that generated schedules are ALWAYS 100% FEASIBLE!

def solve_ga(
    project: Project,
    evaluation_budget: int = 500,
    seed: int = 42,
    population_size: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    elite_count: int = 2,
    tournament_k: int = 3,
) -> AlgorithmResult:
    """
    1. GENETIC ALGORITHM (GA) SOLVER
    Uses population-based evolutionary search with tournament selection, uniform crossover,
    and random-key mutation. Elitism preserves top solutions across generations.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history = []

    # Step 1: Initialize random population of random-key priority vectors
    pop = [[rng.random() for _ in range(n)] for _ in range(population_size)]
    fits, scheds = [], []
    for ind in pop:
        _, ms, sched = Evaluator.evaluate(project, ind)
        fits.append(ms)
        scheds.append(sched)
        evaluations += 1

    # Track overall best solution found
    best_idx = min(range(population_size), key=lambda i: fits[i])
    best_ms, best_vec, best_sched = fits[best_idx], pop[best_idx][:], scheds[best_idx]
    convergence_history.append((evaluations, best_ms))

    # Step 2: Main Evolutionary Loop
    while evaluations < evaluation_budget:
        # Rank population by makespan (lower is better)
        ranked = sorted(range(population_size), key=lambda i: fits[i])
        
        # Elitism: Copy top 'elite_count' solutions directly to next generation
        new_pop = [pop[i][:] for i in ranked[:elite_count]]
        new_fits = [fits[i] for i in ranked[:elite_count]]
        new_scheds = [scheds[i] for i in ranked[:elite_count]]

        # Generate offspring until population capacity or budget exhausted
        while len(new_pop) < population_size and evaluations < evaluation_budget:
            # Tournament Selection for Parent 1 and Parent 2
            s1 = rng.sample(range(population_size), tournament_k)
            p1 = pop[min(s1, key=lambda i: fits[i])][:]
            s2 = rng.sample(range(population_size), tournament_k)
            p2 = pop[min(s2, key=lambda i: fits[i])][:]

            # Uniform Crossover (50% gene swap probability)
            if rng.random() < crossover_rate:
                c1 = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(n)]
                c2 = [p2[i] if rng.random() < 0.5 else p1[i] for i in range(n)]
            else:
                c1, c2 = p1[:], p2[:]

            # Uniform Random Mutation (resets random key gene if random < mutation_rate)
            c1 = [rng.random() if rng.random() < mutation_rate else g for g in c1]
            c2 = [rng.random() if rng.random() < mutation_rate else g for g in c2]

            # Evaluate offspring solutions
            for child in (c1, c2):
                if len(new_pop) >= population_size or evaluations >= evaluation_budget:
                    break
                _, ms, sched = Evaluator.evaluate(project, child)
                evaluations += 1
                new_pop.append(child)
                new_fits.append(ms)
                new_scheds.append(sched)

                # Update global best solution
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
    evaluation_budget: int = 500,
    seed: int = 42,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.01,
) -> AlgorithmResult:
    """
    2. SIMULATED ANNEALING (SA) SOLVER
    
    DEFINITION - Metropolis Acceptance Criterion:
    Standard trajectory-based metaheuristic inspired by thermal annealing in metallurgy.
    If a neighbor solution improves makespan (delta < 0), it is ALWAYS accepted.
    If it worsens makespan (delta > 0), it is accepted probabilistically with P = exp(-delta / T),
    allowing the solver to escape local optima at higher temperatures.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history = []

    # Step 1: Initialize current candidate solution
    curr = [rng.random() for _ in range(n)]
    _, curr_ms, curr_sched = Evaluator.evaluate(project, curr)
    evaluations += 1

    best_vec, best_ms, best_sched = curr[:], curr_ms, curr_sched
    convergence_history.append((evaluations, best_ms))
    temp = initial_temp # Set initial temperature T_0

    # Step 2: Annealing Loop
    while evaluations < evaluation_budget and temp > min_temp:
        cand = curr[:]
        # Neighborhood Move: 50% chance pairwise gene swap, 50% chance Gaussian perturbation
        if rng.random() < 0.5 and n >= 2:
            i, j = rng.sample(range(n), 2)
            cand[i], cand[j] = cand[j], cand[i] # Pairwise swap
        else:
            i = rng.randrange(n)
            cand[i] = max(0.0, min(1.0, cand[i] + rng.gauss(0, 0.1))) # Gaussian noise

        # Evaluate candidate solution
        _, cand_ms, cand_sched = Evaluator.evaluate(project, cand)
        evaluations += 1
        delta = cand_ms - curr_ms # Difference in makespan (cand - curr)

        # Metropolis Acceptance Rule: Accept if better (delta < 0) OR with probability exp(-delta / T)
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            curr, curr_ms, curr_sched = cand, cand_ms, cand_sched

        # Update global best solution found so far
        if curr_ms < best_ms:
            best_vec, best_ms, best_sched = curr[:], curr_ms, curr_sched
            convergence_history.append((evaluations, best_ms))

        temp *= cooling_rate # Cool down temperature exponentially: T = T * alpha

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
    evaluation_budget: int = 500,
    seed: int = 42,
    colony_size: int = 20,
    limit: int = 15,
) -> AlgorithmResult:
    """
    3. BEE COLONY OPTIMIZATION (BCO / ABC) SOLVER
    
    DEFINITION - Artificial Bee Colony (Karaboga):
    Mimics honeybee foraging behavior with 3 bee roles:
    1. Employed Bees: Perform local search around current food sources using neighbor equation v_ij = x_ij + phi*(x_ij - x_kj).
    2. Onlooker Bees: Select promising food sources probabilistically via roulette wheel selection based on fitness.
    3. Scout Bees: Abandon food sources that fail to improve after 'limit' trials and reset with random search vectors.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history = []

    # Step 1: Initialize food sources (colony)
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

    # Step 2: Main Artificial Bee Colony Loop
    while evaluations < evaluation_budget:
        # PHASE 1: Employed Bee Phase (Local search around each food source)
        for i in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            partner_idx = rng.choice([j for j in range(colony_size) if j != i])
            cand = sources[i][:]
            idx = rng.randrange(n)
            phi = rng.uniform(-1.0, 1.0)
            # Differential variation formula: v_ij = x_ij + phi * (x_ij - x_kj)
            cand[idx] = max(0.0, min(1.0, cand[idx] + phi * (cand[idx] - sources[partner_idx][idx])))

            _, cand_ms, cand_sched = Evaluator.evaluate(project, cand)
            evaluations += 1

            # Greedy selection between current source and candidate
            if cand_ms <= fits[i]:
                sources[i], fits[i], scheds[i], trials[i] = cand, cand_ms, cand_sched, 0
            else:
                trials[i] += 1

            if fits[i] < best_ms:
                best_ms, best_vec, best_sched = fits[i], sources[i][:], scheds[i]
                convergence_history.append((evaluations, best_ms))

        # PHASE 2: Onlooker Bee Phase (Probabilistic selection based on fitness)
        max_ms = max(fits) if fits else 1
        weights = [(max_ms - f + 1) for f in fits]
        tot = sum(weights) or 1.0
        probs = [w / tot for w in weights] # Roulette wheel probabilities

        for _ in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            # Roulette Wheel Selection of food source
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

        # PHASE 3: Scout Bee Phase (Reset stagnant food sources exceeding 'limit' trials)
        for i in range(colony_size):
            if evaluations >= evaluation_budget:
                break
            if trials[i] >= limit: # Food source exhausted!
                sources[i] = [rng.random() for _ in range(n)] # Re-initialize with random vector
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
    evaluation_budget: int = 500,
    seed: int = 42,
    tenure: int = 7,
    n_neighbours: int = 10,
) -> AlgorithmResult:
    """
    4. TABU SEARCH (TS) SOLVER
    
    DEFINITION - Tabu Search & Aspiration Criterion (Glover):
    A memory-based local search metaheuristic that maintains a "Tabu List" of recent move operations
    (e.g., swapping position i and j) to prevent cycling back to recently visited states.
    ASPIRATION CRITERION: If a move is tabu but produces a new GLOBAL BEST makespan, the tabu status
    is overridden and the move is accepted anyway!
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    n = len(project.activities)
    evaluations = 0
    convergence_history = []

    # Step 1: Initialize current solution
    curr = [rng.random() for _ in range(n)]
    _, curr_ms, curr_sched = Evaluator.evaluate(project, curr)
    evaluations += 1

    best_vec, best_ms, best_sched = curr[:], curr_ms, curr_sched
    convergence_history.append((evaluations, best_ms))

    # Tabu list FIFO queue and set for fast O(1) membership checks
    tabu_list: Deque[Tuple[int, int]] = deque()
    tabu_set = set()

    # Step 2: Main Tabu Search Loop
    while evaluations < evaluation_budget:
        neighbours = []
        seen_moves = set()
        attempts = 0
        
        # Generate neighborhood solutions via pairwise index swaps
        while len(neighbours) < n_neighbours and attempts < n_neighbours * 5 and n >= 2:
            attempts += 1
            i, j = sorted(rng.sample(range(n), 2))
            if (i, j) in seen_moves:
                continue
            seen_moves.add((i, j))
            v = curr[:]
            v[i], v[j] = v[j], v[i] # Swap genes at index i and j
            neighbours.append((v, (i, j)))

        best_cand, best_cand_ms, best_cand_sched, best_cand_move = None, None, None, None

        # Evaluate neighborhood candidates
        for vec, move_id in neighbours:
            if evaluations >= evaluation_budget:
                break
            _, ms, sched = Evaluator.evaluate(project, vec)
            evaluations += 1

            is_tabu = move_id in tabu_set
            aspiration = ms < best_ms # Aspiration condition: new global best!

            # Reject move if it is tabu AND does not satisfy aspiration criterion
            if is_tabu and not aspiration:
                continue

            # Select best non-tabu (or aspiration-satisfying) move in neighborhood
            if best_cand_ms is None or ms < best_cand_ms:
                best_cand, best_cand_ms, best_cand_sched, best_cand_move = vec, ms, sched, move_id

        # Diversification / Restart if neighborhood yields no valid move
        if best_cand is None:
            curr = [rng.random() for _ in range(n)] # Random restart
            if evaluations < evaluation_budget:
                _, curr_ms, curr_sched = Evaluator.evaluate(project, curr)
                evaluations += 1
            tabu_list.clear()
            tabu_set.clear()
            continue

        # Move to selected neighbor solution
        curr = best_cand
        if best_cand_ms is not None:
            curr_ms, curr_sched = best_cand_ms, best_cand_sched

        # Add move to Tabu List and enforce maximum tenure length
        if best_cand_move is not None:
            tabu_list.append(best_cand_move)
            tabu_set.add(best_cand_move)
            if len(tabu_list) > tenure:
                tabu_set.discard(tabu_list.popleft()) # Evict oldest move from tabu list

        # Update global best solution found so far
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
    evaluation_budget: int = 500,
    seed: int = 42,
) -> AlgorithmResult:
    """
    5. PRIORITY RULE HEURISTICS (PRH) SOLVER
    
    DEFINITION - Priority Rules:
    Deterministic dispatching rules based on CPM network properties:
    - EST (Earliest Start Time): Prioritize tasks that can start earliest.
    - EFT (Earliest Finish Time): Prioritize tasks that can finish earliest.
    - LST (Latest Start Time): Prioritize tasks with lowest late start (minimizes risk of delay).
    - LFT (Latest Finish Time): Prioritize tasks with lowest late finish.
    Followed by random sampling to consume remaining evaluation budget.
    """
    rng = random.Random(seed)
    start_ts = time.perf_counter()
    evaluations = 0
    convergence_history = []

    # Step 1: Perform CPM forward and backward passes to compute EST, EFT, LST, LFT
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

    # Step 2: Build priority rule dictionaries
    sorted_ids = sorted(project.activities.keys())
    rules = {
        "EST": {a: (horizon + 1 - est[a]) for a in sorted_ids},
        "EFT": {a: (horizon + 1 - eft[a]) for a in sorted_ids},
        "LST": {a: (horizon + 1 - lst[a]) for a in sorted_ids},
        "LFT": {a: (horizon + 1 - lft[a]) for a in sorted_ids},
    }

    best_ms, best_vec, best_sched, best_rule = None, None, None, ""

    # Step 3: Evaluate each of the 4 priority rules
    for rule_name, prio_dict in rules.items():
        if evaluations >= evaluation_budget:
            break
        max_p = max(prio_dict.values()) or 1.0
        vec = [prio_dict[a] / max_p for a in sorted_ids] # Normalize to [0, 1]
        _, ms, sched = Evaluator.evaluate(project, vec)
        evaluations += 1

        if best_ms is None or ms < best_ms:
            best_ms, best_vec, best_sched, best_rule = ms, vec, sched, rule_name
            convergence_history.append((evaluations, best_ms))

    # Step 4: Random sampling loop to use remaining evaluation budget
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
# DEFINITION - Statistical Benchmark Comparison:
# Runs multiple solvers across random seeds on a single project instance to generate
# comparative statistics: best makespan, worst makespan, mean makespan, standard deviation,
# average execution runtime, evaluation counts, and feasibility rate.

@dataclass
class AlgorithmSummary:
    """
    Statistical summary aggregated across multiple benchmark runs for a specific algorithm.
    """
    algorithm: str                                                  # Name of the algorithm evaluated
    best_makespan: int                                              # Best (minimum) makespan found across runs
    mean_makespan: float                                            # Mean makespan value
    worst_makespan: int                                             # Worst (maximum) makespan found
    std_makespan: float                                             # Standard deviation of makespan
    mean_runtime: float                                             # Average runtime in seconds
    mean_evaluations: float                                         # Average SSGS evaluation count
    n_runs: int                                                     # Total benchmark runs executed
    n_feasible: int                                                 # Number of runs producing feasible schedules
    all_feasible: bool                                              # True if 100% of runs produced feasible schedules
    records: List[BenchmarkRecord] = field(default_factory=list)   # Raw BenchmarkRecord items

@dataclass
class ComparisonReport:
    """
    Structured comparison report comparing performance of all 5 metaheuristic solvers.
    """
    instance_id: str                                                # Name/ID of problem dataset instance
    summaries: List[AlgorithmSummary]                               # Summary record for each algorithm
    n_seeds: int                                                    # Number of random seeds tested
    budget: int                                                     # SSGS evaluation budget per run
    winner: str                                                     # Best performing algorithm name
    winner_ms: int                                                  # Best overall makespan achieved

    def to_text(self) -> str:
        """Formats the comparison report into a clean, human-readable ASCII text table."""
        lines = [
            f"=== Comparison Report: {self.instance_id} ===",
            f"Seeds: {self.n_seeds}  |  Evaluation budget: {self.budget} per run",
            "",
            f"{'Algorithm':<12} {'Best':>6} {'Mean':>7} {'Worst':>7} {'Std':>6} {'Runtime(s)':>11} {'Evals':>7} {'Feasible':>9}",
            "-" * 72,
        ]

        # Sort algorithms by best makespan, then mean makespan
        for s in sorted(self.summaries, key=lambda x: (x.best_makespan, x.mean_makespan)):
            # Mark winning algorithm with arrow indicator (◄)
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
    evaluation_budget: int = 500,
    instance_id: str = "instance",
) -> List[BenchmarkRecord]:
    """
    Executes all 5 metaheuristic solvers across multiple random seed values.
    Returns a list of BenchmarkRecord objects containing performance statistics.
    """
    records = []
    for seed in seeds:
        # Loop through each solver registered in SOLVER_REGISTRY
        for algo_name, solver_fn in SOLVER_REGISTRY.items():
            # Run solver algorithm
            res = solver_fn(project, evaluation_budget=evaluation_budget, seed=seed)
            # Perform independent dual-layer constraint check
            feasible = Evaluator.check_constraints(project, res.schedule)
            # Append benchmark record
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


def compare_algorithms(records: List[BenchmarkRecord], instance_id: str = "instance", budget: int = 500) -> ComparisonReport:
    """
    Builds a statistical ComparisonReport object aggregating benchmark results across algorithms.
    Computes mean, min, max, std, runtime averages, and determines the overall winning algorithm.
    """
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
# SECTION 8: PRINTABLE PDF REPORT GENERATOR ENGINE
# ==============================================================================
# DEFINITION - ReportLab Document Engine:
# Generates publication-ready PDF reports containing executive summaries, feature tables,
# and high-resolution Gantt chart images using ReportLab Platypus flowables and Matplotlib.

def generate_pdf_report(
    project: Any,
    features: Dict[str, Any],
    best_res: Optional[Any] = None,
    all_results: Optional[Dict[str, Any]] = None,
    proj_name: str = "Project",
) -> bytes:
    """
    Builds a printable PDF report byte stream using ReportLab and Matplotlib.
    
    Returns:
        bytes: Binary PDF file data stream ready to send to browser or save to disk.
    """
    buffer = io.BytesIO() # In-memory byte buffer
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Define ReportLab paragraph typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
    )
    heading_style = ParagraphStyle(
        "SecHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    caption_style = ParagraphStyle(
        "ChartCaption",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
    )

    story = [] # ReportLab Platypus document element stream

    # Section 1: Title Banner
    story.append(Paragraph(f"AI-RCPSP Project Analysis & Optimization Report", title_style))
    story.append(Paragraph(f"Instance File: <b>{proj_name}</b> | Generated automatically by AI-RCPSP Framework", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12))

    # Section 2: Executive Summary Table
    story.append(Paragraph("1. Executive Summary", heading_style))
    if best_res:
        winner_name = best_res.algorithm
        best_ms = best_res.makespan
        runtime_str = f"{best_res.runtime:.4f} seconds"
        status_str = "<font color='#16a34a'><b>100% FEASIBLE (Precedence & Resource Constraints Satisfied)</b></font>"
    else:
        winner_name = "N/A"
        best_ms = "N/A"
        runtime_str = "N/A"
        status_str = "Optimization pending"

    summary_data = [
        [Paragraph("<b>Winning Solver:</b>", body_style), Paragraph(str(winner_name), body_style)],
        [Paragraph("<b>Optimal Makespan:</b>", body_style), Paragraph(f"<b>{best_ms}</b> time units", body_style)],
        [Paragraph("<b>Solver Runtime:</b>", body_style), Paragraph(runtime_str, body_style)],
        [Paragraph("<b>Feasibility Status:</b>", body_style), Paragraph(status_str, body_style)],
        [Paragraph("<b>Critical Path Bound (CPM):</b>", body_style), Paragraph(f"{int(features.get('critical_path_duration', 0))} time units", body_style)],
    ]

    sum_table = Table(summary_data, colWidths=[160, 380])
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 12))

    # Section 3: Project Topological & Constraint Characteristics Table
    story.append(Paragraph("2. Project Topological & Constraint Characteristics", heading_style))
    char_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Symbol</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Plain-English Context & Definition</b>", body_style)],
        [Paragraph("Real Activities", body_style), Paragraph("N", body_style), Paragraph(str(int(features.get("N", 0))), body_style), Paragraph("Total non-dummy project tasks requiring time and resources.", body_style)],
        [Paragraph("Network Complexity", body_style), Paragraph("NC", body_style), Paragraph(f"{features.get('NC', 0):.2f}", body_style), Paragraph("Average precedence links per activity (graph density).", body_style)],
        [Paragraph("Resource Factor", body_style), Paragraph("RF", body_style), Paragraph(f"{features.get('RF', 0):.2f}", body_style), Paragraph("Average proportion of resource types requested per task.", body_style)],
        [Paragraph("Resource Strength", body_style), Paragraph("RS", body_style), Paragraph(f"{features.get('RS', 0):.2f}", body_style), Paragraph("Resource tightness (1.0 = unconstrained CPM, 0.0 = bottleneck).", body_style)],
        [Paragraph("Critical Path", body_style), Paragraph("CPM", body_style), Paragraph(str(int(features.get("critical_path_duration", 0))), body_style), Paragraph("Theoretical minimum duration assuming infinite resources.", body_style)],
    ]
    char_table = Table(char_data, colWidths=[110, 50, 60, 320])
    char_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(char_table)
    story.append(Spacer(1, 14))

    # Section 4: Rendered Gantt Chart Image Generation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        if best_res and best_res.schedule:
            # Render Matplotlib Gantt chart
            fig, ax = plt.subplots(figsize=(7, 2.5), dpi=200)
            for entry in best_res.schedule:
                if entry.duration > 0:
                    ax.barh(f"Act {entry.activity_id}", entry.duration, left=entry.start_time, color="#0284c7", edgecolor="#0369a1")
            ax.set_xlabel("Time Units")
            ax.set_ylabel("Activity")
            ax.set_title("Project Schedule Gantt Chart", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            gantt_path = tmp_path / "gantt.png"
            plt.savefig(gantt_path)
            plt.close()

            story.append(Paragraph("3. Schedule Gantt Timeline", heading_style))
            story.append(Image(str(gantt_path), width=500, height=180))
            story.append(Paragraph("Figure 1: Optimized activity start and finish times decoded via SSGS.", caption_style))
            story.append(Spacer(1, 12))

        # Build PDF document inside tempdir context so image files exist
        doc.build(story)
        pdf_data = buffer.getvalue() # Extract byte data from buffer
        buffer.close()
        return pdf_data # Return PDF binary data stream



# ==============================================================================
# SECTION 9: COMMAND LINE INTERFACE (CLI) MAIN ENTRY POINT
# ==============================================================================

def main():
    """CLI execution controller function when running backend.py directly as a standalone script."""
    import sys
    print("=" * 72)
    print(" AI-RCPSP Core Backend Engine (Adithya Baiju Module) ")
    print("=" * 72)

    # Step 1: Resolve file path from CLI argument or auto-detect sample file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        possible = list(Path(".").rglob("*.sm")) + list(Path(".").rglob("*.rcp"))
        if possible:
            file_path = str(possible[0])
            print(f"No file argument provided. Auto-selecting instance: {file_path}")
        else:
            print("Usage: python backend.py <path_to_instance.sm>")
            return

    # Step 2: Parse project file
    print(f"\n[1/3] Parsing instance file: {file_path}")
    project = Parser.parse(file_path)
    print(f"      Loaded project: {project.num_activities} activities, {project.num_resources} resources.")

    # Step 3: Extract topological and resource features
    print("\n[2/3] Extracting Project Features...")
    features = extract_features(project)
    print(f"      Paper Metrics -> N: {features.N}, NC: {features.NC:.2f}, RF: {features.RF:.2f}, RS: {features.RS:.2f}")
    print(f"      Critical Path Duration: {features.critical_path_duration}")

    # Step 4: Execute solver benchmark suite
    print("\n[3/3] Executing Metaheuristic Benchmark Solvers (500 eval budget)...")
    records = run_benchmark(project, seeds=[42], evaluation_budget=500, instance_id=Path(file_path).stem)
    report = compare_algorithms(records, instance_id=Path(file_path).stem, budget=500)

    # Step 5: Print summary report to console
    print("\n" + report.to_text())
    print("\n[SUCCESS] Completed backend engine benchmark operations.")


# Entry point guard when backend.py is executed directly from command line
if __name__ == "__main__":
    main()
