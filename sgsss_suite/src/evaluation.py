"""
================================================================================
src/evaluation.py — Central Feasibility Checker & Makespan Evaluator
================================================================================
Provides two core functions used throughout the project:

  1. Evaluator.evaluate(project, priority_vector)
       -> Calls SSGS to decode a priority vector, returns (schedule, makespan).

  2. Evaluator.check_constraints(project, schedule)
       -> Independent dual-layer audit:
         Layer 1 — Precedence: finish(p) <= start(i) for every predecessor p.
         Layer 2 — Resource:   sum demands(t) <= capacity(k) for every (k, t).

No algorithm may bypass this module — it is the single source of truth for
schedule feasibility and fitness evaluation.

References: Mitsou & Koulinas (2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Any

from src.model import Project, ScheduleEntry
from src.ssgs import SSGS


@dataclass
class ScheduleAudit:
    """
    Comprehensive audit report from the dual-layer constraint verification.

    Attributes:
        is_feasible:            True if 100% compliant with both precedence & resource constraints.
        precedence_passed:      True if zero precedence violations exist.
        precedence_violations:  Count of precedence link violations.
        resource_passed:        True if zero resource capacity over-allocations exist.
        resource_violations:    Count of (resource, time_step) capacity over-allocations.
        total_time_steps:       Total horizon analyzed (makespan).
        max_resource_utilization: Peak utilization per resource {resource_id: max_usage / capacity}.
        summary:                Human-readable academic verification summary string.
    """
    is_feasible: bool
    precedence_passed: bool
    precedence_violations: int
    resource_passed: bool
    resource_violations: int
    total_time_steps: int
    max_resource_utilization: Dict[str, float] = field(default_factory=dict)
    summary: str = ""


class Evaluator:
    """
    Central evaluator: decodes priority vectors via SSGS and independently
    validates schedule feasibility.
    """

    # ==============================================================================
    # SECTION SUMMARY: Schedule Decoding & Fitness Evaluation
    # What is in this section: Wraps the SSGS decoder to translate candidate priority
    # vectors into concrete activity schedules and calculate the overall makespan.
    # ==============================================================================

    @staticmethod
    def evaluate(
        project: Project,
        priority_vector: Union[Dict[int, float], List[float]],
    ) -> Tuple[List[ScheduleEntry], int]:
        """
        Evaluate a random-key priority vector.

        Args:
            project:         The RCPSP instance.
            priority_vector: Continuous priorities (one per activity).

        Returns:
            (schedule_entries, makespan)
        """
        schedule, makespan = SSGS.generate_schedule(project, priority_vector)
        return schedule, makespan

    # ==============================================================================
    # SECTION SUMMARY: Dual-Layer Feasibility Audit
    # What is in this section: Performs an independent constraint check verifying:
    #   1. Layer 1 (Precedence): All predecessor tasks finish before successors start.
    #   2. Layer 2 (Resources): Total concurrent resource demands never exceed capacity.
    # Returns a rich ScheduleAudit object or True/False.
    # ==============================================================================

    @staticmethod
    def audit_schedule(
        project: Project,
        schedule: List[ScheduleEntry],
    ) -> ScheduleAudit:
        """
        Perform an exhaustive, dual-layer constraint feasibility audit.

        Layer 1 (Precedence):
            For every activity i and predecessor p:
            finish_time(p) <= start_time(i)

        Layer 2 (Resource Capacity):
            For every renewable resource k and discrete time step t:
            sum demand_k(active_tasks_at_t) <= capacity_k

        Returns:
            ScheduleAudit with granular violation diagnostics and verification status.
        """
        finish_map: Dict[int, int] = {e.activity_id: e.finish_time for e in schedule}
        start_map: Dict[int, int] = {e.activity_id: e.start_time for e in schedule}

        # ── LAYER 1: Precedence Constraint Audit ────────────────────────────
        prec_violations = 0
        for act_id, act in project.activities.items():
            if act_id not in start_map:
                continue
            s_time = start_map[act_id]
            for p in act.predecessors:
                if p in finish_map and finish_map[p] > s_time:
                    prec_violations += 1

        # ── LAYER 2: Resource Capacity Limit Audit ──────────────────────────
        max_t: int = max((e.finish_time for e in schedule), default=0)
        res_violations = 0
        max_util: Dict[str, float] = {}

        for r_id, res in project.resources.items():
            usage: List[int] = [0] * (max_t + 1)
            for e in schedule:
                req = project.activities[e.activity_id].resource_requirements.get(r_id, 0)
                if req > 0:
                    for t in range(e.start_time, e.finish_time):
                        usage[t] += req
            peak = max(usage) if usage else 0
            max_util[str(r_id)] = (peak / res.capacity) if res.capacity > 0 else 0.0
            overflows = sum(1 for u in usage if u > res.capacity)
            res_violations += overflows

        prec_passed = (prec_violations == 0)
        res_passed = (res_violations == 0)
        is_feas = prec_passed and res_passed

        summary = (
            "100% FEASIBLE ✓ (0 Precedence & 0 Resource Overflows)"
            if is_feas else
            f"INFEASIBLE ✗ ({prec_violations} Precedence Violations, {res_violations} Resource Overflows)"
        )

        return ScheduleAudit(
            is_feasible=is_feas,
            precedence_passed=prec_passed,
            precedence_violations=prec_violations,
            resource_passed=res_passed,
            resource_violations=res_violations,
            total_time_steps=max_t,
            max_resource_utilization=max_util,
            summary=summary,
        )

    @staticmethod
    def check_constraints(
        project: Project,
        schedule: List[ScheduleEntry],
    ) -> bool:
        """
        Dual-Layer Constraint Validation helper.
        Returns True if the schedule is 100% feasible; False on any violation.
        """
        audit = Evaluator.audit_schedule(project, schedule)
        return audit.is_feasible

