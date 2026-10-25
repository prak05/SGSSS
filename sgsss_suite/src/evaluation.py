"""Independent feasibility check of a finished schedule (does not trust the decoder)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from src.model import Project


@dataclass
class Audit:
    precedence_violations: List[str]
    resource_violations: List[str]
    makespan: int

    @property
    def feasible(self) -> bool:
        return not self.precedence_violations and not self.resource_violations


def audit(p: Project, starts: Sequence[int]) -> Audit:
    finish = [s + d for s, d in zip(starts, p.durations)]
    prec = [f"{p.task_names[i]} starts at {starts[i]} before {p.task_names[q]} finishes at {finish[q]}"
            for i in range(p.n) for q in p.preds[i] if starts[i] < finish[q]]
    makespan = max(finish)
    res = []
    for k, cap in enumerate(p.capacities):
        usage = [0] * (makespan + 1)
        for i in range(p.n):
            for t in range(starts[i], finish[i]):
                usage[t] += p.demands[i][k]
        res += [f"{p.resource_names[k]} uses {u}/{cap} at t={t}" for t, u in enumerate(usage) if u > cap]
    return Audit(prec, res, makespan)


def resource_profile(p: Project, starts: Sequence[int]) -> List[List[int]]:
    """usage[k][t] for plotting."""
    makespan = max(s + d for s, d in zip(starts, p.durations))
    usage = [[0] * makespan for _ in p.capacities]
    for i in range(p.n):
        for t in range(starts[i], starts[i] + p.durations[i]):
            for k, r in enumerate(p.demands[i]):
                usage[k][t] += r
    return usage


def total_finish(p: Project, starts: Sequence[int]) -> int:
    """Sum of activity finish times: lower = work is completed earlier (less work in progress)."""
    return sum(s + d for s, d in zip(starts, p.durations))


def levelling(p: Project, starts: Sequence[int]) -> float:
    """Sum over resources and periods of (usage / capacity)^2: lower = smoother workload."""
    return round(sum((u / c) ** 2 for row, c in zip(resource_profile(p, starts), p.capacities)
                     if c for u in row), 3)


def quality_key(p: Project, starts: Sequence[int], makespan: int):
    """Schedule-quality ranking used everywhere: makespan, then total finish time, then levelling."""
    return makespan, total_finish(p, starts), levelling(p, starts)
