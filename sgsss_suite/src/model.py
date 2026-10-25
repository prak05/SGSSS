"""Core data structures: a Project (the RCPSP instance) and a Result (one solver run).

Activities are stored by index 0..n-1. Index 0 is the dummy start and n-1 the
dummy end in PSPLIB files, but nothing here depends on that.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional


@dataclass(eq=False)   # compared by identity, so it can be cached and hashed
class Project:
    name: str
    durations: List[int]
    demands: List[List[int]]          # demands[i][k] = units of resource k used by activity i
    capacities: List[int]             # capacities[k] = units of resource k available per period
    preds: List[List[int]]            # preds[i] = indices that must finish before i starts
    task_names: Optional[List[str]] = None
    resource_names: Optional[List[str]] = None
    succs: List[List[int]] = field(init=False)

    def __post_init__(self):
        n = len(self.durations)
        self.task_names = self.task_names or [f"A{i + 1}" for i in range(n)]
        self.resource_names = self.resource_names or [f"R{k + 1}" for k in range(len(self.capacities))]
        self.succs = [[] for _ in range(n)]
        for i, ps in enumerate(self.preds):
            for p in ps:
                self.succs[p].append(i)
        for i in range(n):
            for k, r in enumerate(self.demands[i]):
                if r > self.capacities[k]:
                    raise ValueError(f"{self.task_names[i]} needs {r} of {self.resource_names[k]} "
                                     f"but only {self.capacities[k]} exist")
        _ = self.topo_order  # raises on cycles

    @property
    def n(self) -> int:
        return len(self.durations)

    @property
    def real_tasks(self) -> int:
        return sum(1 for d in self.durations if d > 0)

    @cached_property
    def topo_order(self) -> List[int]:
        """Kahn's algorithm. Raises ValueError if the precedence graph has a cycle."""
        indeg = [len(p) for p in self.preds]
        queue = deque(i for i in range(self.n) if indeg[i] == 0)
        order = []
        while queue:
            i = queue.popleft()
            order.append(i)
            for s in self.succs[i]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    queue.append(s)
        if len(order) != self.n:
            raise ValueError("Precedence graph contains a cycle")
        return order

    @cached_property
    def est(self) -> List[int]:
        """Earliest start times ignoring resources (CPM forward pass)."""
        est = [0] * self.n
        for i in self.topo_order:
            est[i] = max((est[p] + self.durations[p] for p in self.preds[i]), default=0)
        return est

    @cached_property
    def cpm_bound(self) -> int:
        """Critical-path length: no schedule can be shorter, even with unlimited resources."""
        return max(e + d for e, d in zip(self.est, self.durations))

    @cached_property
    def lft(self) -> List[int]:
        """Latest finish times that still meet the CPM bound (CPM backward pass)."""
        lft = [self.cpm_bound] * self.n
        for i in reversed(self.topo_order):
            lft[i] = min((lft[s] - self.durations[s] for s in self.succs[i]), default=self.cpm_bound)
        return lft

    @cached_property
    def lst(self) -> List[int]:
        return [f - d for f, d in zip(self.lft, self.durations)]

    @cached_property
    def lower_bound(self) -> int:
        """max(critical path, resource bound): total work on resource k / capacity of k."""
        work = [math.ceil(sum(d * dem[k] for d, dem in zip(self.durations, self.demands)) / c)
                for k, c in enumerate(self.capacities) if c > 0]
        return max([self.cpm_bound] + work)

    @cached_property
    def reversed(self) -> "Project":
        """Same project with every arc flipped (used by backward justification)."""
        return Project(self.name + " (reversed)", self.durations, self.demands, self.capacities,
                       [list(s) for s in self.succs], self.task_names, self.resource_names)


@dataclass
class Result:
    algorithm: str
    makespan: int
    starts: List[int]
    evaluations: int          # schedule decodings used
    best_eval: int            # evaluation at which the final makespan was first reached
    runtime: float            # seconds
    trace: List[int]          # makespan of the schedule decoded at each evaluation
    best_trace: List[int]     # best makespan found so far, after each evaluation
    total_finish: int = 0     # tie-break 1: sum of finish times (lower = work done earlier)
    levelling: float = 0.0    # tie-break 2: sum of squared relative resource usage (lower = smoother)
    mean_evals: float = 0.0   # tie-break 3: mean evaluations to reach the makespan over 5 seeds
    stopped_optimal: bool = False   # stopped early: makespan equals the lower bound (proven optimal)
