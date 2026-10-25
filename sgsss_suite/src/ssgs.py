"""Serial Schedule Generation Scheme (SSGS) — the one decoder every solver uses.

Every solver searches over priority vectors (one number per activity). SSGS turns a
priority vector into a schedule:
  1. Among activities whose predecessors are all scheduled, take the highest priority.
  2. Start it at the earliest time >= all predecessor finishes where every resource
     has enough free capacity for its whole duration.
  3. Repeat until everything is scheduled.
So every decoded schedule respects precedence and capacity by construction.
"""
from __future__ import annotations

import heapq
from typing import List, Sequence, Tuple

from src.model import Project


def decode(p: Project, priority: Sequence[float]) -> Tuple[List[int], int]:
    """Return (start times, makespan) for a priority vector (higher = scheduled first)."""
    free = [[c] * (sum(p.durations) + 1) for c in p.capacities]   # free[k][t]
    finish = [0] * p.n
    starts = [0] * p.n
    waiting = [len(ps) for ps in p.preds]
    ready = [(-priority[i], i) for i in range(p.n) if waiting[i] == 0]
    heapq.heapify(ready)
    while ready:
        _, i = heapq.heappop(ready)
        d = p.durations[i]
        need = [(k, r) for k, r in enumerate(p.demands[i]) if r]
        t = max((finish[q] for q in p.preds[i]), default=0)
        while d and need:
            clash = next((u for k, r in need for u in range(t, t + d) if free[k][u] < r), None)
            if clash is None:
                break
            t = clash + 1               # every start <= clash would overlap the clash period
        for k, r in need:
            for u in range(t, t + d):
                free[k][u] -= r
        starts[i], finish[i] = t, t + d
        for s in p.succs[i]:
            waiting[s] -= 1
            if waiting[s] == 0:
                heapq.heappush(ready, (-priority[s], s))
    return starts, max(finish)


def backward(p: Project, starts: Sequence[int]) -> Tuple[List[int], int]:
    """Right-justify a schedule: decode the reversed project, latest-finishing activity first.

    Mirroring that reversed schedule gives a feasible schedule of the original project
    in which every activity is pushed as late as it can go.
    """
    finish = [s + d for s, d in zip(starts, p.durations)]
    back_starts, ms = decode(p.reversed, finish)
    return [ms - (b + d) for b, d in zip(back_starts, p.durations)], ms


def justify(p: Project, starts: Sequence[int]) -> Tuple[List[int], int]:
    """Forward-backward improvement (Valls et al. 2005): right-justify, then left-justify.

    Re-packing activities in the order of their right-shifted start times closes idle
    gaps left by the first schedule. Costs two decodes.
    """
    right, _ = backward(p, starts)
    return decode(p, [-s for s in right])


def to_priority(p: Project, starts: Sequence[int]) -> List[float]:
    """Priority vector in [0, 1] that reproduces a schedule's activity order."""
    topo_pos = {i: r for r, i in enumerate(p.topo_order)}
    order = sorted(range(p.n), key=lambda i: (starts[i], topo_pos[i]))
    rank = {i: r for r, i in enumerate(order)}
    return [1.0 - rank[i] / max(1, p.n - 1) for i in range(p.n)]
