"""Shared bookkeeping for all solvers: evaluation budget, best-so-far and the iteration trace.

Every solver gets the same budget of SSGS decodes. One decode = one evaluation =
one row in the iteration table shown in the UI. Justification (two decodes) is
charged as two evaluations, so no solver gets free work.
"""
from __future__ import annotations

import random
import time
from typing import Callable, List, Sequence

from src import ssgs
from src.evaluation import levelling, quality_key, total_finish
from src.model import Project, Result


class BudgetSpent(Exception):
    pass


class Search:
    """lexicographic=True (proposed solvers):
      * among schedules with the best makespan, keep the one with the lowest total finish
        time, then the smoothest resource use;
      * stop early once the makespan equals the lower bound (provably optimal) and quality
        has not improved for 10% of the budget.
    The baselines (textbook versions) keep the first schedule that reaches their best
    makespan and always use the whole budget."""

    def __init__(self, project: Project, budget: int, name: str, lexicographic: bool = False):
        self.p, self.budget, self.name, self.lexicographic = project, budget, name, lexicographic
        self.best_ms, self.best_starts, self.best_eval, self.best_key = None, None, 0, None
        self.last_starts: List[int] = []
        self.critical: List[int] = []      # zero-slack activities seen in the last justification
        self.trace: List[int] = []
        self.best_trace: List[int] = []
        self.stopped_optimal = False

    @property
    def used(self) -> int:
        return len(self.trace)

    def _record(self, starts: List[int], ms: int) -> int:
        if self.used >= self.budget:
            raise BudgetSpent
        self.last_starts = starts
        self.trace.append(ms)
        if self.best_ms is None or ms < self.best_ms:
            self.best_ms, self.best_starts, self.best_eval, self.best_key = ms, starts, self.used, None
        elif self.lexicographic and ms == self.best_ms:
            self.best_key = self.best_key or quality_key(self.p, self.best_starts, ms)
            key = quality_key(self.p, starts, ms)
            if key < self.best_key:
                self.best_starts, self.best_eval, self.best_key = starts, self.used, key
        self.best_trace.append(self.best_ms)
        if (self.lexicographic and self.best_ms == self.p.lower_bound
                and self.used - self.best_eval >= max(20, self.budget // 10)):
            self.stopped_optimal = True
            raise BudgetSpent
        return ms

    def evaluate(self, priority: Sequence[float]) -> int:
        """Decode a priority vector, record it, return its makespan."""
        if self.used >= self.budget:
            raise BudgetSpent
        return self._record(*ssgs.decode(self.p, priority))

    def justify(self, starts: Sequence[int]) -> int:
        """Forward-backward improvement, charged as two evaluations (both schedules are
        feasible and recorded). Returns the makespan; the schedule is in last_starts."""
        right, ms = ssgs.backward(self.p, starts)
        self._record(right, ms)
        ms = self._record(*ssgs.decode(self.p, [-s for s in right]))
        self.critical = [i for i in range(self.p.n) if right[i] == self.last_starts[i]]   # no slack
        return ms


def run(name: str, body: Callable[[Search, random.Random], None],
        project: Project, budget: int, seed: int, lexicographic: bool = False) -> Result:
    """Run a solver body until it returns or the budget is spent, then package the Result."""
    search = Search(project, budget, name, lexicographic)
    t0 = time.perf_counter()
    try:
        body(search, random.Random(seed))
    except BudgetSpent:
        pass
    runtime = time.perf_counter() - t0
    return Result(name, search.best_ms, search.best_starts, search.used, search.best_eval, runtime,
                  search.trace, search.best_trace, total_finish(project, search.best_starts),
                  levelling(project, search.best_starts), stopped_optimal=search.stopped_optimal)


def random_keys(rng: random.Random, n: int) -> List[float]:
    return [rng.random() for _ in range(n)]
