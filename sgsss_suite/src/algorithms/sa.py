"""Baseline 2 — Simulated Annealing.

One current solution; a neighbour swaps the priorities of two random activities.
Worse neighbours are accepted with probability exp(-delta / T), and the temperature
cools geometrically so that it falls to 1% of its start value by the end of the budget.
"""
import math

from src.algorithms.common import random_keys, run


def solve_sa(project, budget=500, seed=42, start_temp=0.05):
    def body(s, rng):
        x = random_keys(rng, project.n)
        cur = s.evaluate(x)
        temp = start_temp * cur
        cooling = 0.01 ** (1 / max(1, budget))
        while True:
            y = list(x)
            a, b = rng.randrange(project.n), rng.randrange(project.n)
            y[a], y[b] = y[b], y[a]
            new = s.evaluate(y)
            if new <= cur or rng.random() < math.exp(-(new - cur) / max(temp, 1e-9)):
                x, cur = y, new
            temp *= cooling

    return run("SA", body, project, budget, seed)
