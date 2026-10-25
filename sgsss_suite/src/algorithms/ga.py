"""Baseline 1 — Genetic Algorithm (Hartmann 1998 style, random-key encoding).

Population of priority vectors; tournament selection, uniform crossover,
per-gene random-reset mutation, and the best individual always survives (elitism).
"""
from src.algorithms.common import random_keys, run


def solve_ga(project, budget=500, seed=42, pop_size=30, crossover=0.9, mutation=0.05):
    def body(s, rng):
        n = project.n
        pop = [random_keys(rng, n) for _ in range(pop_size)]
        fit = [s.evaluate(x) for x in pop]

        def pick():                                   # binary tournament
            a, b = rng.randrange(pop_size), rng.randrange(pop_size)
            return pop[a] if fit[a] <= fit[b] else pop[b]

        while True:
            elite = min(range(pop_size), key=fit.__getitem__)
            children, child_fit = [pop[elite]], [fit[elite]]
            while len(children) < pop_size:
                mum, dad = pick(), pick()
                child = ([m if rng.random() < 0.5 else d for m, d in zip(mum, dad)]
                         if rng.random() < crossover else list(mum))
                child = [rng.random() if rng.random() < mutation else g for g in child]
                children.append(child)
                child_fit.append(s.evaluate(child))
            pop, fit = children, child_fit

    return run("GA", body, project, budget, seed)
