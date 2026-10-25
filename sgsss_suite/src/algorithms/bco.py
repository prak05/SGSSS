"""Baseline 3 — Bee Colony Optimisation (Artificial Bee Colony, Karaboga 2005).

Food sources are priority vectors. Employed bees tweak one gene of their source toward
or away from a random other source; onlooker bees do the same on sources picked in
proportion to quality; a source that fails to improve `limit` times is abandoned and
a scout bee replaces it with a random one.
"""
from src.algorithms.common import random_keys, run


def solve_bco(project, budget=500, seed=42, sources=15, limit=None):
    limit = limit or project.n

    def body(s, rng):
        n = project.n
        food = [random_keys(rng, n) for _ in range(sources)]
        fit = [s.evaluate(x) for x in food]
        trials = [0] * sources

        def try_neighbour(i):
            k = rng.choice([j for j in range(sources) if j != i])
            g = rng.randrange(n)
            y = list(food[i])
            y[g] = min(1.0, max(0.0, y[g] + rng.uniform(-1, 1) * (y[g] - food[k][g])))
            new = s.evaluate(y)
            if new <= fit[i]:
                food[i], fit[i], trials[i] = y, new, 0
            else:
                trials[i] += 1

        while True:
            for i in range(sources):                                  # employed bees
                try_neighbour(i)
            weights = [1.0 / f for f in fit]
            for _ in range(sources):                                  # onlooker bees
                try_neighbour(rng.choices(range(sources), weights)[0])
            worst = max(range(sources), key=trials.__getitem__)       # scout bee
            if trials[worst] > limit:
                food[worst], trials[worst] = random_keys(rng, n), 0
                fit[worst] = s.evaluate(food[worst])

    return run("BCO", body, project, budget, seed)
