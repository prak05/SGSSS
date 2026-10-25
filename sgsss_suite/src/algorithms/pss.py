"""Proposed 2 — Particle Swarm Search (PSS), seeded by the trained neural priority model.

Each particle is a priority vector in [0, 1]^n with a velocity:
    v <- w*v + c1*r1*(personal_best - x) + c2*r2*(swarm_best - x),   x <- x + v
  1. Particle 0 is the network's prediction, the next few are classic rules, the rest
     are noisy copies of the prediction (instead of blind random starts).
  2. Inertia w falls linearly from 0.9 to 0.4: explore first, then converge.
  3. Every new swarm best is re-packed by forward-backward justification and the
     swarm then follows the improved schedule. Equal makespans are compared on total
     finish time, so the swarm keeps improving schedule quality.
  4. Lamarckian learning: half of all particle schedules are re-packed by justification
     and the particle jumps to the improved order (setting chosen by tuning on training data).
  5. If the swarm best stalls, the worst half of the swarm is re-scattered around it.
"""
from src.algorithms.common import run
from src.algorithms.gnn import predict
from src.algorithms.rules import normalise, rules
from src.evaluation import total_finish
from src.ssgs import to_priority


def solve_pss(project, budget=500, seed=42, swarm=20, c1=1.5, c2=1.5, vmax=0.2, patience=8, repack=0.5):
    """repack: chance that a particle's schedule is re-packed (justified) and the particle
    moved to the improved order (Lamarckian learning)."""
    p = project
    clip = lambda v, lo, hi: min(hi, max(lo, v))

    def body(s, rng):
        def fitness(x):
            ms = s.evaluate(x)
            return ms, total_finish(p, s.last_starts)

        prior = normalise(predict(p))
        fitness(prior)
        s.justify(s.best_starts)                       # evaluations 1-3: the GNN start, re-packed
        xs = [to_priority(p, s.best_starts)] + [normalise(v) for v in list(rules(p).values())[:4]]
        xs += [[clip(g + rng.gauss(0, 0.1), 0, 1) for g in prior] for _ in range(swarm - len(xs))]
        vs = [[0.0] * p.n for _ in xs]
        pbest = [list(x) for x in xs]
        pfit = [fitness(x) for x in xs]
        gbest, stale = to_priority(p, s.best_starts), 0
        gfit = (s.best_ms, total_finish(p, s.best_starts))
        while True:
            w = 0.9 - 0.5 * min(1.0, s.used / budget)
            for i, (x, v) in enumerate(zip(xs, vs)):
                for g in range(p.n):
                    v[g] = clip(w * v[g] + c1 * rng.random() * (pbest[i][g] - x[g])
                                + c2 * rng.random() * (gbest[g] - x[g]), -vmax, vmax)
                    x[g] = clip(x[g] + v[g], 0.0, 1.0)
                f = fitness(x)
                if repack and rng.random() < repack:
                    s.justify(s.last_starts)
                    f = (s.trace[-1], total_finish(p, s.last_starts))
                    x[:] = to_priority(p, s.last_starts)
                if f < pfit[i]:
                    pbest[i], pfit[i] = list(x), f
                if f < gfit:
                    s.justify(s.last_starts)
                    gbest, stale = to_priority(p, s.best_starts), -1
                    gfit = (s.best_ms, total_finish(p, s.best_starts))
                    pbest[i], pfit[i] = list(gbest), gfit
            stale += 1
            if stale > patience:                       # re-scatter the worst half
                for i in sorted(range(swarm), key=pfit.__getitem__)[swarm // 2:]:
                    xs[i] = [clip(g + rng.gauss(0, 0.15), 0, 1) for g in gbest]
                    vs[i] = [0.0] * p.n
                stale = 0

    return run("PSS", body, project, budget, seed, lexicographic=True)
