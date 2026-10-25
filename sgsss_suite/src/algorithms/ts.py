"""Proposed 1 — Tabu Search (TS), started from the trained neural priority model.

Solution = an activity list (precedence-feasible order); SSGS schedules it in that order.
  1. Start from the network's priority order, improved by forward-backward justification.
  2. Each iteration samples shift moves: take one activity and re-insert it anywhere
     between its last predecessor and first successor. Every move is one evaluation.
  3. Move to the best sampled neighbour, even if it is worse. A moved activity is
     tabu for `tenure` iterations, unless moving it beats the best ever (aspiration).
  4. Each new best schedule is re-packed with justification. Neighbours with equal makespan
     are compared on total finish time, so the search keeps improving schedule quality
     after it reaches its best makespan.
  5. After `patience` iterations without improvement, restart from the best list
     with a few random shifts (diversification).
"""
from src.algorithms.common import run
from src.algorithms.gnn import predict
from src.evaluation import total_finish
from src.ssgs import to_priority


def _order(p, starts):
    prio = to_priority(p, starts)
    return sorted(range(p.n), key=lambda i: -prio[i])


def _shift(p, order, a, rng):
    """Re-insert activity a at a random precedence-feasible position (or None)."""
    rest = [x for x in order if x != a]
    pos = {x: i for i, x in enumerate(rest)}
    lo = max((pos[q] + 1 for q in p.preds[a]), default=0)
    hi = min((pos[q] for q in p.succs[a]), default=len(rest))
    b = rng.randint(lo, hi)
    return None if b == order.index(a) else rest[:b] + [a] + rest[b:]


def _priority(order):
    prio = [0.0] * len(order)
    for rank, a in enumerate(order):
        prio[a] = -rank
    return prio


def solve_ts(project, budget=500, seed=42, neighbours=None, tenure=7, patience=25, start=None,
             focus=0.0, justify_moves=False):
    """start: priority vector to begin from (default: the trained GNN's prediction).
    focus: chance of moving a zero-slack (critical) activity instead of a random one.
    justify_moves: re-pack every accepted move, not only new bests."""
    p = project
    movable = [i for i in range(p.n) if p.durations[i] > 0] or list(range(p.n))
    k = neighbours or max(6, min(20, p.n // 3))

    def body(s, rng):
        s.evaluate(predict(p) if start is None else start)
        s.justify(s.best_starts)
        current = _order(p, s.best_starts)
        tabu, stale, it, empty = {}, 0, 0, 0
        while empty < 50:                              # 50 iterations with no legal move: order is forced
            it += 1
            best_before, found_before = s.best_ms, s.best_eval
            moves = []
            critical = [i for i in s.critical if p.durations[i] > 0]
            for _ in range(k):
                a = rng.choice(critical) if critical and rng.random() < focus else rng.choice(movable)
                new = _shift(p, current, a, rng)
                if new:
                    ms = s.evaluate(_priority(new))
                    moves.append(((ms, total_finish(p, s.last_starts)), a, new, s.last_starts))
            allowed = [m for m in moves if tabu.get(m[1], 0) < it or m[0][0] < best_before] or moves
            empty = 0 if allowed else empty + 1
            if not allowed:
                continue
            _, a, current, starts = min(allowed, key=lambda m: m[0])
            tabu[a] = it + tenure
            if justify_moves and s.best_eval == found_before:
                s.justify(starts)
                current = _order(p, s.last_starts)
            if s.best_eval > found_before:             # new best (makespan or quality)
                s.justify(s.best_starts)
                current, stale = _order(p, s.best_starts), 0
            else:
                stale += 1
            if stale > patience:                       # diversify around the best list
                current = _order(p, s.best_starts)
                for _ in range(max(2, p.n // 10)):
                    current = _shift(p, current, rng.choice(movable), rng) or current
                tabu, stale = {}, 0

    return run("TS", body, project, budget, seed, lexicographic=True)
