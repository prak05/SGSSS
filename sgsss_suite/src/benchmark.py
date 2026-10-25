"""Run solvers under identical conditions, pick the winner, and explain the win from the data.

The winner is decided only by measured schedule quality, in this order:
  1. makespan                (shorter project)
  2. total finish time       (sum of activity finish times: work completed earlier)
  3. resource levelling      (sum of squared relative usage: smoother workload)
  4. reliability             (mean evaluations needed to reach that makespan over 5 seeds;
                              only computed for algorithms still tied after 1-3)
  5. runtime                 (seconds)
Algorithm names play no part in the ranking.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

from src.algorithms import FULL_NAMES, SOLVERS
from src.evaluation import audit
from src.model import Project, Result


def resource_pressure(p: Project) -> float:
    """How strongly resources (not precedence) limit the project: max over resources of
    (total work on it / its capacity) divided by the critical-path length. Above 1 = resources dominate."""
    work = max(sum(d * dem[k] for d, dem in zip(p.durations, p.demands)) / c
               for k, c in enumerate(p.capacities) if c)
    return work / max(1, p.cpm_bound)


def auto_budget(p: Project) -> int:
    """Evaluations per algorithm, set from the project itself (never from the file type):
    10 x activities x (1 + resource pressure, capped at 2), rounded to 50, at least 200.
    A typical 30-activity benchmark gets ~500 (the PSPLIB standard); bigger or tighter projects get more."""
    raw = 10 * p.real_tasks * (1 + min(resource_pressure(p), 2.0))
    return max(200, int(round(raw / 50)) * 50)


def auto_seed(p: Project) -> int:
    """Random seed = fingerprint of the project content (durations, demands, capacities, precedence).
    Names and file format are ignored, so the same project in .sm, .rcp or .json gets the same seed."""
    content = json.dumps([p.durations, p.demands, p.capacities, [sorted(q) for q in p.preds]])
    return int(hashlib.sha256(content.encode()).hexdigest()[:8], 16) % 100000


RELIABILITY_SEEDS = 5


def run_all(p: Project, budget: Optional[int] = None, seed: Optional[int] = None,
            names=None, on_done=None) -> Dict[str, Result]:
    """budget / seed default to auto_budget(p) / auto_seed(p)."""
    budget = budget or auto_budget(p)
    seed = auto_seed(p) if seed is None else seed
    results = {}
    for name in names or SOLVERS:
        results[name] = SOLVERS[name](p, budget=budget, seed=seed)
        if on_done:
            on_done(name, results[name])
    add_reliability(p, results, budget, seed)
    return results


def add_reliability(p: Project, results: Dict[str, Result], budget: int, seed: int) -> None:
    """For algorithms tied with the best on schedule quality, re-run on extra seeds and record the
    mean number of evaluations needed to reach that makespan (budget + 1 when a run misses it)."""
    for r in results.values():
        r.mean_evals = float(r.best_eval)
    top = min(quality(r) for r in results.values())
    tied = [a for a, r in results.items() if quality(r) == top]
    if len(tied) < 2:
        return
    for a in tied:
        needed = [results[a].best_eval]
        for extra in range(1, RELIABILITY_SEEDS):
            trace = SOLVERS[a](p, budget=budget, seed=seed + extra).best_trace
            needed.append(next((t + 1 for t, b in enumerate(trace) if b <= top[0]), budget + 1))
        results[a].mean_evals = sum(needed) / len(needed)


def quality(r: Result):
    return r.makespan, r.total_finish, r.levelling


CRITERIA = [("makespan", "makespan"), ("total_finish", "total finish time"),
            ("levelling", "resource levelling"), ("mean_evals", "reliability (mean evaluations over 5 seeds)"),
            ("runtime", "runtime")]


def rank_key(r: Result):
    return tuple(getattr(r, field) for field, _ in CRITERIA)


def ranking(results: Dict[str, Result]) -> List[str]:
    return sorted(results, key=lambda a: rank_key(results[a]))


def deciding_criterion(a: Result, b: Result) -> str:
    """Name of the first criterion on which a beats b."""
    return next((label for field, label in CRITERIA if getattr(a, field) != getattr(b, field)), "nothing (identical)")


def lead_taken(results: Dict[str, Result], winner: str) -> Optional[int]:
    """First evaluation from which the winner's best-so-far stayed strictly ahead of every rival."""
    w = results[winner].best_trace
    rivals = [r.best_trace for a, r in results.items() if a != winner]
    steps = min(len(w), *(len(r) for r in rivals)) if rivals else len(w)
    lead = None
    for t in range(steps):
        ahead = all(w[t] < r[t] for r in rivals)
        lead = (t + 1 if lead is None else lead) if ahead else None
    return lead


def explain(p: Project, results: Dict[str, Result]) -> dict:
    """Facts behind the outcome, in the order a reader needs them."""
    order = ranking(results)
    win = results[order[0]]
    runner = results[order[1]] if len(order) > 1 else None
    tied = [a for a in order[1:] if results[a].makespan == win.makespan]
    lb = p.lower_bound
    rows = [{
        "rank": i + 1, "algorithm": a, "name": FULL_NAMES[a], "makespan": r.makespan,
        "gap_to_lower_bound_%": round(100 * (r.makespan - lb) / lb, 2) if lb else 0.0,
        "behind_winner": r.makespan - win.makespan, "total_finish": r.total_finish,
        "levelling": r.levelling, "best_found_at_eval": r.best_eval, "mean_evals_5_seeds": r.mean_evals,
        "evaluations": r.evaluations, "stopped_early_optimal": r.stopped_optimal, "runtime_s": round(r.runtime, 3),
        "feasible": audit(p, r.starts).feasible,
    } for i, (a, r) in enumerate((a, results[a]) for a in order)]

    if runner is None:
        how = f"{order[0]} was the only algorithm run."
    elif not tied:
        how = (f"{order[0]} found the shortest schedule ({win.makespan} periods), "
               f"{runner.makespan - win.makespan} shorter than the runner-up {order[1]}.")
    else:
        crit = deciding_criterion(win, runner)
        detail = {"total finish time": f"total finish time {win.total_finish} vs {runner.total_finish}",
                  "resource levelling": f"levelling {win.levelling} vs {runner.levelling}",
                  "reliability (mean evaluations over 5 seeds)":
                      f"needed {win.mean_evals:.1f} evaluations on average over {RELIABILITY_SEEDS} seeds "
                      f"vs {runner.mean_evals:.1f}",
                  "runtime": f"{win.runtime:.3f}s vs {runner.runtime:.3f}s"}.get(crit, crit)
        how = (f"{order[0]} tied on makespan {win.makespan} with {', '.join(tied)}; it won on "
               f"{crit} ({detail} for {order[1]}).")
    lead = lead_taken(results, order[0])
    return {
        "winner": order[0], "makespan": win.makespan, "lower_bound": lb,
        "optimal_proven": win.makespan == lb, "strict_win": not tied, "tied_with": tied,
        "decided_by": "makespan" if not tied else deciding_criterion(win, runner),
        "how": how, "lead_from_eval": lead, "table": rows,
    }
