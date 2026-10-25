#!/usr/bin/env python3
"""Command-line run: python backend.py <instance.sm|.rcp|.json> [budget] [seed]

Budget and seed are set automatically from the project content unless given.

Parses the project, runs all six solvers with the same budget, audits every schedule,
prints the iteration summary and explains how the winner won.
"""
import sys

from src.algorithms import FULL_NAMES, PROPOSED
from src.benchmark import auto_budget, auto_seed, explain, resource_pressure, run_all
from src.parser import parse


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    p = parse(sys.argv[1])
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else auto_budget(p)
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else auto_seed(p)
    print(f"Project {p.name}: {p.real_tasks} activities, {len(p.capacities)} resources, "
          f"critical path {p.cpm_bound}, lower bound {p.lower_bound}")
    print(f"Every solver gets {budget} schedule evaluations (seed {seed}).")
    if len(sys.argv) <= 2:
        print(f"  auto budget = 10 x {p.real_tasks} activities x (1 + resource pressure {resource_pressure(p):.2f})"
              f", rounded to 50;  auto seed = fingerprint of the project content\n")

    def progress(name, r):
        tag = "proposed" if name in PROPOSED else "baseline"
        print(f"  {name:4s} {FULL_NAMES[name]:26s} ({tag})  makespan {r.makespan:4d}  "
              f"best found at evaluation {r.best_eval:5d}  {r.runtime:6.2f}s")

    results = run_all(p, budget, seed, on_done=progress)
    e = explain(p, results)
    checkpoints = sorted({1, 10, 50, 100, budget // 4, budget // 2, budget})
    print("\nBest makespan so far at selected evaluations:")
    print("  eval " + "".join(f"{a:>7s}" for a in results))
    for t in checkpoints:
        print(f"  {t:4d} " + "".join(f"{r.best_trace[min(t, len(r.best_trace)) - 1]:7d}" for r in results.values()))
    print("\nRanking (makespan → total finish time → levelling → reliability over 5 seeds → runtime):")
    for row in e["table"]:
        print(f"  {row['rank']}. {row['algorithm']:4s} makespan {row['makespan']:4d}  gap {row['gap_to_lower_bound_%']:6.2f}%  "
              f"total finish {row['total_finish']:6d}  levelling {row['levelling']:9.2f}  "
              f"evals (5-seed mean) {row['mean_evals_5_seeds']:7.1f}  feasible {row['feasible']}"
              + ("  [stopped early: proven optimal]" if row["stopped_early_optimal"] else ""))
    print(f"\nWinner: {e['winner']} ({FULL_NAMES[e['winner']]}). {e['how']}")
    if e["lead_from_eval"]:
        print(f"It was strictly ahead of every other algorithm from evaluation {e['lead_from_eval']} onwards.")
    if e["optimal_proven"]:
        print("Its makespan equals the lower bound, so it is provably optimal.")


if __name__ == "__main__":
    main()
