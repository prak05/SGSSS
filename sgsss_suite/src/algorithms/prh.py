"""Baseline 4 — Priority Rule Heuristics with multi-pass biased sampling (Kolisch 1996).

First every classic rule (LFT, LST, MSLK, MTS, GRPW, ...) is decoded once. The rest
of the budget samples noisy versions of the rules, so the search stays close to
good rule orders while still trying new ones.
"""
from src.algorithms.common import run
from src.algorithms.rules import normalise, rules


def solve_prh(project, budget=500, seed=42, noise=0.15):
    def body(s, rng):
        rule_vectors = [normalise(v) for v in rules(project).values()]
        for v in rule_vectors:
            s.evaluate(v)
        while True:
            for v in rule_vectors:
                s.evaluate([x + rng.gauss(0, noise) for x in v])

    return run("PRH", body, project, budget, seed)
