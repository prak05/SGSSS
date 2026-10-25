"""Classic priority rules and the per-activity features the neural priority model reads.

Every rule returns one number per activity; higher means "schedule earlier".
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np

from src.model import Project


def _all_successor_counts(p: Project) -> List[int]:
    reach = [0] * p.n                       # bitset of transitive successors
    for i in reversed(p.topo_order):
        for s in p.succs[i]:
            reach[i] |= reach[s] | (1 << s)
    return [bin(r).count("1") for r in reach]


@lru_cache(maxsize=1024)
def rules(p: Project) -> Dict[str, List[float]]:
    d, est, lst, lft = p.durations, p.est, p.lst, p.lft
    mts = _all_successor_counts(p)
    load = [sum(r / c for r, c in zip(p.demands[i], p.capacities) if c) for i in range(p.n)]
    return {
        "LFT (latest finish time)": [-x for x in lft],
        "LST (latest start time)": [-x for x in lst],
        "MSLK (minimum slack)": [-(l - e) for l, e in zip(lst, est)],
        "MTS (most total successors)": [float(x) for x in mts],
        "GRPW (rank positional weight)": [d[i] + sum(d[s] for s in p.succs[i]) for i in range(p.n)],
        "WRUP (resource-weighted successors)": [0.7 * len(p.succs[i]) + 0.3 * load[i] for i in range(p.n)],
        "SPT (shortest processing time)": [-x for x in d],
        "EST (earliest start time)": [-x for x in est],
    }


FEATURES = ["earliest start", "latest start", "latest finish", "slack", "duration",
            "direct successors", "all successors", "rank positional weight",
            "resource load", "predecessors"]


@lru_cache(maxsize=1024)
def features(p: Project) -> np.ndarray:
    """n x 10 matrix, each column scaled to roughly [0, 1] so instances of any size compare."""
    cp = max(1, p.cpm_bound)
    r = rules(p)
    grpw = np.array(r["GRPW (rank positional weight)"], float)
    cols = [
        np.array(p.est) / cp, np.array(p.lst) / cp, np.array(p.lft) / cp,
        (np.array(p.lst) - np.array(p.est)) / cp,
        np.array(p.durations) / max(1, max(p.durations)),
        np.array([len(s) for s in p.succs]) / max(1, max(len(s) for s in p.succs)),
        np.array(r["MTS (most total successors)"]) / max(1, p.n),
        grpw / max(1.0, grpw.max()),
        np.array([sum(x / c for x, c in zip(p.demands[i], p.capacities) if c) for i in range(p.n)])
        / max(1, len(p.capacities)),
        np.array([len(q) for q in p.preds]) / max(1, max(len(q) for q in p.preds)),
    ]
    return np.stack(cols, axis=1)


def normalise(values: List[float]) -> List[float]:
    """Rank-scale any priority list into [0, 1] (keeps the order, removes the units)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = rank / max(1, len(values) - 1)
    return out
