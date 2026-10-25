#!/usr/bin/env python3
"""Compare TS / PSS settings on a fixed sample of TRAINING instances (never the test set).

For each setting: mean makespan, and on how many instances it matches or beats the best of the four
baselines. Used to choose the default settings in ts.py and pss.py.
Usage: python scripts/tune.py [--sample 60] [--workers 8]
"""
import argparse
import random
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.algorithms import BASELINES, SOLVERS  # noqa: E402
from src.algorithms.pss import solve_pss  # noqa: E402
from src.algorithms.ts import solve_ts  # noqa: E402
from src.benchmark import auto_budget, auto_seed  # noqa: E402
from src.parser import parse  # noqa: E402

SETTINGS = {
    "TS default": (solve_ts, {}),
    "TS focus 0.5": (solve_ts, {"focus": 0.5}),
    "TS tenure 12": (solve_ts, {"tenure": 12, "patience": 40}),
    "PSS default (repack 0.5)": (solve_pss, {}),
    "PSS repack 0.8": (solve_pss, {"repack": 0.8}),
    "PSS swarm 30": (solve_pss, {"swarm": 30}),
}


def job(path):
    p = parse(path)
    budget, seed = auto_budget(p), auto_seed(p)
    base = min(SOLVERS[a](p, budget, seed).makespan for a in BASELINES)
    return base, {k: fn(p, budget, seed, **kw).makespan for k, (fn, kw) in SETTINGS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    files = sorted(f for f in (ROOT / "data" / "train").iterdir() if f.suffix in (".sm", ".rcp", ".json"))
    files = random.Random(7).sample(files, args.sample)
    with Pool(args.workers) as pool:
        out = pool.map(job, files)
    print(f"{len(files)} training instances; best-baseline mean makespan {sum(b for b, _ in out) / len(out):.2f}")
    for k in SETTINGS:
        ms = [o[k] for _, o in out]
        print(f"  {k:26s} mean {sum(ms) / len(ms):7.2f}   <= best baseline {sum(o[k] <= b for b, o in out)}"
              f"/{len(out)}   < best baseline {sum(o[k] < b for b, o in out)}/{len(out)}")


if __name__ == "__main__":
    main()
