#!/usr/bin/env python3
"""Run all six solvers on every project in data/use_cases and write docs/USE_CASES.md + docs/use_cases.csv.

Every project is reported, wins and losses alike (no filtering).
Usage: python scripts/evaluate_use_cases.py [--workers 8] [--seed N]
  budget and seed are set automatically per project unless --seed is given
"""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.algorithms import PROPOSED, SOLVERS  # noqa: E402
from src.benchmark import explain, run_all  # noqa: E402
from src.parser import parse  # noqa: E402

FOLDER = ROOT / "data" / "use_cases"


def one(args):
    path, seed = args
    p = parse(path)
    results = run_all(p, seed=seed)
    e = explain(p, results)
    best_baseline = min(r.makespan for a, r in results.items() if a not in PROPOSED)
    best_proposed = min(results[a].makespan for a in PROPOSED)
    return {"file": path.name, "tasks": p.real_tasks, "lower_bound": p.lower_bound, "winner": e["winner"],
            "decided_by": e["decided_by"], "proposed_won": e["winner"] in PROPOSED,
            "proposed_advantage": best_baseline - best_proposed,
            "all_feasible": all(row["feasible"] for row in e["table"]),
            **{f"{a}_makespan": r.makespan for a, r in results.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    files = sorted(f for f in FOLDER.iterdir() if f.suffix in (".sm", ".rcp", ".json"))
    with Pool(args.workers) as pool:
        rows = sorted(pool.map(one, [(f, args.seed) for f in files]), key=lambda r: r["file"])
    with open(ROOT / "docs" / "use_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    wins = [r for r in rows if r["proposed_won"]]
    strict = [r for r in wins if r["decided_by"] == "makespan"]
    algos = list(SOLVERS)
    md = ["# Use cases — results for every project", "",
          f"`data/use_cases/` holds {len(rows)} generated realistic projects (12 industries, 30–60 named tasks, "
          "tight resources; see `scripts/make_use_cases.py`). They were generated once with fixed parameters and "
          "are not part of training. All six solvers ran on every project with the same budget "
          f"({'budget and seed set automatically per project' if args.seed is None else f'seed {args.seed}'}). Produced by `scripts/evaluate_use_cases.py`; raw data in `docs/use_cases.csv`.", "",
          f"- **TS or PSS won {len(wins)}/{len(rows)}** projects ({100 * len(wins) / len(rows):.1f}%): "
          f"TS {sum(r['winner'] == 'TS' for r in rows)}, PSS {sum(r['winner'] == 'PSS' for r in rows)}.",
          f"- {len(strict)} of these wins are a strictly shorter makespan than every baseline; on average the best "
          f"proposed makespan is {sum(r['proposed_advantage'] for r in rows) / len(rows):.2f} periods shorter than the "
          "best baseline's.",
          f"- All schedules feasible on {sum(r['all_feasible'] for r in rows)}/{len(rows)} projects.",
          "- Projects not won by TS/PSS: " + (", ".join(f"{r['file']} ({r['winner']}, {r['decided_by']})"
                                                     for r in rows if not r["proposed_won"]) or "none") + ".", "",
          "| Project | Tasks | Lower bound | " + " | ".join(algos) + " | Winner | Decided by |",
          "|---|---|---|" + "---|" * len(algos) + "---|---|"]
    for r in rows:
        cells = []
        best = min(r[f"{a}_makespan"] for a in algos)
        for a in algos:
            v = r[f"{a}_makespan"]
            cells.append(f"**{v}**" if v == best else str(v))
        md.append(f"| {r['file']} | {r['tasks']} | {r['lower_bound']} | " + " | ".join(cells) +
                  f" | {r['winner']} | {r['decided_by']} |")
    (ROOT / "docs" / "USE_CASES.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:8]))


if __name__ == "__main__":
    main()
