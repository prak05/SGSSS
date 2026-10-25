"""Read and write RCPSP instances in three formats.

  .sm   PSPLIB (Kolisch & Sprecher 1996)
  .rcp  Patterson / RanGen: "n K", capacities, then per job "dur demands... nsucc succs..."
  .json {"name", "resources": {name: cap}, "activities": [{"id", "name", "duration",
         "predecessors", "resource_requirements"}]}  (PSPLIB-style "jobs/succs" JSON also accepted)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.model import Project


def parse(path) -> Project:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    reader = {".sm": _read_sm, ".rcp": _read_rcp, ".json": _read_json}.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file type {path.suffix!r}; use .sm, .rcp or .json")
    return reader(text, path.stem)


def _from_successors(name, durations, demands, caps, succs, **names) -> Project:
    preds = [[] for _ in durations]
    for i, ss in enumerate(succs):
        for s in ss:
            preds[s].append(i)
    return Project(name, durations, demands, caps, preds, **names)


def _read_sm(text: str, name: str) -> Project:
    lines = text.splitlines()
    k = int(re.search(r"-\s*renewable\s*:\s*(\d+)", text).group(1))
    jobs = int(re.search(r"jobs \(incl\. supersource/sink \)\s*:\s*(\d+)", text).group(1))

    def rows_after(header):
        start = next(i for i, l in enumerate(lines) if l.startswith(header))
        return [l.split() for l in lines[start + 1:] if l.split() and l.split()[0].isdigit()]

    succs = [[int(x) - 1 for x in r[3:3 + int(r[2])]] for r in rows_after("PRECEDENCE RELATIONS")[:jobs]]
    req = rows_after("REQUESTS/DURATIONS")[:jobs]
    durations = [int(r[2]) for r in req]
    demands = [[int(x) for x in r[3:3 + k]] for r in req]
    caps = [int(x) for x in rows_after("RESOURCEAVAILABILITIES")[0][:k]]
    return _from_successors(name, durations, demands, caps, succs)


def _read_rcp(text: str, name: str) -> Project:
    tok = [int(t) for t in text.split()]
    n, k = tok[0], tok[1]
    caps, pos = tok[2:2 + k], 2 + k
    durations, demands, succs = [], [], []
    for _ in range(n):
        durations.append(tok[pos])
        demands.append(tok[pos + 1:pos + 1 + k])
        ns = tok[pos + 1 + k]
        succs.append([s - 1 for s in tok[pos + 2 + k:pos + 2 + k + ns]])
        pos += 2 + k + ns
    return _from_successors(name, durations, demands, caps, succs)


def _read_json(text: str, name: str) -> Project:
    d = json.loads(text)
    if "jobs" in d:  # PSPLIB-style JSON: parallel lists, 1-based job ids
        index = {j: i for i, j in enumerate(d["jobs"])}
        return _from_successors(d.get("name", name), d["durations"], d["demands"], d["capacities"],
                                [[index[s] for s in ss] for ss in d["succs"]])
    res = list(d["resources"].items())
    acts = d["activities"]
    index = {a["id"]: i for i, a in enumerate(acts)}
    return Project(
        d.get("name", name),
        [int(a["duration"]) for a in acts],
        [[int(a.get("resource_requirements", {}).get(r, 0)) for r, _ in res] for a in acts],
        [int(c) for _, c in res],
        [[index[p] for p in a.get("predecessors", [])] for a in acts],
        task_names=[a.get("name", f"A{a['id']}") for a in acts],
        resource_names=[r for r, _ in res],
    )


def write(project: Project, path) -> None:
    """Save a project in the format given by the file extension."""
    path = Path(path)
    p = project
    if path.suffix == ".json":
        out = {"name": p.name,
               "resources": dict(zip(p.resource_names, p.capacities)),
               "activities": [{"id": i + 1, "name": p.task_names[i], "duration": p.durations[i],
                               "predecessors": [q + 1 for q in p.preds[i]],
                               "resource_requirements": dict(zip(p.resource_names, p.demands[i]))}
                              for i in range(p.n)]}
        path.write_text(json.dumps(out, indent=1))
    elif path.suffix == ".rcp":
        rows = [f"{p.n} {len(p.capacities)}", " ".join(map(str, p.capacities))]
        rows += [" ".join(map(str, [p.durations[i], *p.demands[i], len(p.succs[i])] +
                              [s + 1 for s in p.succs[i]])) for i in range(p.n)]
        path.write_text("\n".join(rows) + "\n")
    elif path.suffix == ".sm":
        k = len(p.capacities)
        rows = ["*" * 72, f"file with basedata            : {path.name}", "*" * 72,
                f"projects                      :  1",
                f"jobs (incl. supersource/sink ):  {p.n}",
                f"horizon                       :  {sum(p.durations)}",
                "RESOURCES", f"  - renewable                 :  {k}   R",
                "  - nonrenewable              :  0   N", "  - doubly constrained       :  0   D",
                "*" * 72, "PRECEDENCE RELATIONS:", "jobnr.    #modes  #successors   successors"]
        rows += [f"  {i + 1}        1          {len(p.succs[i])}          " +
                 " ".join(str(s + 1) for s in p.succs[i]) for i in range(p.n)]
        rows += ["*" * 72, "REQUESTS/DURATIONS:",
                 "jobnr. mode duration  " + "  ".join(f"R {r + 1}" for r in range(k)), "-" * 72]
        rows += [f"  {i + 1}      1     {p.durations[i]}       " + "    ".join(map(str, p.demands[i]))
                 for i in range(p.n)]
        rows += ["*" * 72, "RESOURCEAVAILABILITIES:", "  " + "    ".join(f"R {r + 1}" for r in range(k)),
                 "   " + "   ".join(map(str, p.capacities)), "*" * 72]
        path.write_text("\n".join(rows) + "\n")
    else:
        raise ValueError(f"Unsupported file type {path.suffix!r}")
