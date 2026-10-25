#!/usr/bin/env python3
"""Download the public benchmark sets and build data/train and data/test (no overlap).

Sources
  PSPLIB  (Kolisch & Sprecher 1996)  https://www.om-db.wi.tum.de/psplib/  J30, J60, J90, J120
          + proven optimal makespans for J30 and best-known makespans for J60/J90/J120
  RG30    (Vanhoucke et al.)          via github.com/sydney-machine-learning/MEHH_RCPSP

Split (PSPLIB groups instances by "parameter" 1..48; train and test use different groups)
  train  J30 parameters 13-48 (.sm, 360)  +  J60 parameters 7-48 (420: groups 13-18 as .json 60, the rest .sm 360)
         +  RG30 Pat101-400 (.rcp, 300)   = 1080 instances
  test   J30 parameters 1-12 (.sm, 120)   +  J60 parameters 1-4 (.sm, 40)  +  J60 parameters 5-6 as .json (20)
         RG30 Pat1-60 (.rcp, 60)          +  J90 / J120 parameter 1-2, instances 1-5 (.sm, 20)
  data/test/reference.csv lists the optimal / best-known makespan where PSPLIB publishes one.
"""
import csv
import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.parser import parse, write  # noqa: E402

PSPLIB = "https://www.om-db.wi.tum.de/psplib"
RG30 = "https://raw.githubusercontent.com/sydney-machine-learning/MEHH_RCPSP/master/datasets/RG30/set1/Pat{}.rcp"
CACHE = Path("/tmp/rcpsp_download_cache")


def fetch(url: str) -> bytes:
    target = CACHE / url.replace("/", "_").replace(":", "").replace("?", "_").replace("&", "_")
    if not target.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "AI-RCPSP dataset builder"})
        target.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return target.read_bytes()


def psplib_set(name: str) -> dict:
    data = fetch(f"{PSPLIB}/download_dataset.php?set={name}&mode=sm&format=tgz")
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {Path(m.name).name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def reference(name: str, kind: str) -> dict:
    text = fetch(f"{PSPLIB}/download_solution.php?set={name}&mode=sm&type={kind}").decode()
    refs = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            refs[f"{name}{parts[0]}_{parts[1]}"] = int(parts[2])
    return refs


def main():
    CACHE.mkdir(exist_ok=True)
    train, test = ROOT / "data" / "train", ROOT / "data" / "test"
    for d in (train, test):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    refs = {**reference("j30", "opt"), **reference("j60", "hrs"),
            **reference("j90", "hrs"), **reference("j120", "hrs")}
    ref_rows = []

    def put(folder: Path, stem: str, raw: bytes, ext: str):
        tmp = folder / f"{stem}.sm"
        tmp.write_bytes(raw)
        if ext != ".sm":
            write(parse(tmp), folder / f"{stem}{ext}")
            tmp.unlink()
        if folder == test and stem in refs:
            kind = "optimal" if stem.startswith("j30") else "best known"
            ref_rows.append((f"{stem}{ext}", refs[stem], kind))

    sets = {name: psplib_set(name) for name in ("j30", "j60", "j90", "j120")}
    for name, files in sets.items():
        for fname, raw in files.items():
            stem = fname[:-3]
            param, inst = (int(x) for x in stem[len(name):].split("_"))
            if name == "j30":
                put(train if param >= 13 else test, stem, raw, ".sm")
            elif name == "j60" and param >= 7:
                put(train, stem, raw, ".json" if 13 <= param <= 18 else ".sm")
            elif name == "j60" and param <= 6:
                put(test, stem, raw, ".sm" if param <= 4 else ".json")
            elif name in ("j90", "j120") and param <= 2 and inst <= 5:
                put(test, stem, raw, ".sm")

    for i in list(range(1, 61)) + list(range(101, 401)):
        (train if i > 100 else test).joinpath(f"Pat{i}.rcp").write_bytes(fetch(RG30.format(i)))

    with open(test / "reference.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("file", "makespan", "type")] + sorted(ref_rows))
    for d in (train, test):
        files = [f for f in d.iterdir() if f.suffix in (".sm", ".rcp", ".json")]
        for f in files:
            parse(f)                                        # every file must load cleanly
        counts = {ext: sum(f.suffix == ext for f in files) for ext in (".sm", ".rcp", ".json")}
        print(f"{d.relative_to(ROOT)}: {len(files)} instances {counts}")
    print(f"reference makespans: {len(ref_rows)}")


if __name__ == "__main__":
    main()
