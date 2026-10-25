"""Tests: parsing, feasibility, budgets, fair winner selection, GNN, report.  Run: pytest tests -q"""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import ssgs  # noqa: E402
from src.algorithms import SOLVERS, gnn  # noqa: E402
from src.algorithms.rules import rules  # noqa: E402
from src.benchmark import explain, ranking, run_all  # noqa: E402
from src.evaluation import audit  # noqa: E402
from src.model import Project  # noqa: E402
from src.parser import parse, write  # noqa: E402
from src.report import pdf_report  # noqa: E402

J30 = ROOT / "data" / "test" / "j301_1.sm"            # proven optimum 43 (PSPLIB)
USE = ROOT / "data" / "use_cases_small"


@pytest.fixture(scope="module")
def j30():
    return parse(J30)


def test_parse_all_formats():
    for ext in (".sm", ".rcp", ".json"):
        p = parse(USE / f"house_construction{ext}")
        assert p.real_tasks == 18 and p.cpm_bound == 47
    assert parse(J30).n == 32 and parse(J30).capacities == [12, 13, 4, 12]
    assert parse(ROOT / "data" / "test" / "Pat1.rcp").n > 0


def test_round_trip(tmp_path, j30):
    for ext in (".sm", ".rcp", ".json"):
        write(j30, tmp_path / f"x{ext}")
        q = parse(tmp_path / f"x{ext}")
        assert (q.durations, q.demands, q.capacities) == (j30.durations, j30.demands, j30.capacities)
        assert [sorted(x) for x in q.preds] == [sorted(x) for x in j30.preds]


def test_cycle_and_capacity_rejected():
    with pytest.raises(ValueError, match="cycle"):
        Project("c", [1, 1], [[0], [0]], [1], [[1], [0]])
    with pytest.raises(ValueError, match="needs"):
        Project("c", [1], [[5]], [1], [[]])


def test_random_priorities_always_feasible(j30):
    rng = random.Random(0)
    for _ in range(50):
        starts, ms = ssgs.decode(j30, [rng.random() for _ in range(j30.n)])
        a = audit(j30, starts)
        assert a.feasible and a.makespan == ms >= j30.lower_bound


def test_justification_never_worse(j30):
    rng = random.Random(1)
    for _ in range(30):
        starts, ms = ssgs.decode(j30, [rng.random() for _ in range(j30.n)])
        new_starts, new_ms = ssgs.justify(j30, starts)
        assert new_ms <= ms and audit(j30, new_starts).feasible


@pytest.mark.parametrize("name", list(SOLVERS))
def test_solver_respects_budget_and_is_feasible(name, j30):
    r = SOLVERS[name](j30, budget=200, seed=3)
    assert r.evaluations <= 200 and len(r.trace) == len(r.best_trace) == r.evaluations
    assert r.best_trace == sorted(r.best_trace, reverse=True)          # best-so-far never gets worse
    assert r.best_trace[-1] == r.makespan and audit(j30, r.starts).makespan == r.makespan
    assert audit(j30, r.starts).feasible and 43 <= r.makespan       # 43 = proven optimum


def test_winner_depends_only_on_results(j30):
    results = run_all(j30, budget=150, seed=5)
    order = ranking(results)
    renamed = {f"X{i}": results[a] for i, a in enumerate(reversed(list(results)))}
    back = {f"X{i}": a for i, a in enumerate(reversed(list(results)))}
    assert [back[x] for x in ranking(renamed)] == order                # names don't matter
    e = explain(j30, results)
    assert e["winner"] == order[0] and e["how"] and all(row["feasible"] for row in e["table"])


def test_gnn_trained_and_better_than_lft_on_average():
    assert gnn.load_model() is not None, "run scripts/train.py first"
    files = sorted((ROOT / "data" / "test").glob("j30*.sm"))[:40]
    gap_gnn = gap_lft = 0
    for f in files:
        p = parse(f)
        gap_gnn += ssgs.decode(p, gnn.predict(p))[1]
        gap_lft += ssgs.decode(p, rules(p)["LFT (latest finish time)"])[1]
    assert gap_gnn <= gap_lft


def test_pdf_report(j30):
    pdf = pdf_report(j30, run_all(j30, budget=60, seed=1))
    assert pdf[:4] == b"%PDF" and len(pdf) > 5000


def test_gnn_inspect_attention_follows_graph(j30):
    g = gnn.inspect(j30)
    assert g["rank"] and len(g["score"]) == j30.n
    for name, w in g["attention"].items():
        neighbours = j30.preds if "upstream" in name else j30.succs
        for i in range(j30.n):
            assert abs(w[i].sum() - 1) < 1e-5                         # each row is a probability distribution
            assert all(w[i][j] == 0 for j in range(j30.n) if j != i and j not in neighbours[i])


def test_use_cases_parse():
    files = sorted((ROOT / "data" / "use_cases").iterdir())
    assert len(files) >= 50
    for f in files[:6]:
        p = parse(f)
        assert 30 <= p.real_tasks <= 60


def test_auto_budget_and_seed_depend_on_content_not_file_type(tmp_path, j30):
    from src.benchmark import auto_budget, auto_seed
    settings = set()
    for ext in (".sm", ".rcp", ".json"):
        write(j30, tmp_path / f"same{ext}")
        q = parse(tmp_path / f"same{ext}")
        settings.add((auto_budget(q), auto_seed(q)))
    assert len(settings) == 1                                    # same project -> same budget and seed
    tight = parse(ROOT / "data" / "test" / "Pat1.rcp")           # 30 activities, resource-bound
    assert auto_budget(tight) > auto_budget(j30) == 500          # tighter resources -> more evaluations
