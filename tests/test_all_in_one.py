import pytest
from pathlib import Path
from rcpsp_all_in_one import (
    Parser,
    SSGS,
    Evaluator,
    extract_features,
    solve_ga,
    solve_sa,
    solve_bco,
    solve_tabu,
    solve_prh,
    solve_gnn_priority,
    run_benchmark,
    compare_algorithms,
)

SAMPLE_SM = "data/j10_sample.sm"

def test_all_in_one_parse_and_extract():
    if not Path(SAMPLE_SM).exists():
        pytest.skip("Sample SM file not found")
    project = Parser.parse(SAMPLE_SM)
    assert project.num_activities == 12
    assert project.num_resources == 2

    features = extract_features(project)
    assert features.N == 10
    assert features.critical_path_duration == 20

def test_all_in_one_solvers():
    if not Path(SAMPLE_SM).exists():
        pytest.skip("Sample SM file not found")
    project = Parser.parse(SAMPLE_SM)

    res_ga = solve_ga(project, evaluation_budget=100, seed=42)
    assert res_ga.makespan > 0
    assert Evaluator.check_constraints(project, res_ga.schedule)

    res_sa = solve_sa(project, evaluation_budget=100, seed=42)
    assert res_sa.makespan > 0
    assert Evaluator.check_constraints(project, res_sa.schedule)

    res_bco = solve_bco(project, evaluation_budget=100, seed=42)
    assert res_bco.makespan > 0
    assert Evaluator.check_constraints(project, res_bco.schedule)

    res_ts = solve_tabu(project, evaluation_budget=100, seed=42)
    assert res_ts.makespan > 0
    assert Evaluator.check_constraints(project, res_ts.schedule)

    res_prh = solve_prh(project, evaluation_budget=100, seed=42)
    assert res_prh.makespan > 0
    assert Evaluator.check_constraints(project, res_prh.schedule)

    res_gnn = solve_gnn_priority(project, evaluation_budget=100, seed=42)
    assert res_gnn.makespan > 0
    assert Evaluator.check_constraints(project, res_gnn.schedule)

def test_all_in_one_benchmark():
    if not Path(SAMPLE_SM).exists():
        pytest.skip("Sample SM file not found")
    project = Parser.parse(SAMPLE_SM)
    # Default 5 metaheuristics
    records = run_benchmark(project, seeds=[42], evaluation_budget=100)
    assert len(records) == 5

    # Full 6-algorithm suite including GNN
    records_6 = run_benchmark(
        project,
        seeds=[42],
        evaluation_budget=100,
        algorithms=["GA", "SA", "BCO", "PRH", "TS", "GNN-Priority"],
    )
    assert len(records_6) == 6
    assert any(r.algorithm == "GNN-Priority" for r in records_6)

    report = compare_algorithms(records_6)
    assert "Comparison Report" in report.to_text()
