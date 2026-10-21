"""
Tests for src/benchmark.py and src/comparison.py.

Uses two project instances:
  - data/sample.rcp  (small custom format, Phase 1 fixture)
  - data/j10_sample.sm (PSPLIB format, Phase 3 fixture)

Tests cover:
  - All five algorithms run on both instances.
  - Every schedule is feasible.
  - Every record has makespan > 0.
  - Evaluation budget is respected.
  - Convergence history is present.
  - Reproducibility per seed.
  - Multi-seed runs produce consistent structures.
  - Comparison report is generated correctly.
  - Validation helper catches known bad records.
"""

import unittest

from src.parser import Parser
from src.benchmark import (
    run_single,
    run_all_algorithms,
    ALGORITHM_NAMES,
    BenchmarkRecord,
)
from src.comparison import compare, validate_records, ComparisonReport


SEEDS  = [1, 2, 3]
BUDGET = 300  # small but sufficient for tests


class TestBenchmarkRunSingle(unittest.TestCase):

    def setUp(self):
        self.project_rcp = Parser.parse("data/sample.rcp")
        self.project_sm  = Parser.parse("data/j10_sample.sm")

    def _check_record(self, rec: BenchmarkRecord, budget: int):
        self.assertIsInstance(rec, BenchmarkRecord)
        self.assertGreater(rec.makespan, 0)
        self.assertTrue(rec.feasible, f"{rec.label}: expected feasible schedule")
        self.assertLessEqual(rec.evaluations, budget)
        self.assertGreater(len(rec.best_priority_vector), 0)
        self.assertGreater(len(rec.convergence_history), 0)
        self.assertGreaterEqual(rec.runtime, 0.0)

    def test_all_algorithms_on_rcp(self):
        for algo in ALGORITHM_NAMES:
            with self.subTest(algorithm=algo):
                rec = run_single(
                    self.project_rcp, algo, seed=1,
                    evaluation_budget=BUDGET,
                    instance_id="sample_rcp",
                )
                self._check_record(rec, BUDGET)

    def test_all_algorithms_on_sm(self):
        for algo in ALGORITHM_NAMES:
            with self.subTest(algorithm=algo):
                rec = run_single(
                    self.project_sm, algo, seed=1,
                    evaluation_budget=BUDGET,
                    instance_id="j10_sm",
                )
                self._check_record(rec, BUDGET)

    def test_reproducibility_rcp(self):
        for algo in ALGORITHM_NAMES:
            with self.subTest(algorithm=algo):
                r1 = run_single(self.project_rcp, algo, seed=42, evaluation_budget=BUDGET)
                r2 = run_single(self.project_rcp, algo, seed=42, evaluation_budget=BUDGET)
                self.assertEqual(r1.makespan, r2.makespan)

    def test_reproducibility_sm(self):
        for algo in ALGORITHM_NAMES:
            with self.subTest(algorithm=algo):
                r1 = run_single(self.project_sm, algo, seed=42, evaluation_budget=BUDGET)
                r2 = run_single(self.project_sm, algo, seed=42, evaluation_budget=BUDGET)
                self.assertEqual(r1.makespan, r2.makespan)

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            run_single(self.project_rcp, "XYZ", seed=1, evaluation_budget=BUDGET)

    def test_convergence_history_is_monotone(self):
        """Best-so-far makespan must never increase over evaluations."""
        rec = run_single(self.project_sm, "GA", seed=1, evaluation_budget=BUDGET)
        history = rec.convergence_history
        for i in range(1, len(history)):
            self.assertLessEqual(
                history[i][1], history[i - 1][1],
                msg=f"GA convergence history non-monotone at index {i}: "
                    f"{history[i-1]} → {history[i]}"
            )


class TestBenchmarkRunAll(unittest.TestCase):

    def setUp(self):
        self.project = Parser.parse("data/j10_sample.sm")

    def test_record_count(self):
        """run_all_algorithms returns 5 × len(seeds) records."""
        records = run_all_algorithms(
            self.project,
            seeds=SEEDS,
            evaluation_budget=BUDGET,
            instance_id="j10",
        )
        self.assertEqual(len(records), len(ALGORITHM_NAMES) * len(SEEDS))

    def test_all_records_feasible(self):
        records = run_all_algorithms(
            self.project, seeds=[1], evaluation_budget=BUDGET, instance_id="j10"
        )
        for rec in records:
            self.assertTrue(rec.feasible, f"{rec.label}: infeasible")

    def test_all_budgets_respected(self):
        records = run_all_algorithms(
            self.project, seeds=SEEDS, evaluation_budget=BUDGET, instance_id="j10"
        )
        for rec in records:
            self.assertLessEqual(rec.evaluations, BUDGET, f"{rec.label}: over budget")

    def test_instance_id_propagated(self):
        records = run_all_algorithms(
            self.project, seeds=[1], evaluation_budget=BUDGET, instance_id="MY_INST"
        )
        for rec in records:
            self.assertEqual(rec.instance_id, "MY_INST")


class TestMultipleSeeds(unittest.TestCase):
    """
    Verify stochastic algorithms can produce different trajectories
    across seeds while remaining individually reproducible.
    """

    def setUp(self):
        self.project = Parser.parse("data/j10_sample.sm")

    def test_seed_reproducibility(self):
        """Same seed must always produce the same makespan."""
        for algo in ["GA", "SA", "BCO", "TS"]:
            with self.subTest(algorithm=algo):
                r1 = run_single(self.project, algo, seed=7,  evaluation_budget=BUDGET)
                r2 = run_single(self.project, algo, seed=7,  evaluation_budget=BUDGET)
                self.assertEqual(r1.makespan, r2.makespan)

    def test_seeds_produce_valid_records(self):
        for seed in SEEDS:
            for algo in ALGORITHM_NAMES:
                with self.subTest(algorithm=algo, seed=seed):
                    rec = run_single(
                        self.project, algo, seed=seed, evaluation_budget=BUDGET
                    )
                    self.assertGreater(rec.makespan, 0)
                    self.assertTrue(rec.feasible)


class TestComparison(unittest.TestCase):

    def setUp(self):
        self.project = Parser.parse("data/j10_sample.sm")
        self.records = run_all_algorithms(
            self.project,
            seeds=SEEDS,
            evaluation_budget=BUDGET,
            instance_id="j10_test",
        )

    def test_compare_returns_report(self):
        report = compare(self.records, instance_id="j10_test", budget=BUDGET)
        self.assertIsInstance(report, ComparisonReport)

    def test_report_has_all_algorithms(self):
        report = compare(self.records, instance_id="j10_test", budget=BUDGET)
        algo_bases = {s.algorithm.split("[")[0] for s in report.summaries}
        for expected in ALGORITHM_NAMES:
            self.assertIn(expected, algo_bases)

    def test_winner_ms_is_minimum(self):
        report = compare(self.records, instance_id="j10_test", budget=BUDGET)
        all_bests = [s.best_makespan for s in report.summaries]
        self.assertEqual(report.winner_ms, min(all_bests))

    def test_winner_is_valid(self):
        report = compare(self.records, instance_id="j10_test", budget=BUDGET)
        algo_bases = {s.algorithm.split("[")[0] for s in report.summaries}
        # winner must be "ALL" or one of the known algorithm names
        if report.winner != "ALL":
            self.assertIn(report.winner.split("[")[0], algo_bases)

    def test_text_report_not_empty(self):
        report = compare(self.records, instance_id="j10_test", budget=BUDGET)
        text = report.to_text()
        self.assertIn("Best feasible solution found", text)
        self.assertIn("j10_test", text)

    def test_n_seeds_correct(self):
        report = compare(self.records, instance_id="j10_test", budget=BUDGET)
        self.assertEqual(report.n_seeds, len(SEEDS))


class TestValidation(unittest.TestCase):

    def setUp(self):
        from src.benchmark import BenchmarkRecord
        self.good_record = BenchmarkRecord(
            instance_id="x",
            algorithm="GA",
            seed=1,
            makespan=25,
            runtime=0.5,
            evaluations=100,
            feasible=True,
            best_priority_vector=[0.5] * 8,
            convergence_history=[(1, 25)],
        )

    def test_valid_record_no_errors(self):
        errors = validate_records([self.good_record])
        self.assertEqual(errors, [])

    def test_infeasible_record_flagged(self):
        import copy
        bad = copy.copy(self.good_record)
        bad.feasible = False
        errors = validate_records([bad])
        self.assertTrue(any("NOT feasible" in e for e in errors))

    def test_zero_makespan_flagged(self):
        import copy
        bad = copy.copy(self.good_record)
        bad.makespan = 0
        errors = validate_records([bad])
        self.assertTrue(any("makespan" in e for e in errors))


class TestFairnessVerification(unittest.TestCase):
    """
    Verify that all five algorithms receive the same input and
    use the same evaluation budget.
    """

    def setUp(self):
        self.project = Parser.parse("data/j10_sample.sm")

    def test_all_algorithms_same_budget(self):
        budget = 200
        records = run_all_algorithms(
            self.project, seeds=[1], evaluation_budget=budget, instance_id="fair"
        )
        for rec in records:
            self.assertLessEqual(
                rec.evaluations, budget,
                f"{rec.algorithm} exceeded budget {budget}: used {rec.evaluations}"
            )

    def test_all_algorithms_same_instance(self):
        """
        All records from one run_all_algorithms call should reference the same
        instance_id (they were all run on the same project).
        """
        records = run_all_algorithms(
            self.project, seeds=[1], evaluation_budget=200, instance_id="FAIRNESS"
        )
        instance_ids = {rec.instance_id for rec in records}
        self.assertEqual(instance_ids, {"FAIRNESS"})

    def test_six_algorithms_suite(self):
        """All algorithms in ALL_ALGORITHM_NAMES run and return valid records."""
        from src.benchmark import ALL_ALGORITHM_NAMES
        records = run_all_algorithms(
            self.project,
            seeds=[1],
            evaluation_budget=200,
            instance_id="ALL_ALGOS",
            algorithms=ALL_ALGORITHM_NAMES,
        )
        self.assertEqual(len(records), len(ALL_ALGORITHM_NAMES))
        algos_in_records = {r.algorithm for r in records}
        self.assertTrue(any("GNN" in a for a in algos_in_records))
        self.assertTrue(any("PSO" in a for a in algos_in_records))
        for r in records:
            self.assertTrue(r.feasible)
            self.assertGreater(r.makespan, 0)


if __name__ == "__main__":
    unittest.main()
