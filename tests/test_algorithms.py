"""
Phase 2 algorithm tests.

Tests verify:
  1. Each algorithm executes without error.
  2. Returns an AlgorithmResult.
  3. Schedule is feasible (precedence + resource constraints).
  4. Makespan is consistent with the returned schedule.
  5. Evaluation budget is respected.
  6. Convergence history is populated.
  7. Same seed → same makespan (reproducibility).
  8. Algorithms exclusively use the common SSGS (checked via patching).
"""

import unittest
from unittest.mock import patch
from src.parser import Parser
from src.evaluation import Evaluator
from src.model import AlgorithmResult

import src.algorithms.ga   as ga_mod
import src.algorithms.sa   as sa_mod
import src.algorithms.bco  as bco_mod
import src.algorithms.prh  as prh_mod
import src.algorithms.tabu as tabu_mod


BUDGET = 200   # small budget so tests stay fast
SEED   = 42


class _AlgorithmTestBase:
    """
    Mix-in with shared checks.  Concrete subclasses set:
        self.project
        self.result   (AlgorithmResult)
        self.budget   (int)
    """

    def _check_result(self):
        r = self.result

        # Must be an AlgorithmResult
        self.assertIsInstance(r, AlgorithmResult)

        # Makespan must be a positive int
        self.assertIsInstance(r.makespan, int)
        self.assertGreater(r.makespan, 0)

        # Priority vector must have one entry per activity
        n_acts = self.project.num_activities
        self.assertEqual(len(r.best_priority_vector), n_acts)

        # Every priority must be in [0, 1]
        for v in r.best_priority_vector:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        # Evaluation budget respected
        self.assertLessEqual(r.evaluations, self.budget)

        # Convergence history must contain at least one entry
        self.assertGreater(len(r.convergence_history), 0)

        # Runtime is non-negative
        self.assertGreaterEqual(r.runtime, 0.0)

        # Schedule feasibility
        feasible = Evaluator.check_constraints(self.project, r.schedule)
        self.assertTrue(feasible, f"{r.algorithm}: schedule not feasible")

        # Makespan consistency — schedule's max finish time == reported makespan
        if r.schedule:
            actual_ms = max(e.finish_time for e in r.schedule)
            self.assertEqual(
                r.makespan, actual_ms,
                f"{r.algorithm}: reported makespan {r.makespan} != "
                f"schedule finish {actual_ms}"
            )


class TestGA(unittest.TestCase, _AlgorithmTestBase):
    def setUp(self):
        self.project = Parser.parse("data/sample.rcp")
        self.budget  = BUDGET
        self.result  = ga_mod.solve(
            self.project, evaluation_budget=BUDGET, seed=SEED,
            population_size=10,
        )

    def test_result(self):           self._check_result()
    def test_algorithm_name(self):   self.assertEqual(self.result.algorithm, "GA")

    def test_reproducibility(self):
        r2 = ga_mod.solve(self.project, evaluation_budget=BUDGET, seed=SEED, population_size=10)
        self.assertEqual(self.result.makespan, r2.makespan)

    def test_different_seed(self):
        r2 = ga_mod.solve(self.project, evaluation_budget=BUDGET, seed=99, population_size=10)
        # Different seeds may produce different vectors (not guaranteed, but check types)
        self.assertIsInstance(r2.makespan, int)

    def test_uses_evaluator(self):
        """
        Verify GA calls Evaluator.evaluate.
        If this call count is 0, GA is doing its own scheduling — forbidden.
        """
        with patch.object(Evaluator, "evaluate", wraps=Evaluator.evaluate) as mock_eval:
            ga_mod.solve(self.project, evaluation_budget=50, seed=SEED, population_size=10)
            self.assertGreater(mock_eval.call_count, 0)


class TestSA(unittest.TestCase, _AlgorithmTestBase):
    def setUp(self):
        self.project = Parser.parse("data/sample.rcp")
        self.budget  = BUDGET
        self.result  = sa_mod.solve(self.project, evaluation_budget=BUDGET, seed=SEED)

    def test_result(self):           self._check_result()
    def test_algorithm_name(self):   self.assertEqual(self.result.algorithm, "SA")

    def test_reproducibility(self):
        r2 = sa_mod.solve(self.project, evaluation_budget=BUDGET, seed=SEED)
        self.assertEqual(self.result.makespan, r2.makespan)

    def test_uses_evaluator(self):
        with patch.object(Evaluator, "evaluate", wraps=Evaluator.evaluate) as mock_eval:
            sa_mod.solve(self.project, evaluation_budget=50, seed=SEED)
            self.assertGreater(mock_eval.call_count, 0)


class TestBCO(unittest.TestCase, _AlgorithmTestBase):
    def setUp(self):
        self.project = Parser.parse("data/sample.rcp")
        self.budget  = BUDGET
        self.result  = bco_mod.solve(
            self.project, evaluation_budget=BUDGET, seed=SEED, colony_size=5,
        )

    def test_result(self):           self._check_result()
    def test_algorithm_name(self):   self.assertEqual(self.result.algorithm, "BCO")

    def test_reproducibility(self):
        r2 = bco_mod.solve(self.project, evaluation_budget=BUDGET, seed=SEED, colony_size=5)
        self.assertEqual(self.result.makespan, r2.makespan)

    def test_uses_evaluator(self):
        with patch.object(Evaluator, "evaluate", wraps=Evaluator.evaluate) as mock_eval:
            bco_mod.solve(self.project, evaluation_budget=30, seed=SEED, colony_size=5)
            self.assertGreater(mock_eval.call_count, 0)


class TestPRH(unittest.TestCase, _AlgorithmTestBase):
    def setUp(self):
        self.project = Parser.parse("data/sample.rcp")
        self.budget  = BUDGET
        self.result  = prh_mod.solve(self.project, evaluation_budget=BUDGET, seed=SEED)

    def test_result(self):           self._check_result()

    def test_algorithm_name_prefix(self):
        self.assertTrue(
            self.result.algorithm.startswith("PRH"),
            f"Expected algorithm name to start with 'PRH', got '{self.result.algorithm}'"
        )

    def test_reports_best_rule(self):
        # Algorithm name should embed the best rule
        self.assertRegex(self.result.algorithm, r"PRH\[.+\]")

    def test_reproducibility(self):
        r2 = prh_mod.solve(self.project, evaluation_budget=BUDGET, seed=SEED)
        self.assertEqual(self.result.makespan, r2.makespan)

    def test_uses_evaluator(self):
        with patch.object(Evaluator, "evaluate", wraps=Evaluator.evaluate) as mock_eval:
            prh_mod.solve(self.project, evaluation_budget=30, seed=SEED)
            self.assertGreater(mock_eval.call_count, 0)


class TestTabuSearch(unittest.TestCase, _AlgorithmTestBase):
    def setUp(self):
        self.project = Parser.parse("data/sample.rcp")
        self.budget  = BUDGET
        self.result  = tabu_mod.solve(
            self.project, evaluation_budget=BUDGET, seed=SEED,
        )

    def test_result(self):           self._check_result()
    def test_algorithm_name(self):   self.assertEqual(self.result.algorithm, "TS")

    def test_reproducibility(self):
        r2 = tabu_mod.solve(self.project, evaluation_budget=BUDGET, seed=SEED)
        self.assertEqual(self.result.makespan, r2.makespan)

    def test_uses_evaluator(self):
        with patch.object(Evaluator, "evaluate", wraps=Evaluator.evaluate) as mock_eval:
            tabu_mod.solve(self.project, evaluation_budget=50, seed=SEED)
            self.assertGreater(mock_eval.call_count, 0)


class TestComparisonSanity(unittest.TestCase):
    """
    Run all five algorithms on the same project with the same seed and budget.
    Verify we can collect results consistently.  Does NOT declare a winner.
    """

    def setUp(self):
        self.project = Parser.parse("data/sample.rcp")

    def test_all_algorithms_produce_results(self):
        budget = 100
        seed   = 7

        results = [
            ga_mod.solve(self.project, evaluation_budget=budget, seed=seed, population_size=10),
            sa_mod.solve(self.project, evaluation_budget=budget, seed=seed),
            bco_mod.solve(self.project, evaluation_budget=budget, seed=seed, colony_size=5),
            prh_mod.solve(self.project, evaluation_budget=budget, seed=seed),
            tabu_mod.solve(self.project, evaluation_budget=budget, seed=seed),
        ]

        for r in results:
            self.assertIsInstance(r.makespan, int)
            self.assertGreater(r.makespan, 0)
            self.assertLessEqual(r.evaluations, budget)
            feasible = Evaluator.check_constraints(self.project, r.schedule)
            self.assertTrue(feasible, f"{r.algorithm} schedule infeasible")

        # Collect for inspection — no winner declared
        makespans = {r.algorithm: r.makespan for r in results}
        print("\n[Comparison sanity] makespans:", makespans)


if __name__ == "__main__":
    unittest.main()
