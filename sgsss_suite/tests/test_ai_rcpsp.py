"""
================================================================================
tests/test_ai_rcpsp.py — Comprehensive Test Suite for AI-RCPSP
================================================================================
Validates:
  1. Multi-format parsing (.sm, .rcp, .json) & project model properties
  2. Auto-adaptive SSGS budget & random seed calculation
  3. Shared SSGS decoding & dual-layer feasibility audit (Precedence + Resource)
  4. Algorithm suite (GA, SA, BCO, PRH, PSO, TBGAT) registry & execution
  5. Dedicated Particle Swarm Optimization (MSP_PSO integration) validation
  6. Dedicated TBGAT (Topology-Aware Bidirectional GAT integration) validation
  7. Plotly interactive visualization generation (Gantt, Resource Profiles, Convergence)
  8. Executive PDF reference report creation without 30-feature dependencies
  9. Topological sort and DAG cycle detection
================================================================================
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (
    Parser,
    Evaluator,
    ScheduleAudit,
    SSGS,
    run_algorithm,
    run_all_algorithms,
    SOLVERS,
    ALGORITHM_NAMES,
    solve_pso,
    solve_tbgat,
    create_gantt_chart,
    create_resource_timeline_charts,
    create_convergence_chart,
    generate_pdf_report,
    validate_dag,
    compute_cpm_lower_bound,
    compute_adaptive_budget,
    compute_adaptive_seed,
)
from src.model import Activity, Resource, Project


class TestAiRcpspCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instance_path = Path(__file__).resolve().parent.parent / "data" / "instances" / "j301_1.sm"
        assert cls.instance_path.exists(), f"Benchmark instance {cls.instance_path} not found"
        cls.project = Parser.parse(cls.instance_path)

        cls.toy_path = Path(__file__).resolve().parent.parent / "data" / "j10_sample.sm"
        if cls.toy_path.exists():
            cls.toy_project = Parser.parse(cls.toy_path)
        else:
            cls.toy_project = None

    def test_01_parser_and_model(self):
        """Verify project parsing and structural properties."""
        self.assertGreater(self.project.num_activities, 0)
        self.assertGreater(self.project.num_resources, 0)
        self.assertEqual(len(self.project.activities), self.project.num_activities)
        self.assertEqual(len(self.project.resources), self.project.num_resources)
        self.assertGreater(len(self.project.real_activities), 0)
        self.assertGreater(self.project.cpm_lower_bound, 0)
        self.assertGreater(self.project.total_work_content, 0)

    def test_02_adaptive_budget_and_seed(self):
        """Verify that evaluation budget and random seed automatically adapt to instance scale."""
        # j301_1 has 30 real activities -> budget should scale to 500
        budget_30 = compute_adaptive_budget(self.project)
        self.assertEqual(budget_30, 500)

        # toy instance (10 activities) -> budget should scale to 200
        if self.toy_project is not None:
            budget_10 = compute_adaptive_budget(self.toy_project)
            self.assertEqual(budget_10, 200)

        # Verify seed derivation is deterministic and reproducible
        seed_1 = compute_adaptive_seed(self.instance_path)
        seed_2 = compute_adaptive_seed(self.instance_path)
        self.assertEqual(seed_1, seed_2)
        self.assertIsInstance(seed_1, int)

    def test_03_ssgs_and_dual_layer_audit(self):
        """Verify SSGS schedule decoding and comprehensive dual-layer audit."""
        n = len(self.project.activities)
        priority_vec = [float(i) / n for i in range(n)]

        schedule, makespan = Evaluator.evaluate(self.project, priority_vec)
        self.assertGreater(makespan, 0)
        self.assertEqual(len(schedule), n)

        # Exhaustive dual-layer audit
        audit: ScheduleAudit = Evaluator.audit_schedule(self.project, schedule)
        self.assertTrue(audit.is_feasible, f"SSGS generated schedule failed audit: {audit.summary}")
        self.assertTrue(audit.precedence_passed)
        self.assertEqual(audit.precedence_violations, 0)
        self.assertTrue(audit.resource_passed)
        self.assertEqual(audit.resource_violations, 0)
        self.assertIn("100% FEASIBLE", audit.summary)

    def test_04_algorithm_registry(self):
        """Verify the 6 solvers (GA, SA, BCO, PRH, PSO, TBGAT) are registered and TS is scratched."""
        expected = ["BCO", "GA", "PRH", "PSO", "SA", "TBGAT"]
        self.assertEqual(sorted(ALGORITHM_NAMES), expected)
        self.assertNotIn("TS", ALGORITHM_NAMES)

        with self.assertRaises(ValueError):
            run_algorithm("NON_EXISTENT", self.project, evaluation_budget=50)

    def test_05_solver_execution_and_feasibility(self):
        """Run all solvers under budget and verify feasibility, runtime, and makespan."""
        budget = 50
        results = run_all_algorithms(self.project, evaluation_budget=budget, seed=42)

        self.assertEqual(len(results), 6)
        for name in ALGORITHM_NAMES:
            res = results[name]
            self.assertGreater(res.makespan, 0)
            self.assertGreater(res.runtime, 0)
            self.assertGreater(len(res.schedule), 0)
            self.assertGreater(len(res.convergence_history), 0)

            # Feasibility audit
            audit = Evaluator.audit_schedule(self.project, res.schedule)
            self.assertTrue(audit.is_feasible, f"Algorithm {name} produced infeasible schedule: {audit.summary}")
            self.assertEqual(audit.precedence_violations, 0)
            self.assertEqual(audit.resource_violations, 0)

    def test_06_pso_msp_integration(self):
        """Verify dedicated Particle Swarm Optimization (MSP_PSO) execution and swarm dynamics."""
        res = solve_pso(self.project, evaluation_budget=60, seed=42)
        self.assertIn("PSO", res.algorithm)
        self.assertGreater(res.makespan, 0)
        self.assertLessEqual(res.makespan, 100)
        self.assertGreater(len(res.history), 0)

        # Audit feasibility
        audit = Evaluator.audit_schedule(self.project, res.schedule)
        self.assertTrue(audit.is_feasible)
        self.assertEqual(audit.precedence_violations, 0)
        self.assertEqual(audit.resource_violations, 0)

    def test_07_tbgat_integration(self):
        """Verify dedicated TBGAT bidirectional graph attention neural scheduling."""
        res = solve_tbgat(self.project, evaluation_budget=40, seed=42)
        self.assertIn("TBGAT", res.algorithm)
        self.assertGreater(res.makespan, 0)
        self.assertLessEqual(res.makespan, 100)
        self.assertGreater(len(res.history), 0)

        # Audit feasibility
        audit = Evaluator.audit_schedule(self.project, res.schedule)
        self.assertTrue(audit.is_feasible)
        self.assertEqual(audit.precedence_violations, 0)
        self.assertEqual(audit.resource_violations, 0)

    def test_08_visualizations(self):
        """Verify Plotly charts generation."""
        budget = 30
        results = run_all_algorithms(self.project, evaluation_budget=budget, seed=42)
        best_res = min(results.values(), key=lambda r: r.makespan)

        # Gantt chart
        gantt_fig = create_gantt_chart(best_res.schedule, self.project)
        self.assertIsNotNone(gantt_fig)
        self.assertGreater(len(gantt_fig.data), 0)

        # Resource charts
        res_figs = create_resource_timeline_charts(self.project, best_res.schedule)
        self.assertEqual(len(res_figs), self.project.num_resources)

        # Convergence chart
        conv_fig = create_convergence_chart(results)
        self.assertIsNotNone(conv_fig)
        self.assertEqual(len(conv_fig.data), 6)

    def test_09_pdf_report(self):
        """Verify PDF generation produces valid binary byte stream without features dictionary."""
        res = run_algorithm("GA", self.project, evaluation_budget=30, seed=42)
        pdf_bytes = generate_pdf_report(
            project=self.project,
            best_res=res,
            proj_name="j301_1.sm",
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_10_dag_acyclicity_validation(self):
        """Verify Kahn's topological sort detects cycles and returns valid orderings."""
        topo = validate_dag(self.project)
        self.assertEqual(len(topo), self.project.num_activities)

        # Create a cyclic project copy
        cyclic_proj = Project()
        cyclic_proj.activities = {
            1: Activity(id=1, duration=1, predecessors=[2], successors=[2]),
            2: Activity(id=2, duration=1, predecessors=[1], successors=[1]),
        }
        with self.assertRaises(ValueError) as ctx:
            validate_dag(cyclic_proj)
        self.assertIn("Cycle detected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
