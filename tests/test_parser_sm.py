"""
Tests for the PSPLIB .sm parser.

Fixture: data/j10_sample.sm
  - 10 real activities + dummy source (job 1) + dummy sink (job 12) = 12 activities
  - 2 renewable resources: R1 capacity=4, R2 capacity=3
  - Known durations, precedences, and resource requirements
"""

import unittest
from src.parser import Parser
from src.model import Project


class TestSMParser(unittest.TestCase):

    def setUp(self):
        self.project: Project = Parser.parse("data/j10_sample.sm")

    # ---- Structure ----

    def test_activity_count(self):
        """12 jobs including dummy source and dummy sink."""
        self.assertEqual(self.project.num_activities, 12)

    def test_resource_count(self):
        """2 renewable resources."""
        self.assertEqual(self.project.num_resources, 2)

    def test_activity_ids_are_1_based(self):
        ids = sorted(self.project.activities.keys())
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 12)

    # ---- Dummy source (job 1) ----

    def test_source_duration_zero(self):
        self.assertEqual(self.project.activities[1].duration, 0)

    def test_source_has_no_predecessors(self):
        self.assertEqual(self.project.activities[1].predecessors, [])

    def test_source_has_successors(self):
        """Source must have at least one successor."""
        self.assertGreater(len(self.project.activities[1].successors), 0)

    def test_source_resource_requirements_zero(self):
        for req in self.project.activities[1].resource_requirements.values():
            self.assertEqual(req, 0)

    # ---- Dummy sink (job 12) ----

    def test_sink_duration_zero(self):
        self.assertEqual(self.project.activities[12].duration, 0)

    def test_sink_has_no_successors(self):
        self.assertEqual(self.project.activities[12].successors, [])

    def test_sink_has_predecessors(self):
        self.assertGreater(len(self.project.activities[12].predecessors), 0)

    # ---- Resource capacities ----

    def test_r1_capacity(self):
        self.assertEqual(self.project.resources["R1"].capacity, 4)

    def test_r2_capacity(self):
        self.assertEqual(self.project.resources["R2"].capacity, 3)

    # ---- Specific activity durations ----

    def test_job2_duration(self):
        self.assertEqual(self.project.activities[2].duration, 3)

    def test_job3_duration(self):
        self.assertEqual(self.project.activities[3].duration, 2)

    def test_job4_duration(self):
        self.assertEqual(self.project.activities[4].duration, 5)

    def test_job8_duration(self):
        self.assertEqual(self.project.activities[8].duration, 6)

    def test_job11_duration(self):
        self.assertEqual(self.project.activities[11].duration, 3)

    # ---- Specific precedence relations ----

    def test_job2_predecessor_is_source(self):
        self.assertIn(1, self.project.activities[2].predecessors)

    def test_job3_predecessor_is_source(self):
        self.assertIn(1, self.project.activities[3].predecessors)

    def test_job5_predecessors(self):
        """Job 5 follows jobs 2 and 3."""
        preds = set(self.project.activities[5].predecessors)
        self.assertIn(2, preds)
        self.assertIn(3, preds)

    def test_job7_predecessors(self):
        """Job 7 follows jobs 4 and 5."""
        preds = set(self.project.activities[7].predecessors)
        self.assertIn(4, preds)
        self.assertIn(5, preds)

    def test_job10_predecessors(self):
        """Job 10 follows jobs 7 and 8."""
        preds = set(self.project.activities[10].predecessors)
        self.assertIn(7, preds)
        self.assertIn(8, preds)

    def test_sink_predecessor(self):
        """Sink (job 12) follows job 11."""
        self.assertIn(11, self.project.activities[12].predecessors)

    # ---- Resource requirements ----

    def test_job2_r1_requirement(self):
        self.assertEqual(self.project.activities[2].resource_requirements["R1"], 3)

    def test_job2_r2_requirement(self):
        self.assertEqual(self.project.activities[2].resource_requirements["R2"], 0)

    def test_job4_r2_requirement(self):
        """Job 4 requires 3 units of R2 — equal to full capacity."""
        self.assertEqual(self.project.activities[4].resource_requirements["R2"], 3)

    def test_job8_r2_requirement(self):
        self.assertEqual(self.project.activities[8].resource_requirements["R2"], 2)

    # ---- Successor consistency ----

    def test_successors_consistent_with_predecessors(self):
        """For every predecessor link a→b, b must appear in a.successors."""
        for a_id, act in self.project.activities.items():
            for p_id in act.predecessors:
                self.assertIn(
                    a_id,
                    self.project.activities[p_id].successors,
                    msg=f"Activity {a_id} listed {p_id} as predecessor but "
                        f"{p_id}.successors does not include {a_id}",
                )

    # ---- Error handling ----

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            Parser.parse("data/does_not_exist.sm")

    def test_unknown_extension_raises(self):
        """Parser must reject a known-existing file with an unrecognised extension."""
        import shutil, os
        # Create a copy of the fixture with a fake extension so the file exists
        src  = "data/j10_sample.sm"
        fake = "data/j10_sample.xyz"
        shutil.copy(src, fake)
        try:
            with self.assertRaises(ValueError):
                Parser.parse(fake)
        finally:
            os.remove(fake)


class TestSMParserSchedulability(unittest.TestCase):
    """
    Verify the j10 instance can actually be scheduled by the SSGS,
    producing a feasible result.
    """

    def setUp(self):
        from src.evaluation import Evaluator
        import random
        self.project = Parser.parse("data/j10_sample.sm")
        self.Evaluator = Evaluator
        rng = random.Random(0)
        self.priority_vector = {
            a_id: rng.random() for a_id in self.project.activities
        }

    def test_ssgs_produces_feasible_schedule(self):
        from src.ssgs import SSGS
        schedule, makespan = SSGS.generate_schedule(
            self.project, self.priority_vector
        )
        self.assertEqual(len(schedule), self.project.num_activities)
        feasible = self.Evaluator.check_constraints(self.project, schedule)
        self.assertTrue(feasible)
        self.assertGreater(makespan, 0)

    def test_all_activities_scheduled(self):
        from src.ssgs import SSGS
        schedule, _ = SSGS.generate_schedule(self.project, self.priority_vector)
        scheduled_ids = {e.activity_id for e in schedule}
        expected_ids  = set(self.project.activities.keys())
        self.assertEqual(scheduled_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
