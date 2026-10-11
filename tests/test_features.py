"""
Unit tests for the RCPSP feature extraction module (src/features.py) and
dataset serialization utility (src/ml/dataset.py).
"""

import math
import tempfile
import unittest
from pathlib import Path

from src.features import (
    ProjectFeatures,
    compute_cpm_schedule,
    extract_features,
    is_dummy_activity,
    validate_dag,
)
from src.ml.dataset import export_features_to_csv, load_features_from_csv
from src.model import Activity, Project, Resource
from src.parser import Parser


class TestHandCraftedMiniProject(unittest.TestCase):
    """
    Tests on a hand-calculated miniature project where all feature values
    are analytically proven.
    """

    def setUp(self):
        # 5 activities: 1 (start dummy), 2, 3, 4, 5 (finish dummy)
        # 2 resources: R1 (cap=4), R2 (cap=2)
        self.project = Project()
        self.project.resources["R1"] = Resource(id="R1", capacity=4)
        self.project.resources["R2"] = Resource(id="R2", capacity=2)

        self.project.activities[1] = Activity(
            id=1, duration=0, predecessors=[], resource_requirements={"R1": 0, "R2": 0}
        )
        self.project.activities[2] = Activity(
            id=2, duration=4, predecessors=[1], resource_requirements={"R1": 2, "R2": 1}
        )
        self.project.activities[3] = Activity(
            id=3, duration=2, predecessors=[1], resource_requirements={"R1": 1, "R2": 0}
        )
        self.project.activities[4] = Activity(
            id=4, duration=3, predecessors=[2, 3], resource_requirements={"R1": 2, "R2": 2}
        )
        self.project.activities[5] = Activity(
            id=5, duration=0, predecessors=[4], resource_requirements={"R1": 0, "R2": 0}
        )

        # Set successors
        for act in self.project.activities.values():
            act.successors = []
        for a_id, act in self.project.activities.items():
            for p_id in act.predecessors:
                self.project.activities[p_id].successors.append(a_id)

        self.features = extract_features(self.project)

    def test_paper_n_activity_count(self):
        """N must exclude dummy start and dummy finish (3 non-dummy)."""
        self.assertEqual(self.features.N, 3)
        self.assertEqual(self.features.activity_count, 5)

    def test_dummy_activity_detection(self):
        self.assertTrue(is_dummy_activity(self.project.activities[1]))
        self.assertTrue(is_dummy_activity(self.project.activities[5]))
        self.assertFalse(is_dummy_activity(self.project.activities[2]))
        self.assertFalse(is_dummy_activity(self.project.activities[3]))
        self.assertFalse(is_dummy_activity(self.project.activities[4]))

    def test_paper_nc_network_complexity(self):
        """5 edges / 3 non-dummy activities = 1.666667."""
        self.assertEqual(self.features.precedence_edge_count, 5)
        self.assertAlmostEqual(self.features.NC, 5.0 / 3.0, places=5)

    def test_paper_rf_resource_factor(self):
        """Positive resource requests = 5 out of 6 possible -> 5/6 = 0.833333."""
        self.assertAlmostEqual(self.features.RF, 5.0 / 6.0, places=5)

    def test_paper_rs_resource_strength(self):
        """
        R1: k_min=2, k_max=3 (at t in [0,2)), cap=4 -> rs_1 = (4-2)/(3-2) = 2.0 -> clamped to 1.0.
        R2: k_min=2, k_max=2, cap=2 -> rs_2 = 1.0.
        RS = (1.0 + 1.0) / 2 = 1.0.
        """
        self.assertAlmostEqual(self.features.RS, 1.0, places=5)

    def test_durations(self):
        """Durations: [4, 2, 3]. Sum = 9, avg = 3.0, min = 2, max = 4."""
        self.assertEqual(self.features.total_duration, 9)
        self.assertAlmostEqual(self.features.average_duration, 3.0, places=5)
        self.assertEqual(self.features.minimum_duration, 2)
        self.assertEqual(self.features.maximum_duration, 4)
        expected_std = math.sqrt(2.0 / 3.0)
        self.assertAlmostEqual(self.features.duration_std, expected_std, places=5)

    def test_critical_path(self):
        """Critical path duration = 7 (jobs 1 -> 2 -> 4 -> 5)."""
        self.assertEqual(self.features.critical_path_duration, 7)
        # Non-dummy critical activities: 2 and 4 (count = 2)
        self.assertEqual(self.features.critical_path_activity_count, 2)

    def test_parallelism(self):
        """Total duration 9 / critical path 7 = 9/7 = 1.285714."""
        self.assertAlmostEqual(self.features.estimated_parallelism, 9.0 / 7.0, places=5)
        self.assertAlmostEqual(self.features.average_parallel_activities, 9.0 / 7.0, places=5)

    def test_network_depth(self):
        """Path 1 -> 2 -> 4 -> 5 has 3 edges."""
        self.assertEqual(self.features.network_depth, 3)

    def test_sources_and_sinks(self):
        self.assertEqual(self.features.source_count, 1)
        self.assertEqual(self.features.sink_count, 1)

    def test_to_vector_and_names_length_match(self):
        vec = self.features.to_vector()
        names = ProjectFeatures.feature_names()
        self.assertEqual(len(vec), len(names))
        for val in vec:
            self.assertTrue(math.isfinite(val))


class TestFeatureValidationErrors(unittest.TestCase):
    """Verify DAG validation raises descriptive errors."""

    def test_cycle_detection(self):
        project = Project()
        project.activities[1] = Activity(id=1, duration=1, predecessors=[2])
        project.activities[2] = Activity(id=2, duration=1, predecessors=[1])
        project.activities[1].successors = [2]
        project.activities[2].successors = [1]

        with self.assertRaises(ValueError) as ctx:
            extract_features(project)
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_missing_predecessor(self):
        project = Project()
        project.activities[1] = Activity(id=1, duration=1, predecessors=[999])

        with self.assertRaises(ValueError) as ctx:
            extract_features(project)
        self.assertIn("non-existent predecessor", str(ctx.exception).lower())

    def test_empty_project(self):
        project = Project()
        with self.assertRaises(ValueError) as ctx:
            extract_features(project)
        self.assertIn("zero activities", str(ctx.exception).lower())


class TestFeaturesOnFixtures(unittest.TestCase):
    """Verify feature extraction on real benchmark instances (.rcp and .sm)."""

    def test_features_on_sample_rcp(self):
        project = Parser.parse("data/sample.rcp")
        feats = extract_features(project)

        self.assertIsInstance(feats, ProjectFeatures)
        self.assertGreater(feats.N, 0)
        self.assertGreater(feats.NC, 0.0)
        self.assertGreaterEqual(feats.RF, 0.0)
        self.assertLessEqual(feats.RF, 1.0)
        self.assertGreaterEqual(feats.RS, 0.0)
        self.assertLessEqual(feats.RS, 1.0)

        # Determinism check
        feats2 = extract_features(project)
        self.assertEqual(feats.to_vector(), feats2.to_vector())

    def test_features_on_j10_sample(self):
        project = Parser.parse("data/j10_sample.sm")
        feats = extract_features(project)

        # j10 has 12 activities, 10 non-dummy (jobs 2..11)
        self.assertEqual(feats.N, 10)
        self.assertEqual(feats.activity_count, 12)
        self.assertEqual(feats.resource_count, 2)

        # NC = 15 edges / 10 = 1.5
        self.assertAlmostEqual(feats.NC, 1.5, places=5)

        # RF = 0.8
        self.assertAlmostEqual(feats.RF, 0.8, places=5)

        # RS = 0.25 (R1: 0.5, R2: 0.0)
        self.assertAlmostEqual(feats.RS, 0.25, places=5)

        # Critical path duration = 20
        self.assertEqual(feats.critical_path_duration, 20)

        # All values finite
        for val in feats.to_vector():
            self.assertTrue(math.isfinite(val))

    def test_features_on_j30_sample(self):
        project = Parser.parse("data/j30_sample.sm")
        feats = extract_features(project)

        # Standard J30 instance has 30 real activities, 4 resources
        self.assertEqual(feats.N, 30)
        self.assertEqual(feats.activity_count, 32)
        self.assertEqual(feats.resource_count, 4)

        self.assertGreater(feats.NC, 0.0)
        self.assertGreater(feats.RF, 0.0)
        self.assertLessEqual(feats.RF, 1.0)
        self.assertGreaterEqual(feats.RS, 0.0)
        self.assertLessEqual(feats.RS, 1.0)

        for val in feats.to_vector():
            self.assertTrue(math.isfinite(val))


class TestDatasetExport(unittest.TestCase):
    """Verify CSV export and loading in src/ml/dataset.py."""

    def test_export_and_load_csv(self):
        project = Parser.parse("data/j10_sample.sm")
        feats = extract_features(project)

        record = {"instance_id": "j10_sample", **feats.to_dict()}

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "features.csv"
            export_features_to_csv([record], csv_path)

            self.assertTrue(csv_path.exists())
            loaded = load_features_from_csv(csv_path)

            self.assertEqual(len(loaded), 1)
            loaded_rec = loaded[0]
            self.assertEqual(loaded_rec["instance_id"], "j10_sample")
            self.assertEqual(loaded_rec["N"], 10)
            self.assertAlmostEqual(float(loaded_rec["NC"]), 1.5, places=5)
            self.assertAlmostEqual(float(loaded_rec["RF"]), 0.8, places=5)
            self.assertAlmostEqual(float(loaded_rec["RS"]), 0.25, places=5)


if __name__ == "__main__":
    unittest.main()
