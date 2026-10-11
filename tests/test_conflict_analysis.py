"""
Unit tests for src/ml/conflict_analysis.py.

Verifies:
  1. Conflict detection on duplicate feature vectors with conflicting labels.
  2. Consistent duplicate detection (duplicate feature vectors with identical labels).
  3. Unique feature vectors produce zero conflicts.
  4. Empty dataset handling.
"""

import tempfile
import unittest
from pathlib import Path

from src.features import ProjectFeatures
from src.ml.conflict_analysis import analyze_conflicts


class TestConflictAnalysis(unittest.TestCase):

    def test_no_conflicts_on_unique_data(self):
        # Two rows with distinct features
        row1 = {"instance_id": "inst_1", "best_algorithm": "GA", "N": 10, "NC": 1.5, "RF": 0.5, "RS": 0.5}
        row2 = {"instance_id": "inst_2", "best_algorithm": "SA", "N": 20, "NC": 1.8, "RF": 0.75, "RS": 0.2}

        report = analyze_conflicts([row1, row2], feature_columns=["N", "NC", "RF", "RS"])
        self.assertEqual(report.total_instances, 2)
        self.assertEqual(report.unique_feature_vectors, 2)
        self.assertEqual(report.num_duplicate_vectors, 0)
        self.assertEqual(report.num_conflicting_duplicate_groups, 0)
        self.assertEqual(report.conflict_rate, 0.0)

    def test_consistent_duplicates(self):
        # Two rows with identical features and identical labels
        row1 = {"instance_id": "inst_1", "best_algorithm": "GA", "N": 10, "NC": 1.5}
        row2 = {"instance_id": "inst_2", "best_algorithm": "GA", "N": 10, "NC": 1.5}

        report = analyze_conflicts([row1, row2], feature_columns=["N", "NC"])
        self.assertEqual(report.total_instances, 2)
        self.assertEqual(report.unique_feature_vectors, 1)
        self.assertEqual(report.num_duplicate_vectors, 1)
        self.assertEqual(report.num_consistent_duplicate_groups, 1)
        self.assertEqual(report.num_conflicting_duplicate_groups, 0)

    def test_conflicting_duplicates_detected(self):
        # Two rows with identical features but conflicting labels
        row1 = {"instance_id": "inst_1", "best_algorithm": "GA", "N": 30, "NC": 1.6, "RF": 0.5, "RS": 0.5}
        row2 = {"instance_id": "inst_2", "best_algorithm": "TS", "N": 30, "NC": 1.6, "RF": 0.5, "RS": 0.5}
        row3 = {"instance_id": "inst_3", "best_algorithm": "BCO", "N": 30, "NC": 1.6, "RF": 0.5, "RS": 0.5}

        report = analyze_conflicts([row1, row2, row3], feature_columns=["N", "NC", "RF", "RS"])
        self.assertEqual(report.total_instances, 3)
        self.assertEqual(report.unique_feature_vectors, 1)
        self.assertEqual(report.num_duplicate_vectors, 1)
        self.assertEqual(report.num_conflicting_duplicate_groups, 1)
        self.assertEqual(report.num_consistent_duplicate_groups, 0)

        # Check conflict group details
        cg = report.conflict_groups[0]
        self.assertTrue(cg.is_conflicting)
        self.assertEqual(cg.unique_labels, ["BCO", "GA", "TS"])

    def test_empty_dataset(self):
        report = analyze_conflicts([])
        self.assertEqual(report.total_instances, 0)
        self.assertEqual(report.conflict_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
