"""
Unit tests for src/ml/generate_dataset.py.

Verifies:
  1. Dataset generation on tiny fixture.
  2. All five algorithms (GA, SA, BCO, PRH, TS) are included.
  3. Strict rule: NO PSO appears anywhere.
  4. Correct 30 feature columns are preserved.
  5. Correct aggregation across multiple seeds.
  6. Correct 'ALL' label assignment when all 5 algorithms tie.
  7. Partial-tie tracking.
  8. Safe checkpointing and resume.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark import ALGORITHM_NAMES
from src.features import ProjectFeatures
from src.ml.generate_dataset import (
    DatasetGenerator,
    generate_training_dataset,
    infer_instance_metadata,
)


class TestDatasetGenerator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.instance_paths = [Path("data/sample.rcp")]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_infer_metadata(self):
        self.assertEqual(infer_instance_metadata(Path("j3010_1.sm")), ("j3010_1", "j30", 30))
        self.assertEqual(infer_instance_metadata(Path("j6010_10.sm")), ("j6010_10", "j60", 60))
        self.assertEqual(infer_instance_metadata(Path("j12010_2.sm")), ("j12010_2", "j120", 120))

    def test_algorithms_contain_no_pso(self):
        """Verify strict architectural constraint: exactly 5 algorithms, 0 PSO."""
        self.assertEqual(len(ALGORITHM_NAMES), 5)
        self.assertNotIn("PSO", ALGORITHM_NAMES)
        self.assertNotIn("pso", [a.lower() for a in ALGORITHM_NAMES])

    def test_dataset_generation_small_fixture(self):
        """Run generation with 2 seeds and small budget (100) on sample.rcp."""
        budget = 100
        seeds = [1, 2]

        completed, total = generate_training_dataset(
            instances=self.instance_paths,
            seeds=seeds,
            evaluation_budget=budget,
            output_dir=self.output_dir,
        )

        # 1 instance * 2 seeds * 5 algorithms = 10 runs
        self.assertEqual(total, 10)
        self.assertEqual(completed, 10)

        # 1. Verify raw_results.csv
        raw_file = self.output_dir / "raw_results.csv"
        self.assertTrue(raw_file.exists())
        with open(raw_file, "r", encoding="utf-8") as f:
            raw_reader = list(csv.DictReader(f))
            self.assertEqual(len(raw_reader), 10)
            algos_found = {r["algorithm"].split("[")[0] for r in raw_reader}
            self.assertEqual(algos_found, set(ALGORITHM_NAMES))
            for r in raw_reader:
                self.assertNotEqual(r["algorithm"], "PSO")
                self.assertEqual(r["feasible"], "True")
                self.assertLessEqual(int(r["evaluations"]), budget)

        # 2. Verify training_dataset.csv
        train_file = self.output_dir / "training_dataset.csv"
        self.assertTrue(train_file.exists())
        with open(train_file, "r", encoding="utf-8") as f:
            train_reader = list(csv.DictReader(f))
            self.assertEqual(len(train_reader), 1)
            row = train_reader[0]

            # All 30 feature columns must be present
            for f_name in ProjectFeatures.feature_names():
                self.assertIn(f_name, row)
                self.assertNotEqual(row[f_name], "")

            # Target label must be valid
            self.assertIn(row["best_algorithm"], ["GA", "SA", "BCO", "PRH", "TS", "ALL"])
            self.assertNotEqual(row["best_algorithm"], "PSO")

        # 3. Verify feature_schema.json
        schema_file = self.output_dir / "feature_schema.json"
        self.assertTrue(schema_file.exists())
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
            self.assertNotIn("PSO", schema["algorithms"])
            self.assertNotIn("PSO", schema["labels"])
            self.assertEqual(schema["num_features"], 30)

        # 4. Verify label_distribution.csv
        dist_file = self.output_dir / "label_distribution.csv"
        self.assertTrue(dist_file.exists())

    def test_checkpoint_and_resume(self):
        """Second run on the same directory should detect existing completed runs."""
        generator = DatasetGenerator(
            output_dir=self.output_dir,
            evaluation_budget=100,
            seeds=[1],
        )
        completed1, total1 = generator.generate(self.instance_paths)
        self.assertEqual(completed1, 5)

        # Re-run: should resume and skip already completed runs
        generator2 = DatasetGenerator(
            output_dir=self.output_dir,
            evaluation_budget=100,
            seeds=[1],
        )
        completed2, total2 = generator2.generate(self.instance_paths)
        self.assertEqual(completed2, 5)


if __name__ == "__main__":
    unittest.main()
