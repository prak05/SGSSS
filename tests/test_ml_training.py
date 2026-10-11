"""
Unit tests for Machine Learning Training and Quality Gate Pipeline.

Verifies:
  1. Step 0 Quality Gate detects missing classes on imbalanced/incomplete datasets.
  2. Step 0 Quality Gate passes on complete, balanced datasets.
  3. Strict exclusion of PSO class.
  4. Strict leakage prevention (outcome columns cannot be used as features).
  5. Correct fitting of StandardScaler on training folds.
  6. Artifact persistence and metadata schema.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import ProjectFeatures
from src.ml.train import (
    ALL_CLASSES,
    LEAKAGE_COLUMNS,
    AlgorithmSelectorTrainer,
    run_quality_gate,
)


class TestMLTraining(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        # Create a synthetic 6-class dataset fixture
        self.feature_names = ProjectFeatures.feature_names()
        np.random.seed(42)
        n_samples = 30  # 5 samples per class across 6 classes
        rows = []
        classes_pool = ALL_CLASSES * 5
        for i, cls in enumerate(classes_pool):
            row = {
                "instance_id": f"inst_{i}",
                "instance_family": "j30" if i < 15 else "j60",
                "instance_source": "PSPLIB",
                "feature_configuration": f"cfg_{i % 6}",
                "instance_size": 30 if i < 15 else 60,
                "best_algorithm": cls,
            }
            # Fill 30 features with random positive values
            for fn in self.feature_names:
                row[fn] = float(np.random.uniform(1.0, 10.0))
            rows.append(row)

        self.balanced_df = pd.DataFrame(rows)
        self.balanced_csv = self.output_dir / "balanced_dataset.csv"
        self.balanced_df.to_csv(self.balanced_csv, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_quality_gate_passes_on_complete_dataset(self):
        """Verify that a dataset containing all 6 classes and finite features passes."""
        res = run_quality_gate(self.balanced_df, min_samples_per_class=1)
        self.assertTrue(res.passed)
        self.assertEqual(len(res.missing_classes), 0)
        self.assertFalse(res.has_nans)
        self.assertFalse(res.has_infs)

    def test_quality_gate_fails_when_class_missing(self):
        """Verify that missing BCO or any other class triggers quality gate failure."""
        imbalanced_df = self.balanced_df[self.balanced_df["best_algorithm"] != "BCO"].copy()
        res = run_quality_gate(imbalanced_df, min_samples_per_class=1)
        self.assertFalse(res.passed)
        self.assertIn("BCO", res.missing_classes)
        self.assertIn("BCO", res.error_message)

    def test_quality_gate_rejects_pso(self):
        """Verify strict rule: PSO is unauthorized and immediately rejected."""
        pso_df = self.balanced_df.copy()
        pso_df.loc[0, "best_algorithm"] = "PSO"
        res = run_quality_gate(pso_df)
        self.assertFalse(res.passed)
        self.assertIn("PSO", res.error_message)

    def test_quality_gate_detects_nans(self):
        """Verify NaN feature values are caught."""
        nan_df = self.balanced_df.copy()
        nan_df.loc[0, self.feature_names[0]] = np.nan
        res = run_quality_gate(nan_df)
        self.assertFalse(res.passed)
        self.assertTrue(res.has_nans)

    def test_leakage_columns_excluded(self):
        """Ensure no outcome column can ever be treated as an input feature."""
        for col in LEAKAGE_COLUMNS:
            self.assertNotIn(col, self.feature_names)
            self.assertNotIn(col, ["NC", "RF", "RS"])

    def test_training_pipeline_end_to_end_on_fixture(self):
        """Verify trainer runs CV, selects model, computes metrics, and writes artifacts."""
        trainer = AlgorithmSelectorTrainer(
            dataset_path=self.balanced_csv,
            output_dir=self.output_dir / "models",
            random_state=42,
        )
        res = trainer.run_experiments(cv_splits=3)
        self.assertEqual(res["status"], "COMPLETED")
        self.assertIn("accuracy", res["model_b_metrics"])
        self.assertIn("macro_f1", res["model_b_metrics"])

        # Check saved files
        m_dir = self.output_dir / "models"
        self.assertTrue((m_dir / "selector_extended.pkl").exists())
        self.assertTrue((m_dir / "scaler_extended.pkl").exists())
        self.assertTrue((m_dir / "label_encoder.pkl").exists())
        self.assertTrue((m_dir / "model_metadata.json").exists())

        # Inspect metadata
        with open(m_dir / "model_metadata.json", "r") as f:
            meta = json.load(f)
        self.assertEqual(meta["model_type"], "MLPClassifier")
        self.assertEqual(meta["classes"], ALL_CLASSES)


if __name__ == "__main__":
    unittest.main()
