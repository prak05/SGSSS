"""
Unit tests for Machine Learning Prediction API.

Verifies:
  1. Prediction on Project objects.
  2. Output schema: predicted_algorithm, confidence, class_probabilities.
  3. Probability vector sums to 1.0.
  4. Confidence equals max class probability.
  5. Correct adherence to confidence policy thresholds.
  6. Unready model raises clear RuntimeError.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import ProjectFeatures
from src.ml.predict import AlgorithmSelectorPredictor, PredictionResult, predict_algorithm
from src.ml.train import ALL_CLASSES, AlgorithmSelectorTrainer
from src.parser import Parser


class TestMLPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a trained model fixture in a temporary directory
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.models_dir = Path(cls.temp_dir.name) / "models"

        feature_names = ProjectFeatures.feature_names()
        np.random.seed(42)
        rows = []
        for i, cls_name in enumerate(ALL_CLASSES * 5):
            row = {
                "instance_id": f"fixture_{i}",
                "instance_family": "j30",
                "instance_source": "PSPLIB",
                "feature_configuration": f"cfg_{i % 5}",
                "instance_size": 30,
                "best_algorithm": cls_name,
            }
            for fn in feature_names:
                row[fn] = float(np.random.uniform(1.0, 10.0))
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = Path(cls.temp_dir.name) / "fixture.csv"
        df.to_csv(csv_path, index=False)

        trainer = AlgorithmSelectorTrainer(
            dataset_path=csv_path,
            output_dir=cls.models_dir,
            random_state=42,
        )
        trainer.run_experiments(cv_splits=3)

        cls.sample_project = Parser.parse("data/sample.rcp")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_predict_algorithm_end_to_end(self):
        """Test prediction from Project object directly."""
        res: PredictionResult = predict_algorithm(
            project=self.sample_project,
            model_dir=self.models_dir,
        )
        self.assertIsInstance(res, PredictionResult)
        self.assertIn(res.predicted_algorithm, ALL_CLASSES)

        # Probabilities sum to 1.0
        prob_sum = sum(res.class_probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

        # Confidence matches max probability
        max_prob = max(res.class_probabilities.values())
        self.assertAlmostEqual(res.confidence, max_prob, places=4)

        # Feature vector has all 30 features
        self.assertEqual(len(res.feature_vector), 30)

    def test_confidence_policy_tiers(self):
        """Verify confidence policy actions."""
        predictor = AlgorithmSelectorPredictor(
            model_path=self.models_dir / "selector_extended.pkl",
            scaler_path=self.models_dir / "scaler_extended.pkl",
            encoder_path=self.models_dir / "label_encoder.pkl",
            metadata_path=self.models_dir / "model_metadata.json",
            high_confidence_thresh=0.999,  # Force medium/low tier
            medium_confidence_thresh=0.001,
        )
        res = predictor.predict(self.sample_project)
        if res.predicted_algorithm == "ALL":
            self.assertIn("Predicted tie ('ALL')", res.recommended_action)
            self.assertEqual(len(res.recommended_algorithms), 5)
        else:
            self.assertIn("Moderate confidence", res.recommended_action)
            self.assertLessEqual(len(res.recommended_algorithms), 2)

    def test_unready_model_raises_error(self):
        """Verify clear error when model artifacts do not exist."""
        empty_predictor = AlgorithmSelectorPredictor(
            model_path=Path("nonexistent/selector.pkl"),
            scaler_path=Path("nonexistent/scaler.pkl"),
            encoder_path=Path("nonexistent/encoder.pkl"),
        )
        self.assertFalse(empty_predictor.is_ready)
        with self.assertRaises(RuntimeError) as ctx:
            empty_predictor.predict(self.sample_project)
        self.assertIn("Model artifacts not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
