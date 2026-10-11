"""
Inference API for Neural Network Algorithm Selection.

Accepts an RCPSP Project, extracts its deterministic project features,
scales them using the trained StandardScaler, executes inference with the
trained MLPClassifier, and applies the confidence execution policy.

Architectural Guarantees:
  - Exactly 5 algorithms + 'ALL' (Zero PSO).
  - No outcome columns used as features.
  - Returns probability distribution across all classes.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.features import ProjectFeatures, extract_features
from src.model import Project


@dataclass
class PredictionResult:
    """Structured algorithm prediction output."""
    predicted_algorithm: str
    confidence: float
    class_probabilities: Dict[str, float]
    feature_vector: Dict[str, float]
    feature_names: List[str]
    model_version: str
    recommended_action: str
    recommended_algorithms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_algorithm": self.predicted_algorithm,
            "confidence": round(self.confidence, 4),
            "class_probabilities": {k: round(v, 4) for k, v in self.class_probabilities.items()},
            "feature_vector": self.feature_vector,
            "feature_names": self.feature_names,
            "model_version": self.model_version,
            "recommended_action": self.recommended_action,
            "recommended_algorithms": self.recommended_algorithms,
        }


class AlgorithmSelectorPredictor:
    """
    Inference engine for project algorithm selection.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/selector_extended.pkl",
        scaler_path: Union[str, Path] = "models/scaler_extended.pkl",
        encoder_path: Union[str, Path] = "models/label_encoder.pkl",
        metadata_path: Union[str, Path] = "models/model_metadata.json",
        high_confidence_thresh: float = 0.80,
        medium_confidence_thresh: float = 0.50,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.encoder_path = Path(encoder_path)
        self.metadata_path = Path(metadata_path)

        self.high_confidence_thresh = high_confidence_thresh
        self.medium_confidence_thresh = medium_confidence_thresh

        self.model = None
        self.scaler = None
        self.encoder = None
        self.metadata = {}
        self.feature_names = ProjectFeatures.feature_names()

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained model artifacts if present."""
        if not (self.model_path.exists() and self.scaler_path.exists() and self.encoder_path.exists()):
            return

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(self.encoder_path, "rb") as f:
            self.encoder = pickle.load(f)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None and self.encoder is not None

    def predict(self, project: Project) -> PredictionResult:
        """
        Predict the best algorithm for an RCPSP project.

        Parameters
        ----------
        project : Project
            The parsed RCPSP project model.

        Returns
        -------
        PredictionResult
            Predicted winner, probability vector, confidence, and recommended action.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Model artifacts not found or model not yet trained. "
                "Ensure dataset passes Step 0 Quality Gate and train.py has completed successfully."
            )

        # 1. Feature extraction (Deterministic 30 features)
        features: ProjectFeatures = extract_features(project)
        feat_dict = features.to_dict()

        # Vector in canonical order
        feat_vals = np.array([feat_dict[fn] for fn in self.feature_names], dtype=float).reshape(1, -1)

        # 2. Preprocessing scaling
        feat_scaled = self.scaler.transform(feat_vals)

        # 3. Model inference
        probs = self.model.predict_proba(feat_scaled)[0]
        classes = self.encoder.classes_

        class_probs = {cls_name: float(probs[i]) for i, cls_name in enumerate(classes)}

        best_idx = int(np.argmax(probs))
        predicted_algo = str(classes[best_idx])
        confidence = float(probs[best_idx])

        # 4. Confidence Execution Policy
        # Rank classes by probability
        ranked_classes = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        # Exclude 'ALL' from candidate execution list if selecting specific algorithms to run
        active_algos = [c for c, _ in ranked_classes if c != "ALL"]

        if predicted_algo == "ALL":
            action = f"Predicted tie ('ALL'): execute all five candidates"
            recommended = ["GA", "SA", "BCO", "PRH", "TS"]
        elif confidence >= self.high_confidence_thresh:
            action = f"High confidence ({confidence:.2f}): execute single predicted winner '{predicted_algo}'"
            recommended = [str(predicted_algo)]
        elif confidence >= self.medium_confidence_thresh:
            top_two = [c for c, _ in ranked_classes[:2]]
            if "ALL" in top_two:
                action = f"Moderate confidence ({confidence:.2f}) includes tie ('ALL'): execute all five candidates"
                recommended = ["GA", "SA", "BCO", "PRH", "TS"]
            else:
                action = f"Moderate confidence ({confidence:.2f}): execute top 2 candidates {top_two}"
                recommended = [str(c) for c in top_two]
        else:
            action = f"Low confidence ({confidence:.2f}): execute all five algorithms"
            recommended = ["GA", "SA", "BCO", "PRH", "TS"]

        return PredictionResult(
            predicted_algorithm=predicted_algo,
            confidence=confidence,
            class_probabilities=class_probs,
            feature_vector=feat_dict,
            feature_names=self.feature_names,
            model_version=self.metadata.get("model_version", "1.0"),
            recommended_action=action,
            recommended_algorithms=recommended,
        )


# ---------------------------------------------------------------------------
# Public Helper Function
# ---------------------------------------------------------------------------

def predict_algorithm(project: Project, model_dir: Union[str, Path] = "models") -> PredictionResult:
    """
    Convenience function to predict algorithm for a Project.

    Parameters
    ----------
    project : Project
        Parsed RCPSP project instance.
    model_dir : Union[str, Path]
        Directory where trained models are located.

    Returns
    -------
    PredictionResult
    """
    m_dir = Path(model_dir)
    predictor = AlgorithmSelectorPredictor(
        model_path=m_dir / "selector_extended.pkl",
        scaler_path=m_dir / "scaler_extended.pkl",
        encoder_path=m_dir / "label_encoder.pkl",
        metadata_path=m_dir / "model_metadata.json",
    )
    return predictor.predict(project)
