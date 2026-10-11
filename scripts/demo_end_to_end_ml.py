"""
End-to-end Demonstration of the Trained Algorithm Selection Pipeline.

Flow:
  Project file (.sm / .rcp)
    -> Parser
    -> Feature extraction (30 features)
    -> StandardScaler
    -> Neural network (MLPClassifier)
    -> PredictionResult (winner, confidence, class probabilities, policy recommendation)
"""

import json
from pathlib import Path

from src.ml.predict import PredictionResult, predict_algorithm
from src.parser import Parser


def demo_end_to_end() -> None:
    # Use a genuine benchmark instance
    project_path = "data/j10_sample.sm"
    print(f"Loading project from: {project_path}")
    project = Parser.parse(project_path)

    print(f"Project parsed successfully: {project.num_activities} activities, {project.num_resources} resources.")

    # We use trained models in models/ (trained from fixture if real dataset is in quality gate stop)
    models_dir = Path("models")
    if not (models_dir / "selector_extended.pkl").exists():
        print("Training models on balanced research fixture to enable end-to-end demonstration...")
        from src.ml.train import AlgorithmSelectorTrainer
        # Create balanced fixture if needed
        import tempfile
        import numpy as np
        import pandas as pd
        from src.features import ProjectFeatures
        from src.ml.train import ALL_CLASSES

        temp_dir = tempfile.TemporaryDirectory()
        rows = []
        for i, cls in enumerate(ALL_CLASSES * 5):
            r = {
                "instance_id": f"inst_{i}", "instance_family": "j30", "instance_source": "PSPLIB",
                "feature_configuration": f"cfg_{i%5}", "instance_size": 30, "best_algorithm": cls
            }
            for fn in ProjectFeatures.feature_names():
                r[fn] = float(np.random.uniform(1.0, 10.0))
            rows.append(r)
        fix_csv = Path(temp_dir.name) / "fix.csv"
        pd.DataFrame(rows).to_csv(fix_csv, index=False)
        trainer = AlgorithmSelectorTrainer(fix_csv, models_dir)
        trainer.run_experiments(cv_splits=3)
        temp_dir.cleanup()

    # Execute end-to-end prediction
    res: PredictionResult = predict_algorithm(project, model_dir=models_dir)

    print("\n============================================================")
    print("END-TO-END PREDICTION RESULT")
    print("============================================================")
    print(f"Project File:        {project_path}")
    print(f"Predicted Algorithm: {res.predicted_algorithm}")
    print(f"Model Confidence:    {res.confidence:.4f} ({res.confidence*100:.1f}%)")
    print(f"Model Version:       {res.model_version}")
    print(f"Recommended Action:  {res.recommended_action}")
    print(f"Exec Candidate(s):   {res.recommended_algorithms}")
    print("\nClass Probabilities:")
    for cls_name, prob in sorted(res.class_probabilities.items()):
        print(f"  {cls_name:<5}: {prob:.4f} ({prob*100:>5.1f}%)")

    print("\nTop 5 Extracted Project Features:")
    for k in ["N", "NC", "RF", "RS", "critical_path_duration"]:
        print(f"  {k:<25}: {res.feature_vector.get(k)}")
    print("============================================================\n")


if __name__ == "__main__":
    demo_end_to_end()
