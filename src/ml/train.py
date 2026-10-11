"""
Neural Network Training and Evaluation Pipeline for Algorithm Selection.

Implements:
  1. Step 0 Dataset Quality Gate (strictly stops if features contain NaN/Inf or classes are missing).
  2. Target definition: best_algorithm in {GA, SA, BCO, PRH, TS, ALL} (Zero PSO).
  3. Strict leakage prevention (excludes all outcome/performance fields).
  4. Model A (Paper-aligned baseline: NC, RF, RS) vs Model B (Extended 30-feature model).
  5. Cross-validation: Grouped/Stratified CV with per-fold StandardScaler.
  6. Architecture selection over controlled candidate set.
  7. Permutation feature importance.
  8. Artifact persistence: models/*.pkl and model_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.features import ProjectFeatures

# ---------------------------------------------------------------------------
# Constants & Allowed Classes
# ---------------------------------------------------------------------------

ALL_CLASSES = ["ALL", "BCO", "GA", "PRH", "SA", "TS"]
PAPER_BASELINE_FEATURES = ["NC", "RF", "RS"]
EXTENDED_FEATURES = ProjectFeatures.feature_names()

LEAKAGE_COLUMNS = {
    "GA_best_ms", "GA_mean_ms", "GA_mean_rt",
    "SA_best_ms", "SA_mean_ms", "SA_mean_rt",
    "BCO_best_ms", "BCO_mean_ms", "BCO_mean_rt",
    "PRH_best_ms", "PRH_mean_ms", "PRH_mean_rt",
    "TS_best_ms", "TS_mean_ms", "TS_mean_rt",
    "best_feasible_makespan", "runtime", "evaluations",
    "convergence_history", "label_basis", "tied_best_algorithms",
    "best_priority_vector", "notes", "makespan", "feasible",
}


# ---------------------------------------------------------------------------
# Step 0: Quality Gate
# ---------------------------------------------------------------------------

@dataclass
class QualityGateResult:
    passed: bool
    total_instances: int
    family_distribution: Dict[str, int]
    label_distribution: Dict[str, int]
    missing_classes: List[str]
    has_nans: bool
    has_infs: bool
    duplicate_feature_vectors: int
    error_message: Optional[str] = None


def run_quality_gate(df: pd.DataFrame, min_samples_per_class: int = 1) -> QualityGateResult:
    """
    Validate dataset readiness before machine learning training.

    Enforces:
      - All 30 features exist and contain finite, non-NaN values.
      - Exactly the allowed classes are present.
      - No required class has zero examples.
      - Sufficient samples exist for training.
    """
    total = len(df)
    families = df["instance_family"].value_counts().to_dict() if "instance_family" in df else {}

    # Check for NaN / Inf in features
    avail_features = [f for f in EXTENDED_FEATURES if f in df.columns]
    feature_df = df[avail_features]
    has_nans = bool(feature_df.isna().any().any())
    has_infs = bool(np.isinf(feature_df.to_numpy(dtype=float)).any()) if not has_nans else False

    # Check duplicate feature vectors
    duplicate_features = int(feature_df.duplicated().sum())

    # Check labels
    if "best_algorithm" not in df.columns:
        return QualityGateResult(
            passed=False,
            total_instances=total,
            family_distribution=families,
            label_distribution={},
            missing_classes=ALL_CLASSES,
            has_nans=has_nans,
            has_infs=has_infs,
            duplicate_feature_vectors=duplicate_features,
            error_message="Target column 'best_algorithm' not found in dataset.",
        )

    # Class counts
    label_counts = {cls_name: 0 for cls_name in ALL_CLASSES}
    actual_counts = df["best_algorithm"].value_counts().to_dict()
    for k, v in actual_counts.items():
        if k in label_counts:
            label_counts[k] = v
        else:
            # Foreign class (e.g. PSO)
            return QualityGateResult(
                passed=False,
                total_instances=total,
                family_distribution=families,
                label_distribution=actual_counts,
                missing_classes=[],
                has_nans=has_nans,
                has_infs=has_infs,
                duplicate_feature_vectors=duplicate_features,
                error_message=f"Unauthorized class '{k}' detected. PSO and non-standard classes are strictly prohibited.",
            )

    missing = [k for k, v in label_counts.items() if v < min_samples_per_class]

    # Verification decision
    errors = []
    if has_nans:
        errors.append("Dataset features contain NaN values.")
    if has_infs:
        errors.append("Dataset features contain Infinite values.")
    if missing:
        errors.append(f"Classes with zero or insufficient examples: {missing}.")

    passed = len(errors) == 0
    err_msg = "; ".join(errors) if errors else None

    return QualityGateResult(
        passed=passed,
        total_instances=total,
        family_distribution=families,
        label_distribution=label_counts,
        missing_classes=missing,
        has_nans=has_nans,
        has_infs=has_infs,
        duplicate_feature_vectors=duplicate_features,
        error_message=err_msg,
    )


# ---------------------------------------------------------------------------
# Evaluation Metrics Helper
# ---------------------------------------------------------------------------

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    """Calculate multi-class evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist()
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

class AlgorithmSelectorTrainer:
    """Trains and evaluates Model A (baseline) and Model B (extended)."""

    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path = Path("models"),
        random_state: int = 42,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.df: Optional[pd.DataFrame] = None
        self.label_encoder = LabelEncoder()

    def load_and_validate(self) -> QualityGateResult:
        """Load dataset and execute Step 0 Quality Gate."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        self.df = pd.read_csv(self.dataset_path)
        gate_res = run_quality_gate(self.df, min_samples_per_class=1)

        print("\n============================================================")
        print("STEP 0 — DATASET QUALITY GATE CHECK")
        print("============================================================")
        print(f"Total instances evaluated: {gate_res.total_instances}")
        print(f"Instance families:         {gate_res.family_distribution}")
        print("\nLabel Distribution:")
        for lbl in ["GA", "SA", "BCO", "PRH", "TS", "ALL"]:
            cnt = gate_res.label_distribution.get(lbl, 0)
            pct = (cnt / gate_res.total_instances * 100.0) if gate_res.total_instances > 0 else 0.0
            print(f"  {lbl:<5} count: {cnt:>3} ({pct:>5.1f}%)")

        print(f"\nMissing / Non-finite features: NaN={gate_res.has_nans}, Inf={gate_res.has_infs}")
        print(f"Duplicate feature vectors:    {gate_res.duplicate_feature_vectors}")

        if not gate_res.passed:
            print("\n------------------------------------------------------------")
            print("QUALITY GATE FAILED:")
            print(f"Reason: {gate_res.error_message}")
            print("\nDataset is not yet suitable for multiclass training.")
            print("TRAINING STOPPED (strict Step 0 safety rule).")
            print("============================================================\n")
        else:
            print("\nQUALITY GATE PASSED: Dataset is ready for training.")
            print("============================================================\n")

        return gate_res

    def run_experiments(
        self,
        cv_splits: int = 5,
        force_train: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute training, cross-validation, and model selection.
        """
        gate = self.load_and_validate()
        if not gate.passed and not force_train:
            return {
                "status": "STOPPED",
                "reason": "Quality gate failed: Dataset is not yet suitable for multiclass training.",
                "quality_gate": asdict(gate),
            }

        # Encode labels
        y_raw = self.df["best_algorithm"].values
        # Ensure all standard classes exist in label encoder classes
        self.label_encoder.fit(ALL_CLASSES)
        y = self.label_encoder.transform(y_raw)

        # Groups for GroupKFold if feature_configuration exists
        has_groups = "feature_configuration" in self.df.columns
        groups = self.df["feature_configuration"].values if has_groups else None

        # Verify no outcome columns are in features
        for col in LEAKAGE_COLUMNS:
            assert col not in PAPER_BASELINE_FEATURES, f"Leakage detected: {col} in baseline"
            assert col not in EXTENDED_FEATURES, f"Leakage detected: {col} in extended"

        X_baseline = self.df[PAPER_BASELINE_FEATURES].values.astype(float)
        X_extended = self.df[EXTENDED_FEATURES].values.astype(float)

        # Candidate architectures
        candidate_archs = [(16,), (32,), (32, 16), (64, 32), (64, 32, 16)]
        print("Evaluating candidate neural-network architectures on Model B...")

        best_arch = (64, 32)
        best_arch_f1 = -1.0

        # Select CV strategy
        # If class counts allow StratifiedKFold (all >= n_splits), use it.
        # Otherwise fallback to KFold with shuffle.
        min_cls_count = min(gate.label_distribution.values()) if gate.label_distribution else 0
        if min_cls_count >= cv_splits:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            cv_name = f"StratifiedKFold({cv_splits})"
        elif has_groups and len(set(groups)) >= cv_splits:
            cv = GroupKFold(n_splits=cv_splits)
            cv_name = f"GroupKFold({cv_splits})"
        else:
            cv = KFold(n_splits=min(cv_splits, len(self.df)), shuffle=True, random_state=self.random_state)
            cv_name = f"KFold({min(cv_splits, len(self.df))})"

        print(f"Validation strategy: {cv_name}")

        arch_results = {}
        for arch in candidate_archs:
            fold_f1s = []
            for train_idx, val_idx in cv.split(X_extended, y, groups=groups):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_extended[train_idx])
                X_va = scaler.transform(X_extended[val_idx])

                clf = MLPClassifier(
                    hidden_layer_sizes=arch,
                    activation="relu",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=self.random_state,
                )
                clf.fit(X_tr, y[train_idx])
                y_pred = clf.predict(X_va)
                fold_f1s.append(f1_score(y[val_idx], y_pred, average="macro", zero_division=0))

            mean_f1 = float(np.mean(fold_f1s))
            arch_results[str(arch)] = round(mean_f1, 4)
            if mean_f1 > best_arch_f1:
                best_arch_f1 = mean_f1
                best_arch = arch

        print(f"Architecture comparison Macro-F1: {arch_results}")
        print(f"Selected architecture: {best_arch} (Macro-F1: {best_arch_f1:.4f})")

        # Cross-validation for Model A (Baseline: NC, RF, RS)
        model_a_preds = np.zeros_like(y)
        for train_idx, val_idx in cv.split(X_baseline, y, groups=groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_baseline[train_idx])
            X_va = scaler.transform(X_baseline[val_idx])
            clf = MLPClassifier(
                hidden_layer_sizes=best_arch,
                activation="relu",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state,
            )
            clf.fit(X_tr, y[train_idx])
            model_a_preds[val_idx] = clf.predict(X_va)

        metrics_a = calculate_metrics(y, model_a_preds, ALL_CLASSES)

        # Cross-validation for Model B (Extended: 30 features)
        model_b_preds = np.zeros_like(y)
        val_fold_scalers = []
        for train_idx, val_idx in cv.split(X_extended, y, groups=groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_extended[train_idx])
            X_va = scaler.transform(X_extended[val_idx])
            clf = MLPClassifier(
                hidden_layer_sizes=best_arch,
                activation="relu",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state,
            )
            clf.fit(X_tr, y[train_idx])
            model_b_preds[val_idx] = clf.predict(X_va)

        metrics_b = calculate_metrics(y, model_b_preds, ALL_CLASSES)

        # Fit final production models on full dataset
        scaler_baseline = StandardScaler().fit(X_baseline)
        model_baseline = MLPClassifier(
            hidden_layer_sizes=best_arch,
            activation="relu",
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        ).fit(scaler_baseline.transform(X_baseline), y)

        scaler_extended = StandardScaler().fit(X_extended)
        model_extended = MLPClassifier(
            hidden_layer_sizes=best_arch,
            activation="relu",
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        ).fit(scaler_extended.transform(X_extended), y)

        # Permutation feature importance on extended model
        perm = permutation_importance(
            model_extended,
            scaler_extended.transform(X_extended),
            y,
            n_repeats=10,
            random_state=self.random_state,
            scoring="accuracy",
        )
        perm_ranking = []
        for i in np.argsort(perm.importances_mean)[::-1]:
            perm_ranking.append({
                "feature": EXTENDED_FEATURES[i],
                "mean_importance": round(float(perm.importances_mean[i]), 5),
                "std_importance": round(float(perm.importances_std[i]), 5),
            })

        # Save artifacts
        with open(self.output_dir / "selector_extended.pkl", "wb") as f:
            pickle.dump(model_extended, f)
        with open(self.output_dir / "scaler_extended.pkl", "wb") as f:
            pickle.dump(scaler_extended, f)
        with open(self.output_dir / "selector_baseline.pkl", "wb") as f:
            pickle.dump(model_baseline, f)
        with open(self.output_dir / "scaler_baseline.pkl", "wb") as f:
            pickle.dump(scaler_baseline, f)
        with open(self.output_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        metadata = {
            "model_version": "1.0",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": "MLPClassifier",
            "solver": "lbfgs",
            "activation": "relu",
            "architecture": list(best_arch),
            "random_state": self.random_state,
            "classes": ALL_CLASSES,
            "features_baseline": PAPER_BASELINE_FEATURES,
            "features_extended": EXTENDED_FEATURES,
            "validation_strategy": cv_name,
            "metrics_model_a_baseline": metrics_a,
            "metrics_model_b_extended": metrics_b,
            "top_features_importance": perm_ranking[:10],
            "confidence_policy": {
                "high_confidence_threshold": 0.80,
                "medium_confidence_threshold": 0.50,
                "policy_rules": [
                    "confidence >= 0.80: execute predicted top algorithm",
                    "0.50 <= confidence < 0.80: execute top 2 predicted algorithms",
                    "confidence < 0.50: execute all 5 algorithms",
                ],
            },
        }

        with open(self.output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Artifacts successfully saved to {self.output_dir}/")

        return {
            "status": "COMPLETED",
            "model_a_metrics": metrics_a,
            "model_b_metrics": metrics_b,
            "best_architecture": best_arch,
            "feature_importance": perm_ranking,
            "metadata_file": str(self.output_dir / "model_metadata.json"),
        }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train AI-RCPSP algorithm selection models.")
    parser.add_argument("--dataset", type=str, default="data/training/training_dataset.csv", help="Path to training CSV.")
    parser.add_argument("--output", type=str, default="models", help="Artifacts directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed.")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV splits.")
    parser.add_argument("--force", action="store_true", help="Force training even if gate fails (testing only).")

    args = parser.parse_args()
    trainer = AlgorithmSelectorTrainer(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        random_state=args.seed,
    )
    res = trainer.run_experiments(cv_splits=args.cv, force_train=args.force)
    if res["status"] == "STOPPED":
        sys.exit(0)


if __name__ == "__main__":
    main()
