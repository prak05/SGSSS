"""
Conflict Analysis for RCPSP Training Datasets.

Examines whether distinct project instances share identical or near-identical
feature vectors, and categorizes whether they produce:
  1. Identical feature vectors with identical labels (consistent duplicates).
  2. Identical feature vectors with conflicting labels (one-to-many conflicts).

Background:
  As discussed in Mitsou & Koulinas (2026), project instances generated with
  identical macroscopic parameters (N, NC, RF, RS) can produce label conflicts
  due to latent network differences. Our extended 30-feature vector provides
  richer discriminative power, but any remaining conflicts must be measured
  and documented transparently.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.features import ProjectFeatures


@dataclass
class ConflictGroup:
    """A cluster of project instances sharing the exact same feature vector."""
    feature_signature: str
    instances: List[str]
    labels: List[str]
    is_conflicting: bool

    @property
    def unique_labels(self) -> List[str]:
        return sorted(list(set(self.labels)))


@dataclass
class ConflictReport:
    """Comprehensive summary of dataset feature-space consistency."""
    total_instances: int
    unique_feature_vectors: int
    num_duplicate_vectors: int
    num_consistent_duplicate_groups: int
    num_conflicting_duplicate_groups: int
    conflict_groups: List[ConflictGroup] = field(default_factory=list)

    @property
    def conflict_rate(self) -> float:
        return (
            self.num_conflicting_duplicate_groups / self.total_instances
            if self.total_instances > 0
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_instances": self.total_instances,
            "unique_feature_vectors": self.unique_feature_vectors,
            "num_duplicate_vectors": self.num_duplicate_vectors,
            "num_consistent_duplicate_groups": self.num_consistent_duplicate_groups,
            "num_conflicting_duplicate_groups": self.num_conflicting_duplicate_groups,
            "conflict_rate": round(self.conflict_rate, 4),
            "conflicts": [
                {
                    "instances": cg.instances,
                    "labels": cg.labels,
                    "unique_labels": cg.unique_labels,
                    "is_conflicting": cg.is_conflicting,
                }
                for cg in self.conflict_groups
                if cg.is_conflicting
            ],
        }

    def to_text(self) -> str:
        lines = [
            "============================================================",
            "FEATURE-SPACE CONFLICT ANALYSIS REPORT",
            "============================================================",
            f"Total instances evaluated: {self.total_instances}",
            f"Unique feature vectors:    {self.unique_feature_vectors}",
            f"Duplicate vector groups:   {self.num_duplicate_vectors}",
            f"  - Consistent duplicates: {self.num_consistent_duplicate_groups}",
            f"  - Conflicting duplicates:  {self.num_conflicting_duplicate_groups}",
            f"Conflict rate:             {self.conflict_rate:.2%}",
            "------------------------------------------------------------",
        ]

        conflicts = [cg for cg in self.conflict_groups if cg.is_conflicting]
        if conflicts:
            lines.append("Conflicting Instances Found:")
            for idx, cg in enumerate(conflicts, 1):
                lines.append(
                    f"  Group {idx}: {len(cg.instances)} instances -> "
                    f"Labels: {cg.unique_labels}"
                )
                for inst, lbl in zip(cg.instances, cg.labels):
                    lines.append(f"    - {inst}: {lbl}")
        else:
            lines.append("No conflicting feature vectors found (all feature vectors are unique or consistent).")

        lines.append("============================================================")
        return "\n".join(lines)


def analyze_conflicts(
    dataset: Union[str, Path, List[Dict[str, Any]]],
    feature_columns: Optional[List[str]] = None,
    round_decimals: int = 4,
) -> ConflictReport:
    """
    Analyze dataset for duplicate feature vectors and label conflicts.

    Parameters
    ----------
    dataset : Union[str, Path, List[Dict[str, Any]]]
        Path to training_dataset.csv or list of row dictionaries.
    feature_columns : Optional[List[str]]
        Subset of features to compare. Defaults to all 30 ProjectFeatures.
    round_decimals : int
        Decimal precision for floating-point feature comparison.

    Returns
    -------
    ConflictReport
    """
    rows: List[Dict[str, Any]] = []
    if isinstance(dataset, (str, Path)):
        p = Path(dataset)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset}")
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    else:
        rows = list(dataset)

    if not rows:
        return ConflictReport(
            total_instances=0,
            unique_feature_vectors=0,
            num_duplicate_vectors=0,
            num_consistent_duplicate_groups=0,
            num_conflicting_duplicate_groups=0,
            conflict_groups=[],
        )

    cols = feature_columns or ProjectFeatures.feature_names()

    # Group by rounded feature signature tuple
    signature_map: Dict[Tuple, List[Dict[str, Any]]] = {}

    for row in rows:
        sig_items = []
        for c in cols:
            val = row.get(c, 0.0)
            try:
                sig_items.append(round(float(val), round_decimals))
            except (ValueError, TypeError):
                sig_items.append(str(val))
        sig = tuple(sig_items)
        signature_map.setdefault(sig, []).append(row)

    total_instances = len(rows)
    unique_vectors = len(signature_map)

    conflict_groups: List[ConflictGroup] = []
    consistent_groups = 0
    conflicting_groups = 0

    for sig, group_rows in signature_map.items():
        if len(group_rows) > 1:
            instances = [r.get("instance_id", "unknown") for r in group_rows]
            labels = [r.get("best_algorithm", "unknown") for r in group_rows]
            unique_lbls = set(labels)
            is_conflicting = len(unique_lbls) > 1

            if is_conflicting:
                conflicting_groups += 1
            else:
                consistent_groups += 1

            conflict_groups.append(
                ConflictGroup(
                    feature_signature=str(sig),
                    instances=instances,
                    labels=labels,
                    is_conflicting=is_conflicting,
                )
            )

    return ConflictReport(
        total_instances=total_instances,
        unique_feature_vectors=unique_vectors,
        num_duplicate_vectors=len(conflict_groups),
        num_consistent_duplicate_groups=consistent_groups,
        num_conflicting_duplicate_groups=conflicting_groups,
        conflict_groups=conflict_groups,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze feature-space conflicts in an RCPSP training dataset."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to training_dataset.csv",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write conflict report as JSON.",
    )
    args = parser.parse_args()

    report = analyze_conflicts(args.dataset)
    print(report.to_text())

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to {args.json_out}")


if __name__ == "__main__":
    main()
