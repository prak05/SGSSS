"""
Feature dataset serialization and loading utilities.

Note: In Phase 4, this module ONLY handles serialization and loading of
project feature vectors to/from CSV. No ML models, no training, and no
algorithm prediction labels are implemented at this stage.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.features import ProjectFeatures


def export_features_to_csv(
    records: List[Dict[str, Any]],
    filepath: Union[str, Path],
    id_column: Optional[str] = "instance_id",
) -> None:
    """
    Export a list of feature dictionaries to a CSV file.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        List of feature dictionaries, where each dict typically contains an
        identifier (e.g. 'instance_id') plus all fields of ProjectFeatures.
    filepath : Union[str, Path]
        Target CSV file path.
    id_column : Optional[str]
        Key representing the instance identifier if present.
    """
    if not records:
        raise ValueError("Cannot export empty records list to CSV.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine headers deterministically: ID column first if present, then feature names
    all_keys = set()
    for rec in records:
        all_keys.update(rec.keys())

    standard_features = ProjectFeatures.feature_names()
    ordered_headers: List[str] = []

    if id_column and id_column in all_keys:
        ordered_headers.append(id_column)

    for feat in standard_features:
        if feat in all_keys and feat not in ordered_headers:
            ordered_headers.append(feat)

    # Any extra metadata columns
    for k in sorted(all_keys):
        if k not in ordered_headers:
            ordered_headers.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_headers)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def load_features_from_csv(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load feature records from a CSV file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the CSV file.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with numeric fields converted to float or int.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {filepath}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed_row[k] = None
                    continue
                # Try int then float
                try:
                    parsed_row[k] = int(v)
                except ValueError:
                    try:
                        parsed_row[k] = float(v)
                    except ValueError:
                        parsed_row[k] = v
            records.append(parsed_row)

    return records
