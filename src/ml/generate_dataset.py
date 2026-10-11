"""
RCPSP Training Dataset Generator.

Generates a supervised learning dataset from RCPSP benchmark instances by:
  1. Parsing each project instance.
  2. Extracting its deterministic 30-dimensional project feature vector.
  3. Executing all five algorithms (GA, SA, BCO, PRH, TS) across multiple independent seeds.
  4. Validating every schedule for feasibility.
  5. Aggregating multi-seed performance (best, mean, worst, std, runtime).
  6. Deriving the target classification label ('best_algorithm' with 'ALL' and tie-tracking).
  7. Checkpointing raw and aggregated results safely to disk.

Strict Architectural Rule:
  The project contains EXACTLY FIVE algorithms: GA, SA, BCO, PRH, TS.
  Zero PSO algorithms, labels, or references.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.benchmark import ALGORITHM_NAMES, BenchmarkRecord, run_single
from src.features import ProjectFeatures, extract_features
from src.model import Project
from src.parser import Parser


# ---------------------------------------------------------------------------
# Helper: Determine instance metadata
# ---------------------------------------------------------------------------

def infer_instance_metadata(file_path: Path) -> Tuple[str, str, int]:
    """
    Infer instance_id, instance_family, and instance_size from filename.

    Example:
      'j3010_1.sm' -> ('j3010_1', 'j30', 30)
      'j12010_2.sm' -> ('j12010_2', 'j120', 120)
      'sample.rcp'  -> ('sample_rcp', 'custom', 8)
    """
    stem = file_path.stem
    lower = stem.lower()
    if lower.startswith("j120"):
        return stem, "j120", 120
    elif lower.startswith("j90"):
        return stem, "j90", 90
    elif lower.startswith("j60"):
        return stem, "j60", 60
    elif lower.startswith("j30"):
        return stem, "j30", 30
    elif lower.startswith("j10"):
        return stem, "j10", 10
    else:
        return stem, "custom", 0


def infer_extended_metadata(file_path: Path) -> Tuple[str, str, int, str, str]:
    """
    Infer extended metadata for RCPSP instance:
    (instance_id, instance_family, instance_size, instance_source, feature_configuration)
    """
    stem = file_path.stem
    inst_id, fam, size = infer_instance_metadata(file_path)

    if fam in {"j30", "j60", "j90", "j120"}:
        source = "PSPLIB"
        param_group = stem.split("_")[0] if "_" in stem else stem
    else:
        source = "custom"
        param_group = "custom_group"

    return inst_id, fam, size, source, param_group


# ---------------------------------------------------------------------------
# Multiprocessing Worker Function
# ---------------------------------------------------------------------------

def _worker_execute_run(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Top-level worker function for parallel execution.
    Parses instance file and executes a single algorithm run.
    """
    path_str = task["instance_path_str"]
    algo = task["algorithm"]
    seed = task["seed"]
    budget = task["evaluation_budget"]
    inst_id = task["instance_id"]

    project = Parser.parse(path_str)
    rec = run_single(
        project=project,
        algorithm=algo,
        seed=seed,
        evaluation_budget=budget,
        instance_id=inst_id,
    )
    if not rec.feasible:
        rec.notes = "FAILURE: Infeasible schedule produced"

    return {
        "record": rec,
        "inst_size": task["instance_size"],
        "inst_family": task["instance_family"],
        "instance_source": task["instance_source"],
        "feature_configuration": task["feature_configuration"],
        "features_dict": task["features_dict"],
    }


# ---------------------------------------------------------------------------
# Dataset Generator Core
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """
    Manages benchmark execution, checkpointing, and aggregation across
    instances and seeds.
    """

    def __init__(
        self,
        output_dir: Path,
        evaluation_budget: int = 5000,
        seeds: Optional[List[int]] = None,
        num_workers: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_budget = evaluation_budget
        self.seeds = seeds or [1, 2, 3]
        self.num_workers = num_workers

        self.raw_results_file = self.output_dir / "raw_results.csv"
        self.training_dataset_file = self.output_dir / "training_dataset.csv"
        self.feature_schema_file = self.output_dir / "feature_schema.json"
        self.label_dist_file = self.output_dir / "label_distribution.csv"
        self.convergence_dir = self.output_dir / "convergence"
        self.convergence_dir.mkdir(parents=True, exist_ok=True)

        self._existing_raw_keys: Set[Tuple[str, str, int]] = set()
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """Read existing raw_results.csv if present to enable resume."""
        if not self.raw_results_file.exists():
            return

        with open(self.raw_results_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = (
                        row["instance_id"],
                        row["algorithm"].split("[")[0],  # normalise PRH[..] -> PRH
                        int(row["seed"]),
                    )
                    self._existing_raw_keys.add(key)
                except (KeyError, ValueError):
                    continue

    def _save_raw_record(
        self,
        record: BenchmarkRecord,
        instance_size: int,
        instance_family: str,
        instance_source: str,
        feature_configuration: str,
        features_dict: Dict[str, Any],
    ) -> None:
        """Append a single benchmark result to raw_results.csv immediately."""
        file_exists = self.raw_results_file.exists()

        row: Dict[str, Any] = {
            "instance_id": record.instance_id,
            "instance_family": instance_family,
            "instance_source": instance_source,
            "feature_configuration": feature_configuration,
            "instance_size": instance_size,
            "seed": record.seed,
            "evaluation_budget": self.evaluation_budget,
            "algorithm": record.algorithm,
            "makespan": record.makespan,
            "runtime": round(record.runtime, 4),
            "evaluations": record.evaluations,
            "feasible": record.feasible,
        }
        # Add all 30 features
        row.update(features_dict)

        # Raw priority vector and notes
        row["best_priority_vector"] = json.dumps(record.best_priority_vector)
        row["notes"] = record.notes

        fieldnames = list(row.keys())

        with open(self.raw_results_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # Mark as completed
        norm_algo = record.algorithm.split("[")[0]
        self._existing_raw_keys.add((record.instance_id, norm_algo, record.seed))

        # Save convergence history to separate file
        if record.convergence_history:
            conv_file = (
                self.convergence_dir
                / f"{record.instance_id}_{norm_algo}_seed{record.seed}.json"
            )
            with open(conv_file, "w", encoding="utf-8") as cf:
                json.dump(record.convergence_history, cf)

    def generate(self, instance_paths: List[Path]) -> Tuple[int, int]:
        """
        Execute benchmark runs for all instances, seeds, and algorithms.

        Returns
        -------
        Tuple[int, int]
            (completed_runs, total_expected_runs)
        """
        total_runs = len(instance_paths) * len(self.seeds) * len(ALGORITHM_NAMES)
        completed = 0

        # Save schema metadata once
        feature_names = ProjectFeatures.feature_names()
        schema_metadata = {
            "schema_version": "1.1",
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_budget": self.evaluation_budget,
            "seeds": self.seeds,
            "algorithms": ALGORITHM_NAMES,
            "num_features": len(feature_names),
            "features": feature_names,
            "paper_features": ["N", "NC", "RF", "RS"],
            "labels": ["GA", "SA", "BCO", "PRH", "TS", "ALL"],
            "metadata_fields": [
                "instance_id",
                "instance_family",
                "instance_source",
                "feature_configuration",
                "instance_size",
            ],
        }
        with open(self.feature_schema_file, "w", encoding="utf-8") as sf:
            json.dump(schema_metadata, sf, indent=2)

        print(f"Starting dataset generation across {len(instance_paths)} instances...")
        print(f"Algorithms: {ALGORITHM_NAMES} (Count: {len(ALGORITHM_NAMES)})")
        print(f"Seeds: {self.seeds} | Budget: {self.evaluation_budget}")

        # Pre-parse instances to extract features and build execution tasks
        tasks: List[Dict[str, Any]] = []
        for inst_path in instance_paths:
            inst_id, inst_fam, inst_sz, inst_src, feat_cfg = infer_extended_metadata(inst_path)
            project = Parser.parse(str(inst_path))
            features = extract_features(project)
            if inst_sz == 0:
                inst_sz = features.N

            feat_dict = features.to_dict()

            for seed in self.seeds:
                for algo in ALGORITHM_NAMES:
                    checkpoint_key = (inst_id, algo, seed)
                    if checkpoint_key in self._existing_raw_keys:
                        completed += 1
                        continue
                    tasks.append({
                        "instance_path_str": str(inst_path),
                        "instance_id": inst_id,
                        "instance_family": inst_fam,
                        "instance_size": inst_sz,
                        "instance_source": inst_src,
                        "feature_configuration": feat_cfg,
                        "features_dict": feat_dict,
                        "algorithm": algo,
                        "seed": seed,
                        "evaluation_budget": self.evaluation_budget,
                    })

        print(f"Already completed runs in checkpoint: {completed}/{total_runs}")
        print(f"Remaining runs to execute: {len(tasks)}")

        if not tasks:
            print("All tasks already completed according to checkpoint!")
            self.aggregate_training_dataset(instance_paths)
            return completed, total_runs

        # Determine worker count
        workers = self.num_workers
        if workers is None:
            # Default to min(os.cpu_count(), 7) to utilize multi-core without freezing system
            cpu_cnt = os.cpu_count() or 4
            workers = max(1, min(cpu_cnt - 1, 7))

        print(f"Executing with {workers} worker process(es)...")

        if workers == 1:
            # Sequential execution
            for idx, task in enumerate(tasks, 1):
                res = _worker_execute_run(task)
                self._save_raw_record(
                    record=res["record"],
                    instance_size=res["inst_size"],
                    instance_family=res["inst_family"],
                    instance_source=res["instance_source"],
                    feature_configuration=res["feature_configuration"],
                    features_dict=res["features_dict"],
                )
                completed += 1
                if idx % 10 == 0 or idx == len(tasks):
                    print(f"[{completed}/{total_runs}] Done: {task['instance_id']} {task['algorithm']} seed={task['seed']} (ms={res['record'].makespan})")
                    self.aggregate_training_dataset(instance_paths)
        else:
            # Parallel execution with ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(_worker_execute_run, task): task for task in tasks
                }
                for idx, future in enumerate(concurrent.futures.as_completed(future_to_task), 1):
                    task = future_to_task[future]
                    res = future.result()
                    self._save_raw_record(
                        record=res["record"],
                        instance_size=res["inst_size"],
                        instance_family=res["inst_family"],
                        instance_source=res["instance_source"],
                        feature_configuration=res["feature_configuration"],
                        features_dict=res["features_dict"],
                    )
                    completed += 1
                    if idx % 15 == 0 or idx == len(tasks):
                        print(f"[{completed}/{total_runs}] Done: {task['instance_id']} {task['algorithm']} seed={task['seed']} (ms={res['record'].makespan})", flush=True)
                        self.aggregate_training_dataset(instance_paths)

        # Final aggregation pass
        self.aggregate_training_dataset(instance_paths)
        print(f"\nAll runs finished. Completed: {completed}/{total_runs}", flush=True)
        return completed, total_runs

    def aggregate_training_dataset(self, instance_paths: List[Path]) -> None:
        """
        Aggregate multi-seed experimental results from raw_results.csv into
        training_dataset.csv and label_distribution.csv.
        """
        if not self.raw_results_file.exists():
            return

        # Load all raw results
        raw_rows: List[Dict[str, Any]] = []
        with open(self.raw_results_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                raw_rows.append(r)

        # Group by instance_id
        by_instance: Dict[str, List[Dict[str, Any]]] = {}
        for r in raw_rows:
            by_instance.setdefault(r["instance_id"], []).append(r)

        training_rows: List[Dict[str, Any]] = []
        label_counts: Dict[str, int] = {
            "GA": 0, "SA": 0, "BCO": 0, "PRH": 0, "TS": 0, "ALL": 0
        }

        # Index metadata by stem
        inst_meta_map: Dict[str, Tuple[str, str, int, str, str]] = {}
        for p in instance_paths:
            inst_meta_map[p.stem] = infer_extended_metadata(p)

        for inst_id, rows in by_instance.items():
            # Check if all algorithms are present for this instance
            algos_present = {r["algorithm"].split("[")[0] for r in rows}
            if len(algos_present) < len(ALGORITHM_NAMES):
                # Instance not fully evaluated yet
                continue

            first_row = rows[0]
            if inst_id in inst_meta_map:
                _, inst_family, inst_size, inst_source, feat_cfg = inst_meta_map[inst_id]
            else:
                inst_family = first_row.get("instance_family", "custom")
                inst_size = int(first_row.get("instance_size", 0))
                inst_source = first_row.get("instance_source", "PSPLIB")
                feat_cfg = first_row.get("feature_configuration", inst_id.split("_")[0])

            # Aggregate per algorithm
            algo_stats: Dict[str, Dict[str, Any]] = {}
            for algo in ALGORITHM_NAMES:
                algo_rows = [r for r in rows if r["algorithm"].split("[")[0] == algo]
                if not algo_rows:
                    continue

                makespans = [int(r["makespan"]) for r in algo_rows if r["feasible"] == "True"]
                runtimes = [float(r["runtime"]) for r in algo_rows]
                evals = [int(r["evaluations"]) for r in algo_rows]

                if not makespans:
                    # Infeasible fallback
                    makespans = [int(r["makespan"]) for r in algo_rows]

                best_ms = min(makespans)
                mean_ms = sum(makespans) / len(makespans)
                worst_ms = max(makespans)
                mean_rt = sum(runtimes) / len(runtimes)
                mean_ev = sum(evals) / len(evals)

                algo_stats[algo] = {
                    "best_ms": best_ms,
                    "mean_ms": mean_ms,
                    "worst_ms": worst_ms,
                    "mean_runtime": mean_rt,
                    "mean_evaluations": mean_ev,
                    "feasible_count": len([r for r in algo_rows if r["feasible"] == "True"]),
                }

            if len(algo_stats) < len(ALGORITHM_NAMES):
                continue

            # Determine best algorithm and tie policy
            # Primary criterion: Lowest best makespan
            global_min_ms = min(s["best_ms"] for s in algo_stats.values())
            best_candidates = [
                algo for algo, s in algo_stats.items() if s["best_ms"] == global_min_ms
            ]

            if len(best_candidates) == len(ALGORITHM_NAMES):
                # All five algorithms achieved the exact same best makespan
                best_algorithm = "ALL"
                label_basis = "exact_five_way_tie"
                tied_best_algorithms = ",".join(ALGORITHM_NAMES)
            elif len(best_candidates) == 1:
                best_algorithm = best_candidates[0]
                label_basis = "unique_best_makespan"
                tied_best_algorithms = best_candidates[0]
            else:
                # Partial tie: resolve via secondary criterion: lowest mean makespan across seeds
                tied_best_algorithms = ",".join(sorted(best_candidates))
                min_mean_ms = min(algo_stats[a]["mean_ms"] for a in best_candidates)
                mean_tied = [
                    a for a in best_candidates if algo_stats[a]["mean_ms"] == min_mean_ms
                ]
                if len(mean_tied) == 1:
                    best_algorithm = mean_tied[0]
                    label_basis = "partial_tie_broken_by_mean_makespan"
                else:
                    # Tertiary: break tie deterministically via hash to avoid runtime bias
                    idx = hash(inst_id) % len(mean_tied)
                    best_algorithm = sorted(mean_tied)[idx]
                    label_basis = "partial_tie_broken_deterministically"

            label_counts[best_algorithm] = label_counts.get(best_algorithm, 0) + 1

            # Build aggregated row
            agg_row: Dict[str, Any] = {
                "instance_id": inst_id,
                "instance_family": inst_family,
                "instance_source": inst_source,
                "feature_configuration": feat_cfg,
                "instance_size": inst_size,
            }

            # Add all 30 features from first row
            for f_name in ProjectFeatures.feature_names():
                agg_row[f_name] = first_row[f_name]

            # Target label and audit trail
            agg_row["best_algorithm"] = best_algorithm
            agg_row["label_basis"] = label_basis
            agg_row["tied_best_algorithms"] = tied_best_algorithms
            agg_row["best_feasible_makespan"] = global_min_ms

            # Summary stats per algorithm for auditability
            for algo in ALGORITHM_NAMES:
                s = algo_stats[algo]
                agg_row[f"{algo}_best_ms"] = s["best_ms"]
                agg_row[f"{algo}_mean_ms"] = round(s["mean_ms"], 2)
                agg_row[f"{algo}_mean_rt"] = round(s["mean_runtime"], 4)

            training_rows.append(agg_row)

        if not training_rows:
            return

        # Write training_dataset.csv
        fieldnames = list(training_rows[0].keys())
        with open(self.training_dataset_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in training_rows:
                writer.writerow(r)

        # Write label_distribution.csv
        with open(self.label_dist_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "count", "percentage"])
            total_labeled = len(training_rows)
            for lbl, cnt in sorted(label_counts.items()):
                pct = round((cnt / total_labeled) * 100.0, 2) if total_labeled > 0 else 0.0
                writer.writerow([lbl, cnt, pct])


# ---------------------------------------------------------------------------
# Public API function
# ---------------------------------------------------------------------------

def generate_training_dataset(
    instances: List[Union[str, Path]],
    seeds: Optional[List[int]] = None,
    evaluation_budget: int = 5000,
    output_dir: Union[str, Path] = "data/training",
    num_workers: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Generate an RCPSP training dataset across benchmark instances.

    Parameters
    ----------
    instances : List[Union[str, Path]]
        Paths to instance files (.sm, .rcp).
    seeds : Optional[List[int]]
        List of integer seeds. Defaults to [1, 2, 3].
    evaluation_budget : int
        Evaluation budget per algorithm run. Defaults to 5000.
    output_dir : Union[str, Path]
        Output directory for dataset artifacts.
    num_workers : Optional[int]
        Number of parallel worker processes.

    Returns
    -------
    Tuple[int, int]
        (completed_runs, total_expected_runs)
    """
    paths = [Path(p) for p in instances]
    generator = DatasetGenerator(
        output_dir=Path(output_dir),
        evaluation_budget=evaluation_budget,
        seeds=seeds or [1, 2, 3],
        num_workers=num_workers,
    )
    return generator.generate(paths)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AI-RCPSP training dataset from benchmark instances."
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=["data/benchmark_100"],
        help="Instance files or directories containing .sm/.rcp files.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5000,
        help="Evaluation budget per run (default: 5000).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Seeds for independent runs (default: 1 2 3).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training",
        help="Output directory (default: data/training).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: cpu_count - 1).",
    )

    args = parser.parse_args()

    # Collect instance file paths
    instance_paths: List[Path] = []
    for item in args.instances:
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.glob("*.sm")):
                instance_paths.append(f)
            for f in sorted(p.glob("*.rcp")):
                instance_paths.append(f)
        elif p.is_file():
            instance_paths.append(p)

    if not instance_paths:
        print(f"No valid instances found in {args.instances}.")
        return

    generate_training_dataset(
        instances=instance_paths,
        seeds=args.seeds,
        evaluation_budget=args.budget,
        output_dir=args.output,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
