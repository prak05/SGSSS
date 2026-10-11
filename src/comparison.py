"""
Comparison engine for AI-RCPSP.

Takes a list of BenchmarkRecord objects (from src/benchmark.py) and
produces per-algorithm summary statistics across multiple seeds/runs.

Metrics reported
----------------
Algorithm | Best | Mean | Worst | Std | Mean Runtime | Mean Evaluations

Primary ranking  : lowest best makespan.
Secondary ranking: lowest mean makespan.
Tertiary ranking : lowest mean runtime.

Important
---------
- No global optimum is declared unless it has been independently verified.
- If all algorithms produce the same best makespan, they are all ranked equally.
- "Best feasible solution found" is the correct wording for metaheuristic results.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.benchmark import BenchmarkRecord, ALGORITHM_NAMES


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmSummary:
    """Aggregate statistics for one algorithm across multiple runs."""
    algorithm:        str
    best_makespan:    int
    mean_makespan:    float
    worst_makespan:   int
    std_makespan:     float
    mean_runtime:     float
    mean_evaluations: float
    n_runs:           int
    n_feasible:       int
    all_feasible:     bool
    records:          List[BenchmarkRecord] = field(default_factory=list)

    @property
    def feasibility_rate(self) -> float:
        return self.n_feasible / self.n_runs if self.n_runs > 0 else 0.0


@dataclass
class ComparisonReport:
    """Full comparison across all algorithms for one project instance."""
    instance_id:   str
    summaries:     List[AlgorithmSummary]
    n_seeds:       int
    budget:        int
    winner:        str        # algorithm name, or "ALL" if tied
    winner_ms:     int

    def to_text(self) -> str:
        """Return a human-readable comparison table."""
        lines = [
            f"=== Comparison Report: {self.instance_id} ===",
            f"Seeds: {self.n_seeds}  |  Evaluation budget: {self.budget} per run",
            "",
            f"{'Algorithm':<12} {'Best':>6} {'Mean':>7} {'Worst':>7} "
            f"{'Std':>6} {'Runtime(s)':>11} {'Evals':>7} {'Feasible':>9}",
            "-" * 70,
        ]

        for s in sorted(self.summaries, key=lambda x: (x.best_makespan, x.mean_makespan)):
            marker = " ◄" if s.algorithm.split("[")[0] == self.winner.split("[")[0] or self.winner == "ALL" else ""
            lines.append(
                f"{s.algorithm:<12} {s.best_makespan:>6} {s.mean_makespan:>7.1f} "
                f"{s.worst_makespan:>7} {s.std_makespan:>6.2f} "
                f"{s.mean_runtime:>11.3f} {s.mean_evaluations:>7.0f} "
                f"{s.n_feasible}/{s.n_runs}{marker}"
            )

        lines += [
            "-" * 70,
            "",
            f"Best feasible solution found: {self.winner_ms} "
            f"(by {self.winner})",
            "",
            "Note: Metaheuristic results are best feasible solutions found,",
            "      not guaranteed global optima.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------

def compare(
    records: List[BenchmarkRecord],
    instance_id: str = "unnamed",
    budget: int = 5000,
) -> ComparisonReport:
    """
    Build a ComparisonReport from a list of BenchmarkRecord objects.

    Parameters
    ----------
    records     : Output from benchmark.run_all_algorithms().
    instance_id : Label for the report.
    budget      : Evaluation budget used (for display only).

    Returns
    -------
    ComparisonReport
    """
    if not records:
        raise ValueError("Cannot compare: no benchmark records provided.")

    # Group by algorithm
    by_algo: Dict[str, List[BenchmarkRecord]] = {}
    for rec in records:
        # Normalise algorithm name (PRH embeds the best rule: PRH[LST] → PRH)
        base = rec.algorithm.split("[")[0]
        by_algo.setdefault(base, []).append(rec)

    summaries: List[AlgorithmSummary] = []

    for algo_base, recs in by_algo.items():
        feasible_recs = [r for r in recs if r.feasible]
        makespans = [r.makespan for r in feasible_recs] if feasible_recs else [r.makespan for r in recs]
        runtimes  = [r.runtime     for r in recs]
        evals     = [r.evaluations for r in recs]

        best_ms  = min(makespans)
        worst_ms = max(makespans)
        mean_ms  = statistics.mean(makespans)
        std_ms   = statistics.pstdev(makespans) if len(makespans) > 1 else 0.0

        summaries.append(AlgorithmSummary(
            algorithm=recs[0].algorithm,  # preserve full name with rule if PRH
            best_makespan=best_ms,
            mean_makespan=mean_ms,
            worst_makespan=worst_ms,
            std_makespan=std_ms,
            mean_runtime=statistics.mean(runtimes),
            mean_evaluations=statistics.mean(evals),
            n_runs=len(recs),
            n_feasible=len(feasible_recs),
            all_feasible=(len(feasible_recs) == len(recs)),
            records=recs,
        ))

    # ---- Determine winner ----
    # Primary: lowest best makespan; only among fully feasible algorithms.
    feasible_summaries = [s for s in summaries if s.all_feasible]
    if not feasible_summaries:
        feasible_summaries = summaries  # fallback: use all if none are fully feasible

    global_best_ms = min(s.best_makespan for s in feasible_summaries)
    winners = [s for s in feasible_summaries if s.best_makespan == global_best_ms]

    if len(winners) == len(feasible_summaries):
        winner_name = "ALL"
    elif len(winners) == 1:
        winner_name = winners[0].algorithm
    else:
        # Secondary: lowest mean makespan
        winners.sort(key=lambda s: (s.mean_makespan, s.mean_runtime))
        winner_name = winners[0].algorithm

    seeds_used = len(set(r.seed for r in records))

    return ComparisonReport(
        instance_id=instance_id,
        summaries=summaries,
        n_seeds=seeds_used,
        budget=budget,
        winner=winner_name,
        winner_ms=global_best_ms,
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_records(records: List[BenchmarkRecord]) -> List[str]:
    """
    Check every record for constraint violations and budget compliance.

    Returns a list of error strings.  Empty list means all records are valid.
    """
    errors: List[str] = []

    for rec in records:
        label = rec.label

        if not rec.feasible:
            errors.append(f"{label}: schedule is NOT feasible")

        if not rec.best_priority_vector:
            errors.append(f"{label}: empty priority vector")

        if rec.makespan <= 0:
            errors.append(f"{label}: invalid makespan {rec.makespan}")

        if rec.notes and rec.notes.startswith("WARNING"):
            errors.append(f"{label}: {rec.notes}")

    return errors
