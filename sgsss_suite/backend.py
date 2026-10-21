"""
================================================================================
AI-RCPSP CORE BACKEND ENGINE & OPTIMIZATION SUITE
================================================================================
Modular façade wrapping the canonical src package.
Core Metaheuristic Optimization Suite for RCPSP (Resource-Constrained Project
Scheduling Problem):
  • 6 Solvers (GA, SA, BCO, PRH, PSO, TBGAT) decoded via centralized SSGS.
  • Integrated Particle Swarm Optimization (from henryyjiang/MSP_PSO).
  • Integrated Bidirectional Graph Attention Network (from zcaicaros/TBGAT).
  • Instance-adaptive budget and random seed calibration.
  • Rigorous dual-layer constraint feasibility auditing.
  • Deterministically resolves any ties so EXACTLY ONE ALGORITHM WINS.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add current directory to sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from src.model import (
    Activity,
    Resource,
    Project,
    ScheduleEntry,
    AlgorithmResult,
    determine_winner,
    validate_dag,
    compute_cpm_lower_bound,
)
from src.ssgs import SSGS
from src.evaluation import Evaluator, ScheduleAudit
from src.parser import (
    Parser,
    compute_adaptive_budget,
    compute_adaptive_seed,
    extract_instance_metadata,
)
from src.algorithms import (
    solve_ga,
    solve_sa,
    solve_bco,
    solve_prh,
    solve_pso,
    solve_tbgat,
    run_algorithm,
    run_all_algorithms,
    SOLVERS,
    ALGORITHM_NAMES,
)
from src.visualization import (
    create_gantt_chart,
    create_resource_timeline_charts,
    create_convergence_chart,
)
from src.report import generate_pdf_report


# ==============================================================================
# SECTION SUMMARY: Algorithm Registry
# What is in this section: Exposes SOLVER_REGISTRY mapping algorithm names to their
# respective solver functions (GA, SA, BCO, PRH, PSO, TBGAT).
# ==============================================================================
SOLVER_REGISTRY = {
    "GA": solve_ga,
    "SA": solve_sa,
    "BCO": solve_bco,
    "PRH": solve_prh,
    "PSO": solve_pso,
    "TBGAT": solve_tbgat,
}


# ==============================================================================
# SECTION SUMMARY: Command Line Interface (CLI) Execution Pipeline
# What is in this section: End-to-end command-line runner that parses RCPSP instance files,
# auto-adapts the SSGS budget and seed to the file scale, runs all solvers,
# audits constraints for 100% feasibility, and declares the definite single winner.
# ==============================================================================
def main():
    """CLI execution controller with rich graphical steps and mathematical explanations."""
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║        AI-RCPSP: METAHEURISTIC OPTIMIZATION & SSGS SCHEDULER             ║")
    print("║        Academic Major Project Prototype                                  ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print("""
    [PIPELINE FLOW]
    ┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ 1. Parse Graph │───>│ 2. Adaptive SSGS│───>│ 3. 6 Solvers    │
    │    & Validate  │    │    Budget & Seed│    │   (GA/SA/BCO/   │
    └────────────────┘    └─────────────────┘    │  PRH/PSO/TBGAT) │
                                                 └────────┬────────┘
    ┌────────────────┐    ┌─────────────────┐             │
    │ 5. Tournament  │<───│ 4. Dual-Layer   │<────────────┘
    │    Winner      │    │    Audit        │
    └────────────────┘    └─────────────────┘
    """)

    # --------------------------------------------------------------------------
    # Subsection: Instance File Selection & Ingestion
    # --------------------------------------------------------------------------
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        candidates = list(Path(".").rglob("*.sm")) + list(Path(".").rglob("*.rcp"))
        if candidates:
            file_path = str(candidates[0])
            print(f"Auto-selected instance: {file_path}")
        else:
            print("Usage: python backend.py <path_to_instance.sm>")
            return

    # --------------------------------------------------------------------------
    # Subsection: STEP 1 — Parsing & Precedence DAG Graph Construction
    # --------------------------------------------------------------------------
    print("┌──────────────────────────────────────────────────────────────────────────┐")
    print("│ STEP 1: Instance Ingestion & Precedence DAG Graph Construction           │")
    print("└──────────────────────────────────────────────────────────────────────────┘")
    print(f"  • Reading Instance Source: {file_path}")
    project = Parser.parse(file_path)
    topo_order = validate_dag(project)
    print(f"  • Graph Topology         : {project.num_activities} activities (including start/end dummies)")
    print(f"  • Real Activities (N)    : {len(project.real_activities)}")
    print(f"  • Renewable Resources (K): {project.num_resources} distinct renewable types")
    print(f"  • CPM Lower Bound Duration: {project.cpm_lower_bound} days (theoretical zero-resource minimum)")
    print(f"  • Total Work Content     : {project.total_work_content} activity-days")
    print(f"  • DAG Validation Status  : VERIFIED ACYCLIC (Topological sort length: {len(topo_order)})")

    # --------------------------------------------------------------------------
    # Subsection: STEP 2 — Auto-Adaptive SSGS Budget & Seed Calibration
    # --------------------------------------------------------------------------
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│ STEP 2: Auto-Adaptive SSGS Evaluation Budget & Random Seed Calibration    │")
    print("└──────────────────────────────────────────────────────────────────────────┘")
    budget = project.file_budget or compute_adaptive_budget(project)
    seed = project.file_seed or compute_adaptive_seed(file_path)
    print(f"  • Auto-Calibrated Budget : {budget} SSGS evaluations")
    print(f"      - Scale Rationale    : Calibrated dynamically for N={len(project.real_activities)} tasks")
    print(f"  • Auto-Calibrated Seed   : {seed}")
    print(f"      - Seed Rationale     : Derived from instance to guarantee scientific reproducibility")

    # --------------------------------------------------------------------------
    # Subsection: STEP 3 — Metaheuristic Priority Search & SSGS Decoding
    # --------------------------------------------------------------------------
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│ STEP 3: Metaheuristic Priority Search & Centralized SSGS Decoding        │")
    print("└──────────────────────────────────────────────────────────────────────────┘")
    print("  • Solvers to Evaluate: 6 (GA, SA, BCO, PRH, PSO, TBGAT)")
    print("  • Evaluation Scheme  : Continuous Random-Key Vector λ in [0, 1]^N")
    print("  • Shared Decoder     : Serial Schedule Generation Scheme (SSGS)")
    print("  • Explanation: Each algorithm explores priority vectors decoded into active,")
    print("    100% precedence- and resource-feasible schedules by the centralized SSGS.")

    algo_descriptions = {
        "GA": "Genetic Algorithm: Population search, 2-point crossover, Gaussian mutation.",
        "SA": "Simulated Annealing: Metropolis acceptance P=exp(-Δ/T), geometric cooling.",
        "BCO": "Bee Colony Optimization: Food source foraging with employed/onlooker bees.",
        "PRH": "Priority Rule Heuristics: Multi-pass sampling (EST, LST, SPT, MTS, GRPW).",
        "PSO": "Particle Swarm Optimization (MSP_PSO): Cognitive/social velocity updates, basin-hopping.",
        "TBGAT": "TBGAT: Bidirectional forward/backward graph attention network over DAG precedence.",
    }

    results = {}
    for algo_name in ALGORITHM_NAMES:
        print(f"\n  ▶ Executing {algo_name} [{algo_descriptions.get(algo_name, '')}]...")
        res = run_algorithm(algo_name, project, evaluation_budget=budget, seed=seed)
        results[algo_name] = res
        audit = Evaluator.audit_schedule(project, res.schedule)
        print(f"    ✓ Completed: Makespan = {res.makespan} days | Runtime = {res.runtime:.4f}s | "
              f"Best found at eval {res.best_eval}/{budget} | Feasible = {audit.is_feasible}")

    # --------------------------------------------------------------------------
    # Subsection: STEP 4 — Comparative Benchmark Performance Table
    # --------------------------------------------------------------------------
    print("\n" + "═" * 80)
    print(f"  {'Algorithm':<10} │ {'Makespan':<10} │ {'Runtime (s)':<13} │ {'SSGS Evals':<11} │ {'Feasibility Audit':<22}")
    print("─" * 80)
    for name, r in results.items():
        audit = Evaluator.audit_schedule(project, r.schedule)
        feas_str = "100% FEASIBLE ✓" if audit.is_feasible else f"FAIL ({audit.precedence_violations}P/{audit.resource_violations}R)"
        print(f"  {name:<10} │ {r.makespan:<10} │ {r.runtime:<13.4f} │ {r.evaluations:<11} │ {feas_str:<22}")
    print("═" * 80)

    # --------------------------------------------------------------------------
    # Subsection: STEP 5 — Tournament Single-Winner Determination & Audit
    # --------------------------------------------------------------------------
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│ STEP 5: Tournament Winner Determination & Dual-Layer Feasibility Audit   │")
    print("└──────────────────────────────────────────────────────────────────────────┘")
    winner_name, winner_res = determine_winner(results)
    final_audit = Evaluator.audit_schedule(project, winner_res.schedule)

    print(f"  🥇 DEFINITE SINGLE WINNER: {winner_name}")
    print(f"     • Optimal Makespan     : {winner_res.makespan} days")
    print(f"     • Computation Runtime  : {winner_res.runtime:.4f} seconds")
    print(f"     • Earliest Best Eval   : Iteration {winner_res.best_eval} of {budget}")
    print(f"     • Feasibility Audit    : {final_audit.summary}")
    print(f"     • Precedence Layer     : {'PASSED ✓ (0 violations across all DAG arcs)' if final_audit.precedence_passed else 'FAILED'}")
    print(f"     • Resource Layer       : {'PASSED ✓ (Usage <= capacity across all time steps)' if final_audit.resource_passed else 'FAILED'}")
    print("  • Explanation: Ties are deterministically resolved by earliest evaluation")
    print("    count, followed by fastest computation runtime, guaranteeing exactly one winner.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()
