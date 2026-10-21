"""
================================================================================
src/__init__.py — AI-RCPSP Top-Level Architecture
================================================================================
Metaheuristic Optimization & Constraint Evaluation Suite for RCPSP
(Resource-Constrained Project Scheduling Problem).

Core Components:
  • Centralized Serial Schedule Generation Scheme (SSGS) decoder
  • Metaheuristic & Neural Solvers: GA, SA, BCO, PRH, PSO (MSP_PSO), TBGAT
  • Dual-layer Precedence and Renewable Resource Feasibility Audit
  • Multi-format Parser (.sm, .rcp, .json) with instance-adaptive budget & seed
  • Interactive Visualizations (Plotly Gantt, Resource Loading, Convergence)
  • Printable Publication-Grade Executive PDF Reports
================================================================================
"""

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
    run_algorithm,
    run_all_algorithms,
    SOLVERS,
    ALGORITHM_NAMES,
    solve_ga,
    solve_sa,
    solve_bco,
    solve_prh,
    solve_pso,
    solve_tbgat,
)
from src.visualization import (
    create_gantt_chart,
    create_resource_timeline_charts,
    create_convergence_chart,
)
from src.report import generate_pdf_report

__all__ = [
    # Data Models
    "Activity",
    "Resource",
    "Project",
    "ScheduleEntry",
    "AlgorithmResult",
    "determine_winner",
    "validate_dag",
    "compute_cpm_lower_bound",
    # SSGS & Evaluation
    "SSGS",
    "Evaluator",
    "ScheduleAudit",
    # Parser & Adaptors
    "Parser",
    "compute_adaptive_budget",
    "compute_adaptive_seed",
    "extract_instance_metadata",
    # Algorithms
    "run_algorithm",
    "run_all_algorithms",
    "SOLVERS",
    "ALGORITHM_NAMES",
    "solve_ga",
    "solve_sa",
    "solve_bco",
    "solve_prh",
    "solve_pso",
    "solve_tbgat",
    # Visualization & Reporting
    "create_gantt_chart",
    "create_resource_timeline_charts",
    "create_convergence_chart",
    "generate_pdf_report",
]
