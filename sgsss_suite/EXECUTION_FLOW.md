# RCPSP Execution Flow — Code Location Guide

This file provides an architectural and code-navigation map to answer **“Where in the code does this action happen?”** by tracing each execution stage to its source location.

## Quick Map

1. **Website Startup** → [ui.py](ui.py)
2. **Upload or Select Project** → [ui.py](ui.py) & [src/parser.py](src/parser.py)
3. **Parse Project File (.sm, .rcp, .json)** → [src/parser.py](src/parser.py)
4. **Create In-Memory Data Models** → [src/model.py](src/model.py)
5. **Validate DAG Acyclicity & CPM Lower Bound** → [src/model.py](src/model.py)
6. **Auto-Calibrate SSGS Budget & Seed** → [src/parser.py](src/parser.py)
7. **Execute Metaheuristic & Neural Solvers (GA, SA, BCO, PRH, PSO, TBGAT)** → [src/algorithms/](src/algorithms/)
8. **Serial Schedule Generation Scheme (SSGS)** → [src/ssgs.py](src/ssgs.py)
9. **Dual-Layer Feasibility Audit (Precedence & Renewable Limits)** → [src/evaluation.py](src/evaluation.py)
10. **Interactive Plotly Gantt Scheduling** → [src/visualization.py](src/visualization.py)
11. **Renewable Resource Timeline Loading vs Capacity** → [src/visualization.py](src/visualization.py)
12. **Comparative Benchmark & Single Winner Tournament** → [src/model.py](src/model.py) & [backend.py](backend.py)
13. **Generate Publication-Grade Executive PDF Report** → [src/report.py](src/report.py)

---

## 1. Website Startup

**Action:** Configure Streamlit execution environment and page structure.
- Main entrypoint: [ui.py](ui.py)
- Registered Solvers: `ALGORITHMS = ["GA", "SA", "BCO", "PRH", "PSO", "TBGAT"]` in [src/algorithms/__init__.py](src/algorithms/__init__.py)

## 2. Project Input & Ingestion

**Action:** User uploads an instance or triggers a quick-load benchmark file.
- Supported file types: `.sm` (PSPLIB), `.rcp` (Patterson), `.json`.
- Handled in `ui.py` sidebar and parsed by `src.parser.Parser.parse()`.

## 3. Project Data Models

**Action:** Create typed in-memory project representations.
- `Activity`: [src/model.py](src/model.py) (duration, predecessors, successors, resource requirements, demands alias).
- `Resource`: [src/model.py](src/model.py) (id, capacity).
- `Project`: [src/model.py](src/model.py) (activities, resources, renewable capacities, cpm_lower_bound, total_work_content).
- `ScheduleEntry`: [src/model.py](src/model.py) (activity_id, start_time, finish_time, duration, resource_usage).
- `AlgorithmResult`: [src/model.py](src/model.py) (makespan, runtime, evaluations, schedule, convergence history).

## 4. Precedence Graph DAG Validation

**Action:** Validate that the activity precedence graph contains no directed cycles.
- Function: `validate_dag(project)` in [src/model.py](src/model.py).
- Algorithm: Kahn's topological sort. Raises `ValueError` if a cycle is present.

## 5. Auto-Adaptive SSGS Budget & Seed Calibration

**Action:** Dynamically calibrate SSGS evaluation budget and random seed to the instance scale.
- Evaluation budget scaling: `compute_adaptive_budget(project)` in [src/parser.py](src/parser.py).
  - $N \le 15 \implies 200$ SSGS evaluations (toy/small)
  - $N \le 45 \implies 500$ SSGS evaluations (PSPLIB J30 benchmark)
  - $N \le 75 \implies 1000$ SSGS evaluations (PSPLIB J60 benchmark)
  - $N \le 100 \implies 1500$ SSGS evaluations (PSPLIB J90 benchmark)
  - $N > 100 \implies 2500$ SSGS evaluations (PSPLIB J120 benchmark)
- Seed derivation: `compute_adaptive_seed(file_path)` in [src/parser.py](src/parser.py).
  - Preserves explicit header seed if provided, otherwise derives a reproducible seed from file content.

## 6. Metaheuristic & Neural Optimization Suite

**Action:** Execute solvers over continuous random-key space $\lambda \in [0, 1]^N$.
- Central Registry & Dispatcher: [src/algorithms/__init__.py](src/algorithms/__init__.py)
- Solvers:
  1. **GA (Genetic Algorithm)**: [src/algorithms/ga.py](src/algorithms/ga.py) (tournament selection, uniform crossover, Gaussian creep mutation).
  2. **SA (Simulated Annealing)**: [src/algorithms/sa.py](src/algorithms/sa.py) (Metropolis acceptance, geometric cooling schedule).
  3. **BCO (Bee Colony Optimization)**: [src/algorithms/bco.py](src/algorithms/bco.py) (employed bees, onlooker waggle dance, scout restarts).
  4. **PRH (Priority Rule Heuristics)**: [src/algorithms/prh.py](src/algorithms/prh.py) (multi-pass dispatching rules + path relinking).
  5. **PSO (Particle Swarm Optimization)**: [src/algorithms/pso.py](src/algorithms/pso.py) (adapted from `henryyjiang/MSP_PSO`: cognitive/social velocity updates, linear inertia damping $w(t)$, basin-hopping perturbation).
  6. **TBGAT (Bidirectional Graph Attention Network)**: [src/algorithms/tbgat.py](src/algorithms/tbgat.py) (adapted from `zcaicaros/TBGAT`: dual forward EST and backward LST graph attention message-passing, policy scoring head, topological neighborhood search).

*(Note: Tabu Search (TS) has been scratched per user request, replaced by TBGAT and MSP_PSO integrations).*

## 7. Centralized Serial Schedule Generation Scheme (SSGS)

**Action:** Deterministically decode candidate priority vectors into 100% active, feasible schedules.
- Implementation: [src/ssgs.py](src/ssgs.py)
- Single source of truth: `SSGS.generate_schedule(project, priorities)`
- Maintains precedence consistency and renewable capacity bounds at all steps.

## 8. Dual-Layer Feasibility Audit

**Action:** Perform exhaustive, independent verification of constraint satisfaction.
- Implementation: `Evaluator.audit_schedule(project, schedule)` in [src/evaluation.py](src/evaluation.py)
- **Layer 1 (Precedence)**: Verifies $\text{finish}(p) \le \text{start}(s)$ for every directed DAG arc $(p, s)$.
- **Layer 2 (Resource)**: Verifies $\sum_{i \in A_t} r_{ik} \le R_k$ across all discrete steps $t \in [0, \text{makespan}]$.

## 9. Tournament Winner Determination

**Action:** Multi-tier lexicographical tie-breaking guaranteeing exactly one winner.
- Function: `determine_winner(results)` in [src/model.py](src/model.py)
- Criteria: Lowest Makespan $\to$ Earliest Best Evaluation Discovery $\to$ Fastest Computation Runtime.

## 10. Publication-Grade Reporting & Visualizations

- Interactive Gantt chart: `create_gantt_chart()` in [src/visualization.py](src/visualization.py)
- Resource utilization timelines: `create_resource_timeline_charts()` in [src/visualization.py](src/visualization.py)
- Multi-algorithm convergence curves: `create_convergence_chart()` in [src/visualization.py](src/visualization.py)
- Printable PDF report: `generate_pdf_report()` in [src/report.py](src/report.py)
