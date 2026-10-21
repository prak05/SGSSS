# AI-RCPSP Codebase Index & Formula Catalog

This document provides a single, comprehensive reference for the entire **AI-RCPSP** codebase. It details every code file, its specific contents, key algorithms, mathematical formulas, data structures, and functions.

---

## 1. Quick Master Lookup Table

| File Path | Component Category | Primary Purpose | Key Formulas / Algorithms Implemented |
| :--- | :--- | :--- | :--- |
| [`rcpsp_all_in_one.py`](rcpsp_all_in_one.py) | Standalone All-in-One | Complete framework in 1 file | Models, SSGS, 30 Features, Parsers, GA, SA, BCO, TS, PRH, ML predictor, CLI |
| [`src/model.py`](src/model.py) | Data Models | Problem domain dataclasses | `Resource`, `Activity`, `Project`, `ScheduleEntry`, `AlgorithmResult` |
| [`src/ssgs.py`](src/ssgs.py) | Decoder | Serial Schedule Generation Scheme | Precedence EST: $S_i \ge \max C_p$; Time-indexed capacity check: $\sum r_{ik} \le R_k$; Makespan: $\max C_i$ |
| [`src/features.py`](src/features.py) | Feature Engine | Extracts 30 project characteristics | Paper metrics ($N, NC, RF, RS$), CPM ($EST, EFT, LST, LFT, TS$), $EP, RT_k$, Kahn's DAG sort |
| [`src/parser.py`](src/parser.py) | File Parser | Parses `.sm`, `.rcp`, `.json` files | PSPLIB file tokenizing, 0-to-1 job index shift, dummy source/sink activity injection |
| [`src/evaluation.py`](src/evaluation.py) | Evaluator | SSGS wrapper & constraint checker | Dual constraint validation: Precedence ($C_p \le S_i$) and Resource limit ($\sum r_{ik} \le R_k$) |
| [`src/benchmark.py`](src/benchmark.py) | Benchmarking | Multi-algorithm solver runner | Budget enforcement, seed isolation, `BenchmarkRecord` construction |
| [`src/comparison.py`](src/comparison.py) | Analytics | Statistical comparison engine | Mean $\mu$, Std $\sigma$, Feasibility rate, Hierarchical ranking (Best MS $\to$ Mean MS $\to$ Runtime) |
| [`src/algorithms/ga.py`](src/algorithms/ga.py) | Optimization | Genetic Algorithm | Tournament Selection ($k=3$), Uniform Blend Crossover, Creep Mutation ($p_m=0.05$), Elitism ($E=2$) |
| [`src/algorithms/sa.py`](src/algorithms/sa.py) | Optimization | Simulated Annealing | Metropolis Acceptance: $P = \exp(-\Delta / T)$, Geometric Cooling: $T_{k+1} = \alpha T_k$, Swap & Perturb operators |
| [`src/algorithms/bco.py`](src/algorithms/bco.py) | Optimization | Artificial Bee Colony | Employed Bee Perturbation ($v_{i,k} = x_{i,k} + \phi(x_{i,k}-x_{j,k})$), Onlooker Probability $p_i$, Scout reset |
| [`src/algorithms/tabu.py`](src/algorithms/tabu.py) | Optimization | Tabu Search | Pairwise index swap neighborhood, Tabu queue (Tenure $T=7$), Aspiration criterion ($M_{\text{cand}} < M_{\text{best}}$) |
| [`src/algorithms/prh.py`](src/algorithms/prh.py) | Optimization | Priority Rule Heuristics | Rule-to-Priority mapping: EST ($P_i = -EST_i$), EFT, LST, LFT, RAND ($P_i \sim U(0,1)$) |
| [`src/ml/dataset.py`](src/ml/dataset.py) | Machine Learning | Dataset container | PyTorch `Dataset` & NumPy array loader for algorithm selection dataset |
| [`src/ml/generate_dataset.py`](src/ml/generate_dataset.py) | Machine Learning | Dataset generator | Parallel processing across PSPLIB files, metaheuristic execution, feature-label matrix compilation |
| [`src/ml/train.py`](src/ml/train.py) | Machine Learning | Model training pipeline | Multi-model training (RF, XGBoost, LightGBM, MLP, SVM), Stratified K-Fold CV, hyperparameter tuning |
| [`src/ml/predict.py`](src/ml/predict.py) | Machine Learning | Inference engine | Feature extraction $\to$ `StandardScaler` $\to$ MLP prediction $\to$ Confidence Execution Policy |
| [`src/ml/conflict_analysis.py`](src/ml/conflict_analysis.py) | Analytics | Bottleneck analysis | Resource & precedence conflict matrix calculation |
| [`src/ml/generate_quality_report.py`](src/ml/generate_quality_report.py) | Quality Gate | Quality audit report | Dataset validation gate, class distribution audit, missing value checks |
| [`app.py`](app.py) | User Interface | Interactive Web Application | Streamlit GUI, file parsing, feature summary, ML recommendation, Plotly Gantt & Resource charts, NetworkX DAG |
| [`scripts/download_psplib_100.py`](scripts/download_psplib_100.py) | Utility Script | PSPLIB Benchmark Downloader | Automated retrieval of PSPLIB J30/J60/J120 problem sets |
| [`scripts/demo_end_to_end_ml.py`](scripts/demo_end_to_end_ml.py) | Utility Script | ML Workflow Demo | End-to-end pipeline execution demonstration |
| [`scripts/audit_dataset.py`](scripts/audit_dataset.py) | Utility Script | Dataset Auditor | Integrity check for training CSV files |

---

## 2. Detailed File-by-File Catalog & Formulas

---

### 📁 Core Domain Models & Scheduler

#### 📄 [`src/model.py`](src/model.py)
- **Role**: Defines the fundamental domain entities for RCPSP.
- **Classes**:
  - `Resource(id: str, capacity: int)`: Renewable resource type with capacity $R_k$.
  - `Activity(id: int, duration: int, predecessors: List[int], successors: List[int], resource_requirements: Dict[str, int])`: Activity parameters ($d_i, \mathcal{P}_i, \mathcal{S}_i, r_{ik}$).
  - `Project(activities: Dict[int, Activity], resources: Dict[str, Resource])`: Graph container. Properties `num_activities` and `num_resources`.
  - `ScheduleEntry(activity_id: int, start_time: int, finish_time: int, duration: int, resource_usage: Dict[str, int])`: Scheduled task window $[S_i, C_i]$ where $C_i = S_i + d_i$.
  - `AlgorithmResult(algorithm: str, best_priority_vector: List[float], schedule: List[ScheduleEntry], makespan: int, runtime: float, evaluations: int, convergence_history: List[tuple])`: Output contract for all solvers.

---

#### 📄 [`src/ssgs.py`](src/ssgs.py)
- **Role**: Serial Schedule Generation Scheme (SSGS) decoder — converts priority vectors into feasible schedules.
- **Mathematical Formulations & Logic**:
  1. **Precedence Earliest Start Time ($EST_i$)**:
     $$S_i \ge \max_{p \in \mathcal{P}_i} C_p$$
  2. **Resource Capacity Feasibility**:
     $$\sum_{i \in \mathcal{A}(t)} r_{ik} \le R_k \quad \forall k \in \{1, \dots, K\}, \; \forall t \in [S_i, S_i + d_i)$$
     where $\mathcal{A}(t) = \{j \mid S_j \le t < C_j\}$ is the set of active activities at time $t$.
  3. **Priority Selection**:
     $$i^* = \arg\max_{j \in \mathcal{E}} \lambda_j$$
     where $\mathcal{E}$ is the set of eligible activities (uncompleted predecessor count = 0) and $\lambda_j$ is the priority value.
  4. **Project Makespan ($M$)**:
     $$M = \max_{i \in \mathcal{V}} C_i$$

---

#### 📄 [`src/features.py`](src/features.py)
- **Role**: Feature Extraction Engine — computes 30 structural, precedence, resource, and critical path features.
- **Mathematical Formulations**:
  1. **Paper Activity Count ($N$)**: Real activities excluding dummy start and sink:
     $$N = N_{\text{total}} - 2$$
  2. **Network Complexity ($NC$)**: Ratio of precedence edges $|E|$ to real activity count $N$:
     $$NC = \frac{|E|}{N}$$
  3. **Resource Factor ($RF$)**: Average fraction of resource types required per activity:
     $$RF = \frac{1}{N \cdot K} \sum_{i=1}^{N} \sum_{k=1}^{K} \mathbb{I}(r_{ik} > 0)$$
  4. **Resource Strength ($RS$)**: Capacity availability relative to peak CPM demand:
     $$RS_k = \frac{R_k - R_k^{\min}}{R_k^{\max} - R_k^{\min}}, \quad R_k^{\min} = \max_{i} r_{ik}, \quad R_k^{\max} = \max_{t} \sum_{i \in \mathcal{A}_{\text{CPM}}(t)} r_{ik}$$
     $$RS = \frac{1}{K} \sum_{k=1}^K RS_k$$
  5. **Precedence Density ($PD$)**:
     $$PD = \frac{2 \cdot |E|}{N \cdot (N - 1)}$$
  6. **CPM Critical Path Duration ($CP_D$)**:
     $$EST_1 = 0; \quad EST_i = \max_{p \in \mathcal{P}_i} (EST_p + d_p); \quad CP_D = \max_{i} (EST_i + d_i)$$
  7. **Backward Pass & Total Slack ($TS_i$)**:
     $$LFT_{\text{sink}} = CP_D; \quad LFT_i = \min_{s \in \mathcal{S}_i} (LFT_s - d_s); \quad LST_i = LFT_i - d_i; \quad TS_i = LST_i - EST_i$$
  8. **Estimated Parallelism ($EP$)**:
     $$EP = \frac{\sum_{i=1}^N d_i}{CP_D}$$
  9. **Resource Tightness ($RT_k$)**:
     $$RT_k = \frac{\sum_{i=1}^N (d_i \cdot r_{ik})}{R_k \cdot CP_D}$$
  10. **Kahn's DAG Validation**: Topological sort and cycle detection algorithm.

---

#### 📄 [`src/parser.py`](src/parser.py)
- **Role**: Multi-format loader (`.sm`, `.rcp`, `.json`).
- **Functions**:
  - `Parser.parse(file_path)`: Format auto-detection based on file extension.
  - `_parse_sm()`: PSPLIB `.sm` file parser converting 0-based activity indices to 1-based job numbers.
  - `_parse_rcp()`: Custom `.rcp` tokenizing parser.
  - `_parse_json()`: Structured JSON parser.

---

#### 📄 [`src/evaluation.py`](src/evaluation.py)
- **Role**: Solution Evaluator & Constraint Validator.
- **Functions**:
  - `Evaluator.evaluate(project, priority_vector)`: Invokes `SSGS.generate_schedule()` and validates feasibility.
  - `Evaluator.check_constraints(project, schedule)`: Independent double-check ensuring precedence constraints ($C_p \le S_i$) and resource capacities ($\sum_{i \in \mathcal{A}(t)} r_{ik} \le R_k$) are strictly satisfied.

---

#### 📄 [`src/benchmark.py`](src/benchmark.py)
- **Role**: Multi-algorithm benchmark execution manager.
- **Functions & Dataclasses**:
  - `BenchmarkRecord`: Struct containing instance ID, algorithm name, seed, makespan, runtime, evaluations, feasibility status, priority vector, and convergence history.
  - `run_single()`: Executes a single algorithm under seed isolation and budget limit.
  - `run_all_algorithms()`: Executes all 5 algorithms (`GA`, `SA`, `BCO`, `PRH`, `TS`) across multiple random seeds.

---

#### 📄 [`src/comparison.py`](src/comparison.py)
- **Role**: Statistical comparison & ranking engine.
- **Formulas & Ranking Rules**:
  - Mean makespan: $\mu = \frac{1}{M} \sum_{m=1}^M M_m$
  - Sample standard deviation: $\sigma = \sqrt{\frac{1}{M} \sum_{m=1}^M (M_m - \mu)^2}$
  - Feasibility rate: $\text{Rate} = \frac{N_{\text{feasible}}}{N_{\text{runs}}}$
  - Hierarchical Ranking Criteria:
    1. Primary: Lowest best makespan.
    2. Secondary: Lowest mean makespan.
    3. Tertiary: Lowest mean runtime.

---

### 📁 Metaheuristic Optimization Solvers (`src/algorithms/`)

#### 📄 [`src/algorithms/ga.py`](src/algorithms/ga.py)
- **Role**: Genetic Algorithm solver.
- **Operators & Formulas**:
  - Chromosome: Random-key priority vector $\mathbf{x} \in [0, 1]^N$.
  - Tournament Selection ($k=3$): Selects best candidate among $k$ random samples.
  - Uniform Blend Crossover:
    $$c_1[i] = p_1[i] \text{ with prob } 0.5 \text{ else } p_2[i]$$
  - Creep Mutation ($p_m = 0.05$):
    $$x_i' = U(0, 1) \text{ if } U(0, 1) < p_m \text{ else } x_i$$
  - Elitism: Preserves top $E=2$ elite individuals across generations.

---

#### 📄 [`src/algorithms/sa.py`](src/algorithms/sa.py)
- **Role**: Simulated Annealing solver.
- **Operators & Formulas**:
  - Neighborhood Operators:
    - Swap operator: Swaps priority keys of two random activities.
    - Perturbation operator: $x_i \leftarrow \text{clamp}(x_i + \mathcal{N}(0, \sigma^2), 0, 1)$.
  - Metropolis Acceptance Criterion:
    $$P(\text{accept candidate}) = \begin{cases} 1 & \text{if } \Delta \le 0 \\ \exp\left(-\frac{\Delta}{T}\right) & \text{if } \Delta > 0 \end{cases}$$
    where $\Delta = M_{\text{candidate}} - M_{\text{current}}$.
  - Geometric Cooling Schedule:
    $$T_{k+1} = \alpha \cdot T_k \quad (\alpha = 0.995, \; T_0 = 100.0)$$

---

#### 📄 [`src/algorithms/bco.py`](src/algorithms/bco.py)
- **Role**: Artificial Bee Colony (ABC) Optimization solver.
- **Operators & Formulas**:
  - Employed Bee Perturbation:
    $$v_{i, k} = x_{i, k} + \phi \cdot (x_{i, k} - x_{j, k}), \quad \phi \sim U(-1, 1)$$
  - Onlooker Bee Selection Probability:
    $$p_i = \frac{w_i}{\sum_{m=1}^{SN} w_m}, \quad w_i = M_{\max} - M_i + 1$$
  - Scout Bee Phase: Abandon food source if trial counter exceeds $L=15$, re-initializing with a random vector.

---

#### 📄 [`src/algorithms/tabu.py`](src/algorithms/tabu.py)
- **Role**: Tabu Search solver.
- **Operators & Formulas**:
  - Neighborhood Generation: Pairwise index swaps $(i, j)$.
  - Tabu Memory Queue: Fixed tenure length $T_{\text{tenure}} = 7$ storing recent move pairs $(i, j)$.
  - Aspiration Criterion: Accepts tabu move if makespan strictly improves global best:
    $$M_{\text{candidate}} < M_{\text{best}}$$

---

#### 📄 [`src/algorithms/prh.py`](src/algorithms/prh.py)
- **Role**: Priority Rule-based Heuristic solver.
- **Rules & Formulas**:
  - **EST Rule**: $P_i = -EST_i$ (Earliest Start Time first).
  - **EFT Rule**: $P_i = -EFT_i$ (Earliest Finish Time first).
  - **LST Rule**: $P_i = -LST_i$ (Latest Start Time first).
  - **LFT Rule**: $P_i = -LFT_i$ (Latest Finish Time first).
  - **RAND Rule**: $P_i \sim U(0, 1)$ (Random priorities).

---

### 📁 Machine Learning Engine (`src/ml/`)

#### 📄 [`src/ml/dataset.py`](src/ml/dataset.py)
- **Role**: PyTorch & NumPy dataset loader for RCPSP algorithm selection.
- **Classes**: `RCPSPDataset(Dataset)` converting feature matrices and algorithm labels into PyTorch Tensors.

---

#### 📄 [`src/ml/generate_dataset.py`](src/ml/generate_dataset.py)
- **Role**: Parallel dataset generation pipeline.
- **Logic**: Executes all 5 metaheuristics on PSPLIB instances across multiple random seeds, extracts 30 project features, determines winning algorithm label, and exports training CSVs.

---

#### 📄 [`src/ml/train.py`](src/ml/train.py)
- **Role**: Multi-model training and evaluation pipeline.
- **Models Trained**: Random Forest, XGBoost, LightGBM, MLPClassifier, SVM.
- **Evaluation**: Stratified K-Fold Cross Validation, accuracy, macro-F1, confusion matrix generation, model serialization to `models/`.

---

#### 📄 [`src/ml/predict.py`](src/ml/predict.py)
- **Role**: Inference API & Confidence Execution Policy engine.
- **Logic & Policy Rules**:
  1. Feature extraction $\to$ `StandardScaler` normalization.
  2. Model inference producing class probability vector $\mathbf{p} = [p_1, \dots, p_C]$.
  3. **Confidence Execution Policy**:
     - $\text{Confidence} \ge 0.80$: Execute top single predicted algorithm.
     - $0.50 \le \text{Confidence} < 0.80$: Execute top 2 predicted algorithms.
     - $\text{Confidence} < 0.50$ or predicted tie (`ALL`): Execute all 5 algorithms.

---

#### 📄 [`src/ml/conflict_analysis.py`](src/ml/conflict_analysis.py)
- **Role**: Resource and precedence bottleneck analysis engine.
- **Calculations**: Precedence concurrency bounds and resource demand overlap matrices.

---

#### 📄 [`src/ml/generate_quality_report.py`](src/ml/generate_quality_report.py)
- **Role**: Step 0 Quality Gate audit report generator.
- **Checks**: Class balance, feature value ranges, missing values, duplicate records.

---

### 📁 User Interface & Utility Scripts

#### 📄 [`app.py`](app.py)
- **Role**: Interactive Streamlit web application.
- **Features**:
  - File uploader supporting `.sm`, `.rcp`, `.json`.
  - Feature summary dashboard displaying paper metrics ($N, NC, RF, RS$).
  - AI Model Recommendation with confidence scores.
  - Interactive solver execution (`GA`, `SA`, `BCO`, `PRH`, `TS`).
  - Visualizations: Plotly Gantt charts, resource utilization curves, NetworkX DAG precedence graphs.

---

#### 📄 [`scripts/download_psplib_100.py`](scripts/download_psplib_100.py)
- **Role**: Automated PSPLIB benchmark dataset downloader. downloads J30, J60, J120 benchmark instances into `data/`.

---

#### 📄 [`scripts/demo_end_to_end_ml.py`](scripts/demo_end_to_end_ml.py)
- **Role**: Demonstration script executing full end-to-end workflow: parsing $\to$ feature extraction $\to$ model prediction $\to$ algorithm execution.

---

#### 📄 [`scripts/audit_dataset.py`](scripts/audit_dataset.py)
- **Role**: CLI tool for auditing dataset CSV file validity.

---

### 📁 Unit & Integration Test Suite (`tests/`)

| Test File | Target Module Tested | Description |
| :--- | :--- | :--- |
| `tests/test_algorithms.py` | `src/algorithms/*` | Tests convergence, deterministic execution with seeds, and schedule validity for GA, SA, BCO, TS, PRH |
| `tests/test_benchmark.py` | `src/benchmark.py` | Tests benchmark runner execution, result recording, and seed isolation |
| `tests/test_conflict_analysis.py` | `src/ml/conflict_analysis.py` | Tests conflict matrix generation and bottleneck metrics |
| `tests/test_constraints.py` | `src/evaluation.py` | Tests constraint checking logic against valid and invalid schedules |
| `tests/test_dataset.py` | `src/ml/dataset.py` | Tests dataset loading, PyTorch tensor conversion, and batching |
| `tests/test_dataset_resume.py` | `src/ml/generate_dataset.py` | Tests dataset generator resume capability on interrupted runs |
| `tests/test_features.py` | `src/features.py` | Validates formulas for $N, NC, RF, RS$, CPM calculations, and Kahn's DAG sort |
| `tests/test_ml_prediction.py` | `src/ml/predict.py` | Tests feature scaling, MLP inference, and confidence threshold policies |
| `tests/test_ml_training.py` | `src/ml/train.py` | Tests multi-model classifier training pipeline and metric reporting |
| `tests/test_parser.py` | `src/parser.py` | Tests `.rcp` and `.json` file parsing |
| `tests/test_parser_sm.py` | `src/parser.py` | Tests PSPLIB `.sm` file parsing and 1-based job number alignment |
| `tests/test_ssgs.py` | `src/ssgs.py` | Tests Serial Schedule Generation Scheme decoding and deadlock detection |

---
