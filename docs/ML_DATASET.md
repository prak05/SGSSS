# Machine Learning Dataset Documentation (AI-RCPSP)

This document describes the design, schema, aggregation methodology, and data splitting principles for the supervised learning dataset generated in Phase 5.

---

## 1. Original Paper vs. Project Extension

| Dimension | Original Research Paper (Mitsou & Koulinas 2026) | Our Project Extension |
|---|---|---|
| **Algorithms** | 4 algorithms: GA, SA, BCO, PRH | **Exactly 5 algorithms**: GA, SA, BCO, PRH, Tabu Search (TS) |
| **Search Space** | Random-key vector priority representation | Unified random-key representation via common SSGS |
| **Input Features** | 3 macroscopic parameters: $NC, RF, RS$ (plus $N$ for grouping) | **30 deterministic features**: $N, NC, RF, RS$ + 26 topological, resource & CPM features |
| **Instances** | PSPLIB J60 and J120 activity sets | PSPLIB instances across sizes ($J10, J30, J60, J90, J120$) |
| **Evaluation Budget** | 5,000 schedule evaluations per instance | Standardized 5,000 evaluations per algorithm run |
| **Target Classes** | 5 classes: `GA`, `SA`, `BCO`, `PRH`, `ALL` | **6 classes**: `GA`, `SA`, `BCO`, `PRH`, `TS`, `ALL` |
| **Implementation** | MATLAB Neural Network Toolbox | Python 3.11+, modular architecture, safe checkpointing |

> **Critical Note on Non-Existent Algorithms**:  
> Particle Swarm Optimization (PSO) is **not implemented** in this project and is strictly excluded from all datasets, labels, and schemas.

---

## 2. Dataset Files and Directory Structure

The training dataset is structured under [`data/training/`](data/training/):

```
data/training/
├── raw_results.csv          # Granular seed-level results (1 row per instance × algorithm × seed)
├── training_dataset.csv     # Main aggregated dataset for model training (1 row per instance)
├── label_distribution.csv   # Class distribution summary (counts and percentages)
├── feature_schema.json      # Metadata schema definition and deterministic feature ordering
└── convergence/             # JSON files containing per-run (evaluation, best_makespan) traces
```

---

## 3. Label Determination and Tie Policy

### Primary Objective: Minimise Makespan
The classification label represents the algorithm that found the lowest feasible project makespan under identical evaluation conditions ($5,000$ evaluations).

### Multi-Seed Aggregation Methodology
Labels are **never** assigned from a single random run. For each project instance and each algorithm, experiments are executed across independent seeds ($1, 2, 3, 4, 5$).

### The `ALL` Class (Exact Five-Way Tie)
- In accordance with Mitsou & Koulinas (2026), when all candidate algorithms achieve the **exact same best feasible makespan**, the instance is categorized as **`ALL`**.
- This class informs the downstream selector that the project network topology does not favor any specific search strategy over others, eliminating unnecessary search overhead.

### Partial-Tie Resolution Policy
When a subset of algorithms tie for the best makespan:
1. **Primary tie-break**: Lowest mean makespan across all seeds.
2. **Secondary tie-break**: Lowest mean runtime across seeds.
3. **Full audit trail**: The field `tied_best_algorithms` records all algorithms that achieved the minimum makespan (e.g., `"GA,TS"`), and `label_basis` records the decision rationale (`"exact_five_way_tie"`, `"unique_best_makespan"`, or `"partial_tie_broken_by_mean_makespan"`).

---

## 4. Dataset Schema

### `training_dataset.csv`
Each row corresponds to an aggregated project instance:
- **Identification**: `instance_id`, `instance_family`, `instance_size`
- **Features**: Exact 30 ProjectFeatures columns (`N`, `NC`, `RF`, `RS`, `activity_count`, ..., `critical_path_activity_count`)
- **Target Label**: `best_algorithm` (`GA`, `SA`, `BCO`, `PRH`, `TS`, or `ALL`)
- **Audit Columns**:
  - `label_basis`
  - `tied_best_algorithms`
  - `best_feasible_makespan`
  - Individual algorithm best makespan, mean makespan, and mean runtime (`GA_best_ms`, `GA_mean_ms`, etc.)

### `raw_results.csv`
Contains granular data per run:
- `instance_id`, `instance_size`, `seed`, `evaluation_budget`
- `algorithm`, `makespan`, `runtime`, `evaluations`, `feasible`
- All 30 project features
- `best_priority_vector` (JSON serialized float list)
- `notes`

---

## 5. Conflict Analysis (One-to-Many Mappings)

Mitsou & Koulinas (2026) discuss label conflicts that arise when project instances with identical high-level parameters ($NC, RF, RS$) possess different latent structures that favor different algorithms.

Our system provides [`src/ml/conflict_analysis.py`](src/ml/conflict_analysis.py) to measure:
1. **Exact duplicate feature vectors**: Instances sharing identical numerical vectors.
2. **Consistent duplicates**: Instances with identical feature vectors and identical labels.
3. **Conflicting duplicates**: Instances with identical feature vectors but different labels.

By expanding from 3 parameters to 30 topological and resource characteristics, feature-space collisions are drastically reduced. Any collisions that do occur are reported transparently.

---

## 6. Guidelines for Model Training & Data Splitting (Phase 6 Preparation)

To prevent data leakage when training the neural network:
1. **Split by Instance Family**: Never place instances from the same ProGen parameter block (or sister variants like `j6010_1` and `j6010_2`) in both training and testing sets.
2. **Stratification**: Ensure the `ALL` class and minority algorithm classes are represented proportionally across splits.
3. **No ML Artifacts in Phase 5**: No scalers, encoders, or classifiers are trained or persisted during this phase.

---

## 7. Research Validity & Interpretation

The target label `best_algorithm` answers:
> *"Which algorithm found the superior feasible schedule under our standardized experimental protocol (5,000 evaluations, multi-seed search)?"*

It does **not** assert mathematical global optimality, as metaheuristics operate by stochastic heuristic search.
