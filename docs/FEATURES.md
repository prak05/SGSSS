# Project Features Documentation (AI-RCPSP)

This document provides a comprehensive reference for all project characteristics extracted by [`src/features.py`](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/features.py).

The feature extractor translates an RCPSP instance into a fixed-length numerical descriptor used for topological analysis, project characterization, and downstream algorithm selection.

---

## 1. Primary Paper Features

These four characteristics are the primary inputs described and used in the reference paper:
> **Reference Paper:**  
> Nikolaos T. Mitsou and Georgios K. Koulinas (2026), *"AI-Enhanced Metaheuristics Control for Project Scheduling Using Neural Networks"*, IEEE Access, Vol. 14, pp. 33600–33610.  
> DOI: 10.1109/ACCESS.2026.3668729.

---

### 1.1 Activity Count ($N$)
- **Feature Name**: `N`
- **Source**: Research Paper (Mitsou & Koulinas 2026, Section III-A)
- **Definition**: The total number of non-dummy activities in the project network.
- **Paper Citation**: *"Initially, for each project, the number of activities (N) is defined as the total number of activities of the project excluding the dummy activities 'Start' and 'Finish' of the project."*
- **Formula**:
  $$N = J - 2$$
  where $J$ is the total number of activities in standard PSPLIB files including dummy source (job 1) and dummy sink (job $J$).
- **Units**: Integer count
- **Expected Range**: $N \in [1, \infty)$ (Standard PSPLIB benchmarks: $J30 \implies N=30$, $J60 \implies N=60$, $J120 \implies N=120$).
- **Implementation Details**:
  Activities with duration 0, zero demands across all resources, and either in-degree 0 (start) or out-degree 0 (finish) are recognized as dummy activities and excluded from $N$. If a project contains no dummy activities (e.g., custom formats), $N$ equals the total activity count.
- **Edge Cases**:
  If all activities have duration 0 and zero demand, the extractor treats all activities as active to prevent division by zero.

---

### 1.2 Network Complexity ($NC$)
- **Feature Name**: `NC`
- **Source**: Research Paper (Mitsou & Koulinas 2026, Section III-A) & PSPLIB (Kolisch & Sprecher 1997, RanGen 2003)
- **Definition**: The average number of non-redundant precedence relations (precedence arcs) per non-dummy activity.
- **Paper Citation**: *"The network complexity (NC) in project management refers to the number of activities and the total number of different combinations among them when creating a schedule within a project network... describe and explain the constraints related to the average number of successors per unscheduled activity [47]."*
- **Formula**:
  $$NC = \frac{|E|}{N}$$
  where $|E|$ is the total number of directed precedence arcs in the project network, and $N$ is the number of non-dummy activities.
- **Units**: Dimensionless ratio (arcs per activity)
- **Expected Range**: Standard PSPLIB parameters: $NC \in \{1.5, 1.8, 2.1\}$. In general DAGs: $NC \ge 0.0$.
- **Implementation Details**:
  $|E| = \sum_{j} |\text{successors}(j)|$. Division is guarded against $N = 0$.
- **Edge Cases**:
  If $N = 0$, $NC = 0.0$.

---

### 1.3 Resource Factor ($RF$)
- **Feature Name**: `RF`
- **Source**: Research Paper (Mitsou & Koulinas 2026, Section III-A, Eq. 5) & Pascoe (1966)
- **Definition**: The average fraction of resource types requested per activity across the project.
- **Paper Citation**: *"The resource factor (RF) refers to the average percentage of resources consumed across each resource category in the project [45]."*
- **Formula**:
  $$RF = \frac{1}{N \cdot |K|} \sum_{j=1}^{N} \sum_{k=1}^{|K|} Q_{jk}$$
  where $Q_{jk} = 1$ if activity $j$ requires a strictly positive amount of resource $k$ ($r_{jk} > 0$), and $0$ if $r_{jk} = 0$. $|K|$ is the number of renewable resource categories.
- **Units**: Dimensionless fraction $\in [0, 1]$
- **Expected Range**: $RF \in [0.0, 1.0]$. In PSPLIB benchmarks: $RF \in \{0.25, 0.50, 0.75, 1.00\}$.
  - $RF = 0.0 \implies$ no activity requires any resource.
  - $RF = 1.0 \implies$ every non-dummy activity requires every available resource type.
- **Implementation Details**:
  Evaluated only over non-dummy activities. Guarded against $|K| = 0$ or $N = 0$.
- **Edge Cases**:
  If no resources exist in the project, $RF = 0.0$.

---

### 1.4 Resource Strength ($RS$)
- **Feature Name**: `RS`
- **Source**: Research Paper (Mitsou & Koulinas 2026, Section III-A, Eq. 6), Cooper (1976), Kolisch & Sprecher (1997)
- **Definition**: Measures the availability (scarcity vs. abundance) of each renewable resource relative to the unconstrained early-start Critical Path Method (CPM) demand.
- **Paper Citation**: *"Finally, the term resource strength (RS) was introduced by Cooper [46] and measures resource availability KrT as a combination of the minimum Krmin and maximum Krmax levels of resource availability throughout the project duration [46], [47]. $Kr^T = Kr^{min} + \text{round}(RS^T (Kr^{max} - Kr^{min}))$"*
- **Formula**:
  Inverting the generative formula for a project with given capacities $C_k$:
  $$RS_k = \begin{cases} \frac{C_k - K_k^{min}}{K_k^{max} - K_k^{min}} & \text{if } K_k^{max} > K_k^{min} \\ 1.0 & \text{if } K_k^{max} \le K_k^{min} \end{cases}$$
  $$RS = \frac{1}{|K|} \sum_{k=1}^{|K|} \min(1.0, \max(0.0, RS_k))$$
  where:
  - $K_k^{min} = \max_{j} r_{jk}$ is the maximum demand for resource $k$ requested by any single activity.
  - $K_k^{max}$ is the peak instantaneous demand for resource $k$ when all activities are scheduled at their Early Start times ($ES_j$) in the precedence-only CPM schedule.
- **Units**: Dimensionless index $\in [0, 1]$
- **Expected Range**: $RS \in [0.0, 1.0]$. In PSPLIB benchmarks: $RS \in \{0.2, 0.5, 0.7, 1.0\}$.
  - $RS = 0.0 \implies$ minimal feasible capacity; capacity equals the single largest task demand (extreme bottleneck).
  - $RS = 1.0 \implies$ capacity equals or exceeds the peak early-start demand (resource constraints do not delay the CPM schedule).
- **Implementation Details**:
  Peak demand $K_k^{max}$ is sampled across all event transition boundaries $t \in \{ES_j\}$ throughout the unconstrained CPM horizon. Clamped to $[0.0, 1.0]$ to conform to canonical literature standards.
- **Edge Cases**:
  If $K_k^{max} == K_k^{min}$, resource demand cannot cause concurrency conflicts beyond single-activity limits; $RS_k = 1.0$.

---

## 2. Additional Project Features

In addition to the 4 paper features, 22 supplementary characteristics are extracted to provide detailed structural, topological, resource-utilization, and critical-path information.

### 2.1 Basic Properties
| Feature Name | Type | Definition | Units | Range |
|---|---|---|---|---|
| `activity_count` | int | Total number of activities (including start/sink dummies). | count | $[1, \infty)$ |
| `resource_count` | int | Total number of renewable resource types ($|K|$). | count | $[0, \infty)$ |
| `average_duration` | float | Mean duration among non-dummy activities. | time units | $\ge 0.0$ |
| `duration_std` | float | Population standard deviation of non-dummy activity durations. | time units | $\ge 0.0$ |
| `minimum_duration` | int | Shortest duration among non-dummy activities. | time units | $\ge 0$ |
| `maximum_duration` | int | Longest duration among non-dummy activities. | time units | $\ge 0$ |
| `total_duration` | int | Sum of durations across all non-dummy activities. | time units | $\ge 0$ |

### 2.2 Precedence & Network Topology
| Feature Name | Type | Definition | Units | Range |
|---|---|---|---|---|
| `precedence_edge_count` | int | Total number of direct precedence arcs ($|E|$). | count | $[0, \infty)$ |
| `precedence_density` | float | Arc density of the DAG: $\frac{|E|}{V(V-1)/2}$, where $V = \text{activity\_count}$. | ratio | $[0.0, 1.0]$ |
| `average_predecessors` | float | Average in-degree across all activities: $|E| / V$. | arcs/act | $\ge 0.0$ |
| `average_successors` | float | Average out-degree across all activities: $|E| / V$. | arcs/act | $\ge 0.0$ |
| `maximum_predecessors` | int | Maximum in-degree observed in any single activity. | count | $[0, V-1]$ |
| `maximum_successors` | int | Maximum out-degree observed in any single activity. | count | $[0, V-1]$ |
| `source_count` | int | Number of root activities (in-degree = 0). | count | $[1, V]$ |
| `sink_count` | int | Number of leaf activities (out-degree = 0). | count | $[1, V]$ |
| `network_depth` | int | Length of the longest edge path from any source to any sink (topological depth). | edges | $[0, V-1]$ |

### 2.3 Parallelism
| Feature Name | Type | Definition | Units | Range |
|---|---|---|---|---|
| `estimated_parallelism` | float | Ratio of total non-dummy work to critical path makespan: $\frac{\sum d_j}{CPM\_duration}$. | ratio | $[1.0, N]$ |
| `average_parallel_activities` | float | Average number of concurrent tasks executing simultaneously in the early-start CPM schedule: $\frac{\int \text{active}(t) dt}{CPM\_duration} \equiv \frac{\sum d_j}{CPM\_duration}$. | tasks | $[1.0, N]$ |

### 2.4 Resource Demands & Tightness
| Feature Name | Type | Definition | Units | Range |
|---|---|---|---|---|
| `average_resource_demand` | float | Mean resource demand $r_{jk}$ across all non-dummy activities and resources. | resource units | $\ge 0.0$ |
| `resource_demand_std` | float | Population standard deviation of resource demands $r_{jk}$. | resource units | $\ge 0.0$ |
| `resource_utilization_ratio` | float | Average ratio of total resource work $\sum_j (d_j \cdot r_{jk})$ to the theoretical bounding box $(CPM\_duration \cdot C_k)$. | ratio | $[0.0, 1.0]$ |
| `average_resource_tightness` | float | Mean across resources of peak early-start demand relative to capacity: $\frac{1}{|K|}\sum \frac{K_k^{max}}{C_k}$. | ratio | $\ge 0.0$ |
| `maximum_resource_tightness` | float | Maximum across all resources of peak early-start demand relative to capacity: $\max_k \frac{K_k^{max}}{C_k}$. | ratio | $\ge 0.0$ |
| `number_of_unused_resource_dimensions` | int | Number of resource types that have zero demand across all project activities. | count | $[0, |K|]$ |

### 2.5 Critical Path (Schedule-Independent)
| Feature Name | Type | Definition | Units | Range |
|---|---|---|---|---|
| `critical_path_duration` | int | Precedence-only project makespan obtained via Critical Path Method (CPM forward pass, without resource limits). | time units | $\ge 0$ |
| `critical_path_activity_count` | int | Number of non-dummy activities with zero total float ($LS_j == ES_j$). | count | $[1, N]$ |

---

## 3. Deterministic Feature Schema & Ordering

The `ProjectFeatures.to_vector()` and `ProjectFeatures.feature_names()` methods return the features in the exact following deterministic order:

```python
[
    "N",
    "NC",
    "RF",
    "RS",
    "activity_count",
    "resource_count",
    "average_duration",
    "duration_std",
    "minimum_duration",
    "maximum_duration",
    "total_duration",
    "precedence_edge_count",
    "precedence_density",
    "average_predecessors",
    "average_successors",
    "maximum_predecessors",
    "maximum_successors",
    "source_count",
    "sink_count",
    "network_depth",
    "estimated_parallelism",
    "average_parallel_activities",
    "average_resource_demand",
    "resource_demand_std",
    "resource_utilization_ratio",
    "average_resource_tightness",
    "maximum_resource_tightness",
    "number_of_unused_resource_dimensions",
    "critical_path_duration",
    "critical_path_activity_count",
]
```

Total features: **30 features** (4 paper features + 26 engineering features).

---

## 4. DAG Validation Rules

Before feature extraction, the precedence structure is validated against the following conditions:
1. **At least one activity**: Projects with 0 activities raise `ValueError("Project has zero activities...")`.
2. **Referential integrity**: All predecessor IDs must exist in `project.activities`. If an activity references an unknown predecessor ID, a `ValueError("Activity X references non-existent predecessor Y")` is raised.
3. **Acyclic graph**: Directed cycle detection is performed using Kahn's algorithm. If a cycle is detected, topological ordering halts and a descriptive `ValueError` is raised listing the cyclic activities.
