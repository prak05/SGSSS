# 🧠 Graph Neural Network (GNN) in AI-RCPSP: Concept, Architecture & Workflow

> **Executive Summary:**  
> This document explains what Graph Neural Networks (GNNs) are, why they were introduced into the AI-RCPSP project, how they fundamentally differ from traditional MLPs, how the message-passing mechanism works, how training and inference are set up, and how GNN priority seeding delivers guaranteed-feasible project schedules in milliseconds.

---

## 1. 💡 What is a GNN? (In Simple Words)

A **Graph Neural Network (GNN)** is an AI architecture designed specifically to process and learn from networks, flowcharts, and relational structures (graphs).

- **Standard Neural Networks (MLPs / CNNs):** Expect fixed tables of independent rows or rigid pixel grids. They cannot naturally navigate variable-sized networks of arrows and dependencies.
- **Graph Neural Networks (GNNs):** Directly ingest **nodes** (activities/tasks) and **edges** (the dependency arrows connecting them). Instead of treating numbers in isolation, a GNN learns by having connected tasks communicate with their neighbors.

---

## 2. ❓ How Did It Come into Our Project?

In the **Resource-Constrained Project Scheduling Problem (RCPSP)**, a project is fundamentally a **Directed Acyclic Graph (DAG)**:
- Task $A$ must finish before Task $B$ can start ($A \to B$).
- Task $B$ must finish before Task $C$ can start ($B \to C$).
- Each task consumes specific amounts of renewable resources (e.g., manpower, equipment).

### The Limitation of the Original Paper Approach
The classical methodology (Mitsou & Koulinas, 2026) collapsed this entire dependency network into **30 flat summary numbers** (e.g., average duration, Network Complexity, Resource Factor, Resource Strength). These summary numbers were then fed into a basic Multi-Layer Perceptron (MLP) or Random Forest just to pick an algorithm (`GA`, `SA`, `BCO`, etc.).

### The Core Motivation
Compressing an entire project flowchart into 30 averages destroys local critical chains, path bottlenecks, and localized resource jams.  
**The GNN was introduced to eliminate this loss of structural fidelity** by operating directly on the actual Directed Acyclic Graph topology.

---

## 🖼️ Architectural Workflow Diagram

![GNN vs MLP Architecture Diagram](docs/gnn_vs_mlp_workflow.jpg)

*Figure 1: Architectural comparison between the Traditional MLP Pipeline (lossy tabular compression) and the Proposed GNN Pipeline (topological bidirectional message passing with Dual-Head outputs).*

---

## 3. ⚖️ How Does It Work Differently from the Existing System?

| Dimension | Existing System (Tabular MLP / Paper Approach) | The GNN System (`AI-RCPSP`) |
| :--- | :--- | :--- |
| **Input Representation** | 30 global macro-averages (NC, RF, RS, duration stats). | Full project DAG: every task, its duration, resources, and precedence edges. |
| **City Analogy** | Describing a city solely by *average building height* and *number of intersections*. You lose the street layout! | Walking down the actual streets, inspecting traffic flow, and locating exact intersections that cause jams. |
| **Output Capability** | **Single-Level:** Predicts only which metaheuristic solver to launch (`GA`, `SA`, `BCO`...). | **Dual-Head:**<br>1. Predicts best solver (`GA`, `SA`, `BCO`, `TS`, `PRH`).<br>2. **Predicts node priority scores for every activity.** |
| **Scheduling Speed** | Metaheuristic takes 500+ iterations (**~50–100 ms**). | Direct SSGS seeding generates near-optimal schedules in **~3 ms (1-shot)**. |
| **Scalability** | Macro-averages become less descriptive as project size scales. | Message passing scales naturally with DAG size and topological depth. |

### 🔬 Deep Dive: What the GNN is Doing Differently Than the MLP (In-Depth Technical Comparison)

To understand why the GNN represents a fundamental paradigm shift rather than just an incremental tweak, consider these five core dimensions:

#### 1. Mathematical Input Space: Fixed Vector vs. Relational Topology
* **What the MLP does:**
  The MLP requires a fixed-width vector $\mathbf{z} \in \mathbb{R}^{30}$. To feed an arbitrary project into an MLP, every activity duration, relationship, and resource must be aggregated into scalar summary statistics:
  $$\mathbf{z} = \begin{bmatrix} N, & NC, & RF, & RS, & \mu_{\text{dur}}, & \sigma_{\text{dur}}, & \dots \end{bmatrix}^T$$
  Two completely different projects—one with a strict serial chain of 30 tasks, and another with three wide parallel branches of 10 tasks each—can produce nearly identical macro-statistics. The MLP is mathematically blind to the difference.
* **What the GNN does:**
  The GNN operates on the actual graph tuple $G = (\mathbf{X}, \mathbf{E})$, where $\mathbf{X} \in \mathbb{R}^{N \times 12}$ preserves the feature vector of every single activity $i$, and $\mathbf{E} \in \mathbb{R}^{2 \times |E|}$ maintains exact pairwise precedence connections. No graph flattening or structural destruction occurs.

#### 2. Permutation Equivariance vs. Arbitrary Feature Order
* **What the MLP does:**
  An MLP treats input dimension index $k$ as a rigid, position-dependent scalar. It has no internal concept of activities, nodes, or re-ordering.
* **What the GNN does:**
  A GNN is strictly **permutation equivariant** at the node level. If the activities in a project are renumbered or reordered, the GNN's predicted priorities reorder in the exact same permutation:
  $$f_{\text{GNN}}(\mathbf{P} \mathbf{X}, \mathbf{P} \mathbf{A} \mathbf{P}^T) = \mathbf{P} f_{\text{GNN}}(\mathbf{X}, \mathbf{A})$$
  This means the model learns invariant structural scheduling principles rather than memorizing task numbering artifacts.

#### 3. Receptive Field: Global Collapse vs. Multi-Hop Neighborhood Expansion
* **What the MLP does:**
  The MLP performs a single global matrix multiplication $\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1$. Every feature is indiscriminately mixed in one step. It cannot capture localized critical paths.
* **What the GNN does:**
  The GNN executes localized **message passing** across $L$ stacked layers:
  * **Layer 1 (1-hop):** Each task exchanges information with its immediate predecessors and successors. It learns about its direct dependencies.
  * **Layer 2 (2-hop):** Each task aggregates information from predecessors of predecessors. It discovers early-stage dependencies and downstream bottlenecks.
  * **Layer 3 (3-hop / Global Context):** The critical chain, parallel fork structures, and convergence hubs are fully mapped into the latent node embedding $\mathbf{h}_i^{(3)}$.

#### 4. Bottleneck & Resource Contention Discovery
* **What the MLP does:**
  The MLP only observes global resource aggregates like Resource Factor ($RF \in [0, 1]$) and Resource Strength ($RS \in [0, 1]$). It knows *whether the project as a whole is resource-tight*, but it has no idea *which specific activities will collide on Resource 1 at time $t=18$*.
* **What the GNN does:**
  Because the GNN's bidirectional convolutional layer aggregates messages along both forward ($u \to v$) and backward ($v \to u$) edges, tasks that lie on high-demand convergence bottlenecks naturally accumulate large gradient activations. The network learns to prioritize tasks that unblock the largest downstream critical chains.

#### 5. Role in the Optimization Ecosystem: Classifier vs. Direct Neuro-Scheduler
* **What the MLP does:**
  The MLP is strictly a **Level 1 Algorithm Selector**. It can only say: *"Run Genetic Algorithm."* It cannot generate a single schedule itself. You still have to spend hundreds of iterations waiting for GA or SA to randomly mutate schedules.
* **What the GNN does:**
  The GNN serves a **dual role**:
  1. **Graph Level:** It predicts the best metaheuristic solver via global pooling.
  2. **Node Level:** It outputs a continuous priority vector $\mathbf{s} \in (0, 1)^N$. Seeding this vector into the Serial Schedule Generation Scheme (SSGS) produces a high-quality, guaranteed-feasible schedule in **~3.2 milliseconds**.

---

## 4. 🚀 Why Is It Better?

1. **Zero Structural Information Loss:**  
   The network directly processes task-to-task relationships, preserving critical chains and localized resource competition that summary statistics wash out.
2. **Instant Schedules (<4 milliseconds):**  
   Metaheuristics (`GA`, `SA`, `BCO`) rely on stochastic trial-and-error loops (evaluating 500+ priority vectors). The GNN infers a high-quality priority vector in a single forward pass, providing an immediate solution.
3. **100% Feasibility Guarantee:**  
   The GNN does not blindly output start times (which could violate constraints). Instead, it outputs continuous **priority scores** to the **Serial Schedule Generation Scheme (SSGS)** decoder. SSGS strictly enforces all precedence and capacity bounds at every time step.

---

## 5. 🔄 How Does the Network Actually Work? ("Message Passing")

Imagine every task in the project is a worker holding a walkie-talkie connected to its direct dependencies:

```
[Task 1 (Site Prep)] ────► [Task 2 (Foundation)] ────► [Task 3 (Framing)]
```

### Step 1: Node Feature Initialization (What Each Task Knows)
Each activity node $i$ starts with a 12-dimensional normalized feature vector:
1. `norm_duration`: Task processing time normalized by maximum activity duration.
2. `mean_resource_ratio`: Average resource demand relative to total capacities.
3. `max_resource_ratio`: Peak resource demand relative to capacities.
4. `est_norm`: Earliest Start Time ($EST$) from unconstrained CPM pass.
5. `eft_norm`: Earliest Finish Time ($EFT$).
6. `lst_norm`: Latest Start Time ($LST$).
7. `lft_norm`: Latest Finish Time ($LFT$).
8. `slack_norm`: Total Slack $(LST - EST)$ indicating schedule flexibility.
9. `in_degree_norm`: Inflow of dependencies (predecessor count).
10. `out_degree_norm`: Outflow of dependencies (successor count).
11. `is_dummy_start`: Milestone indicator for project start.
12. `is_dummy_finish`: Milestone indicator for project completion.

### Step 2: Bidirectional Message Passing (`DirectedGraphConvLayer`)
Information propagates along directed edges through multiple message-passing rounds:
1. **Downstream Pass (Forward Precedence):**  
   Predecessors pass completion times and resource burdens downstream to their successors:  
   *“Here is when I will finish, and here are the resource quantities I will occupy.”*
2. **Upstream Pass (Backward Urgency):**  
   Successors pass deadline pressures upstream to their predecessors:  
   *“I am on the critical path with zero slack. Any delay on your end directly delays project delivery!”*

Each round aggregates incoming and outgoing messages with degree normalization, Layer Normalization, GELU activations, and residual skip connections:

$$h_i^{(l+1)} = \sigma \Big( \text{LayerNorm}\big( W_{self} h_i^{(l)} + \text{Agg}_{p \in Pred(i)}(W_{fwd} h_p^{(l)}) + \text{Agg}_{s \in Succ(i)}(W_{bwd} h_s^{(l)}) + b \big) \Big) + h_i^{(l)}$$

### Step 3: Dual-Head Output Decoding
- **Head 1 (Node Priority Head):**  
  Passes node embeddings through an MLP ($\text{Linear} \to \text{GELU} \to \text{Linear} \to \text{Sigmoid}$) to output a priority score $s_i \in (0, 1)$ for every activity.
- **Head 2 (Graph Readout Head):**  
  Applies global mean-pooling and max-pooling across all node embeddings to produce a project-level embedding, then predicts the optimal metaheuristic algorithm class.

---

## 6. 🏋️ Is Training Required for It?

**Yes, but it is fast, lightweight, and offline.**

- **Training Paradigm (Imitation Learning):**  
  The GNN is trained on benchmark instances (e.g., PSPLIB). For each instance, we determine the best-performing priority vector discovered by classical heuristics and metaheuristics. The network is trained using Mean Squared Error (MSE) loss on node priorities and Cross-Entropy loss on algorithm classification.
- **Training Speed:**  
  Because the architecture uses pure PyTorch tensor operations without heavy C++ graph libraries, training on 80 project graphs for 15 epochs takes **less than 5 seconds** on a standard CPU.
- **Inference (Zero-Retraining):**  
  The trained weights are saved in `models/gnn_dual_head.pt`. During regular operation or in the Streamlit web dashboard, the model loads the pre-trained weights instantly and produces schedule priorities in **0.003 seconds without any retraining**.

---

## 7. 🛠️ How Was It Set Up and Built?

The implementation consists of modular components:

1. **[`src/ml/graph_converter.py`](src/ml/graph_converter.py):**  
   Converts parsed `Project` objects into graph tensors ($\mathbf{X} \in \mathbb{R}^{N \times 12}$ node features, $\mathbf{E} \in \mathbb{R}^{2 \times |E|}$ edge indices).
2. **[`src/ml/gnn_model.py`](src/ml/gnn_model.py):**  
   Contains `DirectedGraphConvLayer`, `RCPSPDualHeadGNN`, and `GNNPriorityScheduler` (which connects neural priorities directly to SSGS).
3. **[`src/ml/train_gnn.py`](src/ml/train_gnn.py):**  
   CLI training script that processes PSPLIB instances, calculates teacher targets, trains the GNN, and saves `models/gnn_dual_head.pt`.
4. **Integration into the 6-Algorithm Suite:**  
   Integrated `GNN-Priority` as the 6th solver in:
   - `app.py` (Streamlit UI with "COMPARE ALL SIX SOLVERS")
   - `Adithya Baiju/ui.py` (Decoupled Streamlit UI)
   - `Adithya Baiju/backend.py` (Core backend engine & `SOLVER_REGISTRY`)
   - `rcpsp_all_in_one.py` (All-in-one CLI and backend engine)
   - `src/benchmark.py` (Benchmarking and comparison engine)
5. **Validation & Unit Testing:**  
   Verified via `tests/test_gnn.py` and `tests/test_benchmark.py`. All 128 tests pass with 100% constraint satisfaction.

---

## 🎙️ 30-Second Viva / Examiner Explanation

> *"In RCPSP, project schedules are governed by a Directed Acyclic Graph (DAG) of task dependencies and resource constraints. Standard MLPs reduce the whole project into 30 flat summary averages, which destroys local bottleneck structures.*  
> *Our Graph Neural Network performs bidirectional message passing directly along the precedence edges. It computes urgency scores for every task and passes them to our SSGS decoder. This delivers a guaranteed-feasible, near-optimal schedule in just 3 milliseconds — over 10x faster than traditional metaheuristics."*
