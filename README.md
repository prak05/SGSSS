# 🏗️ AI-RCPSP: Teacher Presentation Cheat Sheet & Quick Reference

> **Punchy 1-2 Sentence Speaker Scripts** for answering any question during your project viva or lab presentation.

---

## ⚡ Quick Commands

```bash
# 1. Launch Interactive UI (Default)
streamlit run app.py

# 2. Launch Interactive UI (sgsss_suite Module - UI in ui.py)
streamlit run "sgsss_suite/ui.py"

# 3. Run Backend Engine CLI (sgsss_suite Module - Backend in backend.py)
python "sgsss_suite/backend.py" data/j10_sample.sm

# 4. Run All-in-One Benchmark Script
python rcpsp_all_in_one.py data/j10_sample.sm

# 5. Run Unit Tests (119 test cases)
python3 -m unittest discover tests -v
```

---

## 📁 sgsss_suite Directory Architecture

The [`sgsss_suite`](sgsss_suite) directory contains a clean, 2-file decoupled implementation:
- **[`sgsss_suite/ui.py`](ui.py)**: Contains all UI elements, layout, Plotly Gantt & Resource charts, precedence DAG, Teacher Explanation tooltips, and tab controls.
- **[`sgsss_suite/backend.py`](backend.py)**: Contains all non-UI logic (Data models, SSGS decoder, CPM feature extraction, file parsers, 5 metaheuristic solvers, statistical benchmark engine, and printable PDF generator).


---

## 🎙️ Spoken Cheat Sheet for `rcpsp_all_in_one.py`

When the teacher asks: **"Open `rcpsp_all_in_one.py` and explain Section X..."**

### 1. Data Models ([`rcpsp_all_in_one.py:L59-L150`](rcpsp_all_in_one.py#L59-L150))
> 💬 **Speaker Script:**  
> *"Section 1 defines our core data classes: `Activity` stores duration, predecessors, and resource demands $r_{ik}$, while `Project` wraps all tasks and renewable resource capacities $R_k$. `ScheduleEntry` records start/finish time windows, and `AlgorithmResult` captures the best priority vector, final makespan, runtime, and convergence trace."*

### 2. SSGS Decoder Engine ([`rcpsp_all_in_one.py:L152-L250`](rcpsp_all_in_one.py#L152-L250))
> 💬 **Speaker Script:**  
> *"SSGS is our Serial Schedule Generation Scheme decoder. It builds an eligible task queue ordered by random-key priorities and assigns each task its earliest start time $t \ge EST$ where resource capacities are fully available."*

### 3. Feature Extraction & CPM Analysis ([`rcpsp_all_in_one.py:L252-L525`](rcpsp_all_in_one.py#L252-L525))
> 💬 **Speaker Script:**  
> *"Section 3 runs topological validation (`validate_dag`), Critical Path Method ($EST/EFT$ and $LST/LFT$), and extracts 30 project indicators including Network Complexity ($NC$), Resource Factor ($RF$), and Resource Strength ($RS$)."*

### 4. Dual Constraint Evaluator ([`rcpsp_all_in_one.py:L527-L562`](rcpsp_all_in_one.py#L527-L562))
> 💬 **Speaker Script:**  
> *"The evaluator performs dual-layer validation to guarantee that no predecessor completes after a task starts, and cumulative resource usage $\sum r_{ik}$ never exceeds capacity $C_k$ at any time step $t$."*

### 5. Multi-Format File Parser ([`rcpsp_all_in_one.py:L564-L775`](rcpsp_all_in_one.py#L564-L775))
> 💬 **Speaker Script:**  
> *"The `Parser` class handles PSPLIB `.sm` files, Paterson `.rcp` files, and `.json` formats, populating our standardized `Project` object regardless of the input format."*

### 6. The 5 Metaheuristic Solvers ([`rcpsp_all_in_one.py:L777-L1193`](rcpsp_all_in_one.py#L777-L1193))
> 💬 **Speaker Script:**  
> *"Section 6 implements our 5 solvers—Genetic Algorithm, Simulated Annealing, Bee Colony Optimization, Tabu Search, and Priority Rule Heuristics. Crucially, all 5 share the exact same SSGS decoder and evaluation budget for a 100% fair benchmark."*

### 7. Benchmark & Comparison Engine ([`rcpsp_all_in_one.py:L1195-L1328`](rcpsp_all_in_one.py#L1195-L1328))
> 💬 **Speaker Script:**  
> *"The benchmark engine runs all 5 solvers across multiple random seeds, evaluates feasibility, and prints a statistical summary table highlighting the winning algorithm."*

### 8. Command-Line Entry Point ([`rcpsp_all_in_one.py:L1329-L1369`](rcpsp_all_in_one.py#L1329-L1369))
> 💬 **Speaker Script:**  
> *"The `main()` entry point parses the input project file, extracts 30 feature dimensions, runs the 5 solvers under a 500-evaluation budget, and outputs the comparative results."*

---

## 🖥️ Spoken Cheat Sheet for Streamlit UI (`app.py`)

When the teacher asks: **"Show me the UI and explain Tab X..."**

### 🎛️ Sidebar / File Loader ([`app.py:L127-L177`](app.py#L127-L177))
> 💬 **Speaker Script:**  
> *"The sidebar handles file ingestion for `.sm`, `.rcp`, or `.json` files and toggles Explanation Mode to display inline math formulas."*

### 📊 Tab 1: Overview & Precedence Graph ([`app.py:L211-L232`](app.py#L211-L232))
> 💬 **Speaker Script:**  
> *"Tab 1 displays key project metrics alongside an interactive NetworkX Directed Acyclic Graph (DAG) that visualizes activity precedence dependencies."*

### 📈 Tab 2: Characteristics ($N, NC, RF, RS$) ([`app.py:L233-L252`](app.py#L233-L252))
> 💬 **Speaker Script:**  
> *"Tab 2 displays project complexity indicators—$NC$ for graph density, $RF$ for resource demand proportion, and $RS$ for capacity tightness relative to CPM peak demand."*

### ⚡ Tab 3: Optimization Execution ([`app.py:L253-L294`](app.py#L253-L294))
> 💬 **Speaker Script:**  
> *"Tab 3 allows running individual solvers or comparing all 5 solvers simultaneously, displaying the winning algorithm, verified makespan, and execution runtime."*

### 📅 Tab 4: Gantt Schedule Timeline ([`app.py:L295-L301`](app.py#L295-L301))
> 💬 **Speaker Script:**  
> *"Tab 4 plots the final decoded schedule as a Plotly Gantt chart, confirming that tasks execute in valid sequence without precedence conflicts."*

### 🔋 Tab 5: Resource Utilization ([`app.py:L302-L314`](app.py#L302-L314))
> 💬 **Speaker Script:**  
> *"Tab 5 plots aggregate resource consumption over time against a red capacity threshold line, proving 100% resource feasibility."*

### 🏁 Tab 6: Solver Comparison & Convergence ([`app.py:L315-L350`](app.py#L315-L350))
> 💬 **Speaker Script:**  
> *"Tab 6 provides a side-by-side solver performance matrix and convergence curves showing how rapidly each algorithm lowers the makespan."*

### 📄 Tab 7: Printable PDF Report ([`app.py:L351-L401`](app.py#L351-L401))
> 💬 **Speaker Script:**  
> *"Tab 7 exports an executive ReportLab PDF reference document with embedded vector charts, metrics, and summary tables for offline reporting."*

---

## 📐 Mathematical Formulas for Viva

### Network Complexity (NC)
$$\text{NC} = \frac{\text{Precedence Arcs}}{N}$$

### Resource Factor (RF)
$$\text{RF} = \frac{1}{N \cdot K} \sum_{i=1}^N \sum_{k=1}^K \begin{cases} 1 & \text{if } r_{ik} > 0 \\ 0 & \text{otherwise} \end{cases}$$

### Resource Strength (RS)
$$\text{RS} = \frac{R_k - R_k^{\min}}{R_k^{\max} - R_k^{\min}}$$
