# AI-RCPSP: Project Explanation & Viva Guide

## 1. What is the Project?
**AI-RCPSP** (AI-Enhanced Resource-Constrained Project Scheduling) is an intelligent scheduling system. It solves the classic NP-hard RCPSP problem not by using a single algorithm, but by using **Machine Learning (Meta-Learning)** to predict and select the best optimization algorithm for a given project based on its unique characteristics.

## 2. Why does this Project exist? (The Problem)
In project management and manufacturing, scheduling tasks while respecting strict dependencies (precedence) and limited resources (e.g., workers, machines) is computationally extremely difficult (NP-hard). 
Decades of research show that **no single algorithm is best for all projects**. Some algorithms work well for highly complex networks, while others work better for highly constrained resources.
Instead of blindly running one algorithm or wasting time running all of them, our system uses AI to inspect the project and intelligently say: *"For this specific project structure, Simulated Annealing is mathematically most likely to find the best schedule."*

## 3. How does the Project work? (The Pipeline)
The system follows a strict pipeline:
1. **Input**: A standard project file (PSPLIB `.sm` or `.rcp`) is uploaded.
2. **Parsing**: The file is converted into an internal graph object containing Activities and Resources.
3. **Feature Extraction**: The system analyzes the graph and extracts mathematical features (like Network Complexity, Resource Factor).
4. **AI Prediction**: A trained Machine Learning model analyzes these features and predicts the most suitable scheduling algorithm.
5. **Optimization**: The recommended algorithm generates a priority list of activities.
6. **Schedule Generation (SSGS)**: The priority list is converted into a feasible schedule (start/end times) without violating any constraints.
7. **Validation**: The schedule is checked for correctness and visualized.

## 4. What every major Folder / File does
- **`app.py`**: The Streamlit Graphical User Interface. It handles file uploads, connects the backend components, and renders the charts and tables.
- **`src/parser.py`**: Reads `.sm` files and builds the `Project` data model. Contains a robust pure-Python fallback parser.
- **`src/model.py`**: Defines the core data structures (`Project`, `Activity`, `Resource`, `AlgorithmResult`).
- **`src/ssgs.py`**: The Serial Schedule Generation Scheme. This is the absolute core engine that converts an algorithm's priority list into actual timestamps.
- **`src/features.py`**: Calculates the mathematical properties of the project graph (NC, RF, RS, etc.) used by the ML model.
- **`src/evaluator.py`**: A helper that runs all algorithms on a project to compare them and calculate their makespans.
- **`src/algorithms/`**: Contains the implementations of the 5 heuristic algorithms.
- **`src/ml/`**: Contains the code for training the Random Forest model and making predictions based on project features.

## 5. What every major Algorithm does
The project implements **five** specific algorithms. Crucially, these algorithms *do not generate schedules*. They generate **Priority Lists** (permutations of activities), which SSGS then uses to build the schedule.

1. **GA (Genetic Algorithm)**: An evolutionary approach. It creates a "population" of priority lists, crosses them over (combines them), and mutates them over generations to find better lists.
2. **SA (Simulated Annealing)**: Inspired by metallurgy. It starts with one priority list and explores neighbor lists. Early on, it accepts "worse" lists to escape local minimums, but as it "cools down", it only accepts better lists.
3. **BCO (Bee Colony Optimization)**: Swarm intelligence. "Bees" explore different priority lists and share information about good lists with the hive, focusing the search on promising areas.
4. **PRH (Priority Rule Heuristics)**: Fast, deterministic rules. It builds a priority list using standard project management rules (e.g., LFT - Latest Finish Time, SPT - Shortest Processing Time).
5. **TS (Tabu Search)**: A local search method that keeps a "Tabu List" (memory) of recently visited priority lists to prevent the algorithm from getting stuck in infinite loops.

*(Note: PSO is explicitly not part of this project's final scope).*

## 6. How Project Features are calculated
To let the AI understand a project, we convert it into numbers (features):
- **NC (Network Complexity)**: The average number of non-redundant dependencies per activity.
- **RF (Resource Factor)**: The density of resource usage. (How many different types of resources does an average activity require?)
- **RS (Resource Strength)**: How constrained the resources are. (Low RS = highly constrained, High RS = abundant resources).

## 7. How the AI selects an algorithm (Meta-Learning)
We pre-trained a **Random Forest Classifier** on thousands of historical RCPSP instances (like the standard `j30`, `j60`, `j120` datasets). 
For each historical project, we calculated its features (NC, RF, RS) and recorded which of our 5 algorithms found the shortest makespan. 
The Random Forest learned the mathematical relationship between the features and the winning algorithm. When a new project is uploaded, the UI passes its features to this model, which outputs the predicted winner.

## 8. How a schedule is actually generated (SSGS)
Algorithms only output a sorted list of activities (e.g., `[1, 3, 2, 5, 4, 6]`).
The **Serial Schedule Generation Scheme (SSGS)** takes this list and goes one-by-one:
1. Take the next activity in the list.
2. Find the earliest possible time where:
   - All its predecessors have completely finished.
   - There is enough remaining resource capacity to support it for its entire duration.
3. Schedule it at that time and subtract its required resources from the timeline.

## 9. How schedules are compared
The system's ultimate metric is the **Makespan** (the total duration to complete the entire project). 
We run all 5 algorithms on the same project using the same evaluation budget (e.g., 500 iterations). We compare their final makespans and their runtime in seconds.

## 10. What understand from the UI
The UI is built to prove the pipeline works:
- **Project Analysis**: Shows that we successfully parsed the graph and extracted mathematical features.
- **AI Recommendation**: Shows the Machine Learning model making a decision *before* any heavy optimization runs.
- **Result Card**: Proves that the algorithm generated a mathematically sound schedule (Validating Precedence and Resources).
- **Gantt & Resource Charts**: Visually prove that the schedule is correct (no overlaps where dependencies exist, no red lines breached on resources).
- **Comparison Engine**: Proves our algorithms are actively competing and finding different, valid solutions.

## 11. What is from the research paper vs. Our Extension
- **From Operations Research Literature**: The definitions of GA, SA, TS, BCO, PRH, the PSPLIB datasets, the mathematical definitions of NC/RF/RS, and the SSGS algorithm.
- **Our Project Extension (The Novelty)**: Instead of just building a scheduler, we framed it as an **Algorithm Selection Problem (ASP)**. We built the Meta-Learning pipeline (the Random Forest) that dynamically routes the project to the best solver. We also wrapped this entirely into an interactive, end-to-end web application.

## 12. Assumptions and Limitations
- **Single-Mode**: Every activity has exactly one way to be executed (fixed duration, fixed resource demand).
- **Non-Preemptive**: Once an activity starts, it cannot be paused.
- **Renewable Resources Only**: Resources (like workers/machines) replenish every hour/day. We do not model consumable resources (like money/materials).
- **Deterministic**: Durations are fixed, there is no uncertainty or risk of delay modeled.
