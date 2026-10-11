# AI-RCPSP Concept Reference & Teacher Explanation Guide

This document provides foundational and mathematical explanations for all project scheduling and machine learning concepts in the AI-RCPSP system. It is designed to ground the Streamlit UI `ⓘ` popover explanations.

---

## 1. Activity
- **Simple Definition:** A discrete unit of work within a project that takes time to complete and requires resources.
- **Mathematical Meaning:** An entity $j \in \{1, \dots, J\}$ characterized by processing duration $p_j \ge 0$ and resource demands $r_{j,k}$.
- **Tiny Example:** "Pour Concrete Foundation" (takes 4 days, needs 2 workers).
- **How Calculated in System:** Parsed from instance definition into [Activity](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/model.py) object with ID, duration, demands, and precedence lists.
- **Why It Matters:** Activities are the atomic building blocks of the scheduling problem.

---

## 2. Predecessor
- **Simple Definition:** A task that must finish before a subsequent task is allowed to start.
- **Mathematical Meaning:** If $(i, j) \in E$, then $S_j \ge S_i + p_i$, where $S_j$ is start time and $p_i$ is duration.
- **Tiny Example:** "Dig Trench" is a predecessor to "Lay Pipe".
- **How Calculated in System:** Stored in `activity.predecessors` and verified by SSGS and constraint validators.
- **Why It Matters:** Enforces temporal logic and prevents physically impossible schedules.

---

## 3. Successor
- **Simple Definition:** A task that can only start after its preceding task has completely finished.
- **Mathematical Meaning:** Task $j$ is a successor of task $i$ if $(i, j) \in E$.
- **Tiny Example:** "Paint Walls" is a successor to "Install Drywall".
- **How Calculated in System:** Stored in `activity.successors` and populated from project precedence relations.
- **Why It Matters:** Used during forward scheduling to determine when activities enter the eligible decision set.

---

## 4. Duration
- **Simple Definition:** The uninterrupted working time required to complete an activity.
- **Mathematical Meaning:** Non-negative integer or real scalar $p_j$.
- **Tiny Example:** Duration $p_3 = 5$ days.
- **How Calculated in System:** Directly read from project file as `activity.duration`.
- **Why It Matters:** Determines activity span and drives overall project completion time.

---

## 5. Resource
- **Simple Definition:** A reusable asset (such as labor, cranes, or machinery) necessary to execute an activity.
- **Mathematical Meaning:** Discrete renewal resource dimension $k \in \{1, \dots, K\}$.
- **Tiny Example:** "Skilled Electrician" or "Heavy Excavator".
- **How Calculated in System:** Modeled as [Resource](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/model.py) with total renewable capacity $R_k$.
- **Why It Matters:** Resource limits prevent all tasks from running in parallel, creating resource bottlenecks.

---

## 6. Resource Capacity
- **Simple Definition:** The maximum amount of a renewable resource available at any single instant in time.
- **Mathematical Meaning:** Bound $R_k$ such that $\sum_{j \in \mathcal{A}(t)} r_{j,k} \le R_k$ for all time $t$.
- **Tiny Example:** A construction site has at most $R_1 = 3$ cranes available.
- **How Calculated in System:** Read from project header into `project.capacities[k]`.
- **Why It Matters:** Forms the core hard capacity constraints that metaheuristics and SSGS must never violate.

---

## 7. Dummy Start
- **Simple Definition:** A fictitious activity added at the beginning of the project with zero duration and zero resource demand.
- **Mathematical Meaning:** Activity $0$ with $p_0 = 0, r_{0,k} = 0$, acting as a universal predecessor to all initial activities.
- **Tiny Example:** Milestone labeled "Project Kickoff".
- **How Calculated in System:** Activity index $0$ in PSPLIB instances, assigned duration 0.
- **Why It Matters:** Establishes a unique root node for topological sorting and critical path computations.

---

## 8. Dummy Finish
- **Simple Definition:** A fictitious activity added at the very end of the project with zero duration and zero resource demand.
- **Mathematical Meaning:** Activity $J+1$ with $p_{J+1} = 0, r_{J+1,k} = 0$, having all terminal tasks as predecessors.
- **Tiny Example:** Milestone labeled "Project Handover".
- **How Calculated in System:** Activity index $J+1$, whose start time equals project makespan.
- **Why It Matters:** Provides a single sink node to measure exact project completion time.

---

## 9. Makespan
- **Simple Definition:** The total elapsed time from project start to final activity completion.
- **Mathematical Meaning:** Objective function $C_{\max} = \max_j (S_j + p_j) = S_{\text{finish}}$.
- **Tiny Example:** A project starts on day 0 and all tasks finish on day 42 $\implies$ Makespan = 42.
- **How Calculated in System:** Finish time of dummy sink node returned by [evaluate_schedule()](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/evaluation.py).
- **Why It Matters:** The primary optimization objective that all 5 metaheuristics aim to minimize.

---

## 10. $N$ (Non-Dummy Activity Count)
- **Simple Definition:** The total number of genuine, work-bearing tasks in the project (excluding dummy start/finish).
- **Mathematical Meaning:** $N = J - 2$.
- **Tiny Example:** In a 32-node graph with 2 dummies, $N = 30$.
- **How Calculated in System:** Count of activities with non-zero duration or non-dummy indices in [src/features.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/features.py).
- **Why It Matters:** Governs problem scale, search space dimensionality ($N!$), and computational complexity.

---

## 11. $NC$ (Network Complexity)
- **Simple Definition:** The average number of immediate precedence relationships per non-dummy activity.
- **Mathematical Meaning:** $NC = \frac{|E|}{N}$.
- **Tiny Example:** 48 precedence arcs across 30 activities $\implies NC = 1.60$.
- **How Calculated in System:** `precedence_edge_count / N` in [extract_features()](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/features.py).
- **Why It Matters:** High $NC$ means tightly constrained task sequences; low $NC$ offers more scheduling freedom.

---

## 12. $RF$ (Resource Factor)
- **Simple Definition:** The average fraction of available resource types required by each activity.
- **Mathematical Meaning:** $RF = \frac{1}{N \cdot K} \sum_{j=1}^N \sum_{k=1}^K \mathbb{I}(r_{j,k} > 0)$.
- **Tiny Example:** If activities require on average 3 out of 4 resource types, $RF = 0.75$.
- **How Calculated in System:** Proportion of non-zero demand entries in activity-resource matrix.
- **Why It Matters:** Measures how multidimensional resource competition is across the project.

---

## 13. $RS$ (Resource Strength)
- **Simple Definition:** How plentiful resource capacities are compared to minimum and maximum demand.
- **Mathematical Meaning:** $RS_k = \frac{R_k - r_{k}^{\min}}{r_{k}^{\max} - r_{k}^{\min}}$, where $r_k^{\min} = \max_j r_{j,k}$ and $r_k^{\max}$ is peak concurrent demand under earliest start times.
- **Tiny Example:** $RS \approx 1.0 \implies$ resources virtually unconstrained; $RS \approx 0.1 \implies$ extreme scarcity.
- **How Calculated in System:** Calculated in [src/features.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/features.py) using earliest-start resource demand profiles.
- **Why It Matters:** Governs whether a project is primarily precedence-constrained or resource-constrained.

---

## 14. Random-Key Representation
- **Simple Definition:** Encoding a schedule as a continuous vector of real numbers in $[0, 1]$, where values determine priority.
- **Mathematical Meaning:** Genotype $\mathbf{x} = (x_1, x_2, \dots, x_N) \in [0, 1]^N$.
- **Tiny Example:** Activity priorities: $[0.72, 0.15, 0.88]$.
- **How Calculated in System:** Random initialization and mutation of float vectors in GA, SA, BCO, and TS.
- **Why It Matters:** Ensures all genetic operators (crossover, mutation, perturbation) produce valid priorities without infeasible permutations.

---

## 15. Priority Vector
- **Simple Definition:** A ranked list of activity priorities used by the decoder to select which eligible task starts next.
- **Mathematical Meaning:** Ordering $\pi = (\pi_1, \dots, \pi_N)$ derived by sorting activity keys.
- **Tiny Example:** Priority list $(3, 1, 4, 2)$ means activity 3 is considered before activity 1.
- **How Calculated in System:** Passed into SSGS to break ties among precedence-eligible activities.
- **Why It Matters:** Bridges continuous metaheuristic search space with discrete, feasible schedules.

---

## 16. Serial Schedule Generation Scheme (SSGS)
- **Simple Definition:** A deterministic simulation engine that converts an activity priority list into an active, 100% feasible schedule.
- **Mathematical Meaning:** An iterative builder that places activities one by one at their earliest precedence- and resource-valid start times.
- **Tiny Example:** Places task 1 at day 0-4; task 2 at day 4-7 based on crane availability.
- **How Calculated in System:** Implemented in [src/ssgs.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/ssgs.py) using a fast capacity timeline array.
- **Why It Matters:** Guarantees zero constraint violations while searching the space of active schedules.

---

## 17. Genetic Algorithm (GA)
- **Simple Definition:** Evolutionary search inspired by natural selection, maintaining a population of solutions that undergo crossover and mutation.
- **Mathematical Meaning:** Population-based metaheuristic using elitism, two-point crossover, and Gaussian mutation on random-key vectors.
- **Tiny Example:** Combines good traits of Schedule A and Schedule B to discover a superior Schedule C.
- **How Calculated in System:** Implemented in [src/algorithms/ga.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/algorithms/ga.py).
- **Why It Matters:** Particularly dominant on large, highly constrained project instances (J90/J120).

---

## 18. Simulated Annealing (SA)
- **Simple Definition:** Physics-inspired search that mimics cooling metal, accepting worse solutions with decreasing probability to escape local optima.
- **Mathematical Meaning:** Metropolis acceptance probability $P = \exp(-\Delta / T)$ with geometric cooling $T_{k+1} = \alpha T_k$.
- **Tiny Example:** High initial temperature explores widely; low temperature fine-tunes local schedule details.
- **How Calculated in System:** Implemented in [src/algorithms/sa.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/algorithms/sa.py).
- **Why It Matters:** Very fast execution per evaluation; effective on moderately sized problem instances.

---

## 19. Bee Colony Optimization (BCO)
- **Simple Definition:** Swarm-intelligence algorithm inspired by the foraging behavior of honeybees sharing nectar source locations.
- **Mathematical Meaning:** Employed bees perform local search, onlooker bees recruit probabilistically, scout bees explore randomly.
- **Tiny Example:** Bees scout promising priority vectors and recruit neighbors to improve makespan.
- **How Calculated in System:** Implemented in [src/algorithms/bco.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/algorithms/bco.py).
- **Why It Matters:** Balances exploration and exploitation in continuous parameter space.

---

## 20. Priority Rule-based Heuristic (PRH)
- **Simple Definition:** Fast, rule-based dispatching heuristics that prioritize activities using domain rules (e.g. latest finish time).
- **Mathematical Meaning:** Static rule selection $\pi_j = \text{score}(j)$ evaluating rules like LFT, LST, MIS, MTS, and randomized variants.
- **Tiny Example:** Always schedule the activity with the strictest deadline first.
- **How Calculated in System:** Implemented in [src/algorithms/prh.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/algorithms/prh.py).
- **Why It Matters:** Ultra-fast baseline that provides strong initial schedules without lengthy iterative search.

---

## 21. Tabu Search (TS)
- **Simple Definition:** Local neighborhood search that uses a short-term memory list (tabu list) to prevent cycling back to recently visited solutions.
- **Mathematical Meaning:** Neighborhood generation $\mathcal{N}(\mathbf{x})$ with tabu tenure $T_{\text{tenure}}$ and aspiration criteria.
- **Tiny Example:** Forbids reverting recently swapped activity priorities for 10 iterations.
- **How Calculated in System:** Implemented in [src/algorithms/tabu.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/algorithms/tabu.py).
- **Why It Matters:** Our research extension beyond the original 4-algorithm paper, providing systematic memory-based local search.

---

## 22. Neural Network
- **Simple Definition:** An AI model composed of interconnected layers of artificial neurons that learns non-linear patterns from training data.
- **Mathematical Meaning:** Function approximator $\hat{\mathbf{y}} = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2)$.
- **Tiny Example:** Takes 30 project metrics and outputs probability distribution over candidate algorithms.
- **How Calculated in System:** Built with scikit-learn `MLPClassifier` in [src/ml/train.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/ml/train.py).
- **Why It Matters:** Acts as the Level 1 meta-controller, predicting the best algorithm before optimization begins.

---

## 23. Training Instance
- **Simple Definition:** A single project project instance along with its extracted numerical features and experimental winner label.
- **Mathematical Meaning:** Tuple $(\mathbf{x}_i, y_i) \in \mathbb{R}^{30} \times \{\text{GA}, \text{SA}, \text{BCO}, \text{PRH}, \text{TS}, \text{ALL}\}$.
- **Tiny Example:** Instance `j3010_1` with feature vector $\mathbf{x}$ and label `ALL`.
- **How Calculated in System:** Recorded in [data/training/training_dataset.csv](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/data/training/training_dataset.csv).
- **Why It Matters:** Provides the ground-truth empirical evidence required for supervised machine learning.

---

## 24. Feature
- **Simple Definition:** A measurable numerical property or characteristic of a project instance.
- **Mathematical Meaning:** Scalar coordinate $x_k \in \mathbb{R}$ in the feature vector $\mathbf{x}$.
- **Tiny Example:** `precedence_density = 0.0967`.
- **How Calculated in System:** Extracted deterministically via [src/features.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/features.py).
- **Why It Matters:** Informs the neural network about project structure without requiring full schedule evaluation.

---

## 25. Prediction
- **Simple Definition:** The algorithm selected by the trained neural network as most likely to produce the shortest makespan.
- **Mathematical Meaning:** $\hat{y} = \arg\max_{c} P(y = c \mid \mathbf{x})$.
- **Tiny Example:** Given a high-tightness project, the model predicts `GA`.
- **How Calculated in System:** Output of [predict_algorithm()](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/ml/predict.py).
- **Why It Matters:** Automates expert algorithm selection for non-expert users.

---

## 26. Confidence
- **Simple Definition:** A numerical score indicating how certain the neural network is in its top recommendation.
- **Mathematical Meaning:** Maximum class probability $C = \max_{c} P(y = c \mid \mathbf{x}) \in [0, 1]$.
- **Tiny Example:** Confidence = 0.88 means the model assigned 88% probability to the top class.
- **How Calculated in System:** `res.confidence` in [PredictionResult](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/ml/predict.py).
- **Why It Matters:** Governs the confidence policy to determine whether to run 1, 2, or all 5 algorithms.

---

## 27. `ALL` Class
- **Simple Definition:** A special classification label indicating that all five metaheuristics found the identical best makespan.
- **Mathematical Meaning:** Condition where $\text{makespan}(\text{GA}) = \text{makespan}(\text{SA}) = \text{makespan}(\text{BCO}) = \text{makespan}(\text{PRH}) = \text{makespan}(\text{TS})$.
- **Tiny Example:** On small project `j10_sample`, all 5 algorithms achieve makespan 31.
- **How Calculated in System:** Assigned by [src/ml/generate_dataset.py](file:///home/prak05/GITHUB/fuck%20shit/AI-RCPSP/src/ml/generate_dataset.py) under `exact_five_way_tie`.
- **Why It Matters:** Prevents artificial tie-breaking on instances where all algorithms perform equally well, maintaining academic honesty.
