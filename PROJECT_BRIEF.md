# AI-Enhanced RCPSP Scheduler

## Project

Build a working academic software prototype for:

"AI-Enhanced Metaheuristics Control for Resource-Constrained Project Scheduling"

The system is inspired by the provided research paper:

"AI-Enhanced Metaheuristics Control for Project Scheduling Using Neural Networks"

The paper is provided as PAPER.pdf and must be treated as the primary
reference for the original methodology.

---

# MAIN OBJECTIVE

Build a Python application that accepts an RCPSP project input file,
analyzes the project structure, predicts which optimization algorithm
should be used, runs the selected optimizer, and produces an
understandable project schedule.

The final application must be understandable to a person who knows
nothing about RCPSP.

---

# CORE IDEA

The system has TWO levels.

LEVEL 1:
An AI model selects the most suitable optimization algorithm based
on project characteristics.

LEVEL 2:
The selected optimization algorithm searches for a high-quality
feasible project schedule.

The neural network does NOT directly create the project schedule.

The optimization algorithm creates candidate schedules.

---

# ORIGINAL PAPER METHODOLOGY

The original paper uses:

1. Genetic Algorithm (GA)
2. Simulated Annealing (SA)
3. Bee Colony Optimization (BCO)
4. Priority Rule Based Heuristic (PRH)

The paper uses:

- Random-key representation
- Serial Schedule Generation Scheme (SSGS)
- Network Complexity (NC)
- Resource Factor (RF)
- Resource Strength (RS)

The objective is to minimize project makespan while respecting:

- precedence constraints
- resource capacity constraints

---

# EXTENSION FOR THIS PROJECT

Extend the original four algorithms with:

5. Tabu Search (TS)
6. Particle Swarm Optimization (PSO)

Final algorithm set:

- GA
- SA
- BCO
- PRH
- TS
- PSO

Also support:

- richer project features
- algorithm confidence
- algorithm comparison
- Gantt chart
- resource utilization
- human-readable explanations
- optional adaptive algorithm switching

---

# INPUT

The application must eventually support:

1. Patterson RCPSP format
2. PSPLIB .sm format
3. CSV fallback

Do not assume every input format is identical.

Create a modular parser architecture.

---

# RCPSP DATA MODEL

Represent:

## Activity

Each activity should contain:

- activity ID
- duration
- predecessors
- successors
- resource requirements

## Resource

Each resource should contain:

- resource ID
- capacity

## Project

The project should contain:

- activities
- resources
- resource capacities
- precedence graph

---

# RANDOM-KEY REPRESENTATION

Represent a candidate solution as a vector of values in [0,1].

Example:

Activity:
A B C D E

Priority:
0.12 0.81 0.32 0.22 0.74

Sorting these values determines the activity priority order.

Example:

A → D → C → E → B

The priority vector itself is NOT the final schedule.

It must be converted into a feasible schedule using SSGS.

---

# SERIAL SCHEDULE GENERATION SCHEME

Implement one central SSGS engine.

ALL optimization algorithms must use this same SSGS implementation.

Do NOT duplicate scheduling logic inside each algorithm.

SSGS must:

1. determine activity priority order
2. respect precedence relationships
3. find earliest feasible start times
4. respect resource capacities
5. calculate activity start times
6. calculate activity finish times
7. calculate total project makespan

A schedule is feasible only if:

- all precedence constraints are satisfied
- no resource capacity is exceeded

---

# OBJECTIVE FUNCTION

Primary objective:

MINIMIZE MAKESPAN.

For a candidate schedule:

makespan = finish time of final project activity

Lower makespan = better solution.

Do NOT claim that a metaheuristic guarantees the global optimum.

Use wording:

"best feasible solution found"

unless an independently verified optimum is available.

---

# ALGORITHM IMPLEMENTATION

Implement the following.

## 1. Genetic Algorithm

Use:

- population
- fitness evaluation
- selection
- crossover
- mutation
- replacement

Fitness should be based primarily on makespan.

---

## 2. Simulated Annealing

Use:

- current solution
- neighboring solution
- acceptance probability
- temperature
- cooling schedule

Allow occasional worse solutions early in the search to escape
local optima.

---

## 3. Bee Colony Optimization

Implement a practical simplified BCO/ABC-style approach using:

- candidate population
- good solution recruitment
- exploration
- scout behaviour

---

## 4. Priority Rule Heuristic

Implement SSGS using priority rules such as:

- EST
- LST
- EFT
- LFT
- random priority

---

## 5. Tabu Search

Implement:

- current solution
- neighbourhood generation
- tabu list
- best-so-far solution

The tabu mechanism must prevent immediate cycling through recently
visited moves.

---

## 6. Particle Swarm Optimization

Implement a practical discrete/random-key PSO.

Each particle should maintain:

- current position
- personal best
- global best

The position represents random-key priorities.

---

# FAIR COMPARISON

All six algorithms must use:

- the same project input
- the same SSGS
- the same objective function
- the same feasibility rules
- approximately the same evaluation budget

Default evaluation budget:

5000 schedule evaluations.

For every algorithm record:

- algorithm name
- best makespan
- runtime
- number of evaluations
- best priority vector
- final schedule

---

# ALGORITHM COMPARISON

Primary comparison:

lowest makespan.

Secondary comparison:

runtime.

Tertiary comparison:

number of evaluations.

Example:

GA = 152
SA = 156
BCO = 154
PRH = 163
TS = 153
PSO = 155

Winner:

GA

If all six algorithms obtain exactly the same best makespan:

winner = ALL

Do not randomly select one.

---

# PROJECT FEATURES

Calculate at minimum:

- N
- NC
- RF
- RS

Also calculate richer features:

- number of resources
- average duration
- duration standard deviation
- minimum duration
- maximum duration
- average resource demand
- resource utilization
- network density
- precedence density
- average predecessors
- average successors
- critical path estimate

The feature extraction code must be independent of the UI.

---

# MACHINE LEARNING MODEL

Use Python scikit-learn initially.

Model:

StandardScaler
+
MLPClassifier

Input:

project features

Output classes:

- GA
- SA
- BCO
- PRH
- TS
- PSO
- ALL

The model must provide:

- predicted algorithm
- probability for each algorithm
- confidence

Save trained models using joblib.

---

# TRAINING DATA

Training examples should be generated by:

1. obtaining RCPSP project instances
2. calculating their project features
3. running all six algorithms
4. recording each algorithm's best makespan
5. selecting the algorithm with minimum makespan
6. using that algorithm as the training label

If all algorithms tie:

label = ALL

Do not train the model using fabricated labels.

---

# CASE STUDY FROM PAPER

The provided research paper contains a real construction case.

It concerns renovation of a five-star hotel in Greece,
specifically a swimming pool complex.

Published characteristics:

NC = 1.3729
RF = 0.3158
RS = 0.9333

Published CPM duration:

152 days

The original paper reports that GA, BCO, PRH and SA all obtained
152 days for this case.

The paper's classifier predicted the ALL class.

IMPORTANT:

Do not hard-code 152 as our algorithm's output.

The software must independently calculate its schedule.

Display:

"Published reference duration: 152 days"

separately from:

"Our computed schedule: X days"

If the exact machine-readable project input is unavailable,
label any reconstructed input as:

"Demonstration reconstruction based on the published case-study
characteristics."

Never claim that a reconstructed dataset is the authors' exact
original input.

---

# USER INTERFACE

Use Streamlit.

The UI should contain:

## 1. Upload

"Upload Project File"

Supported:
- .sm
- .rcp
- .csv

---

## 2. Project Overview

Show:

- number of activities
- number of resources
- NC
- RF
- RS
- estimated critical path

Use simple language.

---

## 3. AI Recommendation

Example:

Recommended Strategy:

Genetic Algorithm

Confidence:

87.4%

Explanation:

"The AI examined the project's task relationships, resource
availability and structural complexity. Based on projects with
similar characteristics in the training data, Genetic Algorithm
was predicted to be the most suitable search strategy."

---

## 4. Algorithm Comparison

Display:

| Algorithm | Makespan | Runtime | Evaluations | Result |
|-----------|----------|---------|-------------|--------|

Highlight the best feasible result.

---

## 5. Final Schedule

Show:

- activity
- start
- finish
- duration
- resources

---

## 6. Gantt Chart

Display a readable Gantt chart.

---

## 7. Resource Utilization

Display resource usage over project time.

---

## 8. Validation

Display:

✓ Precedence constraints satisfied

✓ Resource constraints satisfied

✓ Schedule feasible

✓ Objective calculated

---

## 9. Explain the Result

A person with zero RCPSP knowledge should be able to understand:

- what the AI selected
- why it selected it
- how the schedule was created
- why the final schedule is better
- what makespan means

Avoid displaying internal class numbers.

---

# ADAPTIVE CONTROLLER

Create an optional basic adaptive controller.

If an algorithm stops improving for a configurable number of
iterations/evaluations:

1. detect stagnation
2. record the current best
3. optionally switch to another algorithm
4. continue searching

Example:

GA stalls
↓
TS activated
↓
TS improves solution
↓
new global best

Keep this modular.

Do not make it mandatory for the initial MVP.

---

# GNN FUTURE EXTENSION

Create an architecture that allows a future GNN selector.

Represent the project as:

nodes = activities
edges = precedence relationships

Node features may include:

- duration
- resource demand
- predecessor count
- successor count

Do NOT implement a full GNN in the first MVP unless the core system
is already stable.

---

# DURATION PREDICTION FUTURE EXTENSION

Keep a modular place for a future duration-prediction component.

It may eventually estimate:

- expected project duration
- uncertainty/confidence

Do not allow this future component to interfere with the core
optimization engine.

---

# SOFTWARE ARCHITECTURE

Use:

src/
    model.py
    parser.py
    ssgs.py
    evaluation.py
    features.py

    algorithms/
        ga.py
        sa.py
        bco.py
        prh.py
        tabu.py
        pso.py

    ml/
        dataset.py
        train.py
        predict.py

app.py

tests/

The UI must not contain optimization logic.

The algorithms must not contain UI logic.

The ML layer must not contain parser logic.

---

# TESTING

Write tests for:

- parser
- SSGS
- precedence feasibility
- resource feasibility
- makespan
- GA
- SA
- BCO
- PRH
- TS
- PSO
- feature extraction
- ML prediction

Every algorithm must produce a feasible schedule.

---

# DEVELOPMENT RULE

Build incrementally.

FIRST:
data model + parser + SSGS + tests

SECOND:
GA + SA + BCO + PRH + tests

THIRD:
TS + PSO + tests

FOURTH:
feature extraction

FIFTH:
training dataset

SIXTH:
ML selector

SEVENTH:
Streamlit UI

EIGHTH:
case study

NINTH:
adaptive controller

TENTH:
future GNN architecture

Never skip tests to move faster.

Run the application after every major milestone.

---

# IMPORTANT

Do not fabricate research results.

Do not hard-code algorithm winners.

Do not hard-code the 152-day case-study result.

Do not claim global optimality for metaheuristics.

Do not create fake ML accuracy numbers.

Do not create fake training data labels.

If something cannot be implemented reliably, explicitly mark it
as a limitation and continue with the working components.
