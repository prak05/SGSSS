# RCPSP Algorithm Reference Repositories

These repositories are REFERENCES ONLY.
Do NOT copy code directly.
Do NOT reproduce repository-specific architecture.
Do NOT copy licenses, comments, datasets, or implementation details unnecessarily.
The project must contain an original unified implementation using:

Random-Key Representation
↓
Common Serial Schedule Generation Scheme (SSGS)
↓
Common Feasibility Evaluator
↓
Common Makespan Objective

## 1. Genetic Algorithm
Primary: https://github.com/derhendrik/RCPSP_GA
Secondary: https://github.com/juliusf/Genetic-SRCPSP

Study:
- chromosome representation
- population initialization
- selection
- crossover
- mutation
- fitness evaluation
- convergence

---

## 2. Simulated Annealing
Primary: https://github.com/mbenahmed1/ScheduleMe

Study:
- neighborhood generation
- acceptance probability
- temperature
- cooling schedule
- termination
- RCPSP sequence representation

---

## 3. Bee Colony Optimization
RCPSP references: https://github.com/sydney-machine-learning/MEHH_RCPSP

Literature references:
- Artificial Bee Colony for RCPSP
- Random-key ABC for RCPSP
- RCPSP ABC using serial schedule generation scheme

Implementation must contain:
- employed/scout/recruit/onlooker-style search logic as appropriate
- exploration
- exploitation
- abandonment/scout mechanism
- random-key candidate representation

Do NOT copy an unrelated ABC implementation blindly.

---

## 4. Priority Rule Heuristic
Primary implementation reference: https://github.com/sydney-machine-learning/MEHH_RCPSP

Study:
- EST
- LST
- EFT
- LFT
- random priority
- priority-based serial schedule generation

The original research paper is the authoritative specification.

---

## 5. Tabu Search
Primary: https://github.com/ThomasK33/rcpsp.rs
Secondary: https://github.com/CTU-IIG/RCPSPCpu

Study:
- neighborhood moves
- tabu list
- tabu tenure
- aspiration criterion
- local search
- cycling prevention

Our implementation must use the common random-key representation and common SSGS.

---

# Architectural Rule
All five algorithms MUST call the same:
src/ssgs.py

All five algorithms MUST use the same:
src/evaluation.py

All five algorithms MUST represent a candidate solution using:
random-key priority vector

All five algorithms MUST ultimately produce:
priority vector
schedule
makespan
runtime
number of evaluations

This is essential for a fair comparison.
The algorithms must never contain their own independent resource-feasibility or scheduling engines.

