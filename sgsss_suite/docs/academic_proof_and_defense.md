# Viva guide — why the proposed system works, with the evidence

**Project:** AI-RCPSP — GNN-guided Tabu Search (TS) and Particle Swarm Search (PSS) for the RCPSP
**Claim:** on held-out benchmark projects, the proposed solvers produce better schedules than four classic
metaheuristics under the *same* evaluation budget. Every number below is from `docs/RESULTS.md`
(produced by `scripts/evaluate.py`), so it can be re-run in front of the examiner.

---

## 1. The problem in one paragraph
Each activity has a duration, predecessors and a demand for renewable resources whose capacity is limited
in every period. We want the shortest feasible project (minimum makespan, written PS | prec | Cmax). The
problem is NP-hard: for 120 activities there are up to 120! activity orders.

## 2. How every solver works (shared part)
All six solvers search over **priority vectors** (one number per activity). The **Serial Schedule Generation
Scheme** (`src/ssgs.py`) turns a priority vector into a schedule: repeatedly take the highest-priority
activity whose predecessors are done, and start it at the earliest time when every resource has enough
free capacity for its whole duration. So **every schedule is feasible by construction**, and an independent
audit (`src/evaluation.py`) re-checks precedence and capacity anyway.

## 3. What is new in the proposed solvers
| Ingredient | What it does | Where |
|---|---|---|
| **Trained GNN start** | A bidirectional graph attention network reads the precedence graph (attention over predecessors = readiness, over successors = bottleneck pressure) and predicts a good activity order *before any search*. It is trained with REINFORCE on 1080 projects. | `src/algorithms/gnn.py`, `scripts/train.py` |
| **Forward-backward justification** | Shift every activity right as far as possible, then left again: closes idle gaps without breaking constraints (Valls et al. 2005). Charged as 2 evaluations. | `src/ssgs.py` |
| **Tabu Search moves** | Re-insert one activity anywhere between its last predecessor and first successor; tabu memory prevents cycling; aspiration lets a record-breaking move through. | `src/algorithms/ts.py` |
| **Swarm learning** | PSS particles start from the GNN and the classic rules; half of the particles are re-packed by justification and adopt the improved order (Lamarckian). | `src/algorithms/pss.py` |
| **Quality beyond makespan** | Among equal-makespan schedules, keep the one with the lowest total finish time, then the smoothest resource use. Stop early when the makespan equals the lower bound (provably optimal). | `src/algorithms/common.py` |

The baselines (GA, SA, BCO, PRH) are the textbook versions: random or rule-based starts, makespan only, and
they always use the whole budget.

## 4. Why the comparison is fair
1. **Same budget**, set from the project itself: 10 × activities × (1 + resource pressure), so bigger and resource-tight
   projects get more evaluations; the random seed is a fingerprint of the project content (same project in any file
   format → same budget and seed). *Every*
   decode is counted, including justification and the GNN's first schedule.
2. **Same decoder and the same audit** for everyone.
3. **The winner is chosen only from the results**: makespan → total finish time → resource levelling →
   reliability (mean evaluations to reach that makespan over 5 seeds, for algorithms still tied) → runtime.
   No algorithm name appears in the ranking code; `tests/test_ai_rcpsp.py` renames the algorithms and checks
   that the ranking does not change.
4. **No training data in the test**: the GNN is trained on `data/train` (1080 projects: PSPLIB J30 groups 13–48,
   J60 groups 7–48, RG30 Pat101–400) and evaluated on `data/test` (J30 groups 1–12, J60 groups 1–6, J90, J120,
   RG30 Pat1–60). The model kept is the one with the best score on a validation split of the *training* data.
5. **External reference values**: for J30 the proven optimal makespans and for J60/J90/J120 the best-known
   makespans published by PSPLIB (`data/test/reference.csv`).

## 5. The evidence (held-out test set, 260 projects)
**Overall:** a proposed method won **232 of 260** projects (89.2%). **All 260** schedules of all six solvers
passed the independent feasibility audit. Budget and seed were set automatically for each project.

| Algorithm | Wins | Mean gap to optimum / best known (200 projects with a published value) | Reached optimum / best known |
|---|---|---|---|
| **PSS (proposed)** | 136 | **0.58%** | **158/200** |
| **TS (proposed)** | 96 | 1.28% | 139/200 |
| PRH | 6 | 1.72% | 137/200 |
| GA | 8 | 2.13% | 130/200 |
| SA | 10 | 2.77% | 120/200 |
| BCO | 4 | 4.89% | 106/200 |

**By family (wins by TS or PSS):** J30 107/120 · J60 37/40 · J60 as .json 20/20 · J90 10/10 · J120 10/10 · RG30 48/60.
The bigger the project, the clearer the advantage: J90 and J120 were won 20/20.

**How the wins were decided:** 75 by a strictly shorter makespan. The rest were makespan ties, decided by total
finish time (70), resource levelling (5), reliability over 5 seeds (33), or runtime (49, mostly loosely constrained
projects where the proposed solvers proved optimality and stopped early).

**Realistic use cases** (`docs/USE_CASES.md`): on 66 generated industry projects (30–60 tasks, never used in training),
TS or PSS won **64/66**, 40 of them with a strictly shorter makespan than every baseline. The two losses
(software_02, shipbuilding_16) are listed.

**Effect of more training (570 → 1080 projects):** the one-shot GNN gap on the test set fell from 19.86% to 19.58%
(LST rule: 20.07%). PSS got closer to the optimum (0.65% → 0.60%; reached 158 → 162/200), but the win count moved
from 234 to 228 because small differences in the starting order change some close races. We kept the larger model
because it scored better on the validation set; the test set was not used to choose it.

**Honest limits:**
- A baseline found a strictly shorter makespan than both proposed solvers on **15/260** projects (all listed in `RESULTS.md`).
- The other 13 non-wins are makespan ties that a baseline won on a tie-break.
- The GNN's one-shot advantage over the best single rule is real but small; most of the gain comes from GNN +
  justification + search together.

## 6. Likely questions

**Why not let the neural network output start times directly?**
A network that outputs start times can violate precedence or capacity. Ours outputs only a *priority order*;
SSGS then builds the schedule, so it is always feasible and the network only has to learn *what to schedule first*.

**Can you show the GNN working?**
Yes, the dashboard tab *GNN inside* walks through it on any project in four plain steps: the 10 facts it reads about
each task, which connected tasks each task listens to (attention, only ever along the project's arrows; a test checks
this), the priority queue it produces with the tasks it moved compared with the textbook LST rule, and the schedule that
queue gives compared with every classic rule. The training history is in `data/models/training_log.json`.

**What does the GNN actually learn, and how do you know it helps?**
It starts out equal to the classic LST rule (the score is the LST rule plus a learned correction, initialised to
zero) and learns when to deviate. One decode of its prediction, with no search, is compared with every classic
rule in the *GNN inside* tab (step 4) and, over all test projects, in `RESULTS.md`.

**Why REINFORCE and not supervised learning?**
There is no "correct" order to copy, only a makespan to minimise, and makespan is not differentiable. REINFORCE
samples orders around the network's scores, decodes them, and makes shorter-makespan orders more likely
(the same family of methods as the TBGAT paper, Zhang et al., UAI 2024). We tried imitating a long Tabu Search
first; it made the one-shot schedules *worse* than the LST rule, so we switched.

**Isn't using total finish time and levelling as tie-breaks biased towards your methods?**
They are applied identically to every algorithm and are standard secondary objectives (work-in-progress and
resource levelling). The proposed solvers win them *because they optimise them*, which is part of the design and
is stated openly. The test set shows how many wins came from a strictly shorter makespan and how many from each tie-break.

**Where does your method lose?**
Every loss is listed in `RESULTS.md` (instance names included). Typical cases: very small or loosely
constrained projects where any random order is already optimal, so a lucky random start is as good as a
trained one, and a few instances where a baseline happened to find a shorter makespan.

**What is the lower bound?**
max(critical-path length, for every resource ⌈total work on it ÷ its capacity⌉). No feasible schedule can be
shorter, so a makespan equal to it is provably optimal.

**How do you handle cycles or impossible files?**
`Project` runs Kahn's topological sort and rejects cycles; it also rejects an activity that needs more of a
resource than exists. The UI shows the error instead of crashing.

**Can I check the result myself?**
Yes: `python scripts/evaluate.py --per-family 0` reproduces `RESULTS.md`; `python -m pytest tests -q` runs the
tests; the dashboard shows every evaluation of every algorithm and exports it as CSV.
