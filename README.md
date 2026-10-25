> **All code lives in [`sgsss_suite/`](sgsss_suite).** Run the commands below from inside that folder. Older prototypes (`app.py`, `rcpsp_all_in_one.py`) are kept for history only.

# AI-RCPSP — GNN-guided Tabu Search and Particle Swarm Search for project scheduling

The **Resource-Constrained Project Scheduling Problem (RCPSP)**: activities have durations, precedence
relations ("B can start only after A finishes") and need renewable resources (workers, machines) whose
capacity is limited in every period. Goal: the shortest possible project (minimum **makespan**).
The problem is NP-hard.

This project proposes two solvers and compares them fairly with four classic baselines:

| | Algorithm | Idea |
|---|---|---|
| **Proposed** | **TS — Tabu Search** | starts from the trained GNN's order; shift moves with tabu memory; justification of every new best |
| **Proposed** | **PSS — Particle Swarm Search** | swarm seeded by the GNN + classic rules; Lamarckian justification of particles |
| Baseline | GA — Genetic Algorithm | random keys, tournament selection, uniform crossover, mutation, elitism |
| Baseline | SA — Simulated Annealing | random swaps, Metropolis acceptance, geometric cooling |
| Baseline | BCO — Bee Colony Optimisation | employed / onlooker / scout bees (ABC) |
| Baseline | PRH — Priority Rule Heuristics | LFT, LST, MSLK, MTS, GRPW, … then biased random sampling |

**What makes the proposed solvers better** (all measured, see `docs/RESULTS.md`):
1. **A trained graph neural network** (bidirectional graph attention, `src/algorithms/gnn.py`) predicts a good
   activity order before any search starts. It was trained with reinforcement learning on **1080 projects**.
2. **Forward-backward justification** re-packs schedules to close idle gaps.
3. **Schedule quality beyond makespan**: among equal-makespan schedules, TS and PSS keep the one with the lowest
   total finish time and the smoothest resource use. The baselines are textbook versions that only minimise makespan.

## Fair comparison — the rules
- **Same budget** for every solver: the number of schedule evaluations (SSGS decodes), set automatically from the
  project's content — 10 × activities × (1 + resource pressure, capped at 2), rounded to 50 (≈500 for a typical
  30-activity benchmark; more for bigger or resource-tight projects). The **random seed** is a fingerprint of the project
  content. Neither depends on the file type: the same project as `.sm`, `.rcp` or `.json` gets the same budget and seed.
  Every decode counts, including both justification passes.
- **Same decoder** (`src/ssgs.py`) for every solver, so every schedule is feasible by construction; an
  **independent audit** (`src/evaluation.py`) re-checks precedence and capacity for every result.
- **Winner from the data only** (`src/benchmark.py`): makespan → total finish time → resource levelling →
  reliability (mean evaluations to reach that makespan over 5 seeds, computed only for algorithms still tied) →
  runtime. Algorithm names play no part; a test renames them to prove it.
- **No training data in the test set**: the GNN is trained on `data/train`; results are measured on `data/test`.

## Data
| Folder | Files | Content |
|---|---|---|
| `data/train` | 1080 | PSPLIB J30 groups 13–48 (.sm, 360), PSPLIB J60 groups 7–48 (.sm 360 + groups 13–18 as .json 60), RG30 Pat101–400 (.rcp, 300) |
| `data/test` | 260 | PSPLIB J30 groups 1–12, J60 groups 1–6 (.sm + .json), J90/J120 samples, RG30 Pat1–60; `reference.csv` = 200 published optimal / best-known makespans |
| `data/use_cases` | 66 | generated realistic projects: 12 industries, 30–60 named tasks, tight resources, 22 per format. Results for **every** one in `docs/USE_CASES.md` |
| `data/use_cases_small` | 24 | 8 hand-written small projects (10–18 tasks) in all three formats; too small to separate the algorithms |
| `data/models` | 2 | `gnn.pt` (trained weights), `training_log.json` (training curve + evaluation) |

Rebuild everything: `python scripts/build_datasets.py` (downloads PSPLIB + RG30) and `python scripts/make_use_cases.py`.

## Run it
```bash
./venv/bin/streamlit run ui.py                                   # dashboard
./venv/bin/python backend.py data/use_cases/house_construction.sm   # command line
./venv/bin/python -m pytest tests -q                             # tests
./venv/bin/python scripts/train.py                               # retrain the GNN on 1080 projects (~7 min)
./venv/bin/python scripts/evaluate.py --per-family 0             # test benchmark -> docs/RESULTS.md
./venv/bin/python scripts/evaluate_use_cases.py                  # use cases -> docs/USE_CASES.md
./venv/bin/python scripts/tune.py                                # compare TS/PSS settings on training data
```
In the dashboard choose a project, keep **Run all algorithms**, press **▶ Run**:
- **Iterations** shows a win tile per algorithm (benchmark wins, how long it led this run, the winner highlighted),
  the convergence curves and a table with one row per evaluation for all six solvers, downloadable as CSV.
- **Results** (kept separate) gives the winner, the ranking, which attribute decided it, the step-by-step evidence
  and the PDF report.
- **GNN inside** explains the neural network in plain language on the chosen project, in four steps: the facts it
  reads about each task, which connected tasks it listens to, the priority queue it produces (and which tasks it
  moved compared with the textbook rule), and the schedule that queue gives compared with every classic rule.
- **Benchmark evidence** lists every win and loss on unseen projects.
- Every tab has its own **📘 Guide to this page**: the key terms used there and every formula on that page, each with
  what it means in plain words and *why* we use it (26 formulas in total). Every table also has column tooltips and a
  "What do these columns mean?" box. All of this text lives in `explanation.py`.

## Code map
| File | What it does |
|---|---|
| `src/model.py` | `Project` (activities, resources, CPM times, lower bound) and `Result` |
| `src/parser.py` | read/write `.sm`, `.rcp`, `.json` |
| `src/ssgs.py` | the SSGS decoder, backward pass, forward-backward justification |
| `src/evaluation.py` | independent audit, resource profile, quality attributes |
| `src/algorithms/common.py` | budget counting and iteration traces shared by all solvers |
| `src/algorithms/rules.py` | classic priority rules and the 10 GNN input features |
| `src/algorithms/gnn.py` | bidirectional graph attention network |
| `src/algorithms/ts.py`, `pss.py` | the proposed solvers |
| `src/algorithms/ga.py`, `sa.py`, `bco.py`, `prh.py` | the baselines |
| `src/benchmark.py` | run all, rank, explain the win |
| `src/charts.py`, `src/report.py` | Plotly charts, PDF report |
| `ui.py`, `backend.py` | dashboard, command line |
| `explanation.py` | every explanation shown in the dashboard: per-tab guides, formulas (what + why), column meanings, terms |
| `.streamlit/config.toml` | dashboard theme |
| `scripts/` | dataset builder, use-case builder, training, tuning, evaluation (test set and use cases) |

See `EXECUTION_FLOW.md` for the step-by-step flow and `docs/academic_proof_and_defense.md` for the viva guide.

## References
Kolisch & Sprecher (1996) PSPLIB · Kolisch (1996) serial SGS · Valls, Ballestín & Quintanilla (2005) justification ·
Hartmann (1998) GA for RCPSP · Karaboga (2005) ABC · Glover (1989) tabu search · Kennedy & Eberhart (1995) PSO ·
Veličković et al. (2018) graph attention networks · Williams (1992) REINFORCE ·
Zhang, Cao, Wu, Song & Sun (UAI 2024) TBGAT — bidirectional graph attention for job-shop scheduling.
