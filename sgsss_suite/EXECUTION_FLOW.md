# Execution flow — where each step happens

```
project file ──► parse ──► Project ──► run 6 solvers (same budget) ──► audit ──► rank ──► explain ──► UI / CLI / PDF
 .sm .rcp .json  parser.py  model.py    algorithms/*  ── each uses ──► ssgs.py    evaluation.py  benchmark.py
```

| # | Step | Where | Notes |
|---|---|---|---|
| 1 | Start the app | `ui.py` (dashboard) or `backend.py` (command line) | |
| 2 | Choose a project | `ui.py` sidebar | use cases, held-out test set, training set, or upload |
| 3 | Read the file | `src/parser.py` → `parse()` | `.sm` PSPLIB, `.rcp` Patterson/RanGen, `.json` (named tasks and resources) |
| 4 | Build the model | `src/model.py` → `Project` | checks for cycles (Kahn's algorithm) and impossible demands; computes CPM times, critical path and lower bound |
| 5 | Set budget & seed | `src/benchmark.py` → `auto_budget()`, `auto_seed()` | from the project content (size, resource pressure; content fingerprint), never the file type; the same for every solver |
| 6 | GNN start (proposed) | `src/algorithms/gnn.py` → `predict()` | bidirectional graph attention over predecessors and successors; weights in `data/models/gnn.pt` |
| 7 | Run the solvers | `src/algorithms/ts.py`, `pss.py` (proposed); `ga.py`, `sa.py`, `bco.py`, `prh.py` (baselines) | all report through `src/algorithms/common.py` → `Search`, which counts evaluations and records every iteration |
| 8 | Decode schedules | `src/ssgs.py` → `decode()`, `justify()` | the only place schedules are built, so every schedule is feasible by construction |
| 9 | Audit | `src/evaluation.py` → `audit()` | independent precedence and capacity check of every final schedule |
| 10 | Rank & explain | `src/benchmark.py` → `ranking()`, `explain()` | makespan → total finish time → levelling → reliability over 5 seeds → runtime; no name-based rules |
| 11 | Show | `ui.py` tabs, `src/charts.py` | precedence graph, iteration table, convergence, Gantt, resource use, how the winner won, GNN inside (`gnn.inspect()`) |
| 12 | Report | `src/report.py` → `pdf_report()` | PDF with ranking, explanation and convergence plot |

Offline steps
| Step | Where |
|---|---|
| Download and split the benchmark data | `scripts/build_datasets.py` → `data/train`, `data/test`, `data/test/reference.csv` |
| Build the realistic projects | `scripts/make_use_cases.py` → `data/use_cases` (66), `data/use_cases_small` (8 × 3 formats) |
| Train the GNN (REINFORCE, 1080 projects) | `scripts/train.py` → `data/models/gnn.pt`, `training_log.json` |
| Compare TS/PSS settings (training data only) | `scripts/tune.py` |
| Results for every use case | `scripts/evaluate_use_cases.py` → `docs/USE_CASES.md`, `docs/use_cases.csv` |
| Measure everything quoted in the docs | `scripts/evaluate.py` → `docs/RESULTS.md`, `docs/results.csv` |
| Tests | `tests/test_ai_rcpsp.py` |
