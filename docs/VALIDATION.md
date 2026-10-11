# Validation Notes — AI-RCPSP

## Project contains exactly FIVE algorithms

| # | Algorithm | Module |
|---|-----------|--------|
| 1 | Genetic Algorithm (GA) | `src/algorithms/ga.py` |
| 2 | Simulated Annealing (SA) | `src/algorithms/sa.py` |
| 3 | Bee Colony Optimisation (BCO) | `src/algorithms/bco.py` |
| 4 | Priority Rule Heuristic (PRH) | `src/algorithms/prh.py` |
| 5 | Tabu Search (TS) | `src/algorithms/tabu.py` |

PSO is **not implemented** in this project.

---

## Input Format — PSPLIB .sm

PSPLIB is the standard benchmark format for RCPSP research.
Files use the `.sm` extension and contain these sections (in order):

```
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          2           2   3
   ...

REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2
   1      1     0       0    0
   ...

RESOURCEAVAILABILITIES:
  R 1  R 2
    4    3
```

**Key conventions:**
- Job numbers are 1-based.
- Job 1 is always the dummy source (duration = 0, no predecessors).
- The last job is always the dummy sink (duration = 0, no successors).
- Successors are listed per-job in the precedence section.
- The `psplib` library returns 0-based indices; the parser converts to 1-based.
- The sink may be absent from the `psplib` activity list and is synthesized.
- Only **renewable** resources are used by this project.

---

## Parser Behaviour

### Extension dispatch
| Extension | Parser used |
|-----------|-------------|
| `.sm`     | PSPLIB via `psplib` library |
| `.rcp`    | Custom hand-written parser |
| Other     | `ValueError` raised |

### Predecessor inversion
`psplib` stores **successor** lists.
The parser inverts these to **predecessor** lists required by the SSGS.

### Sink synthesis
When `psplib` omits the dummy sink (a known behaviour when the last job has
no successors and zero duration), the parser detects the phantom job number
referenced by predecessors and creates the sink Activity entry explicitly.

### Validation performed at parse time
1. At least 2 activities (source + sink minimum).
2. At least 1 renewable resource.
3. All predecessor IDs resolve to known activities.
4. No activity requires more of a resource than its total capacity.

---

## Benchmark Methodology

### Setup
- Each algorithm run is **independent**.
- The same `Project` object is passed to all algorithms — no mutation.
- Each run uses its own `random.Random(seed)` instance — no shared global state.
- `evaluation_budget` is passed identically to every algorithm and every seed.

### Evaluation counting
- Every call to `Evaluator.evaluate()` counts as one evaluation.
- The SSGS call inside `Evaluator.evaluate()` is the only scheduling engine.
- Algorithms must not call the SSGS independently (verified by mock-patch tests).

### Fairness rules
1. All algorithms receive the same `Project` object.
2. All algorithms receive the same `evaluation_budget`.
3. All algorithms use the same `src/ssgs.py` and `src/evaluation.py`.
4. Internal algorithm parameters (population size, tenure, etc.) are
   algorithm-specific and explicit — not hidden.

---

## Seed Handling

- Seeds are integers, passed directly to `random.Random(seed)`.
- The same seed on the same algorithm on the same project always produces
  the same priority vector sequence and therefore the same final makespan.
- Different seeds may (but are not guaranteed to) produce different makespans.
- PRH selects the deterministic rule (EST/EFT/LST/LFT) with zero randomness
  for those rule evaluations; only the RAND trials are seed-dependent.
- SA may terminate before budget exhaustion when the temperature drops below
  `min_temp`. This is expected behaviour and the evaluation count will be
  lower than the budget.

---

## Metrics

| Metric | Definition |
|--------|------------|
| Best makespan | Minimum makespan found across all runs for this algorithm |
| Mean makespan | Average makespan across all feasible runs |
| Worst makespan | Maximum makespan across all runs |
| Std | Population standard deviation of makespan across runs |
| Mean runtime | Average wall-clock time per run (seconds) |
| Mean evaluations | Average SSGS evaluation calls per run |
| Feasibility rate | Fraction of runs that produced a feasible schedule |

---

## Phase 3 Validation Results (j10_sample.sm, budget=5000, seeds=1,2,3)

```
=== Comparison Report: j10_sample ===
Seeds: 3  |  Evaluation budget: 5000 per run

Algorithm      Best    Mean   Worst    Std  Runtime(s)   Evals  Feasible
----------------------------------------------------------------------
GA               31    31.0      31   0.00       0.705    5000  3/3
SA               31    31.0      31   0.00       0.255    1839  3/3
BCO              31    31.0      31   0.00       0.692    5000  3/3
PRH[EST]         31    31.0      31   0.00       0.779    5000  3/3
TS               31    31.0      31   0.00       0.680    5000  3/3
----------------------------------------------------------------------

Best feasible solution found: 31 (by ALL)

Note: Metaheuristic results are best feasible solutions found,
      not guaranteed global optima.
```

All 15 runs (5 algorithms × 3 seeds) produced feasible schedules.
Zero constraint violations were detected.

**SA note:** SA terminated at 1839 evaluations (temperature floor reached
before budget exhaustion). This is correct behaviour — the algorithm's
geometric cooling reduces temperature to below `min_temp` before the
5000-evaluation budget is consumed.

---

## Limitations

1. **Single-mode RCPSP only.** Multi-mode instances are not supported.
2. **Renewable resources only.** Non-renewable resource constraints are
   ignored during parsing and scheduling.
3. **No global optimum guarantee.** Metaheuristics find good feasible
   solutions but cannot prove optimality without exhaustive search or
   mathematical bounds.
4. **Small fixture instance.** `j10_sample.sm` has 10 real activities and is
   primarily for correctness verification. Results on larger instances
   (j30, j60, j120) may differ significantly.
5. **Single-project instances only.** Multi-project files are not handled.
6. **SA early termination.** With default cooling rate 0.995 the budget may
   not be fully consumed on small instances. This is by design.

---

## Phase History

| Phase | Status | Contents |
|-------|--------|----------|
| 1 | ✅ | Parser (.rcp), SSGS, evaluator, model |
| 2 | ✅ | GA, SA, BCO, PRH, TS |
| 3 | ✅ | PSPLIB .sm parser, benchmark runner, comparison engine |
| 4+ | 🔜 | Feature extraction, ML classifier, UI |
