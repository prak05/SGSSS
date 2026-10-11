# Algorithm Notes

Engineering decisions, literature basis, and implementation choices for
each optimisation algorithm in the AI-RCPSP project.

All algorithms share:
- **Representation**: random-key priority vector `[r₁, r₂, …, rₙ]`, rᵢ ∈ [0, 1]
- **Decoder**: `src/ssgs.py` — Serial Schedule Generation Scheme (common, never duplicated)
- **Evaluator**: `src/evaluation.py` — calls SSGS, validates constraints, returns makespan
- **Objective**: minimise makespan (lower = better)
- **Budget**: 5 000 SSGS evaluations by default (configurable)

---

## 1. Genetic Algorithm (`src/algorithms/ga.py`)

### Representation
Random-key chromosome; one float ∈ [0, 1] per activity.

### Literature basis
- *Hartmann (2002)*: random-key GA for RCPSP (canonical reference).
- Repositories studied (ideas only): `derhendrik/RCPSP_GA`, `juliusf/Genetic-SRCPSP`.

### Mechanism
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Initialisation | Uniform random [0,1] | Standard for random-key GA |
| Selection | Tournament (k=3) | Simple, avoids premature convergence |
| Crossover | Uniform blend | Preserves [0,1] bounds; each gene copied from one parent |
| Mutation | Per-gene random reset at rate `mutation_rate` | Simple creep mutation |
| Elitism | Top `elite_count` individuals passed unchanged | Preserves best solutions |
| Termination | Evaluation budget exhausted | Ensures fair comparison |

### Engineering decisions
- Uniform blend crossover chosen over one-point (easier bound maintenance).
- Tournament selection preferred over roulette (avoids fitness-scaling issues).

---

## 2. Simulated Annealing (`src/algorithms/sa.py`)

### Representation
Current solution = random-key vector; one float ∈ [0, 1] per activity.

### Literature basis
- Kirkpatrick et al. (1983): canonical SA.
- Repository studied: `mbenahmed1/ScheduleMe` (RCPSP SA framework — ideas only).

### Mechanism
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Initial solution | Uniform random | Simple, unbiased start |
| Neighbourhood | Swap two keys OR Gaussian perturb one key | Mix of structural and continuous moves |
| Acceptance | Boltzmann: P = exp(−Δ/T) | Standard SA; allows escaping local optima |
| Cooling | Geometric: T ← α·T | Simple, predictable decay |
| Termination | Budget exhausted OR T < T_min | Dual stopping condition |

### Parameters (defaults)
- `initial_temp = 100.0`
- `cooling_rate = 0.995`
- `min_temp = 0.01`

### Engineering decisions
- Gaussian perturbation σ=0.1 is an engineering choice (not from paper).
- Cooling rate tuned empirically so budget is consumed before T_min on medium instances.

---

## 3. Bee Colony Optimisation (`src/algorithms/bco.py`)

### Variant chosen
**Artificial Bee Colony (ABC) adapted to random-key RCPSP** — not a generic
ABC implementation.

### Literature basis
- Karaboga & Basturk (2007): canonical ABC algorithm.
- Sonmez (2013): RCPSP bee-algorithm comparison — *ScienceDirect*.
- Küçükoğlu et al. (2026): random-key ABC with serial SGS for RCPSP — *Springer*.
- Repository studied for priority-rule ideas: `sydney-machine-learning/MEHH_RCPSP`.

### Mechanism
| Phase | Role | Behaviour |
|-------|------|-----------|
| Employed bees | Exploitation | Perturb each food source using ABC φ-formula |
| Onlooker bees | Biased exploitation | Preferentially perturb better sources (makespan-weighted) |
| Scout bees | Exploration | Abandon exhausted sources (trial counter ≥ limit); replace with random |

#### ABC perturbation formula
```
v_i = x_i + φ · (x_i − partner_i)   where φ ∈ [−1, 1]
```
Clamped to [0, 1]. Only one gene per candidate is modified per trial.

#### Abandonment criterion
If a food source is not improved for `limit` consecutive trials, it is
replaced by a random candidate (scout phase). `limit = 15` by default.

### Paper requirement vs. engineering decision
| Item | Status |
|------|--------|
| Random-key representation | Paper requirement (RCPSP random-key literature) |
| Serial SGS decoding | Paper requirement |
| ABC employed/onlooker/scout structure | Literature basis (Küçükoğlu 2026) |
| Makespan-weighted onlooker selection | Engineering choice (monotone weight function) |
| φ-formula perturbation | Literature basis (Karaboga 2007) |
| `limit` parameter | Engineering choice (default 15) |

---

## 4. Priority Rule-based Heuristic (`src/algorithms/prh.py`)

### Representation
Priority dict `{act_id: float}` derived from a rule, normalised to [0, 1],
decoded through the common SSGS.

### Literature basis
- Research paper (primary authority): EST, LST, EFT, LFT, RANDOM rules.
- UGent RCPSP research group: extensive priority-rule documentation.
- Repository studied: `sydney-machine-learning/MEHH_RCPSP`.

### Rules implemented
| Rule | Priority computation | Type |
|------|---------------------|------|
| EST | Earliest Start Time (CPM forward pass) | Deterministic |
| EFT | Earliest Finish Time | Deterministic |
| LST | Latest Start Time (CPM backward pass) | Deterministic |
| LFT | Latest Finish Time | Deterministic |
| RAND | Uniform random [0,1] | Stochastic |

### Mechanism
1. Evaluate EST, EFT, LST, LFT rules once each (4 evaluations).
2. Use remaining budget on repeated RAND trials.
3. Return the result with lowest makespan, naming the winning rule.

### Engineering decisions
- LFT backward pass uses max(EFT) as project horizon (CPM horizon).
- PRH algorithm name encodes winning rule: `PRH[LST]`, `PRH[RAND]`, etc.
- Budget distribution (4 deterministic + remainder to RAND) is an engineering choice.

---

## 5. Tabu Search (`src/algorithms/tabu.py`)

### Representation
Random-key vector; one float ∈ [0, 1] per activity.

### Literature basis
- Glover (1989): canonical Tabu Search.
- Repositories studied: `ThomasK33/rcpsp.rs` (TS for RCPSP), `CTU-IIG/RCPSPCpu` (parallel TS benchmark).

### Mechanism
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Neighbourhood | Pairwise swap of random-key positions | Preserves [0,1]; well-defined tabu ID |
| Tabu ID | (min(i,j), max(i,j)) swap pair | Compact, unambiguous |
| Tabu tenure | Fixed `tenure` iterations | Simple short-term memory |
| Aspiration | Accept tabu move if ms < global best | Standard criterion |
| Restart | Random reinitialisation if entire neighbourhood is tabu | Escape total stagnation |

### Parameters (defaults)
- `tenure = 7`
- `n_neighbours = 10`

### Engineering decisions
- Swap neighbourhood preferred over key-perturbation (easier to define stable tabu IDs).
- Tenure value of 7 is an engineering choice; should be tuned per instance size.
- Restart on full-tabu-neighbourhood is an engineering choice not in every TS formulation.

---

## Common Architecture Note

```
Algorithm
   ↓
generate candidate priority vector
   ↓
Evaluator.evaluate(project, priority_vector)
   ↓
SSGS.generate_schedule(project, priority_vector)
   ↓
feasible schedule + makespan
```

No algorithm contains any resource-feasibility logic or scheduling logic.
All constraint checking is performed exclusively by `src/evaluation.py` and `src/ssgs.py`.
