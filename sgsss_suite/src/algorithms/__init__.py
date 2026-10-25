"""All solvers share one signature: solve(project, budget, seed) -> Result."""
from src.algorithms.bco import solve_bco
from src.algorithms.ga import solve_ga
from src.algorithms.prh import solve_prh
from src.algorithms.pss import solve_pss
from src.algorithms.sa import solve_sa
from src.algorithms.ts import solve_ts

SOLVERS = {"TS": solve_ts, "PSS": solve_pss,
           "GA": solve_ga, "SA": solve_sa, "BCO": solve_bco, "PRH": solve_prh}

FULL_NAMES = {"TS": "Tabu Search", "PSS": "Particle Swarm Search", "GA": "Genetic Algorithm",
              "SA": "Simulated Annealing", "BCO": "Bee Colony Optimisation",
              "PRH": "Priority Rule Heuristics"}

PROPOSED = ("TS", "PSS")      # use the trained priority model + justification
BASELINES = ("GA", "SA", "BCO", "PRH")
