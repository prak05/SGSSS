"""Plain-English explanations shown in the dashboard (ui.py).

Every table column, every tab and every technical term has an entry here, so a first-time
user can understand what they are looking at. Edit the text here — the GUI picks it up.
"""

# ── What each tab is for (shown at the top of the tab) ───────────────────────────────
TABS = {
    "overview": "Start here. This page describes the project you picked: its activities, how they depend on each "
                "other and which resources they need. Nothing has been optimised yet.",
    "iterations": "The race, evaluation by evaluation. Every algorithm gets the same number of attempts "
                  "(evaluations). Each attempt produces one complete, valid schedule; the table records its length "
                  "and the best length found so far.",
    "schedules": "The actual timetable each algorithm produced: when every activity starts and ends (Gantt chart) "
                 "and how much of each resource is in use at every moment.",
    "results": "The verdict, kept separate from everything else: who won, by how much, why, and the evidence. "
               "The winner is chosen only from measured numbers, never from algorithm names.",
    "gnn": "What the 'smart starting point' is and how it works, in plain words, using the project you picked.",
    "evidence": "Results on projects the network never saw: 260 benchmark projects with published reference values "
                "and 66 realistic industry projects. Every win and every loss is listed.",
}

# ── The algorithms ──────────────────────────────────────────────────────────────────
ALGORITHMS = {
    "TS": ("Tabu Search", "Proposed",
           "Starts from the trained network's order, then keeps moving single activities to better positions. "
           "Remembers recent moves (the tabu list) so it does not go in circles."),
    "PSS": ("Particle Swarm Search", "Proposed",
            "A swarm of 20 candidate orders that move towards their own best and the swarm's best. Seeded by the "
            "trained network; half the schedules are re-packed to close gaps."),
    "GA": ("Genetic Algorithm", "Baseline",
           "A population of random orders that are combined (crossover) and randomly changed (mutation); "
           "the best survive."),
    "SA": ("Simulated Annealing", "Baseline",
           "One random order changed step by step; worse changes are sometimes accepted early on, less and "
           "less as the 'temperature' cools."),
    "BCO": ("Bee Colony Optimisation", "Baseline",
            "Artificial bees improve food sources (orders); good sources attract more bees, exhausted ones "
            "are abandoned."),
    "PRH": ("Priority Rule Heuristics", "Baseline",
            "Classic textbook rules (e.g. 'latest finish time first'), then noisy variations of them."),
}

# ── Glossary ────────────────────────────────────────────────────────────────────────
TERMS = {
    "Activity": "One task of the project, e.g. 'Foundation'. Start and Finish are zero-length markers.",
    "Precedence": "'B after A' means B may start only when A has finished.",
    "Renewable resource": "Workers or machines available in a fixed amount in every period (the capacity). "
                          "Activities running at the same time must not use more than the capacity.",
    "Makespan": "Total project length: the finish time of the last activity. Lower is better. This is the main goal.",
    "Lower bound": "A length no schedule can beat: the longer of (a) the critical path and (b) total work on a "
                   "resource divided by its capacity. A makespan equal to it is provably optimal.",
    "Critical path": "The longest chain of dependent activities. Even with unlimited resources the project "
                     "cannot be shorter.",
    "Slack": "How many periods an activity can slip without delaying the project (ignoring resources). "
             "0 = critical.",
    "Evaluation budget": "Evaluations each algorithm may use, set automatically from the project: "
                         "10 × activities × (1 + resource pressure, capped at 2), rounded to 50, at least 200. "
                         "Bigger projects and tighter resources get more. Identical for all six algorithms.",
    "Resource pressure": "How strongly resources limit the project: for the busiest resource, its total work "
                         "÷ its capacity, divided by the critical-path length. Above 1 = resources, not precedence, "
                         "decide the project length.",
    "Random seed": "Starting point of the random number generator. Set automatically as a fingerprint of the "
                   "project's content (durations, demands, capacities, precedence) — so the same project gets the "
                   "same seed whether it comes as .sm, .rcp or .json, and every run is reproducible.",
    "Evaluation": "One attempt: an algorithm proposes an activity order and the decoder turns it into a complete "
                  "schedule. All algorithms get the same number of evaluations (the budget).",
    "SSGS (decoder)": "Serial Schedule Generation Scheme: takes an order and places each activity at the earliest "
                      "time that respects precedence and capacity. Every schedule is therefore valid.",
    "Justification": "A repair step: push every activity as late as possible, then as early as possible. "
                     "Closes idle gaps. Costs two evaluations.",
    "Total finish time": "Sum of all activities' finish times. Lower means work gets done earlier on average. "
                         "First tie-break when makespans are equal.",
    "Resource levelling": "Sum of squared resource usage (as a share of capacity) over all periods. Lower means a "
                          "smoother workload without peaks. Second tie-break.",
    "Reliability": "If algorithms are still tied, each is re-run with 4 more random seeds; the one that needs "
                   "fewer evaluations on average to reach that makespan wins. Third tie-break.",
    "GNN": "Graph Neural Network — the project's 'experienced planner'. It looks at the whole task network and "
           "suggests which tasks to schedule first, before the search algorithms start. It learned this from "
           "1,080 past projects.",
    "Attention": "How much a task takes a connected task into account when the GNN decides its priority. "
                 "A task only ever looks at tasks directly connected to it.",
    "Gap to lower bound": "(makespan − lower bound) ÷ lower bound, in %. 0% = provably optimal.",
    "Optimum / best known": "For the J30 benchmark the optimal makespans are proven and published; for J60/J90/J120 "
                            "the best makespans anyone has published.",
}

# ── Table columns: header → explanation (shown as tooltips and in the legend below each table) ──
COLUMNS = {
    # project table
    "Activity": "Name of the task.",
    "Duration": "How many periods the activity takes.",
    "Must follow": "Activities that must finish before this one may start.",
    "Earliest start": "Earliest possible start if resources were unlimited (critical-path method).",
    "Slack": TERMS["Slack"],
    # iteration table
    "Evaluation": "Attempt number (1 = the first schedule each algorithm built).",
    "best": "Shortest makespan this algorithm had found up to and including this evaluation. It can only go down.",
    "tried": "Makespan of the schedule built at exactly this evaluation (can go up and down while it explores).",
    "New best": "The makespan this algorithm improved to at that evaluation.",
    "At evaluation": "The evaluation at which the improvement happened.",
    # ranking table
    "Rank": "Final position: 1 = winner.",
    "Algorithm": "Short name (TS, PSS = proposed; GA, SA, BCO, PRH = baselines).",
    "Name": "Full name of the algorithm.",
    "Type": "Proposed (this project's method) or Baseline (classic method to compare against).",
    "Makespan": TERMS["Makespan"],
    "Gap to lower bound (%)": TERMS["Gap to lower bound"],
    "Behind winner": "How many periods longer than the winner's schedule (0 = same length).",
    "Total finish time": TERMS["Total finish time"],
    "Levelling": TERMS["Resource levelling"],
    "Best found at": "Evaluation at which this algorithm first reached its final makespan.",
    "Reliability (avg. evals, 5 seeds)": TERMS["Reliability"] + " Shown as the first-run value when not needed.",
    "Evaluations used": "Attempts actually used. Fewer than the budget means it stopped early because the "
                        "schedule was provably optimal.",
    "Stopped early (optimal)": "Yes = the makespan equalled the lower bound, so searching further was pointless.",
    "Runtime (s)": "Wall-clock seconds. Last tie-break only.",
    "Valid": "Independent check that no activity starts before its predecessors finish and no resource is "
             "over capacity. Should always be ✔.",
    # GNN tables
    "earliest start": "Earliest start ÷ critical-path length (0 = can start at once).",
    "latest start": "Latest start without delaying the project ÷ critical-path length.",
    "latest finish": "Latest finish without delaying the project ÷ critical-path length.",
    "slack": "Slack ÷ critical-path length (0 = critical).",
    "duration": "Duration ÷ longest duration in the project.",
    "direct successors": "Number of activities waiting directly on this one (scaled).",
    "all successors": "Number of activities that depend on it directly or indirectly ÷ number of activities.",
    "rank positional weight": "Own duration + durations of direct successors (scaled).",
    "resource load": "Average share of each resource's capacity this activity needs.",
    "predecessors": "Number of activities it waits for (scaled).",
    "Task": "Name of the task.",
    "Classic rule score": "Priority the classic 'latest start time first' rule would give (higher = sooner).",
    "Network's adjustment": "What the trained network adds (+) or takes away (−) based on what it learned.",
    "Final priority": "Classic rule score + the network's adjustment. Higher = scheduled sooner.",
    "Queue position (network)": "Where the task stands in the network's suggested order (1 = first in line).",
    "Queue position (classic rule)": "Where it would stand using the classic rule alone.",
    "Moved": "↑ = the network moved it earlier in the queue, ↓ = later, blank = unchanged.",
    "View": "Which attention: layer 1 or 2, looking upstream (at predecessors) or downstream (at successors).",
    "Neighbour": "The activity being listened to (including itself).",
    "Weight": "Share of attention (0–1); the weights in one view add up to 1.",
    "LST term": "Score from the classic 'latest start time first' rule (higher = schedule earlier).",
    "Learned correction": "What the trained network adds or subtracts from the LST score.",
    "Final score": "LST term + learned correction. The decoder schedules higher scores first.",
    "GNN rank": "Position in the GNN's order (1 = considered first).",
    "LST rank": "Position under the plain LST rule.",
    "Changed by GNN": "✔ where the network moved the activity compared with the plain rule.",
}

AUTO_SETTINGS = ("budget = 10 × {n} activities × (1 + resource pressure {pressure:.2f}), rounded to 50; "
                 "seed = fingerprint of the project content. The file type plays no part — the same project as "
                 ".sm, .rcp or .json gets the same values.")

# ── Per-tab guide: the terms and formulas used on each page, and WHY each formula is used ──
# Each formula: (name, LaTeX, what it means in plain words, why we use it)
HELP = {
    "overview": {
        "terms": ["Activity", "Precedence", "Renewable resource", "Critical path", "Slack", "Lower bound",
                  "Resource pressure", "Evaluation budget", "Random seed"],
        "formulas": [
            ("Earliest start (critical-path method)",
             r"ES_j = \max_{i \in \mathrm{Pred}(j)} \left( ES_i + d_i \right)",
             "A task can start once its slowest predecessor has finished (dᵢ = duration of task i).",
             "It gives the earliest possible start of every task when resources are unlimited — the starting point "
             "for slack, the critical path and the lower bound."),
            ("Critical path length",
             r"CP = \max_j \left( ES_j + d_j \right)",
             "The finish time of the last task if resources were unlimited.",
             "No real schedule can be shorter, so it tells us how close to perfect a schedule is."),
            ("Slack",
             r"\mathrm{Slack}_j = LS_j - ES_j",
             "How long a task can be delayed (latest start − earliest start) without delaying the project.",
             "Slack 0 = critical task. Critical tasks are coloured red in the graph because delaying them delays "
             "everything."),
            ("Lower bound",
             r"LB = \max\left( CP,\ \max_k \left\lceil \frac{\sum_j d_j\, r_{jk}}{R_k} \right\rceil \right)",
             "Either the critical path, or the total work needing resource k (duration × units, rⱼₖ) divided by how "
             "many units exist (Rₖ) — whichever is larger.",
             "A guaranteed minimum project length. If an algorithm reaches it, its schedule is provably optimal "
             "and the algorithm can stop."),
            ("Resource pressure",
             r"\rho = \frac{1}{CP}\ \max_k \frac{\sum_j d_j\, r_{jk}}{R_k}",
             "How busy the busiest resource is compared with the critical path. ρ > 1: resources, not the task "
             "order, decide the project length.",
             "Tight-resource projects are harder, so they need more search effort — ρ feeds the budget formula."),
            ("Evaluation budget",
             r"B = \max\Big(200,\ 50 \cdot \mathrm{round}\big(\tfrac{10\, n\, (1 + \min(\rho, 2))}{50}\big)\Big)",
             "10 attempts per task (n = number of tasks), up to 3× more for resource-tight projects, rounded to 50.",
             "Bigger and harder projects get more attempts, and every algorithm gets exactly the same number — "
             "so the comparison is fair. A typical 30-task benchmark gets 500, the usual research standard."),
            ("Random seed",
             r"\mathrm{seed} = \mathrm{SHA256}(\text{durations, demands, capacities, precedence}) \bmod 10^5",
             "A fingerprint of the project's content. File names, task names and file type are ignored.",
             "The same project always gets the same seed — in .sm, .rcp or .json — so every run can be repeated "
             "exactly."),
        ],
    },
    "iterations": {
        "terms": ["Evaluation", "Makespan", "SSGS (decoder)", "Justification"],
        "formulas": [
            ("Best so far",
             r"b_t = \min_{s \le t} m_s",
             "After attempt t, the best makespan found in attempts 1…t (mₛ = makespan of attempt s).",
             "Attempts go up and down while an algorithm explores; the best-so-far line only goes down and shows "
             "real progress."),
            ("“Led this run” (tiles)",
             r"L_a = \#\left\{ t : b^{a}_t = \min_x b^{x}_t \right\}",
             "The number of attempts at which algorithm a had the best (or joint-best) result of all six.",
             "Shows who was ahead during the race, not just at the finish line."),
            ("Particle Swarm Search — how a particle moves",
             r"v \leftarrow w\,v + c_1 r_1 (p_{best} - x) + c_2 r_2 (g_{best} - x), \qquad x \leftarrow x + v",
             "Each candidate order x drifts towards its own best (p_best) and the swarm's best (g_best); w falls "
             "from 0.9 to 0.4, c₁ = c₂ = 1.5, r₁, r₂ are random.",
             "Balances exploring new orders (early, high w) with refining the best one found (late, low w)."),
            ("Simulated Annealing — accepting a worse step",
             r"P(\text{accept}) = e^{-\Delta / T}",
             "A move that makes the schedule Δ periods longer is accepted with this probability; temperature T "
             "falls over time.",
             "Accepting some worse moves early helps escape dead ends; later it settles on the best area."),
            ("Tabu Search — the tabu rule",
             r"\text{move}(a)\ \text{allowed} \iff \text{it}_{now} > \text{it}_{last}(a) + 7 \ \lor\ m_{new} < m_{best}",
             "A task that was just moved may not be moved again for 7 iterations, unless doing so beats the best "
             "schedule ever found.",
             "Stops the search from undoing its own moves and going round in circles."),
        ],
    },
    "schedules": {
        "terms": ["SSGS (decoder)", "Renewable resource", "Makespan"],
        "formulas": [
            ("Placing a task (SSGS)",
             r"S_j = \min \Big\{ t \ge \max_{i \in \mathrm{Pred}(j)} F_i \ :\ u_k(\tau) + r_{jk} \le R_k\ \ "
             r"\forall k,\ \forall \tau \in [t, t + d_j) \Big\}",
             "A task starts at the earliest time after all its predecessors finish when, for its whole duration, "
             "every resource still has enough free units.",
             "This is what makes every schedule valid by construction — no task starts too early and no resource "
             "is ever over its limit."),
            ("Finish time and project length",
             r"F_j = S_j + d_j, \qquad C_{max} = \max_j F_j",
             "A task finishes its duration after it starts; the project ends when the last task ends.",
             "C_max (the makespan) is the number every algorithm tries to make as small as possible."),
            ("Resource use per period",
             r"u_k(t) = \sum_{j\,:\,S_j \le t < F_j} r_{jk} \ \le\ R_k",
             "Units of resource k busy at time t = the sum over all tasks running at that moment.",
             "The resource chart plots exactly this; the independent check confirms it never exceeds capacity."),
        ],
    },
    "results": {
        "terms": ["Makespan", "Lower bound", "Gap to lower bound", "Total finish time", "Resource levelling",
                  "Reliability"],
        "formulas": [
            ("Gap to lower bound",
             r"\mathrm{Gap} = \frac{C_{max} - LB}{LB} \times 100\%",
             "How far a schedule is above the guaranteed minimum, in percent.",
             "Makes results comparable across projects of different sizes; 0% = provably optimal."),
            ("Total finish time (tie-break 1)",
             r"TF = \sum_j F_j",
             "The sum of all tasks' finish times.",
             "Two schedules can end on the same day; the one with lower TF completes its work earlier on average — "
             "less work in progress, earlier hand-overs."),
            ("Resource levelling (tie-break 2)",
             r"L = \sum_k \sum_t \left( \frac{u_k(t)}{R_k} \right)^2",
             "Squared share of each resource in use, added over all periods.",
             "Squaring punishes peaks: a smooth workload scores lower than one that swings between idle and "
             "overloaded — easier for real crews."),
            ("Reliability (tie-break 3)",
             r"\bar{E} = \frac{1}{5} \sum_{s=1}^{5} E_s",
             "Eₛ = attempts needed to reach the tied makespan with random seed s (5 seeds).",
             "One lucky run proves little; an algorithm that reaches the result quickly with every seed is "
             "genuinely more reliable."),
            ("Choosing the winner",
             r"\text{winner} = \arg\min_a \big( C_{max},\ TF,\ L,\ \bar{E},\ \text{runtime} \big)",
             "Compare makespan first; only if tied compare TF, then L, then reliability, then runtime.",
             "A fixed, published order applied to every algorithm the same way — the algorithm's name never "
             "matters."),
        ],
    },
    "gnn": {
        "terms": ["GNN", "Attention"],
        "formulas": [
            ("Scaling a fact to 0–1",
             r"x = \frac{\text{value}}{\text{largest value in the project}}",
             "Each fact is divided by the biggest such value in the project (times by the critical-path length).",
             "Lets the same network handle a 10-task and a 120-task project — it compares proportions, not raw "
             "numbers."),
            ("Attention — how much task i listens to task j",
             r"\alpha_{ij} = \frac{e^{\,q_i \cdot k_j / \sqrt{d}}}{\sum_{l \in N(i)} e^{\,q_i \cdot k_l / \sqrt{d}}}",
             "A similarity score between task i's 'question' qᵢ and neighbour j's 'answer' kⱼ, turned into shares "
             "that add up to 1 over i's connected tasks N(i).",
             "Not every neighbour matters equally; the network learns which ones to listen to, and only connected "
             "tasks are allowed."),
            ("Updating a task's picture",
             r"h_i \leftarrow \mathrm{Norm}\Big( h_i + f\big( \textstyle\sum_{j} \alpha^{\uparrow}_{ij} v_j \ \Vert\ "
             r"\sum_{j} \alpha^{\downarrow}_{ij} v_j \big) \Big)",
             "Combine what it heard from the tasks before it (↑) and after it (↓), and add it to what it already "
             "knew.",
             "Adding (instead of replacing) keeps the task's own facts; doing it twice spreads information two "
             "steps along the network."),
            ("Final priority",
             r"\mathrm{priority}_i = w \cdot \left( -\frac{LS_i}{CP} \right) + \mathrm{correction}_i",
             "The textbook 'latest start' rule plus a learned correction.",
             "Starting from a proven rule makes the network safe — untrained it equals the rule — and the learned "
             "correction is where it improves on it."),
        ],
    },
    "evidence": {
        "terms": ["Optimum / best known", "Gap to lower bound", "Lower bound"],
        "formulas": [
            ("Gap to the optimum / best known",
             r"\mathrm{Gap}^{*} = \frac{C_{max} - C^{*}}{C^{*}} \times 100\%",
             "C* = the proven optimal (J30) or best published (J60/J90/J120) makespan for that benchmark.",
             "Compares our schedules with the best results the research community has found — an outside yardstick."),
            ("Win rate",
             r"\text{win rate}_a = \frac{\#\ \text{projects won by } a}{\#\ \text{projects}}",
             "Share of unseen projects on which algorithm a was ranked first.",
             "Tested on projects never used for training, so it measures real performance, not memorisation."),
        ],
    },
}

# ── GNN tab, in plain language ───────────────────────────────────────────────────────
GNN_INTRO = ("**Why is there a neural network in a scheduling project?**  \n"
             "Tabu Search and Particle Swarm Search improve a schedule step by step. Where they *start* matters a lot: "
             "a good first guess saves hundreds of attempts. The GNN gives them that good first guess.  \n"
             "Think of it as an **experienced project manager** who glances at the plan and says *“start these tasks "
             "first”* — except it learned that instinct from **1,080 past projects** instead of years on site. "
             "It runs once, in a fraction of a second, and its suggestion becomes attempt no. 1 of TS and PSS.")

GNN_STEPS = [
    ("👀", "Look at every task", "Read the facts about each task",
     "Before deciding anything, the planner needs to know each task: how long it takes, how much slack it has, "
     "how many tasks are waiting for it, and how many workers/machines it needs.",
     "For every task it writes down 10 simple facts as numbers between 0 and 1 (0 = low, 1 = high), so a "
     "2-day task in a small project and a 20-day task in a big one can be compared fairly."),
    ("🤝", "Ask the connected tasks", "Learn from the tasks before and after it",
     "A task's importance depends on its neighbours. A short task can still be urgent if many big tasks are "
     "waiting for it to finish.",
     "Each task 'asks' the tasks directly before it (am I ready to start?) and directly after it (who is waiting "
     "for me?), and decides how much weight to give each answer. It does this twice, so information travels two "
     "steps along the network."),
    ("📋", "Decide the order", "Turn everything into a priority list",
     "The schedule builder needs one clear queue: which task to place first, second, third…",
     "It starts from a well-known textbook rule (“tasks with the latest possible start time can wait”) and then "
     "applies corrections it learned — moving some tasks up or down the queue."),
    ("🗓️", "Build the schedule", "Turn the queue into a real timetable",
     "A priority list is only useful if it gives a short project. This step checks that.",
     "The schedule builder takes tasks in queue order and places each at the earliest time its predecessors are "
     "finished and enough workers/machines are free. We then compare the resulting project length with what the "
     "classic textbook rules would give."),
]

FEATURE_PLAIN = {
    "earliest start": "How early it can start", "latest start": "How late it may start",
    "latest finish": "How late it may finish", "slack": "Spare time (slack)", "duration": "How long it takes",
    "direct successors": "Tasks waiting directly on it", "all successors": "All tasks depending on it",
    "rank positional weight": "Work it unlocks next", "resource load": "Workers/machines it needs",
    "predecessors": "Tasks it waits for",
}

# ── The win tiles above the iteration table ─────────────────────────────────────────
TILES = ("**How to read the tiles:** one tile per algorithm. *Benchmark wins* = how many of the 326 unseen "
         "projects (260 benchmark + 66 use cases) it won in our full evaluation. *Led this run* = at how many "
         "evaluations of this run it held the best (or joint-best) makespan. The highlighted tile is the "
         "winner of this run (see the Results tab for why).")


def explain_column(name: str) -> str:
    """Look up a column; dynamic names like 'TS best' or 'Crews (cap 8)' are matched by pattern."""
    if name in COLUMNS:
        return COLUMNS[name]
    if name.endswith(" best") or name.endswith(" tried"):
        algo, kind = name.rsplit(" ", 1)
        return f"{ALGORITHMS.get(algo, (algo,))[0]}: {COLUMNS[kind]}"
    if "(cap" in name:
        return f"Units of this resource the activity needs in every period it runs; 'cap' = units available."
    return ""

for _raw, _plain in FEATURE_PLAIN.items():          # plain-language feature names reuse the detailed text
    COLUMNS[_plain] = COLUMNS[_raw]
