# RCPSP Execution Flow — Code Location Guide

This file is written so that anyone can answer **“Where in the code does this action happen?”** by following the action to its exact source location.

## Quick map

1. **Open the website** → [ui.py](ui.py#L61-L66)
2. **Upload or select a project** → [ui.py](ui.py#L311-L346)
3. **Parse the project file** → [ui.py](ui.py#L373-L380) and [backend.py](backend.py#L595-L611)
4. **Create the project data model** → [backend.py](backend.py#L73-L159)
5. **Calculate N, NC, RF, RS, and CPM** → [backend.py](backend.py#L392-L506)
6. **Choose a solver** → [ui.py](ui.py#L467-L489)
7. **Run the selected solver** → [ui.py](ui.py#L280-L286)
8. **Build the SSGS schedule** → [backend.py](backend.py#L210-L315)
9. **Validate precedence and resources** → [backend.py](backend.py#L550-L583)
10. **Show the Gantt chart** → [ui.py](ui.py#L518)
11. **Show resource profiles** → [ui.py](ui.py#L530-L532)
12. **Compare solvers** → [ui.py](ui.py#L537-L582)
13. **Generate the PDF** → [ui.py](ui.py#L609-L630)

---

## 1. Website startup

**Action:** Configure the Streamlit page.

- Page title and layout: [ui.py](ui.py#L61-L66)
- Importing the backend functions: [ui.py](ui.py#L34-L55)
- Solver names exposed by the UI: [ui.py](ui.py#L57-L58)

**What happens:** Streamlit creates the page, imports the backend engine, and defines the five solver names available in the interface.

## 2. Project input

**Action:** User chooses a demo file or uploads a project file.

- The upload control is created here: [ui.py](ui.py#L311-L313)
- Demo buttons select a file path: [ui.py](ui.py#L326-L334)
- Uploaded bytes are saved to a temporary file: [ui.py](ui.py#L335-L346)
- The selected path and project name are stored in session state: [ui.py](ui.py#L315-L324)

The supported formats are `.sm`, `.rcp`, and `.json`.

## 3. Project data models

**Action:** Create the in-memory project structure.

- Resource model: [backend.py](backend.py#L73-L83)
- Activity model: [backend.py](backend.py#L85-L102)
- Project model: [backend.py](backend.py#L104-L119)
- Schedule entry model: [backend.py](backend.py#L123-L138)
- Solver result model: [backend.py](backend.py#L142-L159)

The model stores:

- Activity duration
- Predecessor and successor links
- Resource requirements
- Resource capacity
- Schedule start and finish times
- Solver result and convergence history

## 4. Parse the uploaded file

**Action:** Convert the selected file into a `Project` object.

- The UI calls the parser: [ui.py](ui.py#L373-L380)
- The parser selects the format: [backend.py](backend.py#L595-L611)
- `.sm` parsing: [backend.py](backend.py#L613-L689)
- `.rcp` parsing: [backend.py](backend.py#L810-L836)
- `.json` parsing: [backend.py](backend.py#L800-L808)

The parser builds `project.activities` and `project.resources`. It also converts predecessor/successor links into the project graph.

## 5. Validate the project graph

**Action:** Check that the project contains no cycles.

- Validation function: [backend.py](backend.py#L369-L389)
- The UI calls feature extraction after parsing: [ui.py](ui.py#L386-L389)

The function uses Kahn's topological sorting algorithm. If the graph contains a cycle, it raises an error.

## 6. Calculate project characteristics

**Action:** Compute N, NC, RF, RS, and CPM.

- Main feature extraction function: [backend.py](backend.py#L392-L506)
- The values are displayed in the UI: [ui.py](ui.py#L436-L459)

| Metric | Where it is calculated | Meaning |
|---|---|---|
| N | [backend.py](backend.py#L398-L399) | Number of real activities |
| NC | [backend.py](backend.py#L402-L404) | Network complexity |
| CPM | [backend.py](backend.py#L405-L424) | Critical-path duration |
| RS | [backend.py](backend.py#L438-L465) | Resource strength |
| RF | [backend.py](backend.py#L466-L470) | Resource factor |

The CPM calculation uses:

- Forward pass: [backend.py](backend.py#L405-L413)
- Critical-path duration: [backend.py](backend.py#L414-L415)
- Backward pass: [backend.py](backend.py#L417-L424)

## 7. Select and run a solver

**Action:** User selects one solver or all five solvers.

- Solver selection controls: [ui.py](ui.py#L467-L476)
- Running selected or all solvers: [ui.py](ui.py#L478-L489)
- Solver dispatch function: [ui.py](ui.py#L280-L286)

The dispatch function finds the solver in `SOLVER_REGISTRY` and calls it with the project, evaluation budget, and random seed.

The solver implementations are:

- GA: [backend.py](backend.py#L839-L913)
- SA: [backend.py](backend.py#L915-L969)
- BCO: [backend.py](backend.py#L971-L1077)
- TS: [backend.py](backend.py#L1079-L1166)
- PRH: [backend.py](backend.py#L1168-L1299)

Each solver returns an `AlgorithmResult` containing:

- Best priority vector
- Schedule
- Makespan
- Runtime
- Evaluation count
- Convergence history

## 8. Build the SSGS schedule

**Action:** Convert priority vectors into activity start and finish times.

- Main SSGS method: [backend.py](backend.py#L210-L315)
- Priority vector mapping: [backend.py](backend.py#L218-L219)
- Resource timeline creation: [backend.py](backend.py#L223-L227)
- Activity selection: [backend.py](backend.py#L239-L252)
- Earliest feasible time calculation: [backend.py](backend.py#L267-L290)
- Finish time calculation: [backend.py](backend.py#L293-L294)
- Resource allocation: [backend.py](backend.py#L297-L301)

The function works in this order:

1. Select an activity that is ready according to predecessors.
2. Choose the activity with the highest priority.
3. Calculate the earliest time satisfying precedence.
4. Check resource capacity for every time unit.
5. Record the activity's start, finish, duration, and resource usage.
6. Update the resource timeline and predecessor completion data.

## 9. Validate the schedule

**Action:** Confirm that the result satisfies all constraints.

- Schedule evaluation: [backend.py](backend.py#L550-L553)
- Precedence validation: [backend.py](backend.py#L556-L571)
- Resource validation: [backend.py](backend.py#L573-L583)

The validation checks:

1. Every predecessor finishes before its successor starts.
2. Total usage of each resource never exceeds that resource's capacity.

## 10. Show the Gantt chart

**Action:** Display activity execution over time.

- Gantt chart creation: [ui.py](ui.py#L110-L195)
- Gantt chart display in the website: [ui.py](ui.py#L518)

The chart uses each activity's `start_time`, `finish_time`, and duration. It displays activities as horizontal bars along a time axis.

## 11. Show resource profiles

**Action:** Compare resource demand with available capacity.

- Resource chart creation: [ui.py](ui.py#L197-L249)
- Resource chart display: [ui.py](ui.py#L530-L532)

For every resource, the code calculates usage over time and draws a horizontal line at the resource capacity.

## 12. Compare solver results

**Action:** Compare makespan, runtime, evaluations, and feasibility.

- Comparison table: [ui.py](ui.py#L537-L563)
- Convergence chart: [ui.py](ui.py#L565-L582)

The comparison table uses the results stored in `st.session_state["results"]`.

The convergence chart plots:

- X-axis: SSGS evaluation count
- Y-axis: best makespan found so far
- One line for each solver

## 13. Generate the PDF report

**Action:** Create and download a printable report.

- Generate PDF button and call: [ui.py](ui.py#L602-L618)
- Download button: [ui.py](ui.py#L620-L630)
- PDF generator implementation: [backend.py](backend.py#L1388-L1435)

The UI passes the project, extracted features, best schedule, all solver results, and project name into the PDF generator.

---

## Short “point me to the code” answers

### If someone asks: “Where does the project file enter the system?”

Look at [ui.py](ui.py#L311-L313). The uploaded file is then saved into `st.session_state["uploaded_path"]` at [ui.py](ui.py#L335-L346).

### If someone asks: “Where does the parser run?”

Look at [ui.py](ui.py#L373-L380). The actual parsing happens in [backend.py](backend.py#L595-L611).

### If someone asks: “Where are N, NC, RF, RS, and CPM calculated?”

Look at [backend.py](backend.py#L392-L506).

### If someone asks: “Where does the solver run?”

Look at [ui.py](ui.py#L478-L489), then [ui.py](ui.py#L280-L286). The actual solver implementation is in the files linked above under the solver implementations.

### If someone asks: “Where is the schedule created?”

Look at [backend.py](backend.py#L210-L315).

### If someone asks: “Where is resource feasibility checked?”

Look at [backend.py](backend.py#L267-L301) while scheduling and [backend.py](backend.py#L556-L583) during validation.

### If someone asks: “Where are the charts shown?”

- Gantt chart: [ui.py](ui.py#L518)
- Resource charts: [ui.py](ui.py#L530-L532)
- Solver comparison: [ui.py](ui.py#L537-L582)

### If someone asks: “Where is the PDF generated?”

Look at [ui.py](ui.py#L609-L630) and the generator in [backend.py](backend.py#L1388-L1435).
