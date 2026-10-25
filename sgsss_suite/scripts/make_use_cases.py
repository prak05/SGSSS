#!/usr/bin/env python3
"""Build the use-case projects.

data/use_cases_small/  8 hand-written projects (10-18 tasks) in all three formats. They are easy to read
                       but too small to separate the algorithms (all six usually reach the same makespan).
data/use_cases/        66 generated realistic projects: 12 industries, 30-60 named tasks in phases,
                       tight resources (PSPLIB resource strength 0.25-0.40), fixed seed, rotating
                       .json / .sm / .rcp. Parameters were fixed before any results were seen, and
                       scripts/evaluate_use_cases.py reports the result of every project (no filtering).

A dummy Start and Finish are added automatically, as PSPLIB does.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import Project  # noqa: E402
from src.parser import parse, write  # noqa: E402

USE_CASES = {
    "house_construction": (
        {"Carpenters": 4, "Electricians": 2, "Plumbers": 2, "Crane": 1},
        [("Site survey", 2, [], [1, 0, 0, 0]), ("Permits", 5, [], [0, 0, 0, 0]),
         ("Excavation", 3, ["Site survey", "Permits"], [1, 0, 0, 1]),
         ("Foundation", 5, ["Excavation"], [2, 0, 1, 1]), ("Framing", 8, ["Foundation"], [4, 0, 0, 1]),
         ("Roof", 6, ["Framing"], [3, 0, 0, 1]), ("Windows & doors", 3, ["Framing"], [2, 0, 0, 0]),
         ("Rough plumbing", 4, ["Framing"], [0, 0, 2, 0]), ("Rough wiring", 4, ["Framing"], [0, 2, 0, 0]),
         ("Insulation", 3, ["Rough plumbing", "Rough wiring", "Roof"], [2, 0, 0, 0]),
         ("Drywall", 5, ["Insulation"], [3, 0, 0, 0]), ("Kitchen fit-out", 4, ["Drywall"], [2, 1, 1, 0]),
         ("Bathroom fit-out", 4, ["Drywall"], [1, 0, 2, 0]), ("Final electrics", 2, ["Drywall"], [0, 2, 0, 0]),
         ("Flooring", 3, ["Kitchen fit-out", "Bathroom fit-out"], [3, 0, 0, 0]),
         ("Painting", 4, ["Flooring", "Final electrics"], [2, 0, 0, 0]),
         ("Landscaping", 3, ["Roof"], [1, 0, 0, 1]),
         ("Inspection", 1, ["Painting", "Landscaping", "Windows & doors"], [0, 1, 1, 0])]),
    "software_release": (
        {"Backend devs": 4, "Frontend devs": 3, "QA engineers": 2, "DevOps": 1},
        [("Requirements", 3, [], [1, 1, 0, 0]), ("Architecture", 4, ["Requirements"], [2, 1, 0, 1]),
         ("Database schema", 3, ["Architecture"], [2, 0, 0, 0]), ("Auth service", 5, ["Architecture"], [2, 0, 0, 0]),
         ("Payments API", 7, ["Database schema"], [3, 0, 0, 0]), ("Search API", 6, ["Database schema"], [2, 0, 0, 0]),
         ("UI design system", 4, ["Requirements"], [0, 2, 0, 0]),
         ("Checkout pages", 6, ["UI design system", "Payments API"], [0, 3, 0, 0]),
         ("Search pages", 4, ["UI design system", "Search API"], [0, 2, 0, 0]),
         ("Account pages", 4, ["UI design system", "Auth service"], [0, 2, 0, 0]),
         ("CI pipeline", 3, ["Architecture"], [0, 0, 0, 1]),
         ("Staging environment", 2, ["CI pipeline"], [0, 0, 0, 1]),
         ("Integration tests", 5, ["Checkout pages", "Search pages", "Account pages", "Staging environment"], [1, 1, 2, 0]),
         ("Load testing", 3, ["Integration tests"], [1, 0, 1, 1]),
         ("Security review", 3, ["Integration tests"], [1, 0, 1, 0]),
         ("Bug fixing", 4, ["Load testing", "Security review"], [3, 2, 1, 0]),
         ("Documentation", 3, ["Integration tests"], [1, 1, 0, 0]),
         ("Release", 1, ["Bug fixing", "Documentation"], [1, 1, 1, 1])]),
    "hospital_wing_renovation": (
        {"Builders": 6, "Electricians": 3, "Medical gas fitters": 2, "Inspectors": 1},
        [("Patient relocation", 3, [], [0, 0, 0, 0]), ("Asbestos survey", 2, [], [1, 0, 0, 1]),
         ("Strip-out", 5, ["Patient relocation", "Asbestos survey"], [5, 1, 0, 0]),
         ("Structural repairs", 6, ["Strip-out"], [4, 0, 0, 0]), ("New partitions", 5, ["Structural repairs"], [4, 0, 0, 0]),
         ("Ventilation ducts", 6, ["Structural repairs"], [2, 1, 0, 0]),
         ("Electrical first fix", 5, ["New partitions"], [0, 3, 0, 0]),
         ("Medical gas pipework", 6, ["New partitions"], [1, 0, 2, 0]),
         ("Nurse call system", 3, ["Electrical first fix"], [0, 2, 0, 0]),
         ("Ceilings", 4, ["Ventilation ducts", "Electrical first fix", "Medical gas pipework"], [3, 0, 0, 0]),
         ("Infection-control flooring", 4, ["Ceilings"], [3, 0, 0, 0]),
         ("Electrical second fix", 3, ["Ceilings", "Nurse call system"], [0, 3, 0, 0]),
         ("Gas outlets & testing", 3, ["Ceilings"], [0, 0, 2, 1]),
         ("Equipment install", 4, ["Infection-control flooring", "Electrical second fix"], [2, 1, 0, 0]),
         ("Deep clean", 2, ["Equipment install", "Gas outlets & testing"], [2, 0, 0, 0]),
         ("Commissioning", 3, ["Deep clean"], [0, 1, 1, 1]), ("Handover", 1, ["Commissioning"], [0, 0, 0, 1])]),
    "wedding_planning": (
        {"Planners": 2, "Decorators": 3, "Caterers": 4},
        [("Book venue", 2, [], [1, 0, 0]), ("Guest list", 2, [], [1, 0, 0]),
         ("Invitations", 3, ["Guest list"], [1, 1, 0]), ("Menu tasting", 2, ["Book venue"], [1, 0, 2]),
         ("Theme & decor plan", 3, ["Book venue"], [1, 2, 0]), ("Order flowers", 1, ["Theme & decor plan"], [1, 1, 0]),
         ("Seating plan", 2, ["Invitations"], [1, 0, 0]), ("Build stage", 3, ["Theme & decor plan"], [0, 3, 0]),
         ("Lighting & sound", 2, ["Build stage"], [0, 2, 0]), ("Prep food", 3, ["Menu tasting"], [0, 0, 4]),
         ("Decorate hall", 2, ["Build stage", "Order flowers", "Seating plan"], [1, 3, 0]),
         ("Set tables", 1, ["Decorate hall"], [0, 1, 2]), ("Rehearsal", 1, ["Lighting & sound", "Decorate hall"], [2, 0, 0]),
         ("Serve dinner", 1, ["Prep food", "Set tables"], [1, 0, 4])]),
    "factory_line_install": (
        {"Millwrights": 4, "Electricians": 3, "Controls engineers": 2, "Forklift": 1},
        [("Floor marking", 1, [], [1, 0, 0, 0]), ("Remove old line", 4, [], [3, 1, 0, 1]),
         ("Anchor bolts", 2, ["Floor marking", "Remove old line"], [2, 0, 0, 0]),
         ("Conveyor sections", 5, ["Anchor bolts"], [4, 0, 0, 1]), ("Robot cells", 6, ["Anchor bolts"], [3, 0, 0, 1]),
         ("Packing machine", 4, ["Anchor bolts"], [2, 0, 0, 1]), ("Power distribution", 4, ["Remove old line"], [0, 3, 0, 0]),
         ("Cable trays", 3, ["Conveyor sections", "Power distribution"], [1, 2, 0, 0]),
         ("Safety fencing", 3, ["Robot cells"], [2, 0, 0, 0]),
         ("Wiring & terminations", 5, ["Cable trays", "Robot cells", "Packing machine"], [0, 3, 0, 0]),
         ("PLC programming", 6, ["Power distribution"], [0, 0, 2, 0]),
         ("Robot programming", 5, ["Wiring & terminations", "PLC programming"], [0, 0, 2, 0]),
         ("Safety validation", 2, ["Safety fencing", "Robot programming"], [0, 1, 1, 0]),
         ("Dry run", 2, ["Safety validation"], [1, 1, 2, 0]),
         ("Production trial", 3, ["Dry run"], [2, 1, 1, 1])]),
    "film_production": (
        {"Camera crews": 2, "Actors": 5, "Editors": 2, "Studio": 1},
        [("Script lock", 4, [], [0, 0, 0, 0]), ("Casting", 5, ["Script lock"], [0, 1, 0, 0]),
         ("Location scouting", 3, ["Script lock"], [1, 0, 0, 0]), ("Set building", 6, ["Script lock"], [0, 0, 0, 1]),
         ("Rehearsals", 3, ["Casting"], [0, 4, 0, 0]),
         ("Studio scenes", 8, ["Set building", "Rehearsals"], [2, 5, 0, 1]),
         ("Location scenes", 6, ["Location scouting", "Rehearsals"], [1, 3, 0, 0]),
         ("Pickup shots", 2, ["Studio scenes", "Location scenes"], [1, 2, 0, 0]),
         ("Rough cut", 6, ["Studio scenes", "Location scenes"], [0, 0, 2, 0]),
         ("Visual effects", 8, ["Rough cut"], [0, 0, 1, 0]), ("Score recording", 4, ["Rough cut"], [0, 0, 0, 1]),
         ("Sound design", 5, ["Rough cut", "Pickup shots"], [0, 0, 1, 0]),
         ("ADR dialogue", 2, ["Rough cut"], [0, 2, 0, 1]),
         ("Final mix", 3, ["Score recording", "Sound design", "ADR dialogue"], [0, 0, 1, 1]),
         ("Colour grade", 3, ["Visual effects"], [0, 0, 1, 0]),
         ("Delivery", 1, ["Final mix", "Colour grade"], [0, 0, 1, 0])]),
}


def build(name, resources, tasks) -> Project:
    names = ["Start"] + [t[0] for t in tasks] + ["Finish"]
    idx = {nm: i for i, nm in enumerate(names)}
    has_succ = {p for t in tasks for p in t[2]}
    preds = [[]] + [[idx[q] for q in t[2]] or [0] for t in tasks]
    preds.append([idx[t[0]] for t in tasks if t[0] not in has_succ])
    k = len(resources)
    return Project(name, [0] + [t[1] for t in tasks] + [0],
                   [[0] * k] + [list(t[3]) for t in tasks] + [[0] * k],
                   list(resources.values()), preds, names, list(resources))


DOMAINS = {
    "construction": (["Crews", "Cranes", "Electricians", "Plumbers"],
                     [("Site", ["survey", "clearance", "fencing", "utilities"]),
                      ("Groundworks", ["excavation", "piling", "drainage", "foundations"]),
                      ("Structure", ["steel frame", "concrete slab", "columns", "roof deck"]),
                      ("Envelope", ["cladding", "glazing", "roofing", "insulation"]),
                      ("Services", ["wiring", "plumbing", "HVAC ducts", "fire system"]),
                      ("Fit-out", ["partitions", "ceilings", "flooring", "painting"]),
                      ("Handover", ["snagging", "testing", "inspection", "cleaning"])]),
    "software": (["Backend devs", "Frontend devs", "QA engineers", "DevOps"],
                 [("Discovery", ["requirements", "user research", "architecture", "estimates"]),
                  ("Platform", ["database", "auth service", "API gateway", "message queue"]),
                  ("Features", ["payments", "search", "notifications", "reporting"]),
                  ("Frontend", ["design system", "checkout UI", "dashboard UI", "mobile views"]),
                  ("Quality", ["integration tests", "load tests", "security review", "bug fixing"]),
                  ("Release", ["staging", "documentation", "migration", "go-live"])]),
    "manufacturing": (["Millwrights", "Electricians", "Controls engineers", "Forklifts"],
                      [("Preparation", ["layout", "removal", "floor works", "anchors"]),
                       ("Mechanical", ["conveyors", "robot cells", "press line", "packers"]),
                       ("Electrical", ["power", "cable trays", "terminations", "lighting"]),
                       ("Controls", ["PLC code", "robot code", "HMI screens", "safety PLC"]),
                       ("Commissioning", ["dry run", "safety test", "trial run", "ramp-up"])]),
    "shipbuilding": (["Welders", "Fitters", "Painters", "Dock crane"],
                     [("Blocks", ["keel block", "bow block", "stern block", "mid block"]),
                      ("Erection", ["block joining", "deck plating", "bulkheads", "superstructure"]),
                      ("Outfitting", ["engine install", "piping", "cabling", "accommodation"]),
                      ("Coating", ["blasting", "primer", "hull paint", "deck paint"]),
                      ("Trials", ["harbour trial", "sea trial", "class survey", "delivery"])]),
    "data_centre": (["Electricians", "Mechanical techs", "Network engineers", "Lifts"],
                    [("Shell", ["slab", "raised floor", "walls", "roof"]),
                     ("Power", ["substation", "UPS", "generators", "busways"]),
                     ("Cooling", ["chillers", "CRAH units", "pipework", "containment"]),
                     ("IT", ["racks", "cabling", "core switches", "servers"]),
                     ("Testing", ["load bank", "failover test", "integrated test", "handover"])]),
    "hospital": (["Builders", "Electricians", "Gas fitters", "Inspectors"],
                 [("Decant", ["relocation", "surveys", "isolation", "strip-out"]),
                  ("Build", ["structure", "partitions", "ventilation", "ceilings"]),
                  ("Services", ["power", "nurse call", "medical gas", "data"]),
                  ("Finish", ["hygienic flooring", "doors", "equipment", "deep clean"]),
                  ("Validation", ["gas testing", "air testing", "commissioning", "handover"])]),
    "aerospace": (["Assembly crews", "Avionics engineers", "Tooling rigs", "Inspectors"],
                  [("Structure", ["fuselage join", "wing mate", "empennage", "landing gear"]),
                   ("Systems", ["hydraulics", "fuel system", "flight controls", "wiring looms"]),
                   ("Avionics", ["cockpit", "radios", "navigation", "software load"]),
                   ("Interior", ["insulation", "panels", "seats", "galleys"]),
                   ("Certification", ["ground tests", "engine runs", "flight test", "paperwork"])]),
    "rail": (["Track gangs", "Signal engineers", "Civil crews", "Tamper"],
             [("Possession", ["isolation", "survey", "access", "protection"]),
              ("Civils", ["drainage", "embankment", "bridge repair", "platforms"]),
              ("Track", ["ballast", "sleepers", "rails", "tamping"]),
              ("Signalling", ["cabling", "signals", "interlocking", "testing"]),
              ("Reopening", ["inspection", "speed trial", "handback", "monitoring"])]),
    "film": (["Camera crews", "Actors", "Editors", "Studio"],
             [("Pre-production", ["script", "casting", "locations", "sets"]),
              ("Shoot", ["studio scenes", "location scenes", "stunts", "pickups"]),
              ("Edit", ["assembly", "rough cut", "fine cut", "picture lock"]),
              ("Post", ["VFX", "sound design", "score", "colour grade"]),
              ("Delivery", ["mix", "subtitles", "masters", "marketing cut"])]),
    "oil_gas": (["Pipefitters", "Welders", "Instrument techs", "Heavy lift"],
                [("Shutdown", ["isolation", "purging", "scaffolding", "permits"]),
                 ("Inspection", ["vessels", "exchangers", "piping", "valves"]),
                 ("Repair", ["vessel repair", "pipe spools", "valve overhaul", "pump overhaul"]),
                 ("Reinstate", ["hydrotest", "insulation", "instruments", "boxing up"]),
                 ("Start-up", ["leak test", "purge", "start-up", "monitoring"])]),
    "event": (["Planners", "Riggers", "Caterers", "AV technicians"],
              [("Plan", ["venue", "budget", "guest list", "suppliers"]),
               ("Build", ["staging", "rigging", "power", "seating"]),
               ("Production", ["lighting", "sound", "video", "decor"]),
               ("Hospitality", ["menu", "kitchen setup", "bars", "service"]),
               ("Show", ["rehearsal", "doors", "show", "load-out"])]),
    "pharma": (["Process engineers", "QA staff", "Validation team", "Cleanroom"],
               [("Design", ["process design", "risk assessment", "URS", "layout"]),
                ("Install", ["reactors", "filling line", "utilities", "HVAC"]),
                ("Qualify", ["IQ", "OQ", "PQ", "cleaning validation"]),
                ("Batches", ["engineering batch", "validation batch 1", "validation batch 2", "stability"]),
                ("Approval", ["dossier", "audit", "release", "transfer"])]),
}


def generate(index: int) -> Project:
    """One realistic project: named tasks in phases, arcs into the next phase, tight resources."""
    rng = random.Random(2026 + index)
    domain = list(DOMAINS)[index % len(DOMAINS)]
    resources, phases = DOMAINS[domain]
    n = rng.randint(30, 60)
    per_phase = [n // len(phases) + (1 if i < n % len(phases) else 0) for i in range(len(phases))]
    tasks, by_phase = [], []
    for (phase, words), count in zip(phases, per_phase):
        names = [f"{phase}: {words[j % len(words)]}" + (f" {j // len(words) + 1}" if j >= len(words) else "")
                 for j in range(count)]
        by_phase.append(names)
        for name in names:
            prev = by_phase[-2] if len(by_phase) > 1 else []
            preds = rng.sample(prev, min(len(prev), rng.randint(1, 3)))
            used = rng.sample(range(len(resources)), rng.randint(1, 3))
            demand = [rng.randint(1, 6) if k in used else 0 for k in range(len(resources))]
            tasks.append((name, rng.randint(1, 10), preds, demand))
    loose = build(domain, {r: 10 ** 6 for r in resources}, tasks)
    strength = rng.uniform(0.25, 0.40)                 # PSPLIB resource strength: 0 = tightest
    caps = {}
    for k, r in enumerate(resources):
        r_min = max(t[3][k] for t in tasks)
        r_max = max(sum(loose.demands[i][k] for i in range(loose.n)
                        if loose.est[i] <= t < loose.est[i] + loose.durations[i]) for t in range(loose.cpm_bound))
        caps[r] = max(r_min, round(r_min + strength * (r_max - r_min)))
    return build(f"{domain}_{index + 1:02d}", caps, tasks)


def write_all(p: Project, folder: Path, exts) -> None:
    for ext in exts:
        write(p, folder / f"{p.name}{ext}")
        assert parse(folder / f"{p.name}{ext}").cpm_bound == p.cpm_bound, p.name


def main(count: int = 66):
    small = ROOT / "data" / "use_cases_small"
    small.mkdir(parents=True, exist_ok=True)
    for name, spec in USE_CASES.items():
        write_all(build(name, *spec), small, (".json", ".sm", ".rcp"))
    out = ROOT / "data" / "use_cases"
    out.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        p = generate(index)
        write_all(p, out, ([".json", ".sm", ".rcp"][index % 3],))
        print(f"{p.name}: {p.real_tasks} tasks, capacities {p.capacities}, lower bound {p.lower_bound}")


if __name__ == "__main__":
    main()
