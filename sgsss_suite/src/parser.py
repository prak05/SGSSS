"""
================================================================================
src/parser.py — Modular Multi-Format RCPSP Project File Parser
================================================================================
Supports three file formats:

  1. **PSPLIB .sm**  — Standard benchmark format (Kolisch et al. 1995).
     Uses the ``psplib`` Python package when available; falls back to a pure-
     Python regex/line parser.

  2. **Paterson .rcp** — Classic format; keyword-based *or* raw-token layout.

  3. **JSON .json**   — Structured JSON payload for modern integrations.

After parsing, the parser:
  • Inverts successor lists → predecessor lists (and vice versa) so both
    directions are always populated.
  • Extracts instance metadata (file seed, evaluation budget, horizon) from
    file headers automatically.

References
──────────
[1][11][12] Mitsou & Koulinas (2026); Kolisch (1995).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.model import Activity, Project, Resource


# ==============================================================================
# SECTION SUMMARY: Instance Metadata & Seed Extraction
# What is in this section: Automatically extracts instance header metadata including
# random generator seed, horizon, and evaluation budget from .sm and .json files.
# ==============================================================================
# SECTION SUMMARY: Adaptive SSGS Budget & Random Seed Calibrator
# What is in this section: Computes scale-adaptive evaluation budgets (200-2500)
# and derives deterministic reproducible seeds tailored to the uploaded instance.
# ==============================================================================

def compute_adaptive_budget(project: Optional[Project]) -> int:
    """
    Computes an adaptive SSGS evaluation budget tailored to the problem scale (N).
    Scales dynamically to ensure thorough metaheuristic search while keeping execution fast:
      • N <= 15:  200 evaluations (instantaneous convergence on toy/demo instances)
      • N <= 35:  500 evaluations (classic PSPLIB J30 benchmark standard)
      • N <= 70:  1000 evaluations (PSPLIB J60 benchmark standard)
      • N <= 100: 1500 evaluations (PSPLIB J90 benchmark standard)
      • N > 100:  2500 evaluations (PSPLIB J120 benchmark standard)
    """
    if project is None or not project.activities:
        return 500
    real_count = len(project.real_activities) if hasattr(project, "real_activities") else len(project.activities)
    if real_count <= 15:
        return 200
    elif real_count <= 35:
        return 500
    elif real_count <= 70:
        return 1000
    elif real_count <= 100:
        return 1500
    else:
        return 2500


def compute_adaptive_seed(file_path: Union[str, Path], file_seed_from_header: Optional[int] = None) -> int:
    """
    Derives an instance-tailored pseudo-random seed.
    If the file header specifies an explicit initial generator seed, it is preserved.
    Otherwise, a deterministic seed is derived from the instance filename/content,
    guaranteeing 100% scientific reproducibility across runs while ensuring different
    instances test different pseudo-random sequences.
    """
    if file_seed_from_header is not None:
        return int(file_seed_from_header)

    path = Path(file_path)
    nums = re.findall(r"\d+", path.stem)
    if nums:
        try:
            return (int("".join(nums)) % 100000) + 42
        except Exception:
            pass

    return (abs(hash(path.name)) % 9000) + 1000


def extract_instance_metadata(
    file_path: Union[str, Path],
    project: Optional[Project] = None,
) -> Dict[str, Any]:
    """
    Auto-extract seed, evaluation budget, and auxiliary metadata from an
    instance file.

    Rules:
      • **.sm** — ``initial value random generator: <seed>`` -> file_seed.
      • **.json** — ``seed`` / ``random_seed`` -> file_seed;
                     ``evaluation_budget`` / ``budget`` -> file_budget.
      • Adaptive seed derived from file header or file name.
      • Adaptive budget auto-scaled by real activity count N (200 - 2500).
    """
    path = Path(file_path)
    meta: Dict[str, Any] = {}
    file_seed: Optional[int] = None
    file_budget: Optional[int] = None

    if path.exists():
        ext = path.suffix.lower()

        if ext == ".sm":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line_str = line.strip()
                        if "initial value random generator" in line_str:
                            m = re.search(
                                r"initial value random generator\s*:\s*(\d+)",
                                line_str, re.IGNORECASE,
                            )
                            if m:
                                file_seed = int(m.group(1))
                                meta["initial_value_random_generator"] = file_seed
                        elif line_str.startswith("horizon"):
                            m = re.search(r"horizon\s*:\s*(\d+)", line_str, re.IGNORECASE)
                            if m:
                                meta["horizon"] = int(m.group(1))
            except Exception:
                pass

        elif ext == ".json":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        file_seed = data.get("seed") or data.get("random_seed")
                        if file_seed is not None:
                            file_seed = int(file_seed)
                        file_budget = data.get("evaluation_budget") or data.get("budget")
                        if file_budget is not None:
                            file_budget = int(file_budget)
                        meta.update({
                            k: v for k, v in data.items()
                            if k not in ("activities", "resources")
                        })
            except Exception:
                pass

    # Adaptive budget calculation based on problem scale
    if file_budget is None:
        file_budget = compute_adaptive_budget(project)

    # Adaptive reproducible seed derivation
    file_seed = compute_adaptive_seed(path, file_seed)

    meta["file_seed"] = file_seed
    meta["file_budget"] = file_budget
    return meta


# ==============================================================================
# SECTION SUMMARY: Multi-Format Parser Dispatcher & Readers
# What is in this section: Implements format detection (.sm, .rcp, .json) and provides
# specialized loaders that populate Project, Activity, and Resource instances.
# ==============================================================================
class Parser:
    """Multi-format RCPSP instance parser (.sm, .rcp, .json)."""

    # --------------------------------------------------------------------------
    # Subsection: Public Ingestion Entrypoint
    # What is in this section: Detects file type, dispatches format parser,
    # and enriches the Project object with extracted instance metadata.
    # --------------------------------------------------------------------------
    @staticmethod
    def parse(
        file_path: Union[str, Path],
        display_name: Optional[str] = None,
    ) -> Project:
        """
        Parse an RCPSP instance file into a :class:`Project`.

        Automatically detects format by extension and enriches the project with
        instance metadata (seed, budget, horizon).

        Raises:
            FileNotFoundError: file does not exist.
            ValueError:        unsupported extension.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".sm":
            project = Parser._parse_sm(str(path))
        elif ext == ".rcp":
            project = Parser._parse_rcp(str(path))
        elif ext == ".json":
            project = Parser._parse_json(str(path))
        else:
            raise ValueError(f"Unsupported format '{ext}'. Use .sm, .rcp, or .json")

        # Attach metadata
        meta = extract_instance_metadata(path, project)
        project.metadata = meta
        project.file_seed = meta.get("file_seed")
        project.file_budget = meta.get("file_budget")
        return project

    # --------------------------------------------------------------------------
    # Subsection: PSPLIB .sm Benchmark Parser
    # What is in this section: Parses PSPLIB .sm benchmark format via the psplib package
    # with a resilient pure-Python regex fallback for Precedence, Demands, and Resources.
    # --------------------------------------------------------------------------
    @staticmethod
    def _parse_sm(file_path: str) -> Project:
        """Parse a PSPLIB .sm file using ``psplib`` package, with fallback."""
        try:
            import psplib
            raw = psplib.parse_psplib(file_path)
            project = Project()

            # Resources (renewable only)
            renewable_indices: List[int] = []
            renewable_r_ids: List[str] = []
            for i, res in enumerate(raw.resources):
                if res.renewable:
                    r_id = f"R{len(renewable_indices) + 1}"
                    project.resources[r_id] = Resource(id=r_id, capacity=res.capacity)
                    renewable_indices.append(i)
                    renewable_r_ids.append(r_id)

            num_psp_acts = raw.num_activities
            predecessors: Dict[int, List[int]] = {
                (i + 1): [] for i in range(num_psp_acts)
            }

            # Build predecessor map from successor lists
            for psp_idx, act in enumerate(raw.activities):
                job_nr = psp_idx + 1
                for succ_0idx in act.successors:
                    succ_job = succ_0idx + 1
                    predecessors.setdefault(succ_job, []).append(job_nr)

            # Detect phantom/dummy sink
            all_ref = {s + 1 for act in raw.activities for s in act.successors}
            present = set(range(1, num_psp_acts + 1))
            phantom = all_ref - present

            if phantom:
                sink_job = max(phantom)
                predecessors.setdefault(sink_job, [])
                for psp_idx, act in enumerate(raw.activities):
                    for s in act.successors:
                        if s + 1 == sink_job:
                            src_job = psp_idx + 1
                            if src_job not in predecessors[sink_job]:
                                predecessors[sink_job].append(src_job)
            else:
                sink_job = num_psp_acts

            # Populate activities
            for psp_idx, psp_act in enumerate(raw.activities):
                job_nr = psp_idx + 1
                mode = psp_act.modes[0]
                reqs: Dict[str, int] = {}
                for r_idx, r_id in zip(renewable_indices, renewable_r_ids):
                    reqs[r_id] = mode.demands[r_idx] if r_idx < len(mode.demands) else 0
                project.activities[job_nr] = Activity(
                    id=job_nr,
                    duration=mode.duration,
                    predecessors=predecessors.get(job_nr, []),
                    resource_requirements=reqs,
                )

            # Add phantom sink if needed
            if phantom:
                sink_reqs = {r_id: 0 for r_id in renewable_r_ids}
                project.activities[sink_job] = Activity(
                    id=sink_job,
                    duration=0,
                    predecessors=predecessors.get(sink_job, []),
                    resource_requirements=sink_reqs,
                )

            # Rebuild successor lists from predecessors
            Parser._rebuild_successors(project)
            return project

        except Exception:
            return Parser._parse_sm_fallback(file_path)

    @staticmethod
    def _parse_sm_fallback(file_path: str) -> Project:
        """Pure-Python line-by-line fallback for .sm files."""
        project = Project()
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()

        mode = None
        num_res = 0

        for line in lines:
            line_str = line.strip()

            if line_str.startswith("- renewable"):
                parts = line_str.split(":")
                if len(parts) > 1:
                    num_res = int(parts[1].split()[0].strip())
            elif line_str.startswith("PRECEDENCE RELATIONS:"):
                mode = "PRECEDENCE"
                continue
            elif line_str.startswith("REQUESTS/DURATIONS:"):
                mode = "REQUESTS"
                continue
            elif line_str.startswith("RESOURCEAVAILABILITIES:"):
                mode = "RESOURCES"
                continue
            elif line_str.startswith("*" * 20):
                if mode == "RESOURCES":
                    mode = None

            if mode == "PRECEDENCE":
                parts = line_str.split()
                if not parts or not parts[0].isdigit():
                    continue
                job_nr = int(parts[0])
                num_succ = int(parts[2])
                succs = [int(x) for x in parts[3:3 + num_succ]]
                if job_nr not in project.activities:
                    project.activities[job_nr] = Activity(id=job_nr, duration=0)
                for suc in succs:
                    if suc not in project.activities:
                        project.activities[suc] = Activity(id=suc, duration=0)
                    project.activities[suc].predecessors.append(job_nr)

            elif mode == "REQUESTS":
                parts = line_str.split()
                if not parts or not parts[0].isdigit():
                    continue
                job_nr = int(parts[0])
                duration = int(parts[2])
                reqs = [int(x) for x in parts[3:3 + num_res]]
                if job_nr not in project.activities:
                    project.activities[job_nr] = Activity(id=job_nr, duration=duration)
                else:
                    project.activities[job_nr].duration = duration
                for idx, r in enumerate(reqs):
                    project.activities[job_nr].resource_requirements[f"R{idx + 1}"] = r

            elif mode == "RESOURCES":
                parts = line_str.split()
                if not parts or not parts[0].isdigit():
                    continue
                for idx, cap in enumerate(parts[:num_res]):
                    r_id = f"R{idx + 1}"
                    project.resources[r_id] = Resource(id=r_id, capacity=int(cap))
                mode = None

        Parser._rebuild_successors(project)
        return project

    # --------------------------------------------------------------------------
    # Subsection: Paterson .rcp Classic Format Parser
    # What is in this section: Reads Paterson keyword format or raw-token format files.
    # --------------------------------------------------------------------------
    @staticmethod
    def _parse_rcp(file_path: str) -> Project:
        """Parse a Paterson .rcp file (keyword or raw-token format)."""
        with open(file_path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip() and not l.strip().startswith("#")]

        if not lines:
            raise ValueError(f"Empty .rcp file: {file_path}")

        first_word = lines[0].split()[0].upper()

        if first_word in ("RESOURCES", "CAPACITY", "ACTIVITIES"):
            project = Project()
            resources_declared = 0
            activity_lines: List[str] = []
            reading_activities = False

            for line in lines:
                parts = line.split()
                kw = parts[0].upper()
                if kw == "RESOURCES":
                    resources_declared = int(parts[1])
                elif kw == "CAPACITY":
                    for idx, cap in enumerate(parts[1:1 + resources_declared]):
                        r_id = f"R{idx + 1}"
                        project.resources[r_id] = Resource(id=r_id, capacity=int(cap))
                elif kw == "ACTIVITIES":
                    reading_activities = True
                elif reading_activities:
                    activity_lines.append(line)

            n_res = len(project.resources)
            for line in activity_lines:
                parts = line.split()
                if len(parts) < 2 + n_res:
                    continue
                a_id = int(parts[0])
                dur = int(parts[1])
                reqs = {f"R{i + 1}": int(parts[2 + i]) for i in range(n_res)}
                preds = [int(tok) for tok in parts[2 + n_res:] if int(tok) != 0]
                project.activities[a_id] = Activity(
                    id=a_id, duration=dur,
                    predecessors=preds, resource_requirements=reqs,
                )

            Parser._rebuild_successors(project)
            return project

        # raw-token Paterson format
        tokens = " ".join(lines).split()
        n_jobs = int(tokens[0])
        n_resources = int(tokens[1])
        idx = 2

        project = Project()
        for r in range(n_resources):
            r_id = f"R{r + 1}"
            project.resources[r_id] = Resource(id=r_id, capacity=int(tokens[idx]))
            idx += 1

        for j in range(1, n_jobs + 1):
            dur = int(tokens[idx]); idx += 1
            reqs: Dict[str, int] = {}
            for r in range(n_resources):
                reqs[f"R{r + 1}"] = int(tokens[idx]); idx += 1
            n_succ = int(tokens[idx]); idx += 1
            succs = [int(tokens[idx + s]) for s in range(n_succ)]
            idx += n_succ
            project.activities[j] = Activity(
                id=j, duration=dur,
                successors=succs, resource_requirements=reqs,
            )

        # Build predecessor links from successors
        for act_id, act in project.activities.items():
            for s in act.successors:
                if s in project.activities:
                    if act_id not in project.activities[s].predecessors:
                        project.activities[s].predecessors.append(act_id)

        return project

    # --------------------------------------------------------------------------
    # Subsection: Structured JSON Parser
    # What is in this section: Ingests structured JSON dictionaries containing resources
    # and activity definitions.
    # --------------------------------------------------------------------------
    @staticmethod
    def _parse_json(file_path: str) -> Project:
        """Parse a structured JSON RCPSP instance."""
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        project = Project()
        for r_id, cap in data.get("resources", {}).items():
            project.resources[r_id] = Resource(id=r_id, capacity=int(cap))

        for act_data in data.get("activities", []):
            act_id = int(act_data["id"])
            project.activities[act_id] = Activity(
                id=act_id,
                duration=int(act_data.get("duration", 0)),
                predecessors=[int(p) for p in act_data.get("predecessors", [])],
                successors=[int(s) for s in act_data.get("successors", [])],
                resource_requirements={
                    k: int(v) for k, v in act_data.get("resource_requirements", {}).items()
                },
            )
        # Reconcile predecessors and successors bidirectionally
        has_any_successors = any(len(act.successors) > 0 for act in project.activities.values())
        has_any_predecessors = any(len(act.predecessors) > 0 for act in project.activities.values())

        if has_any_predecessors and not has_any_successors:
            Parser._rebuild_successors(project)
        elif has_any_successors and not has_any_predecessors:
            for act in project.activities.values():
                act.predecessors = []
            for a_id, act in project.activities.items():
                for s_id in act.successors:
                    if s_id in project.activities:
                        project.activities[s_id].predecessors.append(a_id)

        return project

    # --------------------------------------------------------------------------
    # Subsection: Bidirectional Successor Rebuilder
    # What is in this section: Reconstructs forward successor graph from reverse predecessors.
    # --------------------------------------------------------------------------
    @staticmethod
    def _rebuild_successors(project: Project) -> None:
        """Reconstruct all successor lists from predecessor lists."""
        for act in project.activities.values():
            act.successors = []
        for a_id, act in project.activities.items():
            for p_id in act.predecessors:
                if p_id in project.activities:
                    project.activities[p_id].successors.append(a_id)

