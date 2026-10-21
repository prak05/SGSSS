"""
Modular parser for RCPSP project files.

Supported formats
-----------------
1. PSPLIB .sm format — parsed via the psplib library.
2. Custom .rcp format — hand-written tokeniser (used for internal tests).
3. (Future) CSV/JSON — placeholder.

Design rules
------------
- The output is always our internal Project/Activity/Resource data model.
- Activity IDs are 1-based (matching PSPLIB job numbers) in the .sm parser,
  and follow the file's numbering in the .rcp parser.
- Predecessor lists (not successor lists) are what the SSGS requires.
- Dummy source  (duration=0, no predecessors) receives the lowest activity ID.
- Dummy sink    (duration=0, no successors)    receives the highest activity ID.
- psplib returns 0-based successor indices and may omit the sink; we handle both.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.model import Activity, Project, Resource


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class Parser:
    @staticmethod
    def parse(file_path: Union[str, Path], display_name: Optional[str] = None) -> Project:
        """
        Detect the file format from the extension and delegate to the
        appropriate sub-parser.

        Parameters
        ----------
        file_path : str or Path
            Path to the project file.
        display_name : str, optional
            Human-friendly file name to display in error messages (e.g. uploaded file name).

        Returns
        -------
        Project
            Populated internal project model.

        Raises
        ------
        ValueError
            If the extension is not recognised.
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {file_path}")

        name_to_show = display_name or path.name

        ext = path.suffix.lower()
        if ext == ".sm":
            return Parser._parse_sm(str(path))
        elif ext == ".rcp":
            return Parser._parse_rcp(str(path))
        elif ext == ".json":
            return Parser._parse_json(str(path), display_name=name_to_show)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}' for '{name_to_show}'. "
                "Supported: .sm (PSPLIB), .rcp (custom), .json (JSON)"
            )


# ---------------------------------------------------------------------------
# PSPLIB .sm parser  (uses the psplib library as a reading backend)
# ---------------------------------------------------------------------------

    @staticmethod
    def _parse_sm(file_path: str) -> Project:
        try:
            import psplib
            return Parser._parse_sm_psplib(file_path)
        except ImportError:
            return Parser._parse_sm_fallback(file_path)

    @staticmethod
    def _parse_sm_psplib(file_path: str) -> Project:
        """
        Parse a PSPLIB-format .sm file and convert it to our Project model.

        PSPLIB activity indices are 0-based inside psplib.
        We map them to 1-based job numbers matching the file's jobnr. column.

        Dummy source = job 1   (always has duration 0 and no predecessors).
        Dummy sink   = job N+2 (synthesized if psplib omits it).
        """
        import psplib
        raw = psplib.parse_psplib(file_path)

        # psplib uses 0-based activity indices; successors are also 0-based indices.
        num_psp_acts = raw.num_activities  # may be missing the sink

        # ---- Resources ----
        project = Project()
        for i, res in enumerate(raw.resources):
            if res.renewable:  # we only handle renewable resources
                r_id = f"R{i + 1}"
                project.resources[r_id] = Resource(id=r_id, capacity=res.capacity)

        if not project.resources:
            raise ValueError(
                f"No renewable resources found in '{file_path}'. "
                "Only renewable resources are supported."
            )

        renewable_indices = [
            i for i, r in enumerate(raw.resources) if r.renewable
        ]
        renewable_r_ids  = [f"R{i + 1}" for i in range(len(renewable_indices))]

        # ---- Build predecessor map from psplib's successor lists ----
        # psplib stores 0-based successor indices.
        # We convert to 1-based job numbers (jobnr = psp_idx + 1).
        predecessors: Dict[int, List[int]] = {
            (i + 1): [] for i in range(num_psp_acts)
        }
        for psp_idx, act in enumerate(raw.activities):
            job_nr = psp_idx + 1
            for succ_0idx in act.successors:
                succ_job = succ_0idx + 1
                if succ_job not in predecessors:
                    # psplib may reference the sink which isn't in the list
                    predecessors[succ_job] = []
                predecessors[succ_job].append(job_nr)

        # Determine the sink job number: it's the one referenced as a successor
        # but not present in the psplib activity list (or the last activity).
        all_referenced_jobs = set()
        for act in raw.activities:
            for s in act.successors:
                all_referenced_jobs.add(s + 1)
        present_jobs = set(range(1, num_psp_acts + 1))
        phantom_jobs = all_referenced_jobs - present_jobs  # typically the sink

        # Synthesize sink entry if missing
        if phantom_jobs:
            sink_job = max(phantom_jobs)
            if sink_job not in predecessors:
                predecessors[sink_job] = []
            # Add predecessors for sink from psplib activities that list it
            for psp_idx, act in enumerate(raw.activities):
                for s in act.successors:
                    if s + 1 == sink_job:
                        src_job = psp_idx + 1
                        if src_job not in predecessors[sink_job]:
                            predecessors[sink_job].append(src_job)
        else:
            # Sink is already present in psplib's list (last entry, dur=0)
            sink_job = num_psp_acts

        # ---- Activities ----
        # Insert real activities from psplib list
        for psp_idx, psp_act in enumerate(raw.activities):
            job_nr = psp_idx + 1
            mode = psp_act.modes[0]  # single-mode RCPSP

            # Resource requirements: only renewable resources, in order
            reqs: Dict[str, int] = {}
            for r_idx, r_id in zip(renewable_indices, renewable_r_ids):
                if r_idx < len(mode.demands):
                    reqs[r_id] = mode.demands[r_idx]
                else:
                    reqs[r_id] = 0

            act = Activity(
                id=job_nr,
                duration=mode.duration,
                predecessors=predecessors.get(job_nr, []),
                resource_requirements=reqs,
            )
            project.activities[job_nr] = act

        # Insert synthesized sink if it was missing
        if phantom_jobs:
            sink_reqs = {r_id: 0 for r_id in renewable_r_ids}
            project.activities[sink_job] = Activity(
                id=sink_job,
                duration=0,
                predecessors=predecessors.get(sink_job, []),
                resource_requirements=sink_reqs,
            )

        # ---- Populate successors ----
        for act in project.activities.values():
            act.successors = []
        for a_id, act in project.activities.items():
            for p_id in act.predecessors:
                if p_id in project.activities:
                    project.activities[p_id].successors.append(a_id)

        # ---- Validation ----
        Parser._validate_project(project, file_path)
        return project

    @staticmethod
    def _parse_sm_fallback(file_path: str) -> Project:
        project = Project()
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        mode = None
        num_res = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith("- renewable"):
                num_res = int(line.split(":")[1].split()[0].strip())
            elif line.startswith("PRECEDENCE RELATIONS:"):
                mode = "PRECEDENCE"
                continue
            elif line.startswith("REQUESTS/DURATIONS:"):
                mode = "REQUESTS"
                continue
            elif line.startswith("RESOURCEAVAILABILITIES:"):
                mode = "RESOURCES"
                continue
            elif line.startswith("************************************************************************"):
                if mode == "RESOURCES":
                    mode = None
                
            if mode == "PRECEDENCE":
                parts = line.split()
                if not parts or not parts[0].isdigit():
                    continue
                job_nr = int(parts[0])
                num_succ = int(parts[2])
                succs = [int(x) for x in parts[3:3+num_succ]]
                if job_nr not in project.activities:
                    project.activities[job_nr] = Activity(id=job_nr, duration=0, predecessors=[], resource_requirements={})
                for suc in succs:
                    if suc not in project.activities:
                        project.activities[suc] = Activity(id=suc, duration=0, predecessors=[], resource_requirements={})
                    project.activities[suc].predecessors.append(job_nr)
                    
            elif mode == "REQUESTS":
                parts = line.split()
                if not parts or not parts[0].isdigit():
                    continue
                job_nr = int(parts[0])
                duration = int(parts[2])
                reqs = [int(x) for x in parts[3:3+num_res]]
                
                if job_nr not in project.activities:
                    project.activities[job_nr] = Activity(id=job_nr, duration=duration, predecessors=[], resource_requirements={})
                else:
                    project.activities[job_nr].duration = duration
                    
                for idx, r in enumerate(reqs):
                    project.activities[job_nr].resource_requirements[f"R{idx+1}"] = r
                    
            elif mode == "RESOURCES":
                parts = line.split()
                if not parts or not parts[0].isdigit():
                    continue
                for idx, cap in enumerate(parts[:num_res]):
                    r_id = f"R{idx+1}"
                    project.resources[r_id] = Resource(id=r_id, capacity=int(cap))
                mode = None
                
        for act in project.activities.values():
            act.successors = []
        for a_id, act in project.activities.items():
            for p_id in act.predecessors:
                if p_id in project.activities:
                    project.activities[p_id].successors.append(a_id)

        Parser._validate_project(project, file_path)
        return project


# ---------------------------------------------------------------------------
# Custom .rcp parser  (supports the hand-crafted internal test format)
# ---------------------------------------------------------------------------

    @staticmethod
    def _parse_rcp(file_path: str) -> Project:
        """
        Parse the custom .rcp format used for internal test instances.

        Format
        ------
        # Comments start with #
        RESOURCES <n>
        CAPACITY  <c1> <c2> …
        ACTIVITIES <m>
        <id> <dur> <r1> <r2> … <pred1> <pred2> …
        """
        project = Project()

        with open(file_path, "r") as fh:
            lines = fh.readlines()

        non_empty = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        if not non_empty:
            raise ValueError(f"Empty RCP file: '{file_path}'")

        # Check if file is standard numeric Patterson benchmark format
        first_parts = non_empty[0].split()
        if len(first_parts) >= 2 and first_parts[0].isdigit() and first_parts[1].isdigit():
            n_act = int(first_parts[0])
            n_res = int(first_parts[1])
            caps = list(map(int, non_empty[1].split()[:n_res]))
            for idx, cap in enumerate(caps):
                r_id = f"R{idx + 1}"
                project.resources[r_id] = Resource(id=r_id, capacity=cap)

            for idx, line in enumerate(non_empty[2:2 + n_act], start=1):
                parts = list(map(int, line.split()))
                dur = parts[0]
                reqs = {f"R{k + 1}": parts[1 + k] for k in range(n_res)}
                n_succ = parts[1 + n_res]
                succs = parts[2 + n_res: 2 + n_res + n_succ]
                project.activities[idx] = Activity(
                    id=idx,
                    duration=dur,
                    resource_requirements=reqs,
                    successors=succs,
                    predecessors=[]
                )

            for a_id, act in project.activities.items():
                for s_id in act.successors:
                    if s_id in project.activities:
                        if a_id not in project.activities[s_id].predecessors:
                            project.activities[s_id].predecessors.append(a_id)

            return project

        # Otherwise parse keyword-based RCP format
        resources_declared = 0
        activity_lines: List[str] = []
        reading_activities = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            kw = parts[0].upper()

            if kw == "RESOURCES":
                resources_declared = int(parts[1])

            elif kw == "CAPACITY":
                if len(parts) - 1 != resources_declared:
                    raise ValueError(
                        f"CAPACITY line has {len(parts) - 1} values but "
                        f"RESOURCES declared {resources_declared}."
                    )
                for idx, cap in enumerate(parts[1:]):
                    r_id = f"R{idx + 1}"
                    project.resources[r_id] = Resource(
                        id=r_id, capacity=int(cap)
                    )

            elif kw == "ACTIVITIES":
                reading_activities = True

            elif reading_activities:
                activity_lines.append(line)

        n_res = len(project.resources)
        if n_res == 0:
            raise ValueError(
                f"No resources found in '{file_path}'. "
                "Check RESOURCES and CAPACITY lines."
            )


        for line in activity_lines:
            parts = line.split()
            # Minimum fields: id  dur  [res...]  [pred | 0]
            min_fields = 2 + n_res + 1
            if len(parts) < min_fields:
                raise ValueError(
                    f"Activity line too short in '{file_path}': '{line}'"
                )

            a_id = int(parts[0])
            dur  = int(parts[1])
            reqs = {}
            r_ids = list(project.resources.keys())
            for i, r_id in enumerate(r_ids):
                reqs[r_id] = int(parts[2 + i])

            # Predecessors start after resource columns
            preds = []
            for tok in parts[2 + n_res:]:
                val = int(tok)
                if val != 0:
                    preds.append(val)

            project.activities[a_id] = Activity(
                id=a_id,
                duration=dur,
                predecessors=preds,
                resource_requirements=reqs,
            )

        # Populate successors
        for act in project.activities.values():
            act.successors = []
        for a_id, act in project.activities.items():
            for p_id in act.predecessors:
                if p_id in project.activities:
                    project.activities[p_id].successors.append(a_id)

        Parser._validate_project(project, file_path)
        return project


# ---------------------------------------------------------------------------
# JSON parser (supports dictionary/object project definitions)
# ---------------------------------------------------------------------------

    @staticmethod
    def _parse_json(file_path: str, display_name: str = "") -> Project:
        """Parse an RCPSP project from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return Parser._parse_json_data(data, file_path=file_path, display_name=display_name)

    @staticmethod
    def _parse_json_data(data: Union[dict, list], file_path: str = "", display_name: str = "") -> Project:
        """
        Parse project structure from a Python dictionary or list (loaded from JSON).
        """
        filename = display_name or (Path(file_path).name if file_path else "JSON")

        # If root is a list, pick the first project instance object in the list
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                raise ValueError(f"JSON list is empty or contains non-object items in '{filename}'.")

        if isinstance(data, dict) and "resources" not in data and "activities" not in data:
            # Check for container keys: "projects", "instances", "data", "benchmark_instances", "items"
            found_inner = False
            for key in ["projects", "instances", "data", "benchmark_instances", "items"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        data = val[0]
                        found_inner = True
                        break
                    elif isinstance(val, dict) and len(val) > 0:
                        first_val = next(iter(val.values()))
                        if isinstance(first_val, dict):
                            data = first_val
                            found_inner = True
                            break
            if not found_inner and isinstance(data, dict) and len(data) > 0:
                # Check if data is a dict of instances: {"j1201_1": {...}, "j1201_2": {...}}
                first_val = next(iter(data.values()))
                if isinstance(first_val, dict) and (
                    "resources" in first_val or "activities" in first_val or "jobs" in first_val or "tasks" in first_val
                ):
                    data = first_val

        project = Project()

        if not isinstance(data, dict):
            raise ValueError(f"JSON root must be an object/dict in '{filename}'.")

        # 1. Parse Resources
        raw_resources = (
            data.get("resources")
            or data.get("resource_capacities")
            or data.get("capacities")
            or data.get("resource_limits")
            or {}
        )
        if isinstance(raw_resources, dict):
            for r_id, res_info in raw_resources.items():
                if isinstance(res_info, dict):
                    cap = int(res_info.get("capacity", 0))
                else:
                    cap = int(res_info)
                project.resources[str(r_id)] = Resource(id=str(r_id), capacity=cap)
        elif isinstance(raw_resources, list):
            for idx, res_info in enumerate(raw_resources):
                if isinstance(res_info, dict):
                    r_id = str(res_info.get("id", f"R{idx+1}"))
                    cap = int(res_info.get("capacity", 0))
                else:
                    r_id = f"R{idx+1}"
                    cap = int(res_info)
                project.resources[r_id] = Resource(id=r_id, capacity=cap)

        if not project.resources:
            data_str = str(data).lower()
            is_metadata = (
                "solutions" in data
                or "instance_set" in data
                or "lb_info" in data_str
                or "lower_bound" in data_str
                or "upper_bound" in data_str
                or "best_known" in data_str
                or "merged" in filename.lower()
                or "solutions" in filename.lower()
                or "bounds" in filename.lower()
                or ("lb" in data_str and "ub" in data_str)
                or ("makespan" in data_str and "activities" not in data and "jobs" not in data)
            )
            if is_metadata:
                raise ValueError(
                    f"The uploaded JSON file ('{filename}') is a PSPLIB benchmark bounds/solutions metadata file "
                    "(containing upper/lower bounds, optimal makespans, or benchmark results), not a project instance definition file. "
                    "Project files require 'activities' (durations, precedence) and 'resources' (capacities). "
                    "Please upload a valid project instance file (.sm, .rcp, or single project definition .json)."
                )
            raise ValueError(
                f"No resources found in JSON file '{filename}'. "
                "Project JSON files must define 'resources' (e.g. {'resources': {'R1': 4, ...}})."
            )

        # 2. Parse Activities
        raw_activities = (
            data.get("activities")
            or data.get("jobs")
            or data.get("tasks")
            or data.get("nodes")
            or []
        )
        activity_list = []
        if isinstance(raw_activities, dict):
            for key, act_data in raw_activities.items():
                if isinstance(act_data, dict):
                    if "id" not in act_data:
                        try:
                            act_data["id"] = int(key)
                        except ValueError:
                            pass
                    activity_list.append(act_data)
        elif isinstance(raw_activities, list):
            activity_list = raw_activities
        else:
            raise ValueError(
                f"Invalid 'activities' field in JSON file '{filename}'. Must be a list or dictionary."
            )

        if not activity_list:
            raise ValueError(f"No activities found in JSON file '{filename}'.")

        successors_map: Dict[int, List[int]] = {}

        for act_info in activity_list:
            if not isinstance(act_info, dict):
                continue
            a_id = int(act_info["id"])
            dur = int(act_info.get("duration", 0))
            preds = [int(p) for p in act_info.get("predecessors", [])]
            succs = [int(s) for s in act_info.get("successors", [])]

            reqs_raw = (
                act_info.get("resource_requirements")
                or act_info.get("requests")
                or act_info.get("demands")
                or act_info.get("resources", {})
            )
            reqs: Dict[str, int] = {}
            if isinstance(reqs_raw, dict):
                for r_id in project.resources:
                    reqs[r_id] = int(reqs_raw.get(r_id, 0))
            elif isinstance(reqs_raw, list):
                for i, r_id in enumerate(project.resources.keys()):
                    reqs[r_id] = int(reqs_raw[i]) if i < len(reqs_raw) else 0

            project.activities[a_id] = Activity(
                id=a_id,
                duration=dur,
                predecessors=preds,
                resource_requirements=reqs,
            )

            if succs:
                successors_map[a_id] = succs

        # Infer predecessors from successors if any provided
        for src_id, succ_list in successors_map.items():
            for dest_id in succ_list:
                if dest_id in project.activities:
                    if src_id not in project.activities[dest_id].predecessors:
                        project.activities[dest_id].predecessors.append(src_id)

        # Populate successors
        for act in project.activities.values():
            act.successors = []
        for a_id, act in project.activities.items():
            for p_id in act.predecessors:
                if p_id in project.activities:
                    if a_id not in project.activities[p_id].successors:
                        project.activities[p_id].successors.append(a_id)

        Parser._validate_project(project, file_path)
        return project


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

    @staticmethod
    def _validate_project(project: Project, source: str = "") -> None:
        """
        Run basic structural checks on a parsed project.
        Raises ValueError on the first problem found.
        """
        label = f"'{source}'" if source else "project"

        if project.num_activities < 2:
            raise ValueError(
                f"Parsed {label} has only {project.num_activities} activity/ies. "
                "A valid RCPSP instance needs at least a source and a sink."
            )

        if project.num_resources == 0:
            raise ValueError(f"Parsed {label} has no resources.")

        # All predecessor IDs must resolve to known activities
        for a_id, act in project.activities.items():
            for p_id in act.predecessors:
                if p_id not in project.activities:
                    raise ValueError(
                        f"Activity {a_id} references unknown predecessor {p_id} "
                        f"in {label}."
                    )

        # Resource requirements must not exceed capacity (trivial sanity)
        for a_id, act in project.activities.items():
            for r_id, req in act.resource_requirements.items():
                cap = project.resources[r_id].capacity
                if req > cap:
                    raise ValueError(
                        f"Activity {a_id} requires {req} of {r_id} but capacity "
                        f"is only {cap} in {label}."
                    )
