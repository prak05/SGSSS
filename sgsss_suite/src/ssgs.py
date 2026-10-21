"""
================================================================================
src/ssgs.py — Serial Schedule Generation Scheme (Central Decoder)
================================================================================
The single, shared decoder used by ALL metaheuristic solvers.

Algorithm outline (one SSGS call):
  1.  Normalise the priority vector into {activity_id -> float}.
  2.  Schedule the dummy source at t = 0.
  3.  Maintain a set of eligible activities (all predecessors finished).
  4.  At each stage pick the eligible activity with the highest priority.
  5.  Compute its Earliest Start Time (EST) = max predecessor finish times.
  6.  Slide EST forward until a contiguous resource-feasible window is found.
  7.  Allocate resources, record the ScheduleEntry, update successor counters.
  8.  Repeat until all activities are scheduled.

Guarantees:
  • Output is always 100% precedence-feasible (DAG order respected).
  • Output is always 100% resource-feasible (no over-allocation).
  • Deterministic: same priority vector => same schedule.

References
──────────
[1][9][14] Mitsou & Koulinas (2026); Kolisch (1996).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

from src.model import Activity, Project, ScheduleEntry


class SSGS:
    """
    Serial Schedule Generation Scheme — the centralised schedule decoder.

    Every algorithm calls SSGS.generate_schedule(project, priority_vector)
    and receives back a guaranteed-feasible (schedule, makespan) pair.
    """

    # ==============================================================================
    # SECTION SUMMARY: Priority Vector Normalization
    # What is in this section: Converts raw priority arrays or dictionaries into a
    # normalized {activity_id: priority_value} mapping for deterministic decoding.
    # ==============================================================================

    @staticmethod
    def _build_priority_dict(
        project: Project,
        priority_vector: Union[Dict[int, float], List[float]],
    ) -> Dict[int, float]:
        """
        Normalise priority_vector into {activity_id: priority_float}.

        Accepts either a dict already keyed by activity ID or a flat
        list whose elements are mapped to sorted activity IDs in order.
        """
        if isinstance(priority_vector, dict):
            return priority_vector

        sorted_ids = sorted(project.activities.keys())
        if len(priority_vector) != len(sorted_ids):
            raise ValueError(
                f"priority_vector length {len(priority_vector)} ≠ "
                f"activity count {len(sorted_ids)}"
            )
        return dict(zip(sorted_ids, priority_vector))

    # ==============================================================================
    # SECTION SUMMARY: Serial Schedule Generation Engine
    # What is in this section: Implements the active scheduling loop: starts dummy
    # task at t=0, dynamically tracks predecessor completions, iteratively picks
    # the highest-priority ready task, slides its start time until all resource
    # constraints are satisfied, and records the final makespan.
    # ==============================================================================

    @staticmethod
    def generate_schedule(
        project: Project,
        priority_vector: Union[Dict[int, float], List[float]],
    ) -> Tuple[List[ScheduleEntry], int]:
        """
        Decode a random-key priority vector into a 100 % feasible schedule.

        Args:
            project:         The RCPSP instance.
            priority_vector: Random-key priorities (one float per activity).

        Returns:
            (schedule_entries, makespan) where makespan = max finish time.
        """
        # Step 1 — normalise priorities
        priority = SSGS._build_priority_dict(project, priority_vector)

        # Step 2 — identify the dummy source (lowest activity ID, usually 1)
        start_act: int = min(project.activities.keys())

        # Step 3 — initial time-horizon estimate (grows dynamically if needed)
        horizon: int = sum(a.duration for a in project.activities.values()) + 100

        # Step 4 — per-resource availability timeline: avail[r][t] = remaining cap
        avail: Dict[str, List[int]] = {
            r_id: [res.capacity] * horizon
            for r_id, res in project.resources.items()
        }

        # Step 5 — bookkeeping structures
        schedule_entries: List[ScheduleEntry] = []
        finish_times: Dict[int, int] = {start_act: 0}
        completed: set = {start_act}

        # Step 6 — schedule the dummy source at t = 0
        schedule_entries.append(
            ScheduleEntry(
                activity_id=start_act,
                start_time=0,
                finish_time=0,
                duration=0,
                resource_usage={},
            )
        )

        # Step 7 — predecessor counters (excluding the source itself)
        uncompleted_preds: Dict[int, int] = {
            a: sum(1 for p in act.predecessors if p != start_act)
            for a, act in project.activities.items()
            if a != start_act
        }

        # Step 8 — initial eligible set: activities whose predecessors are done
        eligible: List[int] = [a for a, cnt in uncompleted_preds.items() if cnt == 0]

        # ── MAIN DECODING LOOP ─────────────────────────────────────────────
        while eligible:
            # Pick the eligible activity with the HIGHEST priority
            eligible.sort(key=lambda a: priority.get(a, 0.0), reverse=True)
            selected_id: int = eligible.pop(0)
            act: Activity = project.activities[selected_id]

            # --- Precedence constraint: EST = max(finish(p) for p in preds) ---
            est: int = max((finish_times[p] for p in act.predecessors), default=0)
            dur: int = act.duration

            # Non-zero resource requirements for this activity
            reqs: List[Tuple[str, int]] = [
                (r_id, req) for r_id, req in act.resource_requirements.items()
                if req > 0
            ]

            # --- Resource constraint: slide start_time >= est until feasible ---
            start_time: int = est
            if dur > 0 and reqs:
                while True:
                    # Grow the availability arrays if we exceed the horizon
                    if start_time + dur >= horizon:
                        new_h = horizon + dur + 500
                        for r_id in avail:
                            cap = project.resources[r_id].capacity
                            avail[r_id].extend([cap] * (new_h - len(avail[r_id])))
                        horizon = new_h

                    # Check all resource dimensions for [start_time, start_time+dur)
                    slot_ok = True
                    for r_id, req in reqs:
                        r_avail = avail[r_id]
                        for t in range(start_time, start_time + dur):
                            if r_avail[t] < req:
                                slot_ok = False
                                break
                        if not slot_ok:
                            break

                    if slot_ok:
                        break               # feasible window found
                    start_time += 1          # try next time slot

            # Finish time
            finish_time: int = start_time + dur
            finish_times[selected_id] = finish_time

            # Allocate resources across active duration window
            if dur > 0 and reqs:
                for r_id, req in reqs:
                    r_avail = avail[r_id]
                    for t in range(start_time, finish_time):
                        r_avail[t] -= req

            # Record the schedule entry
            schedule_entries.append(
                ScheduleEntry(
                    activity_id=selected_id,
                    start_time=start_time,
                    finish_time=finish_time,
                    duration=dur,
                    resource_usage=act.resource_requirements.copy(),
                )
            )

            # Mark completed and update eligible set
            completed.add(selected_id)
            for s in act.successors:
                if s in uncompleted_preds:
                    uncompleted_preds[s] -= 1
                    if uncompleted_preds[s] == 0:
                        eligible.append(s)

        # Makespan = max finish time across all entries
        makespan: int = max((e.finish_time for e in schedule_entries), default=0)
        return schedule_entries, makespan
