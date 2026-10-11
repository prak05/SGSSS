from typing import Dict, List, Tuple, Union

from src.model import Project, ScheduleEntry


class SSGS:
    """
    Serial Schedule Generation Scheme — the one and only scheduling decoder.

    All optimisation algorithms must call this class.  No algorithm module
    may contain its own resource-feasibility or scheduling logic.

    Priority-vector contract
    ========================
    The caller passes a *priority_vector* which may be either:
      • dict[int, float]  — mapping activity_id → priority value in [0, 1]
      • list[float]       — values in order of sorted(project.activities.keys()),
                            i.e. position 0 corresponds to the lowest activity ID.

    Higher priority value → scheduled earlier among eligible activities.
    """

    @staticmethod
    def _build_priority_dict(
        project: Project,
        priority_vector: Union[Dict[int, float], List[float]],
    ) -> Dict[int, float]:
        """Normalise any supported priority-vector form into a dict."""
        if isinstance(priority_vector, dict):
            return priority_vector
        # List form: map values to activity IDs in sorted key order
        sorted_ids = sorted(project.activities.keys())
        if len(priority_vector) != len(sorted_ids):
            raise ValueError(
                f"priority_vector length {len(priority_vector)} does not match "
                f"number of activities {len(sorted_ids)}"
            )
        return dict(zip(sorted_ids, priority_vector))

    @staticmethod
    def generate_schedule(
        project: Project,
        priority_vector: Union[Dict[int, float], List[float]],
    ) -> Tuple[List[ScheduleEntry], int]:
        """
        Decode a random-key priority vector into a feasible project schedule.

        Returns
        -------
        schedule : list[ScheduleEntry]
            Start/finish times for every activity, in scheduling order.
        makespan : int
            Finish time of the last activity (project duration).
        """
        priority = SSGS._build_priority_dict(project, priority_vector)

        start_act = min(project.activities.keys())

        # Capacity timeline array per resource
        # Preallocate sufficient horizon with dynamic expansion if needed
        horizon = sum(a.duration for a in project.activities.values()) + 100
        avail: Dict[str, List[int]] = {
            r_id: [res.capacity] * horizon for r_id, res in project.resources.items()
        }

        schedule_entries: List[ScheduleEntry] = []
        finish_times: Dict[int, int] = {start_act: 0}
        completed: set = {start_act}

        # Dummy start activity always scheduled at time 0 with zero duration
        schedule_entries.append(
            ScheduleEntry(
                activity_id=start_act,
                start_time=0,
                finish_time=0,
                duration=0,
                resource_usage={},
            )
        )

        # Track uncompleted predecessor counts for O(1) eligibility updates
        uncompleted_preds = {
            a: sum(1 for p in act.predecessors if p != start_act)
            for a, act in project.activities.items()
            if a != start_act
        }
        eligible = [a for a, count in uncompleted_preds.items() if count == 0]

        while eligible:
            # Select highest-priority eligible activity
            eligible.sort(key=lambda a: priority.get(a, 0.0), reverse=True)
            selected_id = eligible.pop(0)
            act = project.activities[selected_id]

            # 1. Respect precedence: must start after all predecessors finish
            est = max((finish_times[p] for p in act.predecessors), default=0)
            dur = act.duration
            reqs = [
                (r_id, req)
                for r_id, req in act.resource_requirements.items()
                if req > 0
            ]

            # 2. Respect resource capacities via fast array lookup
            start_time = est
            if dur > 0 and reqs:
                while True:
                    if start_time + dur >= horizon:
                        add_h = horizon + dur + 500
                        for r_id in avail:
                            cap = project.resources[r_id].capacity
                            avail[r_id].extend([cap] * (add_h - len(avail[r_id])))
                        horizon = add_h

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
                        break
                    start_time += 1

            finish_time = start_time + dur
            finish_times[selected_id] = finish_time

            # Deduct resource occupancy
            if dur > 0 and reqs:
                for r_id, req in reqs:
                    r_avail = avail[r_id]
                    for t in range(start_time, finish_time):
                        r_avail[t] -= req

            schedule_entries.append(
                ScheduleEntry(
                    activity_id=selected_id,
                    start_time=start_time,
                    finish_time=finish_time,
                    duration=dur,
                    resource_usage=act.resource_requirements.copy(),
                )
            )

            completed.add(selected_id)
            # Update successors
            for s in act.successors:
                if s in uncompleted_preds:
                    uncompleted_preds[s] -= 1
                    if uncompleted_preds[s] == 0:
                        eligible.append(s)

        if len(schedule_entries) != len(project.activities):
            raise RuntimeError(
                "SSGS deadlock: could not schedule all activities. "
                "Check for cycles or missing predecessor data."
            )

        makespan = max(finish_times.values())
        return schedule_entries, makespan
