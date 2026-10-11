from typing import List, Tuple, Dict
from src.model import Project, ScheduleEntry
from src.ssgs import SSGS

class Evaluator:
    @staticmethod
    def evaluate(project: Project, priority_vector: List[float]) -> Tuple[bool, int, List[ScheduleEntry]]:
        """
        Evaluate a candidate priority vector using SSGS.
        Returns:
            feasible (bool): Always True if SSGS succeeds (SSGS guarantees feasibility).
            makespan (int): The final makespan of the project.
            schedule (List[ScheduleEntry]): The computed feasible schedule.
        """
        schedule, makespan = SSGS.generate_schedule(project, priority_vector)
        
        # Double check constraints for safety
        feasible = Evaluator.check_constraints(project, schedule)
        
        return feasible, makespan, schedule

    @staticmethod
    def check_constraints(project: Project, schedule: List[ScheduleEntry]) -> bool:
        """
        Validates a generated schedule to ensure:
        1. Precedence constraints are met.
        2. Resource constraints are met at any time t.
        """
        # Map finish times
        finish_times = {entry.activity_id: entry.finish_time for entry in schedule}
        start_times = {entry.activity_id: entry.start_time for entry in schedule}
        
        # Precedence check
        for act_id, act in project.activities.items():
            if act_id not in start_times:
                return False # Activity not scheduled
            
            for pred in act.predecessors:
                if finish_times[pred] > start_times[act_id]:
                    return False
        
        # Resource capacity check
        makespan = max(finish_times.values()) if finish_times else 0
        
        for t in range(makespan):
            current_usage = {r_id: 0 for r_id in project.resources}
            for entry in schedule:
                if entry.start_time <= t < entry.finish_time:
                    for r_id, usage in entry.resource_usage.items():
                        current_usage[r_id] += usage
            
            for r_id, cap in project.resources.items():
                if current_usage[r_id] > cap.capacity:
                    return False
                    
        return True
