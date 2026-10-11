from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Resource:
    id: str
    capacity: int

@dataclass
class Activity:
    id: int
    duration: int
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    resource_requirements: Dict[str, int] = field(default_factory=dict)

@dataclass
class Project:
    activities: Dict[int, Activity] = field(default_factory=dict)
    resources: Dict[str, Resource] = field(default_factory=dict)
    
    @property
    def num_activities(self) -> int:
        return len(self.activities)
    
    @property
    def num_resources(self) -> int:
        return len(self.resources)

@dataclass
class ScheduleEntry:
    activity_id: int
    start_time: int
    finish_time: int
    duration: int
    resource_usage: Dict[str, int]

@dataclass
class AlgorithmResult:
    algorithm: str
    best_priority_vector: List[float]
    schedule: List[ScheduleEntry]
    makespan: int
    runtime: float
    evaluations: int
    convergence_history: List[tuple] = field(default_factory=list)
