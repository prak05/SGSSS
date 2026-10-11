import unittest
from src.parser import Parser
from src.evaluation import Evaluator
from src.model import ScheduleEntry

class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.project = Parser.parse('data/sample.rcp')

    def test_infeasible_schedule_precedence(self):
        # Fake an invalid schedule
        schedule = [
            ScheduleEntry(0, 0, 0, 0, {}),
            ScheduleEntry(1, 0, 4, 4, {"R1": 2, "R2": 1}),
            ScheduleEntry(2, 0, 3, 3, {"R1": 1, "R2": 1}),
            ScheduleEntry(4, 0, 5, 5, {"R1": 2, "R2": 1}), # Started before 1 finished (pred of 4 is 1)
        ]
        
        feasible = Evaluator.check_constraints(self.project, schedule)
        self.assertFalse(feasible)

    def test_infeasible_schedule_resource(self):
        # Fake an invalid schedule due to resource overallocation
        # 1 and 4 both use R1=2, total=4 > capacity=3
        schedule = [
            ScheduleEntry(0, 0, 0, 0, {}),
            ScheduleEntry(1, 0, 4, 4, {"R1": 2, "R2": 1}),
            ScheduleEntry(4, 0, 5, 5, {"R1": 2, "R2": 1}),
        ]
        
        # Modify pred just for testing resource overlap without failing precedence 
        self.project.activities[4].predecessors = []
        
        feasible = Evaluator.check_constraints(self.project, schedule)
        self.assertFalse(feasible)

if __name__ == '__main__':
    unittest.main()
