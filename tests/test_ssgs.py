import unittest
from src.parser import Parser
from src.ssgs import SSGS
from src.evaluation import Evaluator

class TestSSGS(unittest.TestCase):
    def setUp(self):
        self.project = Parser.parse('data/sample.rcp')

    def test_ssgs_feasibility(self):
        # Generate random priority vector
        import random
        random.seed(42)
        priorities = {a_id: random.random() for a_id in self.project.activities.keys()}
        
        schedule, makespan = SSGS.generate_schedule(self.project, priorities)
        
        # Check that we got a schedule for all activities
        self.assertEqual(len(schedule), self.project.num_activities)
        
        # Evaluate to ensure constraints are respected
        feasible = Evaluator.check_constraints(self.project, schedule)
        self.assertTrue(feasible)

if __name__ == '__main__':
    unittest.main()
