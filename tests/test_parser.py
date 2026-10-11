import unittest
import os
from src.parser import Parser

class TestParser(unittest.TestCase):
    def test_parse_sample_rcp(self):
        project = Parser.parse('data/sample.rcp')
        self.assertEqual(project.num_activities, 8)
        self.assertEqual(project.num_resources, 2)
        
        self.assertEqual(project.resources['R1'].capacity, 3)
        self.assertEqual(project.resources['R2'].capacity, 2)
        
        act_4 = project.activities[4]
        self.assertEqual(act_4.duration, 5)
        self.assertEqual(act_4.resource_requirements['R1'], 2)
        self.assertEqual(act_4.resource_requirements['R2'], 1)
        self.assertEqual(act_4.predecessors, [1])

    def test_parse_sample_json(self):
        project = Parser.parse('data/sample.json')
        self.assertEqual(project.num_activities, 8)
        self.assertEqual(project.num_resources, 2)
        
        self.assertEqual(project.resources['R1'].capacity, 3)
        self.assertEqual(project.resources['R2'].capacity, 2)
        
        act_2 = project.activities[2]
        self.assertEqual(act_2.duration, 4)
        self.assertEqual(act_2.resource_requirements['R1'], 2)
        self.assertEqual(act_2.resource_requirements['R2'], 1)
        self.assertEqual(act_2.predecessors, [1])

    def test_parse_merged_metadata_json_error(self):
        import tempfile, json
        merged_data = {"j1201_1": {"lb": 68, "ub": 68, "makespan": 68}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(merged_data, f)
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                Parser.parse(tmp_path, display_name="j120_sm_merged.json")
            self.assertIn("j120_sm_merged.json", str(ctx.exception))
            self.assertIn("PSPLIB benchmark bounds/solutions metadata file", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
