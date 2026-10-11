import csv
import json
import tempfile
from pathlib import Path
from src.ml.generate_dataset import DatasetGenerator, generate_training_dataset
from src.benchmark import ALGORITHM_NAMES

def test_resume_behavior():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create a mock raw_results.csv with one completed run
        raw_csv = tmp_path / "raw_results.csv"
        with open(raw_csv, "w") as f:
            f.write("instance_id,instance_family,instance_source,feature_configuration,instance_size,seed,evaluation_budget,algorithm,makespan,runtime,evaluations,feasible,N,NC,RF,RS,best_priority_vector,notes\n")
            f.write("j301_1,j30,PSPLIB,j301,30,1,5000,GA,100,0.1,5000,True,30,1.5,0.5,0.2,[],\n")
        
        gen = DatasetGenerator(output_dir=tmp_path, evaluation_budget=5000, seeds=[1, 2, 3])
        assert ("j301_1", "GA", 1) in gen._existing_raw_keys
        assert len(gen._existing_raw_keys) == 1

if __name__ == "__main__":
    test_resume_behavior()
    print("Test passed.")
