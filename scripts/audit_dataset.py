import csv
import collections
import pprint
from pathlib import Path

def audit_dataset():
    raw_file = Path("data/training/raw_results.csv")
    if not raw_file.exists():
        print("No raw_results.csv found.")
        return
        
    counts = collections.defaultdict(list)
    with open(raw_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["instance_id"], row["algorithm"].split("[")[0], int(row["seed"]))
            counts[row["instance_id"]].append(key)
            
    print(f"Total instances in raw results: {len(counts)}")
    complete_instances = 0
    algorithms = {"GA", "SA", "BCO", "PRH", "TS"}
    seeds = {1, 2, 3}
    expected_runs = len(algorithms) * len(seeds)
    
    for inst, runs in counts.items():
        if len(runs) == expected_runs:
            complete_instances += 1
        elif len(runs) > expected_runs:
            print(f"Instance {inst} has {len(runs)} runs! > {expected_runs}")
            
    print(f"Complete instances (15 runs): {complete_instances}")
    
    training_file = Path("data/training/training_dataset.csv")
    if training_file.exists():
        with open(training_file, "r") as f:
            reader = csv.DictReader(f)
            labels = collections.Counter(r["best_algorithm"] for r in reader)
        print(f"Labels in training dataset: {labels}")

if __name__ == "__main__":
    audit_dataset()
