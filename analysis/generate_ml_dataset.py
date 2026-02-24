import os
import sys
import csv
import numpy as np

# Fix project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from analysis.swarm_core import run_simulation

# Parameter ranges
N_values = [10, 20, 30, 40, 50]
distances = [2, 3, 4, 5]
comm_ranges = [5, 10, 15, 20]

dataset = []

print("Generating dataset...")

for N in N_values:
    for d in distances:
        for comm in comm_ranges:
            metrics, _ = run_simulation(N, distance=d, comm_range=comm)

            dataset.append([
                N,
                d,
                comm,
                float(metrics["Connectivity Ratio"]),
                float(metrics["Minimum Inter-Drone Distance"])
            ])

# Save to CSV
output_file = "analysis/swarm_ml_dataset.csv"

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["N", "distance", "comm_range", "connectivity", "min_distance"])
    writer.writerows(dataset)

print(f"Dataset saved to {output_file}")