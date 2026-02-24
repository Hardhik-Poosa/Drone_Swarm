import os
import sys

# Fix project root path FIRST
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from swarm_core import run_simulation
from utils.network_metrics import is_fully_connected

# -----------------------------------------
# Experiment parameters
# -----------------------------------------
N_values = [30, 50]
comm_ranges = [5, 10, 15, 20]

results = {}

for N in N_values:
    results[N] = []
    for comm in comm_ranges:
        metrics, positions = run_simulation(N, distance=5, comm_range=comm)
        full_conn = is_fully_connected(positions, comm)
        results[N].append((metrics["Connectivity Ratio"], full_conn))

# -----------------------------------------
# Plot Connectivity vs Communication Range
# -----------------------------------------
for N in N_values:
    pair_conn = [r[0] for r in results[N]]
    plt.plot(comm_ranges, pair_conn, marker='o', label=f"N={N}")

plt.title("Connectivity Ratio vs Communication Range")
plt.xlabel("Communication Range")
plt.ylabel("Connectivity Ratio")
plt.legend()
plt.grid()
plt.show()

# -----------------------------------------
# Density Tradeoff Experiment
# -----------------------------------------
print("\n--- Density Tradeoff Experiment ---")

distances = [5, 3, 2]
N = 30
comm_range = 10

for d in distances:
    metrics, _ = run_simulation(N, distance=d, comm_range=comm_range)
    print(f"Distance={d}")
    print(metrics)
    print()