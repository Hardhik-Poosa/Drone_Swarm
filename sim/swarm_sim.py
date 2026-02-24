import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# -------------------------------------------------
# Fix project root path
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.shape_generator import generate_shape
from utils.evaluate_swarm import evaluate_formation

# -------------------------------------------------
# ---------------- USER INPUT ---------------------
# -------------------------------------------------
N_DRONES = 5                 # \ CHANGE THIS FREELY
SHAPE = "circle"               # line | circle | v | grid
DISTANCE = 5               # spacing between drones
# -------------------------------------------------

# -------------------------------------------------
# Generate target formation (USER-CONTROLLED)
# -------------------------------------------------
target_formation = generate_shape(
    shape=SHAPE,
    n_drones=N_DRONES,
    distance=DISTANCE
)

# -------------------------------------------------
# Initialize random starting positions
# -------------------------------------------------
np.random.seed(42)
current_positions = np.random.uniform(-5, 5, size=(N_DRONES, 2))

# -------------------------------------------------
# Assign drones to targets (Hungarian Algorithm)
# -------------------------------------------------
cost_matrix = np.linalg.norm(
    current_positions[:, None, :] - target_formation[None, :, :],
    axis=2
)

row_ind, col_ind = linear_sum_assignment(cost_matrix)
target_formation = target_formation[col_ind]

# -------------------------------------------------
# Simulation parameters
# -------------------------------------------------
step_size = 0.08
num_frames = 120

plt.figure(figsize=(6, 6))

# -------------------------------------------------
# Swarm motion loop
# -------------------------------------------------
for frame in range(num_frames):
    direction = target_formation - current_positions
    current_positions += step_size * direction

    plt.clf()
    plt.scatter(
        current_positions[:, 0],
        current_positions[:, 1],
        c="blue",
        s=60,
        label="Drones"
    )

    plt.scatter(
        target_formation[:, 0],
        target_formation[:, 1],
        c="red",
        marker="x",
        s=80,
        label="Target Formation"
    )

    for i in range(N_DRONES):
        plt.text(
            current_positions[i, 0] + 0.05,
            current_positions[i, 1] + 0.05,
            str(i),
            fontsize=9
        )

    plt.title(f"Drone Swarm Motion Simulation ({SHAPE}, N={N_DRONES})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.gca().set_aspect("equal")
    plt.pause(0.05)

plt.show()

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
metrics = evaluate_formation(current_positions, target_formation)

print("\n--- Swarm Evaluation Metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("---------------------------------")
