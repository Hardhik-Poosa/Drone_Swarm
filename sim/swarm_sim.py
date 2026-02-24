import os
import sys

# -------------------------------------------------
# Fix project root path
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from utils.semantic_image_to_formation import image_to_semantic_outline
from utils.evaluate_swarm import evaluate_formation

# -------------------------------------------------
# ---------------- USER INPUT ---------------------
# -------------------------------------------------

IMAGE_PATH = "D:\\drone_swram\\input_images\\516AaQ6o17L.webp"
N_DRONES = 300
SCALE_FACTOR = 5

# -------------------------------------------------
# Generate target formation from semantic model
# -------------------------------------------------

target_formation = image_to_semantic_outline(
    IMAGE_PATH,
    n_drones=N_DRONES,
    scale_factor=SCALE_FACTOR
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

plt.figure(figsize=(8, 8))

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
        s=40,
        label="Drones"
    )

    plt.scatter(
        target_formation[:, 0],
        target_formation[:, 1],
        c="red",
        marker="x",
        s=50,
        label="Target Formation"
    )

    # Show labels only if N is small
    if N_DRONES <= 100:
        for i in range(N_DRONES):
            plt.text(
                current_positions[i, 0] + 0.05,
                current_positions[i, 1] + 0.05,
                str(i),
                fontsize=7
            )

    plt.title(f"Drone Swarm Motion Simulation (N={N_DRONES})")
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