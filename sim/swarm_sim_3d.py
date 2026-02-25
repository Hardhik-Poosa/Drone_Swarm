import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D

# Fix project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.semantic_image_to_formation import image_to_semantic_outline
from utils.formation_3d import lift_to_true_3d
import pandas as pd

# ---------------- USER INPUT ----------------
IMAGE_PATH = "input_images/Cristiano_Ronaldo.webp"
BASE_DRONES = 300
SCALE_FACTOR = 6
HEIGHT_SCALE = 10
LAYERS = 4
# --------------------------------------------

# Generate 2D formation
formation_2d = image_to_semantic_outline(
    IMAGE_PATH,
    n_drones=BASE_DRONES,
    scale_factor=SCALE_FACTOR
)

# Lift to TRUE 3D + volumetric filling
target_formation = lift_to_true_3d(
    formation_2d,
    IMAGE_PATH,
    height_scale=HEIGHT_SCALE,
    layers=LAYERS
)

N_DRONES = len(target_formation)

# Initialize random 3D positions
np.random.seed(42)
current_positions = np.random.uniform(-5, 5, size=(N_DRONES, 3))

# Hungarian assignment (3D)
cost_matrix = np.linalg.norm(
    current_positions[:, None, :] - target_formation[None, :, :],
    axis=2
)

row_ind, col_ind = linear_sum_assignment(cost_matrix)
target_formation = target_formation[col_ind]

# Simulation parameters
step_size = 0.05
repulsion_strength = 0.02
repulsion_radius = 0.6
num_frames = 150

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

for frame in range(num_frames):

    # ---------------- Attraction ----------------
    attraction = target_formation - current_positions

    # ---------------- Repulsion ----------------
    repulsion = np.zeros_like(current_positions)

    for i in range(N_DRONES):
        diff = current_positions[i] - current_positions
        dist = np.linalg.norm(diff, axis=1)

        mask = (dist > 0) & (dist < repulsion_radius)

        if np.any(mask):
            repulsion[i] += np.sum(
                (diff[mask] / (dist[mask][:, None] + 1e-6)) *
                (1 / (dist[mask][:, None]**2)),
                axis=0
            )

    # Update positions
    current_positions += step_size * attraction + repulsion_strength * repulsion

    ax.cla()

    ax.scatter(
        current_positions[:, 0],
        current_positions[:, 1],
        current_positions[:, 2],
        c="blue",
        s=8
    )

    ax.set_title("3D Volumetric Drone Swarm (With Collision Avoidance)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])

    # Cinematic rotating camera
    ax.view_init(
        elev=30 + 10*np.sin(frame/20),
        azim=frame
    )

    plt.pause(0.05)

plt.show()

# Export final positions
os.makedirs("output", exist_ok=True)
np.savetxt("output/drone_positions_3d.csv",
           current_positions,
           delimiter=",")


df = pd.DataFrame(current_positions, columns=["x", "y", "z"])
df.to_csv("output/drone_positions_3d.csv", index=False)


print("3D Drone positions exported to output/drone_positions_3d.csv")