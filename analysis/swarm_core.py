import os
import sys

# Fix project root path FIRST
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.shape_generator import generate_shape
from utils.evaluate_swarm import evaluate_formation

def run_simulation(N, distance, comm_range):
    
    target = generate_shape("grid", n_drones=N, distance=distance)

    np.random.seed(42)
    current = np.random.uniform(-5, 5, size=(N, 2))

    cost = np.linalg.norm(current[:, None, :] - target[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    target = target[col_ind]

    step_size = 0.08
    num_frames = 120

    for _ in range(num_frames):
        direction = target - current
        current += step_size * direction

    metrics = evaluate_formation(current, target, comm_range=comm_range)

    return metrics, current