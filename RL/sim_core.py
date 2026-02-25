"""
rl/sim_core.py
==============
Step-wise simulation primitives extracted from sim/swarm_sim.py and
sim/swarm_sim_3d.py so that Gymnasium-style environments can call
reset() → step() in a tight RL loop.

Key design choices
------------------
- Pure NumPy (no matplotlib / no model downloads) → safe inside RL episodes.
- 3D repulsion uses scipy.spatial.cKDTree → O(N log N) instead of O(N²).
- The RL *action* is added on top of the built-in physics so the agent only
  needs to learn residual corrections; base physics handles convergence.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_initial_positions(N: int, dims: int = 2, seed=None) -> np.ndarray:
    """Random starting cloud, matching the seed convention of the original sims."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-5, 5, size=(N, dims)).astype(np.float32)


def hungarian_assign_targets(positions: np.ndarray,
                             targets: np.ndarray) -> np.ndarray:
    """
    Reorder *targets* so that each drone is paired with the nearest waypoint
    (Hungarian / optimal assignment).  Mirrors the assignment in swarm_core.py.
    Returned array has the same shape as targets but rows permuted.
    """
    cost = np.linalg.norm(
        positions[:, None, :] - targets[None, :, :], axis=2
    )
    _, col_ind = linear_sum_assignment(cost)
    return targets[col_ind].astype(np.float32)


# ---------------------------------------------------------------------------
# 2-D step
# ---------------------------------------------------------------------------

def step_2d(
    positions: np.ndarray,
    targets: np.ndarray,
    actions: np.ndarray,
    step_size: float = 0.08,
) -> np.ndarray:
    """
    One simulation frame in 2-D.

    Physics (mirrors swarm_sim.py):
        attraction = target - current
        new_pos    = current + step_size * attraction

    RL extension:
        The agent adds a small velocity delta (actions) on top of the
        base-physics move.  This residual formulation keeps learning stable.

    Parameters
    ----------
    positions : (N, 2)  current drone positions
    targets   : (N, 2)  assigned waypoints
    actions   : (N, 2)  per-drone velocity delta from RL policy (clipped)
    step_size : float   attraction gain

    Returns
    -------
    new_positions : (N, 2)
    """
    attraction = targets - positions
    new_positions = positions + step_size * attraction + actions
    return new_positions.astype(np.float32)


# ---------------------------------------------------------------------------
# 3-D step
# ---------------------------------------------------------------------------

def step_3d(
    positions: np.ndarray,
    targets: np.ndarray,
    actions: np.ndarray,
    step_size: float = 0.05,
    repulsion_strength: float = 0.02,
    repulsion_radius: float = 0.6,
) -> np.ndarray:
    """
    One simulation frame in 3-D.

    Physics (mirrors swarm_sim_3d.py but with cKDTree for scalability):
        attraction = target - current
        repulsion  = Σ (diff / dist³) for neighbours within repulsion_radius

    RL extension:
        actions are added on top (residual control).

    Parameters
    ----------
    positions         : (N, 3)
    targets           : (N, 3)
    actions           : (N, 3)  per-drone 3-D velocity delta from policy
    step_size         : float
    repulsion_strength: float
    repulsion_radius  : float

    Returns
    -------
    new_positions : (N, 3)
    """
    attraction = targets - positions

    # --- Vectorised repulsion via cKDTree (O(N log N)) ---
    repulsion = np.zeros_like(positions)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=repulsion_radius, output_type="ndarray")

    if len(pairs) > 0:
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        diff = positions[i_idx] - positions[j_idx]           # (M, 3)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)   # (M, 1)
        dist_safe = dist + 1e-6
        force = diff / (dist_safe * (dist_safe ** 2))         # 1/d² direction
        np.add.at(repulsion, i_idx, force)
        np.add.at(repulsion, j_idx, -force)

    new_positions = (
        positions
        + step_size * attraction
        + repulsion_strength * repulsion
        + actions
    )
    return new_positions.astype(np.float32)


# ---------------------------------------------------------------------------
# Fast connectivity helper (used in reward + termination)
# ---------------------------------------------------------------------------

def fast_connectivity_ratio(positions: np.ndarray,
                            comm_range: float) -> float:
    """
    Fraction of all possible drone pairs within comm_range.
    Uses cKDTree → O(N log N) rather than O(N²).
    """
    N = len(positions)
    if N < 2:
        return 1.0
    tree = cKDTree(positions)
    pairs_within = len(tree.query_pairs(r=comm_range))
    total_possible = N * (N - 1) / 2
    return pairs_within / total_possible


def fast_min_distance(positions: np.ndarray) -> float:
    """Minimum inter-drone distance using cKDTree k=2 query."""
    if len(positions) < 2:
        return float("inf")
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)   # k=2: nearest is self (dist=0), 2nd is neighbour
    return float(dists[:, 1].min())
