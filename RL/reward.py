"""
rl/reward.py
============
Reward computation for the Drone Swarm MARL environments.

Objective: maximise connectivity while enforcing collision safety.

Reward formula (team-level, shared equally across all agents):
    R = w_conn  *  connectivity_ratio            (↑ want high)
      - w_safety * unsafe_pair_ratio             (↓ penalise collisions)
      - w_conv   * mean_convergence_error        (↓ formation accuracy shaping)
      + w_bonus  * [fully-connected bonus]       (sparse terminal bonus)

All terms are normalised to [0, 1] range so weights are interpretable.
"""

import numpy as np
from scipy.spatial import cKDTree

from rl.sim_core import fast_connectivity_ratio, fast_min_distance


def compute_reward(
    positions: np.ndarray,
    targets: np.ndarray,
    comm_range: float,
    min_safe_dist: float = 0.5,
    w_conn: float = 1.0,
    w_safety: float = 2.0,
    w_conv: float = 0.1,
    w_bonus: float = 0.5,
    fully_connected_threshold: float = 1.0,
) -> tuple[float, dict]:
    """
    Compute the shared team reward for one step.

    Parameters
    ----------
    positions   : (N, D)  current drone positions
    targets     : (N, D)  assigned waypoints
    comm_range  : float   communication radius
    min_safe_dist: float  minimum safe inter-drone distance
    w_conn      : float   weight for connectivity reward
    w_safety    : float   weight for safety penalty
    w_conv      : float   weight for convergence shaping
    w_bonus     : float   bonus for achieving full connectivity
    fully_connected_threshold : float  connectivity_ratio above which bonus fires

    Returns
    -------
    reward : float
    info   : dict  (detailed breakdown for logging)
    """
    N = len(positions)
    tree = cKDTree(positions)

    # --- connectivity -------------------------------------------------------
    connectivity_ratio = fast_connectivity_ratio(positions, comm_range)

    # --- safety (fraction of pairs below min_safe_dist) --------------------
    unsafe_pairs = len(tree.query_pairs(r=min_safe_dist))
    total_possible = max(N * (N - 1) / 2, 1)
    unsafe_pair_ratio = unsafe_pairs / total_possible

    # --- convergence shaping -----------------------------------------------
    convergence_error = float(
        np.mean(np.linalg.norm(positions - targets, axis=1))
    )
    # Normalise: divide by an expected "initial" error scale so term ~1 at start
    INIT_ERROR_SCALE = 10.0
    convergence_norm = min(convergence_error / INIT_ERROR_SCALE, 1.0)

    # --- full connectivity bonus (sparse) -----------------------------------
    fully_connected_bonus = (
        w_bonus if connectivity_ratio >= fully_connected_threshold else 0.0
    )

    # --- total reward -------------------------------------------------------
    reward = (
        w_conn   * connectivity_ratio
        - w_safety * unsafe_pair_ratio
        - w_conv   * convergence_norm
        + fully_connected_bonus
    )

    info = {
        "connectivity_ratio": connectivity_ratio,
        "unsafe_pair_ratio": unsafe_pair_ratio,
        "convergence_error": convergence_error,
        "min_distance": fast_min_distance(positions),
        "fully_connected_bonus": fully_connected_bonus,
        "reward": reward,
    }
    return float(reward), info


def terminal_bonus(
    positions: np.ndarray,
    targets: np.ndarray,
    comm_range: float,
    min_safe_dist: float = 0.5,
    connectivity_target: float = 0.95,
    convergence_target: float = 0.5,
) -> tuple[float, bool]:
    """
    Extra reward at episode end if both connectivity and convergence goals met.
    Returns (bonus, success_flag).
    """
    connectivity_ratio = fast_connectivity_ratio(positions, comm_range)
    convergence_error = float(
        np.mean(np.linalg.norm(positions - targets, axis=1))
    )
    min_dist = fast_min_distance(positions)

    success = (
        connectivity_ratio >= connectivity_target
        and convergence_error <= convergence_target
        and min_dist >= min_safe_dist
    )
    bonus = 2.0 if success else 0.0
    return bonus, success
