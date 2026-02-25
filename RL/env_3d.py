"""
rl/env_3d.py
============
3-D Multi-Agent RL environment for Drone Swarm AI.

Identical structure to SwarmEnv2D but :
  - positions / targets are 3-D
  - base physics includes collision-repulsion (from swarm_sim_3d.py)
  - observation adds a z-component for target-relative and neighbour vectors
  - action dim = 3

Observation per drone  (dim = 4 + K*4 = 28 for K=6)
------------------------------------------------------
  [0:3]    relative vector to assigned target (dx, dy, dz)
  [3]      distance to assigned target
  [4:4+K*4] for each of K nearest neighbours:
               relative position (dx, dy, dz) + distance  (4 values)

Action per drone  (dim = 3)
----------------------------
  (dvx, dvy, dvz)  velocity delta clipped to [-MAX_V, MAX_V]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from scipy.spatial import cKDTree

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_OK = True
except ImportError:
    _GYM_OK = False
    gym    = None  # type: ignore
    spaces = None  # type: ignore

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as RLlibMAEnv
    _BASE = RLlibMAEnv
except ImportError:
    if _GYM_OK:
        _BASE = gym.Env
    else:
        _BASE = object

from utils.shape_generator import generate_shape
from rl.sim_core import (
    make_initial_positions,
    hungarian_assign_targets,
    step_3d,
)
from rl.reward import compute_reward, terminal_bonus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_NEIGHBORS  = 6
MAX_V        = 0.15     # slightly smaller for 3-D stability
OBS_DIM      = 4 + K_NEIGHBORS * 4   # 4 + 6*4 = 28


def _generate_shape_3d(shape: str, n_drones: int, distance: float) -> np.ndarray:
    """
    Generate a 3-D target formation.
    Uses the existing 2-D shape_generator and offsets layers for simple 3-D
    shapes.  For RL training this is sufficient; image-derived 3-D formations
    can be passed in via the 'precomputed_targets' option in reset().
    """
    pts_2d = generate_shape(shape, n_drones=n_drones, distance=distance)  # (N,2)
    z      = np.zeros((len(pts_2d), 1), dtype=np.float32)
    return np.hstack([pts_2d, z]).astype(np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SwarmEnv3D(_BASE):
    """
    3-D per-drone MARL environment.

    Config keys
    -----------
    N                   : int     default 20
    distance            : float   default 1.5
    comm_range          : float   default 6.0
    min_safe_dist       : float   default 0.5
    max_steps           : int     default 150
    shape               : str     default "grid"
    step_size           : float   default 0.05
    repulsion_strength  : float   default 0.02
    repulsion_radius    : float   default 0.6
    precomputed_targets : ndarray (N,3) optional; overrides shape generation
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}

        self.N                  = int(cfg.get("N",                  20))
        self.distance           = float(cfg.get("distance",          1.5))
        self.comm_range         = float(cfg.get("comm_range",        6.0))
        self.min_safe_dist      = float(cfg.get("min_safe_dist",     0.5))
        self.max_steps          = int(cfg.get("max_steps",          150))
        self.shape              = str(cfg.get("shape",              "grid"))
        self.step_size          = float(cfg.get("step_size",         0.05))
        self.repulsion_strength = float(cfg.get("repulsion_strength",0.02))
        self.repulsion_radius   = float(cfg.get("repulsion_radius",  0.6))
        self._precomputed       = cfg.get("precomputed_targets", None)
        if self._precomputed is not None:
            self._precomputed = np.array(self._precomputed, dtype=np.float32)
            self.N = len(self._precomputed)

        self._agent_ids = set(range(self.N))

        single_obs = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        single_act = spaces.Box(-MAX_V, MAX_V,  shape=(3,),        dtype=np.float32)
        self.observation_space = single_obs
        self.action_space      = single_act

        self.positions: np.ndarray = np.zeros((self.N, 3), dtype=np.float32)
        self.targets  : np.ndarray = np.zeros((self.N, 3), dtype=np.float32)
        self._t = 0

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if options:
            if "N"        in options:
                self.N        = int(options["N"])
                self._precomputed = None
                self._agent_ids   = set(range(self.N))
            if "shape"    in options: self.shape    = str(options["shape"])
            if "distance" in options: self.distance = float(options["distance"])
            if "precomputed_targets" in options:
                self._precomputed = np.array(
                    options["precomputed_targets"], dtype=np.float32
                )
                self.N            = len(self._precomputed)
                self._agent_ids   = set(range(self.N))

        if self._precomputed is not None:
            self.targets = self._precomputed.copy()
        else:
            self.targets = _generate_shape_3d(
                self.shape, self.N, self.distance
            )

        self.positions = make_initial_positions(self.N, dims=3, seed=seed)
        self.targets   = hungarian_assign_targets(self.positions, self.targets)
        self._t        = 0

        return self._build_obs(), {}

    def step(self, action_dict: dict):
        actions = np.array(
            [action_dict.get(i, np.zeros(3, dtype=np.float32))
             for i in range(self.N)],
            dtype=np.float32,
        )
        actions = np.clip(actions, -MAX_V, MAX_V)

        self.positions = step_3d(
            self.positions, self.targets, actions,
            step_size          = self.step_size,
            repulsion_strength = self.repulsion_strength,
            repulsion_radius   = self.repulsion_radius,
        )
        self._t += 1
        done = self._t >= self.max_steps

        reward, info = compute_reward(
            self.positions, self.targets,
            comm_range    = self.comm_range,
            min_safe_dist = self.min_safe_dist,
        )

        if done:
            t_bonus, success = terminal_bonus(
                self.positions, self.targets,
                comm_range    = self.comm_range,
                min_safe_dist = self.min_safe_dist,
            )
            reward += t_bonus
            info["success"] = success
        else:
            info["success"] = False

        obs = self._build_obs()
        rewards     = {i: reward for i in range(self.N)}
        terminateds = {i: done   for i in range(self.N)}
        terminateds["__all__"] = done
        truncateds  = {i: False  for i in range(self.N)}
        truncateds["__all__"]  = False
        infos       = {i: info   for i in range(self.N)}

        return obs, rewards, terminateds, truncateds, infos

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict:
        tree    = cKDTree(self.positions)
        k_query = min(K_NEIGHBORS + 1, self.N)
        dists_all, idx_all = tree.query(self.positions, k=k_query)

        obs_dict = {}
        for i in range(self.N):
            rel_target      = self.targets[i] - self.positions[i]
            rel_target_dist = float(np.linalg.norm(rel_target))

            nn_feats = []
            for k in range(1, k_query):
                nb  = idx_all[i, k]
                d   = dists_all[i, k]
                rel = self.positions[nb] - self.positions[i]
                nn_feats.extend([rel[0], rel[1], rel[2], d])

            while len(nn_feats) < K_NEIGHBORS * 4:
                nn_feats.extend([0.0, 0.0, 0.0, 0.0])

            obs = np.array(
                [rel_target[0], rel_target[1], rel_target[2], rel_target_dist]
                + nn_feats,
                dtype=np.float32,
            )
            obs_dict[i] = obs

        return obs_dict

    # ------------------------------------------------------------------
    # Curriculum helper
    # ------------------------------------------------------------------

    def set_curriculum(self, N: int, shape: str, distance: float):
        self.N        = N
        self.shape    = shape
        self.distance = distance
        self._agent_ids = set(range(N))
        self._precomputed = None
