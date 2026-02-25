"""
rl/env_2d.py
============
2-D Multi-Agent Reinforcement Learning environment for Drone Swarm AI.

Each drone is an independent agent sharing ONE policy (parameter sharing /
MAPPO-style).  Centralised training is handled by RLlib; execution is fully
decentralised (each agent observes only local information).

Observation per drone  (dim = 3 + K*3 = 21 for K=6)
------------------------------------------------------
  [0:2]   relative vector to assigned target  (dx, dy)
  [2]     distance to assigned target
  [3:3+K*3]  for each of K nearest neighbours:
               relative position (dx, dy) + distance  (3 values)

Action per drone  (dim = 2)
----------------------------
  (dvx, dvy)  velocity delta clipped to [-MAX_V, MAX_V]
  Added on top of base attraction physics (residual control).

Team reward (shared equally)
-----------------------------
  See rl/reward.py for the full formula.
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
        _BASE = gym.Env  # plain Gymnasium fallback (no Ray)
    else:
        _BASE = object   # absolute fallback (missing both; import won't crash)

from utils.shape_generator import generate_shape
from rl.sim_core import (
    make_initial_positions,
    hungarian_assign_targets,
    step_2d,
)
from rl.reward import compute_reward, terminal_bonus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_NEIGHBORS = 6          # number of nearest neighbours in obs
MAX_V       = 0.2        # max action magnitude per axis
OBS_DIM     = 3 + K_NEIGHBORS * 3   # 3 + 6*3 = 21

SHAPES = ["grid", "circle", "line", "v"]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SwarmEnv2D(_BASE):
    """
    2-D per-drone MARL environment.

    Config keys (pass via RLlib env_config dict or constructor)
    -----------------------------------------------------------
    N            : int   number of drones  (default 20)
    distance     : float target spacing    (default 1.5)
    comm_range   : float connectivity radius (default 5.0)
    min_safe_dist: float collision threshold (default 0.5)
    max_steps    : int   episode length    (default 120)
    shape        : str   target formation shape (default "grid")
    step_size    : float attraction gain   (default 0.08)
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}

        self.N             = int(cfg.get("N",             20))
        self.distance      = float(cfg.get("distance",     1.5))
        self.comm_range    = float(cfg.get("comm_range",   5.0))
        self.min_safe_dist = float(cfg.get("min_safe_dist", 0.5))
        self.max_steps     = int(cfg.get("max_steps",    120))
        self.shape         = str(cfg.get("shape",        "grid"))
        self.step_size     = float(cfg.get("step_size",   0.08))

        # RLlib multi-agent bookkeeping
        self._agent_ids = set(range(self.N))

        single_obs   = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,),  dtype=np.float32)
        single_act   = spaces.Box(-MAX_V, MAX_V,   shape=(2,),        dtype=np.float32)
        self.observation_space = single_obs
        self.action_space      = single_act

        # State
        self.positions: np.ndarray = np.zeros((self.N, 2), dtype=np.float32)
        self.targets  : np.ndarray = np.zeros((self.N, 2), dtype=np.float32)
        self._t = 0

    # ------------------------------------------------------------------
    # Gymnasium / RLlib API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """
        options may contain:
            "N"       : override drone count (curriculum)
            "shape"   : override target shape  (curriculum)
            "distance": override target spacing (curriculum)
        """
        if options:
            if "N"        in options: self.N        = int(options["N"])
            if "shape"    in options: self.shape    = str(options["shape"])
            if "distance" in options: self.distance = float(options["distance"])
            self._agent_ids = set(range(self.N))

        self.targets   = generate_shape(
            self.shape, n_drones=self.N, distance=self.distance
        ).astype(np.float32)
        self.positions = make_initial_positions(self.N, dims=2, seed=seed)
        self.targets   = hungarian_assign_targets(self.positions, self.targets)
        self._t        = 0

        obs = self._build_obs()
        return obs, {}

    def step(self, action_dict: dict):
        # Collect actions; missing agents get zero action
        actions = np.array(
            [action_dict.get(i, np.zeros(2, dtype=np.float32))
             for i in range(self.N)],
            dtype=np.float32,
        )
        # Clip to action bounds
        actions = np.clip(actions, -MAX_V, MAX_V)

        # Physics step
        self.positions = step_2d(
            self.positions, self.targets, actions, self.step_size
        )
        self._t += 1
        done = self._t >= self.max_steps

        # Step reward (shared)
        reward, info = compute_reward(
            self.positions, self.targets,
            comm_range=self.comm_range,
            min_safe_dist=self.min_safe_dist,
        )

        # Terminal bonus
        if done:
            t_bonus, success = terminal_bonus(
                self.positions, self.targets,
                comm_range=self.comm_range,
                min_safe_dist=self.min_safe_dist,
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
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict:
        tree = cKDTree(self.positions)
        # Extra safety: K can't exceed N-1
        k_query = min(K_NEIGHBORS + 1, self.N)
        dists_all, idx_all = tree.query(self.positions, k=k_query)

        obs_dict = {}
        for i in range(self.N):
            rel_target     = self.targets[i] - self.positions[i]
            rel_target_dist = float(np.linalg.norm(rel_target))

            nn_feats = []
            # Skip index 0 (self), take up to K neighbours
            for k in range(1, k_query):
                nb   = idx_all[i, k]
                d    = dists_all[i, k]
                rel  = self.positions[nb] - self.positions[i]
                nn_feats.extend([rel[0], rel[1], d])

            # Pad with zeros if fewer than K neighbours
            while len(nn_feats) < K_NEIGHBORS * 3:
                nn_feats.extend([0.0, 0.0, 0.0])

            obs = np.array(
                [rel_target[0], rel_target[1], rel_target_dist] + nn_feats,
                dtype=np.float32,
            )
            obs_dict[i] = obs

        return obs_dict

    # ------------------------------------------------------------------
    # Curriculum helper (called by CurriculumCallback in train.py)
    # ------------------------------------------------------------------

    def set_curriculum(self, N: int, shape: str, distance: float):
        self.N        = N
        self.shape    = shape
        self.distance = distance
        self._agent_ids = set(range(N))
