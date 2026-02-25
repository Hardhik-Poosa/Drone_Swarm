import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

K_NEIGHBORS = 6
OBS_DIM     = 3 + K_NEIGHBORS * 3   # 21
MAX_V       = 0.2


def _make_targets(shape, N, distance):
    from utils.shape_generator import generate_shape
    return generate_shape(shape, n_drones=N, distance=distance).astype(np.float32)


class SwarmSB3Env(gym.Env):
    """
    Single-drone-perspective Gymnasium env for Stable Baselines 3.
    obs = (21,)  per-drone local observation  [shape-agnostic]
    act = (2,)   per-drone velocity delta
    The same policy is applied independently to every drone at inference time
    (parameter sharing / CTDE). Works for ANY swarm size without retraining.
    """
    metadata = {"render_modes": []}

    def __init__(self, env_config=None):
        super().__init__()
        cfg = env_config or {}
        self.N          = int(cfg.get("N", 20))
        self.distance   = float(cfg.get("distance", 1.5))
        self.comm_range = float(cfg.get("comm_range", 5.0))
        self.max_steps  = int(cfg.get("max_steps", 120))
        self.shape      = str(cfg.get("shape", "grid"))
        self.step_size  = float(cfg.get("step_size", 0.08))
        self.min_safe   = float(cfg.get("min_safe_dist", 0.5))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Box(-MAX_V, MAX_V,   shape=(2,),        dtype=np.float32)
        self.positions = np.zeros((self.N, 2), dtype=np.float32)
        self.targets   = np.zeros((self.N, 2), dtype=np.float32)
        self._t = 0
        self._focal = 0

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(seed)
        if options:
            if "shape"    in options: self.shape    = str(options["shape"])
            if "distance" in options: self.distance = float(options["distance"])
        self.targets   = _make_targets(self.shape, self.N, self.distance)
        self.positions = rng.uniform(-5, 5, size=(self.N, 2)).astype(np.float32)
        cost = np.linalg.norm(
            self.positions[:, None, :] - self.targets[None, :, :], axis=2)
        _, col = linear_sum_assignment(cost)
        self.targets = self.targets[col]
        self._t = 0
        self._focal = int(rng.integers(0, self.N))
        return self._obs(self._focal), {}

    def step(self, action):
        action = np.clip(action, -MAX_V, MAX_V).astype(np.float32)
        # Physics for ALL drones
        attraction = self.targets - self.positions
        repulsion  = np.zeros_like(self.positions)
        tree = cKDTree(self.positions)
        pairs = tree.query_pairs(r=self.min_safe * 3, output_type="ndarray")
        if len(pairs):
            i_idx, j_idx = pairs[:, 0], pairs[:, 1]
            diff  = self.positions[i_idx] - self.positions[j_idx]
            dist  = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
            force = diff / (dist * dist ** 2) * 0.02
            np.add.at(repulsion, i_idx,  force)
            np.add.at(repulsion, j_idx, -force)
        self.positions = (self.positions
                          + self.step_size * attraction
                          + repulsion).astype(np.float32)
        # Apply RL action to the focal drone
        self.positions[self._focal] += action
        self._t += 1
        done = self._t >= self.max_steps
        # Reward
        dists_tgt    = np.linalg.norm(self.positions - self.targets, axis=1)
        mean_err     = float(dists_tgt.mean())
        tree2        = cKDTree(self.positions)
        max_pairs    = self.N * (self.N - 1) / 2
        connectivity = len(tree2.query_pairs(r=self.comm_range)) / max(max_pairs, 1)
        unsafe       = len(tree2.query_pairs(r=self.min_safe))
        reward = -0.1 * mean_err + 0.5 * connectivity - 0.2 * (unsafe / max(max_pairs, 1))
        if done and mean_err < 0.5 and connectivity > 0.8:
            reward += 10.0
        self._focal = int(np.argmax(dists_tgt))
        return self._obs(self._focal), reward, done, False, {}

    def _obs(self, focal):
        tree    = cKDTree(self.positions)
        k_query = min(K_NEIGHBORS + 1, self.N)
        dists, idxs = tree.query(self.positions[focal:focal + 1], k=k_query)
        dists, idxs = dists[0], idxs[0]
        rel_target = self.targets[focal] - self.positions[focal]
        rel_dist   = float(np.linalg.norm(rel_target))
        nn_feats   = []
        for k in range(1, k_query):
            nb  = idxs[k]; d = float(dists[k])
            rel = self.positions[nb] - self.positions[focal]
            nn_feats.extend([float(rel[0]), float(rel[1]), d])
        while len(nn_feats) < K_NEIGHBORS * 3:
            nn_feats.extend([0.0, 0.0, 0.0])
        return np.array(
            [float(rel_target[0]), float(rel_target[1]), rel_dist] + nn_feats,
            dtype=np.float32,
        )

    def set_curriculum(self, shape, distance):
        self.shape = shape; self.distance = distance
