"""
RL/rl_controller.py
===================
Loads a trained PPO (Stable Baselines 3) checkpoint and runs per-frame
RL inference on ANY set of drone positions + target formation.

This is what connects the trained RL policy to the live pipeline.
It is completely shape-agnostic — it works whether targets come from
a grid, circle, or Ronaldo's silhouette.

Key design
----------
- Policy input (per drone): [dx, dy, dist_to_target, ...6 neighbour features]
  Total = 21 floats per drone (same as SwarmEnv2D.OBS_DIM = 21)
- Policy output (per drone): [dvx, dvy]  velocity delta in XY
- For 3D pipelines: XY movement = RL policy, Z = physics (attraction only)
- Repulsion is vectorised: O(N log N) via cKDTree
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from scipy.spatial import cKDTree

# ── Constants (must match env_2d.py) ──────────────────────────────────────────
K_NEIGHBORS = 6
OBS_DIM     = 3 + K_NEIGHBORS * 3   # 21
MAX_V       = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Observation builder  (mirrors SwarmEnv2D._build_obs exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _build_obs_flat(positions_2d: np.ndarray,
                    targets_2d: np.ndarray) -> np.ndarray:
    """
    Build a flat (N * OBS_DIM,) observation vector from current XY positions
    and XY targets.  Exactly mirrors SwarmEnv2D._build_obs so the loaded
    policy sees inputs it was trained on.
    """
    N = len(positions_2d)
    tree = cKDTree(positions_2d)
    k_query = min(K_NEIGHBORS + 1, N)
    dists_all, idx_all = tree.query(positions_2d, k=k_query)

    flat = np.zeros(N * OBS_DIM, dtype=np.float32)

    for i in range(N):
        rel_target = targets_2d[i] - positions_2d[i]
        rel_dist   = float(np.linalg.norm(rel_target))

        nn_feats = []
        for k in range(1, k_query):
            nb   = idx_all[i, k]
            d    = dists_all[i, k]
            rel  = positions_2d[nb] - positions_2d[i]
            nn_feats.extend([float(rel[0]), float(rel[1]), float(d)])

        while len(nn_feats) < K_NEIGHBORS * 3:
            nn_feats.extend([0.0, 0.0, 0.0])

        row = np.array(
            [float(rel_target[0]), float(rel_target[1]), rel_dist] + nn_feats,
            dtype=np.float32,
        )
        flat[i * OBS_DIM: i * OBS_DIM + OBS_DIM] = row

    return flat


# ─────────────────────────────────────────────────────────────────────────────
# RLController class
# ─────────────────────────────────────────────────────────────────────────────
class RLController:
    """
    Wraps a saved PPO model and exposes a single method:
        step(positions_3d, targets_3d) -> new_positions_3d

    For 3D: XY movement comes from the RL policy; Z is pure attraction
    physics (depth is not in the 2D training obs, so the policy doesn't
    know about Z).
    """

    def __init__(self, checkpoint_path: str):
        """
        Parameters
        ----------
        checkpoint_path : str  path to .zip file (with or without '.zip')
        """
        from stable_baselines3 import PPO
        path = checkpoint_path
        if not path.endswith(".zip"):
            path = path + ".zip"
        if not os.path.exists(path):
            raise FileNotFoundError(f"RL checkpoint not found: {path}")
        self.model = PPO.load(path)
        print(f"[RL] Loaded policy from {path}")

    def step(self,
             positions_3d: np.ndarray,
             targets_3d: np.ndarray,
             step_size: float = 0.05,
             repulsion_strength: float = 0.015,
             repulsion_radius: float = 0.5,
             z_step_size: float = 0.08) -> np.ndarray:
        """
        One simulation frame driven by the RL policy.

        XY axes: RL policy action + attraction base physics + repulsion
        Z  axis: attraction physics only

        Parameters
        ----------
        positions_3d : (N, 3)
        targets_3d   : (N, 3)
        step_size    : attraction gain for XY
        repulsion_strength : repulsion weight
        repulsion_radius   : repulsion cutoff distance
        z_step_size  : attraction gain for Z (larger = faster depth lock)

        Returns
        -------
        new_positions : (N, 3) float32
        """
        N = len(positions_3d)

        # XY sub-problem: ─────────────────────────────────────
        pos_xy = positions_3d[:, :2]   # (N, 2)
        tgt_xy = targets_3d[:, :2]     # (N, 2)

        # Build observation and get RL actions — run PER DRONE so N is unlimited
        # Each drone's obs is (OBS_DIM=21,); policy outputs (2,) action.
        # This makes the policy completely N-agnostic despite being trained on N=20.
        tree_xy = cKDTree(pos_xy)
        k_query = min(K_NEIGHBORS + 1, N)
        dists_all, idx_all = tree_xy.query(pos_xy, k=k_query)

        actions_xy = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            rel_target = tgt_xy[i] - pos_xy[i]
            rel_dist   = float(np.linalg.norm(rel_target))
            nn_feats = []
            for k in range(1, k_query):
                nb  = idx_all[i, k]
                d   = dists_all[i, k]
                rel = pos_xy[nb] - pos_xy[i]
                nn_feats.extend([float(rel[0]), float(rel[1]), float(d)])
            while len(nn_feats) < K_NEIGHBORS * 3:
                nn_feats.extend([0.0, 0.0, 0.0])
            obs_i = np.array(
                [float(rel_target[0]), float(rel_target[1]), rel_dist] + nn_feats,
                dtype=np.float32,
            )
            act_i, _ = self.model.predict(obs_i, deterministic=True)
            actions_xy[i] = np.clip(act_i[:2], -MAX_V, MAX_V)

        # XY physics step: attraction + RL correction + repulsion
        attraction_xy = tgt_xy - pos_xy

        # Vectorised repulsion (O(N log N))
        repulsion_xy = np.zeros_like(pos_xy)
        if N > 1:
            tree = cKDTree(positions_3d)   # 3D distance for repulsion
            pairs = tree.query_pairs(r=repulsion_radius, output_type="ndarray")
            if len(pairs) > 0:
                i_idx, j_idx = pairs[:, 0], pairs[:, 1]
                diff = positions_3d[i_idx] - positions_3d[j_idx]
                dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
                force_3d = diff / (dist * dist ** 2)     # 1/d² direction
                np.add.at(repulsion_xy, i_idx, force_3d[:, :2] * repulsion_strength)
                np.add.at(repulsion_xy, j_idx, -force_3d[:, :2] * repulsion_strength)

        new_xy = pos_xy + step_size * attraction_xy + actions_xy + repulsion_xy

        # Z sub-problem: pure attraction (policy has no depth info) ──────────
        pos_z  = positions_3d[:, 2:3]
        tgt_z  = targets_3d[:, 2:3]
        new_z  = pos_z + z_step_size * (tgt_z - pos_z)

        new_positions = np.concatenate([new_xy, new_z], axis=1).astype(np.float32)
        return new_positions


# ─────────────────────────────────────────────────────────────────────────────
# Quick-train helper (called by pipeline if no checkpoint found)
# ─────────────────────────────────────────────────────────────────────────────
def auto_train(checkpoint_dir: str,
               total_timesteps: int = 300_000,
               logger=None) -> str:
    """
    Train a PPO model from scratch and save to checkpoint_dir.
    Returns path to saved model.
    """
    import RL   # ensure rl alias is registered
    from RL.train_sb3 import make_env
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    os.makedirs(checkpoint_dir, exist_ok=True)
    _log(f"[RL] No checkpoint found — auto-training for {total_timesteps:,} steps...")
    _log(f"[RL] This runs once and saves to {checkpoint_dir}/best_model.zip")

    # Curriculum stages (fixed N, varying shape/distance)
    stages = [
        ("grid",   1.5, total_timesteps // 4),
        ("circle", 1.5, total_timesteps // 4),
        ("v",      1.5, total_timesteps // 4),
        ("grid",   1.2, total_timesteps // 4),
    ]
    N_FIXED = 20   # fixed throughout so obs space never changes

    env      = DummyVecEnv([make_env(N_FIXED, "grid", 1.5)])
    eval_env = DummyVecEnv([make_env(N_FIXED, "grid", 1.5, seed=99)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=0,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=max(total_timesteps // 20, 2048),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(total_timesteps // 5, 2048),
        save_path=checkpoint_dir,
        name_prefix="swarm_ppo",
    )

    first = True
    for (shape_cur, dist_cur, steps) in stages:
        _log(f"[RL]   Stage: N={N_FIXED}, shape={shape_cur}, steps={steps:,}")
        env_new  = DummyVecEnv([make_env(N_FIXED, shape_cur, dist_cur)])
        eval_new = DummyVecEnv([make_env(N_FIXED, shape_cur, dist_cur, seed=99)])
        model.set_env(env_new)
        eval_cb.eval_env = eval_new
        model.learn(
            total_timesteps=steps,
            callback=[eval_cb, ckpt_cb],
            reset_num_timesteps=first,
            progress_bar=True,
        )
        first = False

    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_path)
    _log(f"[RL] Training complete. Saved to {final_path}.zip")

    # Return best_model if it was saved, else final
    best = os.path.join(checkpoint_dir, "best_model.zip")
    return best if os.path.exists(best) else final_path + ".zip"
