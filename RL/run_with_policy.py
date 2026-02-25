"""
rl/run_with_policy.py
=====================
Drop-in runner: loads a trained RL checkpoint and drives the existing
simulation scripts (2-D or 3-D) using the learned per-drone policy.

The output format and CSV export match the originals so Unity / Blender
workflows are unaffected.

Usage
-----
    # 2-D sim with RL policy
    python rl/run_with_policy.py --checkpoint rl/checkpoints/2d/checkpoint_002000 \
                                  --mode 2d --n-drones 60 --shape grid

    # 3-D sim with RL policy (uses precomputed formation from sim_3d.py output)
    python rl/run_with_policy.py --checkpoint rl/checkpoints/3d/checkpoint_003000 \
                                  --mode 3d --n-drones 80 --shape circle

    # 3-D sim with image-derived formation
    python rl/run_with_policy.py --checkpoint rl/checkpoints/3d/checkpoint_003000 \
                                  --mode 3d --image input_images/516AaQ6o17L.webp \
                                  --base-drones 300
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from utils.evaluate_swarm import evaluate_formation
from rl.sim_core import (
    make_initial_positions,
    hungarian_assign_targets,
    step_2d,
    step_3d,
)


def load_policy(checkpoint_path: str, mode: str):
    """Restore RLlib algo and return a compute_action callable."""
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.policy.policy import PolicySpec
    except ImportError:
        raise RuntimeError(
            "ray[rllib] is required.  pip install 'ray[rllib]>=2.9.0'"
        )

    import warnings, os as _os
    _os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if mode == "2d":
        from rl.env_2d import SwarmEnv2D as EnvClass
        env_config = {"N": 10, "step_size": 0.08}
    else:
        from rl.env_3d import SwarmEnv3D as EnvClass
        env_config = {
            "N": 10, "step_size": 0.05,
            "repulsion_strength": 0.02, "repulsion_radius": 0.6,
        }

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    tmp_env   = EnvClass(config=env_config)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space

    # Must use old API stack to match training config
    algo = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner       = False,
            enable_env_runner_and_connector_v2 = False,
        )
        .environment(EnvClass, env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
        .multi_agent(
            policies={"shared_policy": PolicySpec(
                observation_space=obs_space,
                action_space=act_space,
            )},
            policy_mapping_fn=lambda agent_id, episode, **kw: "shared_policy",
        )
        .build_algo()
    )
    algo.restore(checkpoint_path)
    return algo


def run_2d_with_policy(
    algo,
    N: int,
    shape: str = "grid",
    distance: float = 1.5,
    comm_range: float = 5.0,
    num_frames: int = 120,
    seed: int = 42,
    visualise: bool = True,
):
    from utils.shape_generator import generate_shape

    targets   = generate_shape(shape, n_drones=N, distance=distance).astype(np.float32)
    positions = make_initial_positions(N, dims=2, seed=seed)
    targets   = hungarian_assign_targets(positions, targets)

    if visualise:
        plt.figure(figsize=(8, 8))

    for frame in range(num_frames):
        # --- build obs for every agent ---
        from rl.env_2d import SwarmEnv2D
        tmp = SwarmEnv2D.__new__(SwarmEnv2D)
        tmp.N         = N
        tmp.positions = positions
        tmp.targets   = targets
        obs_dict = tmp._build_obs()

        # --- compute actions ---
        action_dict = {}
        for agent_id, obs in obs_dict.items():
            action_dict[agent_id] = algo.compute_single_action(
                obs, policy_id="shared_policy", explore=False
            )

        actions   = np.array([action_dict[i] for i in range(N)], dtype=np.float32)
        positions = step_2d(positions, targets, actions, step_size=0.08)

        if visualise:
            plt.clf()
            plt.scatter(positions[:, 0], positions[:, 1],
                        c="blue", s=40, label="Drones")
            plt.scatter(targets[:, 0], targets[:, 1],
                        c="red", marker="x", s=50, label="Target")
            plt.title(f"RL-Controlled Swarm 2-D  (N={N}, frame={frame+1})")
            plt.xlabel("X"); plt.ylabel("Y")
            plt.legend(); plt.grid()
            plt.gca().set_aspect("equal")
            plt.pause(0.03)

    if visualise:
        plt.show()

    metrics = evaluate_formation(positions, targets, comm_range=comm_range)
    print("\n--- RL Policy Evaluation Metrics (2-D) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return positions, metrics


def run_3d_with_policy(
    algo,
    N: int,
    shape: str = "grid",
    distance: float = 1.5,
    comm_range: float = 6.0,
    num_frames: int = 150,
    seed: int = 42,
    visualise: bool = True,
    image_path: str | None = None,
    base_drones: int = 300,
):
    from mpl_toolkits.mplot3d import Axes3D

    # --- build targets ---
    if image_path:
        try:
            from utils.semantic_image_to_formation import image_to_semantic_outline
            from utils.formation_3d import lift_to_true_3d
            formation_2d = image_to_semantic_outline(
                image_path, n_drones=base_drones, scale_factor=6
            )
            targets = lift_to_true_3d(formation_2d, image_path,
                                       height_scale=10, layers=4)
            N       = len(targets)
            print(f"[run_3d] Image-derived formation: {N} drones")
        except Exception as e:
            print(f"[run_3d] Image pipeline failed ({e}), falling back to shape")
            image_path = None

    if not image_path:
        from utils.shape_generator import generate_shape
        pts2d   = generate_shape(shape, n_drones=N, distance=distance)
        targets = np.hstack([pts2d, np.zeros((N, 1))]).astype(np.float32)

    targets   = np.array(targets, dtype=np.float32)
    positions = make_initial_positions(N, dims=3, seed=seed)
    targets   = hungarian_assign_targets(positions, targets)

    if visualise:
        fig = plt.figure(figsize=(9, 9))
        ax  = fig.add_subplot(111, projection="3d")

    for frame in range(num_frames):
        from rl.env_3d import SwarmEnv3D
        tmp           = SwarmEnv3D.__new__(SwarmEnv3D)
        tmp.N         = N
        tmp.positions = positions
        tmp.targets   = targets
        obs_dict = tmp._build_obs()

        action_dict = {}
        for agent_id, obs in obs_dict.items():
            action_dict[agent_id] = algo.compute_single_action(
                obs, policy_id="shared_policy", explore=False
            )

        actions   = np.array([action_dict[i] for i in range(N)], dtype=np.float32)
        positions = step_3d(positions, targets, actions,
                             step_size=0.05,
                             repulsion_strength=0.02,
                             repulsion_radius=0.6)

        if visualise:
            ax.cla()
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       c="blue", s=8)
            ax.set_title(f"RL-Controlled Swarm 3-D  (N={N}, frame={frame+1})")
            ax.view_init(elev=30 + 10 * np.sin(frame / 20), azim=frame)
            plt.pause(0.03)

    if visualise:
        plt.show()

    metrics = evaluate_formation(positions, targets, comm_range=comm_range)
    print("\n--- RL Policy Evaluation Metrics (3-D) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs("output", exist_ok=True)
    import pandas as pd
    pd.DataFrame(positions, columns=["x", "y", "z"]).to_csv(
        "output/drone_positions_3d_rl.csv", index=False
    )
    print("RL 3-D positions exported to output/drone_positions_3d_rl.csv")
    return positions, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run drone swarm sim with trained RL policy"
    )
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--mode",        default="2d",  choices=["2d", "3d"])
    parser.add_argument("--n-drones",    default=40,    type=int)
    parser.add_argument("--shape",       default="grid",
                        choices=["grid", "circle", "line", "v"])
    parser.add_argument("--distance",    default=1.5,   type=float)
    parser.add_argument("--comm-range",  default=5.0,   type=float)
    parser.add_argument("--frames",      default=None,  type=int)
    parser.add_argument("--no-vis",      action="store_true")
    parser.add_argument("--image",       default=None,
                        help="[3-D only] image path for formation")
    parser.add_argument("--base-drones", default=300,   type=int,
                        help="[3-D image mode] base drone count")
    args = parser.parse_args()

    algo = load_policy(args.checkpoint, args.mode)

    if args.mode == "2d":
        frames = args.frames or 120
        run_2d_with_policy(
            algo,
            N          = args.n_drones,
            shape      = args.shape,
            distance   = args.distance,
            comm_range = args.comm_range,
            num_frames = frames,
            visualise  = not args.no_vis,
        )
    else:
        frames = args.frames or 150
        run_3d_with_policy(
            algo,
            N           = args.n_drones,
            shape       = args.shape,
            distance    = args.distance,
            comm_range  = args.comm_range,
            num_frames  = frames,
            visualise   = not args.no_vis,
            image_path  = args.image,
            base_drones = args.base_drones,
        )

    try:
        import ray
        ray.shutdown()
    except Exception:
        pass
