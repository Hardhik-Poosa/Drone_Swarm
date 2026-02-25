"""
rl/evaluate.py
==============
Benchmark RL policy against the two existing non-RL baselines in the project.

Baselines
---------
1. Scripted controller   — the original attraction-only physics from
                           analysis/swarm_core.run_simulation()
2. Spacing optimizer     — linear-regression-based spacing predictor from
                           analysis/train_connectivity_model.py (if model
                           file exists), then runs scripted controller with
                           the predicted optimal spacing.

RL policy                — loads a trained RLlib checkpoint and runs the
                           full MARL episode with learned per-drone actions.

Metrics reported (matching utils/evaluate_swarm.py)
----------------------------------------------------
- Connectivity Ratio
- Minimum Inter-Drone Distance
- Convergence Error
- Episode success rate (connectivity >= 0.95 AND min_dist >= 0.5)

Run
---
    python rl/evaluate.py --checkpoint rl/checkpoints/2d/checkpoint_002000 \
                           --mode 2d --scenarios 50

    python rl/evaluate.py --checkpoint rl/checkpoints/3d/checkpoint_003000 \
                           --mode 3d --scenarios 30
"""

import os
import sys
import argparse
import json
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from utils.shape_generator    import generate_shape
from utils.evaluate_swarm     import evaluate_formation
from utils.network_metrics    import is_fully_connected
from analysis.swarm_core      import run_simulation
from rl.sim_core              import (
    make_initial_positions,
    hungarian_assign_targets,
    step_2d,
    step_3d,
    fast_connectivity_ratio,
    fast_min_distance,
)
from rl.curriculum import CurriculumScheduler


# ---------------------------------------------------------------------------
# Helper: scripted baseline
# ---------------------------------------------------------------------------

def run_scripted_2d(N: int, distance: float, comm_range: float, seed: int = 42):
    """Runs the original 2-D scripted controller (swarm_core.run_simulation)."""
    metrics, _ = run_simulation(N, distance=distance, comm_range=comm_range)
    return metrics


def run_scripted_manual_2d(
    N: int, distance: float, comm_range: float, seed: int = 42,
    step_size: float = 0.08, num_frames: int = 120
) -> dict:
    """
    Pure NumPy re-implementation of swarm_core.run_simulation so we can
    share the exact same initial positions as the RL evaluation for a
    fair comparison.
    """
    from scipy.optimize import linear_sum_assignment

    targets   = generate_shape("grid", n_drones=N, distance=distance).astype(np.float32)
    positions = make_initial_positions(N, dims=2, seed=seed)

    # Hungarian assignment
    cost     = np.linalg.norm(positions[:, None, :] - targets[None, :, :], axis=2)
    _, c_ind = linear_sum_assignment(cost)
    targets  = targets[c_ind]

    for _ in range(num_frames):
        positions += step_size * (targets - positions)

    return evaluate_formation(positions, targets, comm_range=comm_range)


def run_scripted_manual_3d(
    N: int, distance: float, comm_range: float, seed: int = 42,
    step_size: float = 0.05, repulsion_strength: float = 0.02,
    repulsion_radius: float = 0.6, num_frames: int = 150
) -> dict:
    targets   = np.hstack([
        generate_shape("grid", n_drones=N, distance=distance),
        np.zeros((N, 1))
    ]).astype(np.float32)
    positions = make_initial_positions(N, dims=3, seed=seed)
    targets   = hungarian_assign_targets(positions, targets)

    for _ in range(num_frames):
        positions = step_3d(
            positions, targets,
            actions            = np.zeros_like(positions),
            step_size          = step_size,
            repulsion_strength = repulsion_strength,
            repulsion_radius   = repulsion_radius,
        )
    return evaluate_formation(positions, targets, comm_range=comm_range)


# ---------------------------------------------------------------------------
# Helper: spacing-optimizer baseline
# ---------------------------------------------------------------------------

def run_spacing_optimizer_2d(N: int, comm_range: float) -> dict | None:
    """
    Load the connectivity regression model and find the optimal spacing,
    then run the scripted controller with that spacing.
    Returns None if the model file does not exist.
    """
    model_file = os.path.join(PROJECT_ROOT, "analysis", "connectivity_model.pkl")
    if not os.path.exists(model_file):
        return None

    try:
        import joblib
        model = joblib.load(model_file)
    except Exception:
        return None

    best_d = None
    for d in np.linspace(1.0, 6.0, 50):
        predicted = model.predict([[N, d, comm_range]])[0]
        if predicted >= 0.8:
            best_d = d
            break

    if best_d is None:
        best_d = 1.5   # fallback

    return run_scripted_manual_2d(N, distance=best_d, comm_range=comm_range)


# ---------------------------------------------------------------------------
# Helper: RL policy rollout
# ---------------------------------------------------------------------------

def _load_rl_algo(checkpoint_path: str, mode: str):
    """
    Build a PPO algo using the old API stack and restore from checkpoint.
    Ray must already be initialised before calling this.
    Returns the restored algo (keep alive for the whole evaluation run).
    """
    try:
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.policy.policy import PolicySpec
    except ImportError:
        raise RuntimeError("ray[rllib] required for RL evaluation.")

    if mode == "2d":
        from rl.env_2d import SwarmEnv2D as EnvClass
    else:
        from rl.env_3d import SwarmEnv3D as EnvClass

    from rl.curriculum import CurriculumScheduler
    scheduler  = CurriculumScheduler(mode=mode)
    init_stage = scheduler.get_stage(0)

    env_config = {
        "N"            : init_stage.N_range[0],
        "shape"        : init_stage.shapes[0],
        "comm_range"   : init_stage.comm_range,
        "distance"     : init_stage.distance,
        "max_steps"    : init_stage.max_steps,
        "min_safe_dist": 0.5,
        "step_size"    : 0.08 if mode == "2d" else 0.05,
    }
    if mode == "3d":
        env_config["repulsion_strength"] = 0.02
        env_config["repulsion_radius"]   = 0.6

    tmp_env   = EnvClass(config=env_config)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space

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


def run_rl_policy(
    checkpoint_path: str,
    mode: str,
    N: int,
    distance: float,
    comm_range: float,
    seed: int = 42,
    algo=None,          # pass pre-built algo to avoid rebuilding every scenario
) -> dict:
    """
    Run a single rollout episode with the trained RL policy.
    If `algo` is None, builds and restores from checkpoint (slow).
    Prefer passing a pre-built algo from _load_rl_algo() for batch evaluation.
    Returns evaluate_formation() metrics dict.
    """
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.policy.policy import PolicySpec
    except ImportError:
        raise RuntimeError("ray[rllib] required for RL evaluation.")

    if mode == "2d":
        from rl.env_2d import SwarmEnv2D as EnvClass
    else:
        from rl.env_3d import SwarmEnv3D as EnvClass

    import warnings, os as _os
    _os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    owns_ray = False
    if algo is None:
        # Fallback: build fresh (single-call usage)
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        owns_ray = True
        algo = _load_rl_algo(checkpoint_path, mode)

    env_config = {
        "N": N, "distance": distance, "comm_range": comm_range,
        "min_safe_dist": 0.5,
        "step_size": 0.08 if mode == "2d" else 0.05,
    }
    if mode == "3d":
        env_config["repulsion_strength"] = 0.02
        env_config["repulsion_radius"]   = 0.6

    # Run one episode
    env = EnvClass(config=env_config)
    obs, _ = env.reset(seed=seed)

    done = False
    while not done:
        action_dict = {}
        for agent_id, agent_obs in obs.items():
            action = algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=False,
            )
            action_dict[agent_id] = action
        obs, _, terminateds, _, info = env.step(action_dict)
        done = terminateds.get("__all__", False)

    metrics = evaluate_formation(
        env.positions, env.targets, comm_range=comm_range
    )
    if owns_ray:
        algo.stop()
        ray.shutdown()
    return metrics


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str | None,
    mode: str = "2d",
    num_scenarios: int = 50,
    seed_base: int = 0,
    results_dir: str = "rl/eval_results",
):
    os.makedirs(results_dir, exist_ok=True)

    scheduler = CurriculumScheduler(mode=mode)
    # Use the hardest stage for evaluation
    eval_stage = scheduler._stages[-1]

    rng    = np.random.default_rng(seed_base)
    Ns     = rng.integers(eval_stage.N_range[0], eval_stage.N_range[1] + 1,
                          size=num_scenarios)
    shapes = rng.choice(eval_stage.shapes, size=num_scenarios)

    agg: dict[str, list] = {
        "scripted"  : [],
        "spacing_opt": [],
        "rl_policy" : [],
    }
    SUCCESS_CONN  = 0.95
    SUCCESS_DIST  = 0.5
    SUCCESS_CONV  = 0.5

    print(f"\n{'='*60}")
    print(f" Evaluation: mode={mode.upper()}  scenarios={num_scenarios}")
    print(f"{'='*60}")

    # Load RL algo once (expensive) and reuse for all scenarios
    rl_algo = None
    if checkpoint_path:
        try:
            import ray
            import warnings, os as _os
            _os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
            rl_algo = _load_rl_algo(checkpoint_path, mode)
            print(f"[eval] RL checkpoint loaded from {checkpoint_path}")
        except Exception as e:
            warnings.warn(f"Could not load RL checkpoint: {e}")
            rl_algo = None

    try:
      for idx in range(num_scenarios):
        N          = int(Ns[idx])
        distance   = eval_stage.distance
        comm_range = eval_stage.comm_range
        seed       = seed_base + idx

        print(f"  Scenario {idx+1:03d}/{num_scenarios}  N={N}  shape={shapes[idx]}", end="  ")

        # 1 — Scripted
        if mode == "2d":
            m_sc = run_scripted_manual_2d(N, distance, comm_range, seed)
        else:
            m_sc = run_scripted_manual_3d(N, distance, comm_range, seed)
        agg["scripted"].append(m_sc)

        # 2 — Spacing optimizer (2-D only)
        if mode == "2d":
            m_opt = run_spacing_optimizer_2d(N, comm_range)
            if m_opt is not None:
                agg["spacing_opt"].append(m_opt)

        # 3 — RL policy (reuse pre-built algo)
        if rl_algo is not None:
            try:
                m_rl = run_rl_policy(
                    checkpoint_path, mode, N, distance, comm_range, seed,
                    algo=rl_algo,
                )
                agg["rl_policy"].append(m_rl)
            except Exception as e:
                warnings.warn(f"RL evaluation failed for scenario {idx+1}: {e}")

        print(
            f"scripted_conn={m_sc['Connectivity Ratio']:.3f}  "
            f"scripted_min_d={m_sc['Minimum Inter-Drone Distance']:.3f}"
        )
    finally:
      if rl_algo is not None:
        try:
            rl_algo.stop()
            ray.shutdown()
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # Aggregate & print
    # ---------------------------------------------------------------------------
    report = {}
    for baseline, results in agg.items():
        if not results:
            continue
        keys = ["Connectivity Ratio", "Minimum Inter-Drone Distance",
                "Convergence Error", "Average Inter-Drone Distance"]
        stats = {}
        for k in keys:
            vals  = [r[k] for r in results if k in r]
            stats[k] = {
                "mean": float(np.mean(vals)),
                "std" : float(np.std(vals)),
                "min" : float(np.min(vals)),
                "max" : float(np.max(vals)),
            }
        # Success rate
        successes = [
            r["Connectivity Ratio"]             >= SUCCESS_CONN
            and r["Minimum Inter-Drone Distance"] >= SUCCESS_DIST
            and r["Convergence Error"]            <= SUCCESS_CONV
            for r in results
        ]
        stats["success_rate"] = float(np.mean(successes))
        report[baseline] = stats

    # Print table
    print(f"\n{'='*60}")
    print(f" Results  (N in {eval_stage.N_range}, comm_range={eval_stage.comm_range})")
    print(f"{'='*60}")
    header = f"{'Baseline':<20s}  {'Connectivity':>12s}  {'Min Dist':>9s}  {'Conv Error':>10s}  {'Success%':>9s}"
    print(header)
    print("-" * len(header))
    for bname, stats in report.items():
        print(
            f"{bname:<20s}  "
            f"{stats['Connectivity Ratio']['mean']:>12.3f}  "
            f"{stats['Minimum Inter-Drone Distance']['mean']:>9.3f}  "
            f"{stats['Convergence Error']['mean']:>10.3f}  "
            f"{100*stats['success_rate']:>8.1f}%"
        )

    # Save
    out_file = os.path.join(results_dir, f"eval_{mode}.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[eval] Full results saved to {out_file}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL vs baselines")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to RLlib checkpoint dir (optional; skips RL if not given)")
    parser.add_argument("--mode",       default="2d",  choices=["2d", "3d"])
    parser.add_argument("--scenarios",  default=50,    type=int)
    parser.add_argument("--seed",       default=0,     type=int)
    parser.add_argument("--results-dir",default="rl/eval_results")
    args = parser.parse_args()

    evaluate(
        checkpoint_path = args.checkpoint,
        mode            = args.mode,
        num_scenarios   = args.scenarios,
        seed_base       = args.seed,
        results_dir     = args.results_dir,
    )
