"""
rl/train.py  (Ray 2.54 compatible — old API stack)
====================================================
MAPPO-style training for Drone Swarm AI using Ray RLlib PPO.

Architecture
------------
- ONE shared policy for all drone agents (parameter sharing).
- PPO with the *old* RLlib API stack
  (enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
  for maximum stability and compatibility with Ray 2.54.
- CurriculumCallback advances the stage each iteration.
- GPU used automatically if torch.cuda.is_available().

Run
---
    # 2-D (start here)
    python rl/train.py --mode 2d --iters 2000 --checkpoint-dir rl/checkpoints/2d

    # 3-D (after 2-D is stable)
    python rl/train.py --mode 3d --iters 3000 --checkpoint-dir rl/checkpoints/3d

    # Resume
    python rl/train.py --mode 2d --resume rl/checkpoints/2d/checkpoint_001000 --iters 2000

Dependencies
------------
    pip install -r requirements_rl.txt
"""

import os
import sys
import argparse
import json
import time
import warnings

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback

# Suppress Ray's many deprecation warnings so output stays readable
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    print("[train.py]  Ray / RLlib not found.  "
          "Install with:  pip install 'ray[rllib]>=2.9.0'")

from rl.curriculum import CurriculumScheduler


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpus() -> int:
    try:
        import torch
        return 1 if torch.cuda.is_available() else 0
    except ImportError:
        return 0


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(DefaultCallbacks):
    """
    After each training iteration:
    - Logs mean reward + active stage (every 10 iters).
    - Pushes updated curriculum config to all env workers.
    Compatible with Ray 2.54 old API stack.
    """

    def on_train_result(self, *, algorithm, result, **kwargs):
        iteration = result.get("training_iteration", 0)
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_reward_mean",
                    result.get("episode_reward_mean", float("nan")))

        mode      = getattr(algorithm, "_rl_mode", "2d")
        scheduler = CurriculumScheduler(mode=mode)
        stage     = scheduler.get_stage(iteration)
        cfg       = scheduler.sample_config(iteration)

        if iteration % 10 == 0:
            print(
                f"  [Curriculum] iter={iteration:04d}  "
                f"stage={stage.name:<28s}  mean_reward={mean_rew:+.3f}"
            )

        # Push new curriculum config to every env so the next episode
        # uses the right N, shape, and comm_range.
        def _update(env):
            try:
                env.reset(options={
                    "N"         : cfg["N"],
                    "shape"     : cfg["shape"],
                    "comm_range": cfg["comm_range"],
                    "distance"  : cfg["distance"],
                    "max_steps" : cfg["max_steps"],
                })
            except Exception:
                pass

        try:
            algorithm.workers.foreach_env(_update)
        except Exception:
            pass   # non-critical


# ---------------------------------------------------------------------------
# Build algorithm config
# ---------------------------------------------------------------------------

def build_config(
    mode               : str = "2d",
    num_rollout_workers: int = 2,
    num_gpus           : int = 0,
) -> "PPOConfig":
    """
    Build a PPOConfig using the OLD RLlib API stack (Ray 2.54 compatible).
    Uses num_epochs / minibatch_size for PPO update parameters.
    """
    assert HAS_RAY, "Ray RLlib required.  pip install 'ray[rllib]>=2.9.0'"

    if mode == "2d":
        from rl.env_2d import SwarmEnv2D as EnvClass
    else:
        from rl.env_3d import SwarmEnv3D as EnvClass

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

    config = (
        PPOConfig()
        # ── Use STABLE old API stack (required for Ray 2.x compatibility) ──
        .api_stack(
            enable_rl_module_and_learner       = False,
            enable_env_runner_and_connector_v2 = False,
        )
        .environment(env=EnvClass, env_config=env_config)
        .framework("torch")
        .env_runners(
            num_env_runners         = num_rollout_workers,
            rollout_fragment_length = "auto",
        )
        # Old-stack PPO training params
        .training(
            gamma             = 0.99,
            lr                = 3e-4,
            clip_param        = 0.2,
            vf_clip_param     = 10.0,
            entropy_coeff     = 0.01,
            num_epochs        = 10,    # PPO update epochs per train batch
            minibatch_size    = 256,   # mini-batch size within each SGD pass
            use_gae           = True,
            vf_loss_coeff     = 0.5,
            grad_clip         = 0.5,
            train_batch_size  = 2000,
            model             = {
                "fcnet_hiddens"   : [256, 256],
                "fcnet_activation": "tanh",
            },
        )
        .resources(num_gpus=num_gpus)
        .multi_agent(
            policies = {
                "shared_policy": PolicySpec(
                    observation_space = obs_space,
                    action_space      = act_space,
                )
            },
            policy_mapping_fn = lambda agent_id, episode, **kw: "shared_policy",
        )
        .callbacks(CurriculumCallback)
        .debugging(log_level="ERROR")
    )
    return config


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    mode               : str        = "2d",
    total_iters        : int        = 2000,
    checkpoint_dir     : str        = "rl/checkpoints",
    resume_path        : str | None = None,
    num_rollout_workers: int        = 2,
    use_gpu            : bool       = True,
    checkpoint_freq    : int        = 100,
):
    assert HAS_RAY, "Install ray[rllib] first."

    num_gpus = _detect_gpus() if use_gpu else 0
    print(f"[train] GPU available: {bool(num_gpus)}")

    ray.init(
        ignore_reinit_error = True,
        include_dashboard   = False,
        log_to_driver       = False,
        num_gpus            = num_gpus,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    config = build_config(
        mode                = mode,
        num_rollout_workers = num_rollout_workers,
        num_gpus            = num_gpus,
    )

    algo = config.build_algo()  # build() is deprecated in Ray 2.x
    algo._rl_mode = mode        # read by CurriculumCallback

    start_iter = 0
    if resume_path:
        print(f"[train] Resuming from {resume_path}")
        algo.restore(resume_path)
        start_iter = algo.iteration

    scheduler = CurriculumScheduler(mode=mode)
    print(scheduler.summary())
    print(
        f"\n[train] Starting MAPPO training - mode={mode.upper()}  "
        f"total_iters={total_iters}  workers={num_rollout_workers}  "
        f"GPU={'yes' if num_gpus else 'no'}\n"
    )

    best_reward   = -float("inf")
    metrics_log   = []

    def _extract_path(c):
        """Extract clean path string from Ray TrainingResult or Checkpoint."""
        if hasattr(c, "checkpoint") and c.checkpoint is not None:
            return getattr(c.checkpoint, "path", str(c.checkpoint))
        return getattr(c, "path", str(c))

    for i in range(start_iter, total_iters):
        t0     = time.time()
        result = algo.train()
        elapsed = time.time() - t0

        # Old API stack: reward keys are nested under 'env_runners'
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_reward_mean",
                    result.get("episode_reward_mean", float("nan")))
        max_rew  = env_runners.get("episode_reward_max",
                    result.get("episode_reward_max",  float("nan")))
        eps_len  = env_runners.get("episode_len_mean",
                    result.get("episode_len_mean",    float("nan")))

        stage = scheduler.get_stage(i)

        # Guard against NaN in round()
        def _safe_round(v, n):
            try: return round(float(v), n)
            except (TypeError, ValueError): return None

        log_entry = {
            "iter"       : i + 1,
            "stage"      : stage.name,
            "mean_reward": _safe_round(mean_rew, 4),
            "max_reward" : _safe_round(max_rew,  4),
            "mean_ep_len": _safe_round(eps_len,  1),
            "elapsed_s"  : round(elapsed, 2),
        }
        metrics_log.append(log_entry)

        # Write metrics file incrementally after every iteration
        with open(os.path.join(checkpoint_dir, f"training_metrics_{mode}.json"), "w") as f:
            json.dump(metrics_log, f, indent=2)

        if (i + 1) % 10 == 0:
            print(
                f"  iter={i+1:04d}  stage={stage.name:<28s}  "
                f"mean_r={mean_rew:+.3f}  ep_len={eps_len:.0f}  "
                f"t={elapsed:.1f}s"
            )

        # Checkpoint whenever reward improves or at scheduled frequency
        should_ckpt = (
            (i + 1) % checkpoint_freq == 0
            or (mean_rew == mean_rew and mean_rew > best_reward)  # not NaN
        )
        if should_ckpt:
            ckpt = algo.save(checkpoint_dir)
            # Ray 2.x returns TrainingResult; actual path is in .checkpoint.path
            ckpt_path = _extract_path(ckpt)
            print(f"  [ckpt] saved -> {ckpt_path}")
            if mean_rew == mean_rew and mean_rew > best_reward:
                best_reward = mean_rew

    # Final checkpoint
    final_ckpt = algo.save(checkpoint_dir)
    final_path = _extract_path(final_ckpt)
    print(f"\n[train] Done.  Final checkpoint: {final_path}")

    log_file = os.path.join(checkpoint_dir, f"training_metrics_{mode}.json")
    print(f"[train] Metrics log -> {log_file}")

    algo.stop()
    ray.shutdown()
    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO drone swarm policy")
    parser.add_argument("--mode",            default="2d",              choices=["2d", "3d"])
    parser.add_argument("--iters",           default=2000,  type=int)
    parser.add_argument("--checkpoint-dir",  default="rl/checkpoints")
    parser.add_argument("--resume",          default=None)
    parser.add_argument("--workers",         default=2,     type=int,
                        help="Rollout workers (0 = single-process, safest on Windows)")
    parser.add_argument("--no-gpu",          action="store_true")
    parser.add_argument("--checkpoint-freq", default=100,   type=int)
    args = parser.parse_args()

    train(
        mode                = args.mode,
        total_iters         = args.iters,
        checkpoint_dir      = args.checkpoint_dir,
        resume_path         = args.resume,
        num_rollout_workers = args.workers,
        use_gpu             = not args.no_gpu,
        checkpoint_freq     = args.checkpoint_freq,
    )
