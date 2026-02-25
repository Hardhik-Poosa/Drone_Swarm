"""
RL/train_sb3.py
===============
Train the Drone Swarm controller using PPO from Stable Baselines 3.

Works with Python 3.13+ (no Ray required).

Usage
-----
    # 2D training — start here
    python RL/train_sb3.py --iters 500 --drones 20 --shape grid

    # Resume from checkpoint
    python RL/train_sb3.py --iters 500 --checkpoint RL/checkpoints/sb3/best_model

    # Evaluate a saved model
    python RL/train_sb3.py --eval --checkpoint RL/checkpoints/sb3/best_model
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum stages: (shape, target_spacing)
# N is FIXED throughout training so the observation space never changes.
# ─────────────────────────────────────────────────────────────────────────────
CURRICULUM = [
    ("grid",   1.8),
    ("grid",   1.5),
    ("circle", 1.5),
    ("line",   1.5),
    ("v",      1.5),
    ("grid",   1.2),
    ("circle", 1.2),
    ("v",      1.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Reward-logging callback
# ─────────────────────────────────────────────────────────────────────────────
class TrainingLogCallback(BaseCallback):
    """Print mean reward and episode stats every N steps."""

    def __init__(self, log_interval: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._ep_rewards: list[float] = []

    def _on_step(self) -> bool:
        # Collect episode rewards from SB3 info buffer
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_interval == 0 and self._ep_rewards:
            mean_r = float(np.mean(self._ep_rewards[-20:]))
            print(f"  [Step {self.num_timesteps:>8,}]  Mean episode reward: {mean_r:.4f}")

        return True   # continue training


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────
def make_env(N: int, shape: str, distance: float, seed: int = 0):
    """Return a Monitor-wrapped SwarmSB3Env factory function."""
    def _init():
        from RL.swarm_env_sb3 import SwarmSB3Env
        env = SwarmSB3Env({
            "N": N,
            "shape": shape,
            "distance": distance,
            "comm_range": 5.0,
            "max_steps": 120,
            "step_size": 0.08,
        })
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("  DRONE SWARM RL TRAINING  (Stable Baselines 3 / PPO)")
    print("=" * 60)
    print(f"  Timesteps      : {args.total_timesteps:,}")
    print(f"  Starting drones: {args.drones}")
    print(f"  Shape          : {args.shape}")
    print(f"  Checkpoint dir : {args.checkpoint_dir}")
    print("=" * 60)

    # Build vectorised env — N is fixed for the whole run
    N0 = args.drones
    env = DummyVecEnv([make_env(N0, args.shape, 1.5)])
    eval_env = DummyVecEnv([make_env(N0, args.shape, 1.5, seed=99)])

    # Build or load model
    if args.checkpoint and os.path.exists(args.checkpoint + ".zip"):
        print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.01,
            clip_range=0.2,
            verbose=0,
            # tensorboard_log requires tensorboard package; omit by default
        )

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.checkpoint_dir,
        eval_freq=max(args.total_timesteps // 20, 4096),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.total_timesteps // 10, 4096),
        save_path=args.checkpoint_dir,
        name_prefix="swarm_ppo",
    )
    log_cb = TrainingLogCallback(log_interval=4096)

    # Curriculum training: split timesteps across stages
    if args.curriculum:
        steps_per_stage = args.total_timesteps // len(CURRICULUM)
        print("\n[INFO] Curriculum training enabled (N fixed, shape/distance varies)")
        for stage_idx, (shape_cur, dist_cur) in enumerate(CURRICULUM):
            print(f"\n  Stage {stage_idx + 1}/{len(CURRICULUM)}: "
                  f"N={N0}, shape={shape_cur}, dist={dist_cur}")
            env_new  = DummyVecEnv([make_env(N0, shape_cur, dist_cur)])
            eval_new = DummyVecEnv([make_env(N0, shape_cur, dist_cur, seed=99)])
            model.set_env(env_new)
            eval_cb.eval_env = eval_new
            model.learn(
                total_timesteps=steps_per_stage,
                callback=[eval_cb, ckpt_cb, log_cb],
                reset_num_timesteps=(stage_idx == 0),
                progress_bar=True,
            )
    else:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_cb, ckpt_cb, log_cb],
            reset_num_timesteps=True,
            progress_bar=True,
        )

    final_path = os.path.join(args.checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"\n[OK] Training complete. Final model saved to {final_path}.zip")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(args):
    print(f"\n[INFO] Evaluating: {args.checkpoint}")
    from RL.swarm_env_sb3 import SwarmSB3Env
    env = SwarmSB3Env({
        "N": args.drones,
        "shape": args.shape,
        "max_steps": 120,
    })
    model = PPO.load(args.checkpoint)

    episode_rewards = []
    for ep in range(args.eval_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            done = terminated or truncated
        episode_rewards.append(total_r)
        print(f"  Episode {ep + 1}: reward = {total_r:.4f}")

    print(f"\n  Mean reward over {args.eval_episodes} episodes: "
          f"{np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Drone Swarm RL Training (Stable Baselines 3 PPO)"
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000,
                        help="Total env steps to train (default 200000)")
    parser.add_argument("--drones",     type=int, default=20,
                        help="Number of drones (default 20)")
    parser.add_argument("--shape",      type=str, default="grid",
                        choices=["grid", "circle", "v", "line"],
                        help="Target formation shape (default grid)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="RL/checkpoints/sb3",
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (without .zip)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum training (harder stages over time)")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation instead of training")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes for evaluation (default 10)")
    args = parser.parse_args()

    if args.eval:
        if not args.checkpoint:
            parser.error("--checkpoint required for --eval")
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
