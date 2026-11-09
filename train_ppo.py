"""v2 weekend 2: train the RL controller with PPO (Stable-Baselines3)
against vectorized JSBSim environments.

Why PPO: it's on-policy and clips its update step, which plays well with
a simulator this stiff -- JSBSim's dynamics can punish a too-aggressive
policy update with a cascade of crashed episodes that then poison the
next batch of data. Off-policy algorithms (SAC, TD3) would reuse old,
possibly-already-stale rollouts from a failed region of the policy; PPO
throws a batch away and tries a smaller step instead, which is a safer
default for a first RL pass on a system that already has a working
PID to fall back on.

Why vectorized envs: each JSBSim episode is CPU-bound (no GPU benefit),
so wall-clock training time scales directly with how many environments
run in parallel. SubprocVecEnv runs each in its own OS process -- real
parallelism, not threads fighting the GIL.
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from flight_env import C172Env
from domain_randomization import DomainRandomizedC172Env


def make_env(rank, seed=0):
    def _init():
        env = C172Env(seed=seed + rank)
        env = DomainRandomizedC172Env(env, seed=seed + rank)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--total-steps", type=int, default=20_000)
    ap.add_argument("--run-name", type=str, default="ppo_smoke")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = f"runs/{args.run_name}"
    os.makedirs(out_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    env = VecMonitor(env, filename=os.path.join(out_dir, "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,            # per env, before each update -> batch of n_steps * n_envs
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=args.seed,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.total_steps // 4 // args.n_envs, 1),
        save_path=out_dir,
        name_prefix="ppo_c172",
    )

    model.learn(total_timesteps=args.total_steps, callback=checkpoint_cb, progress_bar=False)
    model.save(os.path.join(out_dir, "final_model"))
    print(f"\nSaved final model to {out_dir}/final_model.zip")


if __name__ == "__main__":
    main()
