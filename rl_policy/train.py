"""Train a task-conditioned PPO policy in the small CPU proxy environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl_policy.env import GraspHoldEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--error-rest", type=float, default=0.0,
                        help="Synthetic probability that REST is decoded as HAND_IMAGERY")
    parser.add_argument("--error-imagery", type=float, default=0.0,
                        help="Synthetic probability that HAND_IMAGERY is decoded as REST")
    parser.add_argument("--output", type=Path, default=Path("outputs/rl_policy/ppo_clean.zip"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    env = GraspHoldEnv(args.error_rest, args.error_imagery)
    try:
        check_env(env, warn=True)
        env.reset(seed=args.seed)
        model = PPO("MlpPolicy", env, seed=args.seed, n_steps=256, batch_size=64,
                    learning_rate=3e-4, gamma=0.98, verbose=1, device="cpu")
        model.learn(total_timesteps=args.steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output)
        print(f"Saved {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
