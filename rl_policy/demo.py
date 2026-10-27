"""Render a short side-by-side GIF of learned HOLD and GRASP episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from rl_policy import HAND_IMAGERY, REST
from rl_policy.env import GraspHoldEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/rl_policy/demo.gif"))
    args = parser.parse_args()
    model = PPO.load(args.model, device="cpu")
    envs = [GraspHoldEnv(), GraspHoldEnv()]
    observations = []
    done = [False, False]
    try:
        for env, intent in zip(envs, (REST, HAND_IMAGERY)):
            obs, _ = env.reset(seed=0, options={"true_intent": intent, "command": intent})
            observations.append(obs)
        frames = []
        for step in range(9):
            frame = Image.new("RGB", (640, 270), "white")
            draw = ImageDraw.Draw(frame)
            for i, (env, label) in enumerate(zip(envs, ("REST: hold", "HAND IMAGERY: grasp"))):
                frame.paste(Image.fromarray(env.render()), (320 * i, 30))
                draw.text((320 * i + 8, 8), f"{label} | step {step}", fill="black")
            frames.append(frame)
            for i, env in enumerate(envs):
                if not done[i]:
                    action, _ = model.predict(observations[i], deterministic=True)
                    observations[i], _, terminated, truncated, _ = env.step(action)
                    done[i] = terminated or truncated
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=450, loop=0)
        print(f"Saved {args.output}")
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
