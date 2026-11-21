"""Verify the opt-in finger target guard on one policy step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    add_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import resolve_presets
    from fri_bci_sim.tasks.finger_guard import (
        FINGER_TARGET_MAX_M, FINGER_TARGET_MIN_M, NearTableFingerGuardEnvCfg,
    )

    cfg = NearTableFingerGuardEnvCfg()
    cfg.play_mode()
    cfg.scene.num_envs = 1
    cfg.seed = 8
    cfg.sim.device = args.device
    resolve_presets(cfg, selected=("newton_mjwarp",))
    report = {"task": "FRI-Franka-Lift-NearTable-Progress-FingerGuard",
              "training": False, "policy_checkpoint": None}
    with launch_simulation(cfg.sim, args):
        env = gym.make(report["task"], cfg=cfg)
        try:
            env.reset()
            base = env.unwrapped
            robot = base.scene["robot"]
            action = torch.zeros(env.action_space.shape, device=base.device)
            action[:, -2:] = -1.0
            env.step(action)
            targets = tensor(robot.data.joint_pos_target)[0, -2:].detach().cpu().tolist()
            joint_pos = tensor(robot.data.joint_pos)[0, -2:].detach().cpu().tolist()
            report.update(action_shape=list(action.shape), finger_raw_action=[-1.0, -1.0],
                          finger_joint_targets_m=targets, finger_joint_pos_m=joint_pos,
                          target_bounds_m=[FINGER_TARGET_MIN_M, FINGER_TARGET_MAX_M])
            report["pass"] = (list(action.shape) == [1, 9] and
                              all(FINGER_TARGET_MIN_M - 1e-6 <= x <= FINGER_TARGET_MAX_M + 1e-6
                                  for x in targets))
        finally:
            env.close()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise RuntimeError("Finger target guard did not bound the commanded target")


if __name__ == "__main__":
    main()
