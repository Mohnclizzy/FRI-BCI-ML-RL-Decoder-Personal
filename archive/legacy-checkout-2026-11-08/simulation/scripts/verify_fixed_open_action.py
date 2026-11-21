"""One-step GPU check for the fixed-open nine-action playback task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import resolve_presets
    from fri_bci_sim.tasks.fixed_open import (
        FIXED_FINGER_TARGET_M, NearTableFixedOpenEnvCfg,
    )

    cfg = NearTableFixedOpenEnvCfg()
    cfg.play_mode()
    cfg.scene.num_envs = 1
    cfg.seed = 8
    cfg.sim.device = args.device
    resolve_presets(cfg, selected=("newton_mjwarp",))
    report = {"task": "FRI-Franka-Lift-NearTable-Progress-FixedOpen",
              "device": args.device, "training": False, "checkpoint": None}
    with launch_simulation(cfg.sim, args):
        env = gym.make(report["task"], cfg=cfg)
        try:
            env.reset()
            base = env.unwrapped
            robot = base.scene["robot"]
            action = torch.tensor([[0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -1.0, 0.7]],
                                  dtype=torch.float32, device=base.device)
            term = base.action_manager.get_term("action")
            joint_names = list(term._joint_names)
            term.process_actions(action)
            arm_expected = (term.processed_actions[:, :7]
                            + robot.data.joint_pos.torch[:, term._joint_ids][:, :7])
            term.apply_actions()
            applied_targets = tensor(robot.data.joint_pos_target)[0].detach().cpu().tolist()
            arm_applied = applied_targets[:7]
            fingers_applied = applied_targets[7:9]
            env.step(action)
            targets_after_step = tensor(robot.data.joint_pos_target)[0].detach().cpu().tolist()
            arm_expected_values = arm_expected[0].detach().cpu().tolist()
            report.update(action_shape=list(action.shape), action_joint_names=joint_names,
                          raw_action=action[0].detach().cpu().tolist(),
                          arm_expected_targets_m=arm_expected_values,
                          arm_applied_targets_m=arm_applied,
                          finger_applied_targets_m=fingers_applied,
                          targets_after_one_step_m=targets_after_step,
                          fixed_finger_target_m=FIXED_FINGER_TARGET_M)
            report["pass"] = (
                list(action.shape) == [1, 9]
                and all(abs(a-b) < 1e-6 for a, b in zip(arm_applied, arm_expected_values))
                and len(fingers_applied) == 2
                and all(abs(x-FIXED_FINGER_TARGET_M) < 1e-7 for x in fingers_applied)
                and all(abs(x-FIXED_FINGER_TARGET_M) < 1e-7 for x in targets_after_step[7:9])
            )
        finally:
            env.close()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise RuntimeError("Fixed-open action interface verification failed")


if __name__ == "__main__":
    main()
