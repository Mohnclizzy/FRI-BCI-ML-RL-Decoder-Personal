"""Inspect one near-table reset and its first policy step without training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def values(value):
    return tensor(value).detach().cpu().tolist()


def snapshot(base):
    robot, cube = base.scene["robot"], base.scene["object"]
    body = {}
    for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
        ids, _ = robot.find_bodies(name)
        if len(ids) != 1:
            raise RuntimeError(f"Expected one {name} body, found {len(ids)}")
        body[name] = values(robot.data.body_pose_w[0, ids[0]])
    contact = []
    for name in ("panda_leftfinger_object_s", "panda_rightfinger_object_s"):
        matrix = base.scene.sensors[name].data.normal_force_matrix_w
        contact.append(None if matrix is None else float(tensor(matrix)[0].norm().item()))
    return {
        "cube_pose_w": values(cube.data.root_pose_w[0]),
        "body_pose_w": body,
        "joint_names": list(robot.data.joint_names),
        "joint_pos": values(robot.data.joint_pos[0]),
        "finger_contact_n": contact,
    }


def assess(before, after):
    cube = before["cube_pose_w"][:3]
    fingers = [before["body_pose_w"][name][:3] for name in ("panda_leftfinger", "panda_rightfinger")]
    finger_joint_pos = [q for name, q in zip(before["joint_names"], before["joint_pos"])
                        if "finger_joint" in name]
    top = cube[2] + 0.025
    result = {
        "cube_center_z_m": cube[2],
        "cube_top_z_m": top,
        "finger_frame_z_clearance_above_cube_top_m": [point[2] - top for point in fingers],
        "finger_frame_lateral_distance_from_cube_center_m": [math.dist(point[:2], cube[:2]) for point in fingers],
        "finger_joint_pos_m": finger_joint_pos,
        "cube_first_step_jump_m": math.dist(cube, after["cube_pose_w"][:3]),
        "hand_first_step_jump_m": math.dist(before["body_pose_w"]["panda_hand"][:3],
                                             after["body_pose_w"]["panda_hand"][:3]),
    }
    result["geometry_screen_pass"] = (
        abs(cube[2] - 0.28) <= .005 and len(finger_joint_pos) == 2
        and min(finger_joint_pos) >= .035
        and min(result["finger_frame_z_clearance_above_cube_top_m"]) >= .005
        and min(result["finger_frame_lateral_distance_from_cube_center_m"]) >= .03
        and all(force is not None and force < .1 for force in before["finger_contact_n"])
        and result["cube_first_step_jump_m"] <= .02
        and result["hand_first_step_jump_m"] <= .05
    )
    result["limitation"] = "Finger-frame clearance is a diagnostic geometry screen, not collision proof."
    return result


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--play-mode", action="store_true", help="Apply the playback reset configuration")
    add_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"task": "FRI-Franka-Lift-NearTable", "seed": args.seed,
              "play_mode": args.play_mode,
              "training": False, "policy_checkpoint": None, "error": None}
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import resolve_presets
        from fri_bci_sim.tasks.near_table import NearTableLiftEnvCfg

        cfg = NearTableLiftEnvCfg()
        if args.play_mode:
            cfg.play_mode()
        cfg.scene.num_envs = 1
        cfg.seed = args.seed
        cfg.sim.device = args.device
        resolve_presets(cfg, selected=("newton_mjwarp",))
        report["task_config_module"] = NearTableLiftEnvCfg.__module__
        with launch_simulation(cfg.sim, args):
            try:
                env = gym.make("FRI-Franka-Lift-NearTable", cfg=cfg)
                env.reset()
                base = env.unwrapped
                before = snapshot(base)
                action = torch.zeros(env.action_space.shape, device=base.device)
                env.step(action)
                after = snapshot(base)
                report.update(initial=before, first_policy_step=after,
                              assessment=assess(before, after),
                              action_dim=action.shape[-1],
                              observation_dim=base.observation_manager.group_obs_dim["policy"])
            finally:
                if env is not None:
                    env.close()
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        (args.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if not report["assessment"]["geometry_screen_pass"]:
        raise RuntimeError("Near-table reset geometry or first-step continuity failed")
    print(json.dumps(report["assessment"]), flush=True)


if __name__ == "__main__":
    main()
