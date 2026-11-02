"""Bounded scripted Franka grasp/lift diagnostic on the fixed table-cube scene.

The cube is never attached or pose-controlled. Only arm and finger joint targets
are sent. JSONL records physical object state and filtered finger contact.
"""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=23.0)
parser.add_argument("--position-only", action="store_true", help="Diagnose translation tracking without orientation")
parser.add_argument("--plan-only", action="store_true", help="Solve and report joint waypoints without physics playback")
parser.add_argument("--joint-waypoints", action="store_true", help="Play measured joint-space lift reference")
parser.add_argument("--decisions", type=Path, help="compare-commands JSONL; enables recorded EEG command replay")
parser.add_argument("--max-estimate-age-s", type=float, default=1.0)
parser.add_argument("--telemetry-every-steps", type=int, default=1, help="Physics-step sampling interval; default records every step")
parser.add_argument(
    "--condition",
    choices=("raw", "confidence", "temporal", "always-hold", "scripted-positive", "scripted-pause"),
    default="temporal",
    help="EEG gate or matched robot control; positive control requires no decision trace",
)
add_launcher_args(parser)
args = parser.parse_args()

from fri_bci_sim.executor import DecisionReplay, ExecutionMonitor, HoldTargetLatch, SkillExecutor
command_replay = None if args.plan_only else DecisionReplay(
    args.decisions, args.condition, args.max_estimate_age_s
)
if not math.isfinite(args.seconds) or args.seconds <= 0 or args.telemetry_every_steps < 1:
    parser.error("Seconds must be finite/positive and telemetry interval must be positive")

import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import instantiate
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from fri_bci_sim.tasks.lift import TableLiftEnvCfg
from fri_bci_sim.commands import Skill
from isaaclab_tasks.utils import resolve_presets


def phase(t):
    if t < 1.0:
        return "settle", 0.39, 0.0, 0.70, 0.04
    if t < 4.0:
        return "raise", 0.39, 0.0, 0.70, 0.04
    if t < 8.0:
        return "translate", 0.55, -0.10, 0.70, 0.04
    if t < 12.0:
        return "descend", 0.55, -0.10, 0.40, 0.04
    if t < 15.0:
        return "close", 0.55, -0.10, 0.40, 0.0
    if t < 20.0:
        return "lift", 0.55, -0.10, 0.61, 0.0
    return "hold", 0.55, -0.10, 0.61, 0.0


def run(sim, scene):
    robot, cube = scene["robot"], scene["object"]
    # A standalone scene does not run the task's reset manager. Without this,
    # all seven arm joints begin at zero instead of the configured ready pose.
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()
    scene.reset()
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    arm = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    fingers = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    arm.resolve(scene)
    fingers.resolve(scene)
    hand_id = arm.body_ids[0]
    jacobian_id = hand_id - 1 if robot.is_fixed_base else hand_id
    jacobian_joints = [j + robot.num_base_dofs for j in arm.joint_ids]
    q0 = robot.data.joint_pos.torch.clone()
    p0 = robot.data.body_pose_w.torch[0, hand_id, :3].clone()
    j0 = robot.data.body_link_jacobian_w.torch[0, jacobian_id, :3, jacobian_joints].clone()
    jacobian_check = []
    for index in (1, 3, 5):
        perturbed = q0.clone()
        perturbed[0, arm.joint_ids[index]] += 0.01
        robot.write_joint_position_to_sim_index(position=perturbed)
        observed = (robot.data.body_pose_w.torch[0, hand_id, :3] - p0).tolist()
        predicted = (0.01 * j0[:, index]).tolist()
        jacobian_check.append({"joint": index + 1, "observed_delta_w": observed, "predicted_delta_w": predicted})
        robot.write_joint_position_to_sim_index(position=q0)
    print(json.dumps({"jacobian_check": jacobian_check}), flush=True)
    controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="position" if args.position_only else "pose",
            use_relative_mode=False, ik_method="dls", orientation_weight=0.3
        ),
        num_envs=1, device=sim.device,
    )
    planned_joint_pos = {}
    if args.plan_only or args.joint_waypoints:
        q_seed = q0.clone()
        limits = robot.data.soft_joint_pos_limits.torch[:, arm.joint_ids]
        waypoints = []
        for label, x, y, z in (("above", 0.55, -0.10, 0.65), ("grasp", 0.55, -0.10, 0.38), ("lift", 0.55, -0.10, 0.61)):
            goal = torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
            for _ in range(250):
                robot.write_joint_position_to_sim_index(position=q_seed)
                pose_w = robot.data.body_pose_w.torch[:, hand_id]
                root_w = robot.data.root_pose_w.torch
                p_b, quat_b = subtract_frame_transforms(
                    root_w[:, :3], root_w[:, 3:7], pose_w[:, :3], pose_w[:, 3:7]
                )
                jac_w = robot.data.body_link_jacobian_w.torch[:, jacobian_id, :, jacobian_joints]
                rot_b_w = matrix_from_quat(root_w[:, 3:7]).transpose(1, 2)
                jac_b = torch.cat((rot_b_w @ jac_w[:, :3], rot_b_w @ jac_w[:, 3:]), dim=1)
                controller.set_command(goal)
                q_arm = q_seed[:, arm.joint_ids]
                delta = controller.compute(p_b, quat_b, jac_b, q_arm) - q_arm
                q_seed[:, arm.joint_ids] = (q_arm + delta.clamp(-0.05, 0.05)).clamp(
                    limits[..., 0], limits[..., 1]
                )
            robot.write_joint_position_to_sim_index(position=q_seed)
            pose_w = robot.data.body_pose_w.torch[:, hand_id]
            root_w = robot.data.root_pose_w.torch
            p_b, quat_b = subtract_frame_transforms(
                root_w[:, :3], root_w[:, 3:7], pose_w[:, :3], pose_w[:, 3:7]
            )
            waypoints.append({"label": label, "joint_pos": q_seed[0, arm.joint_ids].tolist(),
                              "hand_pos_b": p_b[0].tolist(), "hand_quat_b": quat_b[0].tolist(),
                              "finger_pos_w": [robot.data.body_pose_w.torch[0, i, :3].tolist()
                                               for i in robot.find_bodies(".*finger")[0]]})
            planned_joint_pos[label] = q_seed[:, arm.joint_ids].clone()
        print(json.dumps({"planned_waypoints": waypoints}), flush=True)
        robot.write_joint_position_to_sim_index(position=q0)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
        if args.plan_only:
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            return
    dt = sim.get_physics_dt()
    steps = int(args.seconds / dt)
    # The initialization target is the configured ready command, not the
    # measured pose after settling. Keep this target as the initial HOLD latch.
    ready_target = robot.data.default_joint_pos.torch[:, arm.joint_ids].clone()
    finger_open = torch.full((1, len(fingers.joint_ids)), 0.04, device=sim.device)
    robot.set_joint_position_target_index(target=ready_target, joint_ids=arm.joint_ids)
    robot.set_joint_position_target_index(target=finger_open, joint_ids=fingers.joint_ids)
    init_steps = max(1, int(round(2.0 / dt)))
    init_path = args.output.parent / "initialization.jsonl"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    init_hand = robot.data.body_pose_w.torch[0, hand_id, :3].tolist()
    init_joint = robot.data.joint_pos.torch[0, arm.joint_ids].tolist()
    init_finger = robot.data.joint_pos.torch[0, fingers.joint_ids].tolist()
    max_init_hand_drift = max_init_joint_drift = max_init_finger_drift = 0.0
    with init_path.open("x", encoding="utf-8") as init_file:
        for init_step in range(init_steps):
            # Reissue the same fixed targets; no cube/object pose writes occur.
            robot.set_joint_position_target_index(target=ready_target, joint_ids=arm.joint_ids)
            robot.set_joint_position_target_index(target=finger_open, joint_ids=fingers.joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            cube_pose = cube.data.root_pose_w.torch[0].tolist()
            hand_pose = robot.data.body_pose_w.torch[0, hand_id].tolist()
            joint_pos = robot.data.joint_pos.torch[0, arm.joint_ids].tolist()
            finger_pos = robot.data.joint_pos.torch[0, fingers.joint_ids].tolist()
            contacts = [float(torch.linalg.norm(scene.sensors[name].data.normal_force_matrix_w.torch[0]))
                        for name in ("panda_leftfinger_object_s", "panda_rightfinger_object_s")]
            max_init_hand_drift = max(
                max_init_hand_drift,
                math.dist(hand_pose[:3], init_hand),
            )
            max_init_joint_drift = max(
                max_init_joint_drift,
                max(abs(a - b) for a, b in zip(joint_pos, init_joint)),
            )
            max_init_finger_drift = max(
                max_init_finger_drift,
                max(abs(a - b) for a, b in zip(finger_pos, init_finger)),
            )
            init_file.write(json.dumps({
                "row_kind": "pre_roll", "initialization_time_s": (init_step + 1) * dt,
                "requested_skill": "hold", "executor_state": "initializing",
                "hand_pose_w": hand_pose, "cube_pose_w": cube_pose,
                "joint_pos": joint_pos, "joint_target": ready_target[0].tolist(),
                "finger_joint_pos": finger_pos, "finger_target_width_m": 0.04,
                "finger_contact_n": contacts,
            }) + "\n")
    target_q = ready_target.clone()
    target_latch = HoldTargetLatch(ready_target, 0.04)
    executor = SkillExecutor()
    skill_time = 0.0
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    initial_cube = cube.data.root_pose_w.torch[0].tolist()
    initial_hand = robot.data.body_pose_w.torch[0, hand_id].tolist()
    initial_joint = robot.data.joint_pos.torch[0, arm.joint_ids].tolist()
    monitor = ExecutionMonitor(initial_joint, initial_hand[:3], initial_cube[2])
    with output.open("x", encoding="utf-8") as f:
        f.write(json.dumps({
            "time_s": 0.0, "interval_start_s": 0.0, "row_kind": "initial",
            "phase": "initial", "requested_skill": "hold", "decision_reason": "initial",
            "executor_state": executor.state.value, "hold_position": True,
            "skill_elapsed_s": 0.0, "skill_complete": False,
            "cube_pose_w": initial_cube, "hand_pose_w": initial_hand,
            "joint_pos": initial_joint, "joint_target": target_q[0].tolist(),
            "finger_target_width_m": 0.04,
            "finger_joint_pos": robot.data.joint_pos.torch[0, fingers.joint_ids].tolist(),
            "finger_contact_n": [0.0, 0.0],
        }) + "\n")
        for step in range(steps):
            t = step * dt
            request = command_replay.at(t)
            event = executor.update(request, skill_complete=skill_time >= 24.0-1e-9)
            if not event.hold_position:
                skill_time = min(24.0, skill_time + dt)
                stage, x, y, z, finger_width = phase(skill_time)
            else:
                stage = "hold"
                x = y = z = None
                target_q, finger_width = target_latch.hold()
            hand_pose = robot.data.body_pose_w.torch[:, hand_id]
            root_pose = robot.data.root_pose_w.torch
            hand_pos_b, hand_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7], hand_pose[:, :3], hand_pose[:, 3:7]
            )
            # This orientation points the hand approach axis toward the table.
            current = robot.data.joint_pos.torch[:, arm.joint_ids].clone()
            if not event.hold_position:
                goal = torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
                controller.set_command(goal[:, :3] if args.position_only else goal, ee_quat=hand_quat_b)
                jacobian_w = robot.data.body_link_jacobian_w.torch[:, jacobian_id, :, jacobian_joints]
                rotation_b_w = matrix_from_quat(root_pose[:, 3:7]).transpose(1, 2)
                jacobian = torch.cat((rotation_b_w @ jacobian_w[:, :3],
                                      rotation_b_w @ jacobian_w[:, 3:]), dim=1)
                if args.joint_waypoints:
                    ready = ready_target
                    above, grasp, lifted = (planned_joint_pos[k] for k in ("above", "grasp", "lift"))
                    if skill_time < 1.0:
                        stage, target_q, finger_width = "settle", ready, 0.04
                    elif skill_time < 8.0:
                        stage, target_q, finger_width = "approach", torch.lerp(ready, above, (skill_time - 1.0) / 7.0), 0.04
                    elif skill_time < 13.0:
                        stage, target_q, finger_width = "descend", torch.lerp(above, grasp, (skill_time - 8.0) / 5.0), 0.04
                    elif skill_time < 16.0:
                        stage, target_q, finger_width = "close", grasp, 0.0
                    elif skill_time < 21.0:
                        stage, target_q, finger_width = "lift", torch.lerp(grasp, lifted, (skill_time - 16.0) / 5.0), 0.0
                    else:
                        stage, target_q, finger_width = "hold", lifted, 0.0
                else:
                    if skill_time > 1.0:
                        correction = controller.compute(hand_pos_b, hand_quat_b, jacobian, current) - current
                        target_q += correction.clamp(-0.005, 0.005)
                    arm_limits = robot.data.soft_joint_pos_limits.torch[:, arm.joint_ids]
                    target_q = target_q.clamp(arm_limits[..., 0], arm_limits[..., 1])
                    target_q = current + (target_q - current).clamp(-0.10, 0.10)
            desired, finger_width = target_latch.resolve(event.hold_position, target_q, finger_width)
            robot.set_joint_position_target_index(target=desired, joint_ids=arm.joint_ids)
            robot.set_joint_position_target_index(
                target=torch.full((1, len(fingers.joint_ids)), finger_width, device=sim.device),
                joint_ids=fingers.joint_ids,
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

            cube_pose = cube.data.root_pose_w.torch[0].tolist()
            cube_vel = cube.data.root_vel_w.torch[0].tolist()
            hand = robot.data.body_pose_w.torch[0, hand_id].tolist()
            forces = [
                float(torch.linalg.norm(scene.sensors[name].data.normal_force_matrix_w.torch[0]))
                for name in ("panda_leftfinger_object_s", "panda_rightfinger_object_s")
            ]
            post_joint = robot.data.joint_pos.torch[0, arm.joint_ids].tolist()
            monitor.observe(post_joint, hand[:3], cube_pose[2], forces,
                            holding=event.hold_position, arm_target=desired[0].tolist(),
                            finger_target=finger_width, dt=dt)
            if (step + 1) % args.telemetry_every_steps == 0 or step == steps-1:
                post_root = robot.data.root_pose_w.torch
                post_hand = robot.data.body_pose_w.torch[:, hand_id]
                post_pos_b, post_quat_b = subtract_frame_transforms(
                    post_root[:, :3], post_root[:, 3:7], post_hand[:, :3], post_hand[:, 3:7]
                )
                f.write(json.dumps({
                    "time_s": (step + 1) * dt, "phase": stage,
                    "interval_start_s": t, "row_kind": "post_step",
                    "requested_skill": request.skill.value, "decision_reason": request.reason,
                    "executor_state": executor.state.value, "hold_position": event.hold_position,
                    "skill_elapsed_s": round(skill_time, 4), "skill_complete": skill_time >= 24.0-1e-9,
                    "cube_pose_w": cube_pose, "cube_velocity_w": cube_vel,
                    "hand_pose_w": hand,
                    "root_pose_w": post_root[0].tolist(),
                    "hand_pos_b": post_pos_b[0].tolist(),
                    "hand_quat_b": post_quat_b[0].tolist(),
                    "joint_pos": post_joint,
                    "joint_target": desired[0].tolist(),
                    "finger_target_width_m": finger_width,
                    "finger_joint_pos": robot.data.joint_pos.torch[0, fingers.joint_ids].tolist(),
                    "finger_contact_n": forces,
                }) + "\n")
    print(json.dumps({"output": str(output), **monitor.summary(),
                      "condition": args.condition, "skill_elapsed_s": skill_time,
                      "skill_complete": skill_time >= 24.0-1e-9,
                      "executor_state": executor.state.value,
                      "executor_semantics": "continuous enable/pause; HOLD preserves executed targets",
                      "physics_dt_s": dt, "physics_steps": steps,
                      "requested_duration_s": args.seconds,
                      "simulated_duration_s": steps * dt,
                      "initialization_telemetry": str(init_path),
                      "initialization_duration_s": init_steps * dt,
                      "initialization_steps": init_steps,
                      "initialization_max_hand_drift_m": max_init_hand_drift,
                      "initialization_max_joint_drift_rad": max_init_joint_drift,
                      "initialization_max_finger_drift_rad": max_init_finger_drift,
                      "initial_hold_arm_target": ready_target[0].tolist(),
                      "telemetry_every_steps": args.telemetry_every_steps}))


def main():
    cfg = TableLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    resolve_presets(cfg, selected=("newton_mjwarp",))
    with launch_simulation(cfg.sim, args):
        sim = sim_utils.SimulationContext(cfg.sim)
        scene = instantiate(cfg.scene)
        sim.reset()
        run(sim, scene)


if __name__ == "__main__":
    main()
