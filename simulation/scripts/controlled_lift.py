"""Bounded scripted Franka grasp/lift diagnostic on the fixed table-cube scene.

The cube is never attached or pose-controlled. Only arm and finger joint targets
are sent. JSONL records physical object state and filtered finger contact.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=23.0)
parser.add_argument("--position-only", action="store_true", help="Diagnose translation tracking without orientation")
parser.add_argument("--plan-only", action="store_true", help="Solve and report joint waypoints without physics playback")
parser.add_argument("--joint-waypoints", action="store_true", help="Play measured joint-space lift reference")
add_launcher_args(parser)
args = parser.parse_args()

import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import instantiate
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from fri_bci_sim.tasks.lift import TableLiftEnvCfg
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
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        if args.plan_only:
            return
    dt = sim.get_physics_dt()
    steps = int((24.0 if args.joint_waypoints else args.seconds) / dt)
    target_q = robot.data.joint_pos.torch[:, arm.joint_ids].clone()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_z = None
    sustained_steps = 0
    best_sustained_steps = 0
    max_z = float("-inf")
    max_contact = [0.0, 0.0]
    with output.open("w", encoding="utf-8") as f:
        for step in range(steps):
            t = step * dt
            stage, x, y, z, finger_width = phase(t)
            hand_pose = robot.data.body_pose_w.torch[:, hand_id]
            root_pose = robot.data.root_pose_w.torch
            hand_pos_b, hand_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7], hand_pose[:, :3], hand_pose[:, 3:7]
            )
            # This orientation points the hand approach axis toward the table.
            goal = torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
            controller.set_command(goal[:, :3] if args.position_only else goal, ee_quat=hand_quat_b)
            jacobian_w = robot.data.body_link_jacobian_w.torch[:, jacobian_id, :, jacobian_joints]
            rotation_b_w = matrix_from_quat(root_pose[:, 3:7]).transpose(1, 2)
            jacobian = torch.cat((
                rotation_b_w @ jacobian_w[:, :3],
                rotation_b_w @ jacobian_w[:, 3:],
            ), dim=1)
            current = robot.data.joint_pos.torch[:, arm.joint_ids]
            if args.joint_waypoints:
                ready = q0[:, arm.joint_ids]
                above, grasp, lifted = (planned_joint_pos[k] for k in ("above", "grasp", "lift"))
                if t < 1.0:
                    stage, target_q, finger_width = "settle", ready, 0.04
                elif t < 8.0:
                    stage, target_q, finger_width = "approach", torch.lerp(ready, above, (t - 1.0) / 7.0), 0.04
                elif t < 13.0:
                    stage, target_q, finger_width = "descend", torch.lerp(above, grasp, (t - 8.0) / 5.0), 0.04
                elif t < 16.0:
                    stage, target_q, finger_width = "close", grasp, 0.0
                elif t < 21.0:
                    stage, target_q, finger_width = "lift", torch.lerp(grasp, lifted, (t - 16.0) / 5.0), 0.0
                else:
                    stage, target_q, finger_width = "hold", lifted, 0.0
            else:
                if stage != "settle":
                    correction = controller.compute(hand_pos_b, hand_quat_b, jacobian, current) - current
                    target_q += correction.clamp(-0.005, 0.005)
                arm_limits = robot.data.soft_joint_pos_limits.torch[:, arm.joint_ids]
                target_q = target_q.clamp(arm_limits[..., 0], arm_limits[..., 1])
                # Limit servo error if the incremental IK command outruns the arm.
                target_q = current + (target_q - current).clamp(-0.10, 0.10)
            desired = target_q
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
            max_contact = [max(a, b) for a, b in zip(max_contact, forces)]
            if baseline_z is None and t >= 0.9:
                baseline_z = cube_pose[2]
            max_z = max(max_z, cube_pose[2])
            if baseline_z is not None and cube_pose[2] - baseline_z >= 0.08 and min(forces) >= 0.1:
                sustained_steps += 1
            else:
                sustained_steps = 0
            best_sustained_steps = max(best_sustained_steps, sustained_steps)
            if step % 12 == 0:
                f.write(json.dumps({
                    "time_s": round((step + 1) * dt, 4), "phase": stage,
                    "cube_pose_w": cube_pose, "cube_velocity_w": cube_vel,
                    "hand_pose_w": hand,
                    "root_pose_w": root_pose[0].tolist(),
                    "hand_pos_b": hand_pos_b[0].tolist(),
                    "hand_quat_b": hand_quat_b[0].tolist(),
                    "joint_pos": current[0].tolist(),
                    "joint_target": desired[0].tolist(),
                    "finger_joint_pos": robot.data.joint_pos.torch[0, fingers.joint_ids].tolist(),
                    "finger_contact_n": forces,
                }) + "\n")
    print(json.dumps({
        "output": str(output), "baseline_cube_z_m": baseline_z, "max_cube_z_m": max_z,
        "max_finger_contact_n": max_contact,
        "best_sustained_lift_contact_s": best_sustained_steps * dt,
        "physical_lift_pass": best_sustained_steps * dt >= 0.5,
    }))


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
