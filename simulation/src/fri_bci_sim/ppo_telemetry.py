"""Bounded RSL-RL policy playback with measured cube and finger telemetry.

Loaded by Isaac Lab through ``--external_callback=fri_bci_sim.ppo_telemetry.register``.
The callback keeps upstream checkpoint loading and policy inference intact while
wrapping the environment step for a short, single-environment physical evaluation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gymnasium as gym
import torch


CONTACT_SENSORS = ("panda_leftfinger_object_s", "panda_rightfinger_object_s")
CONTACT_THRESHOLD_N = 0.1
LIFT_THRESHOLD_M = 0.08
SUSTAINED_CONTACT_S = 0.5


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


class _TelemetryWrapper(gym.Wrapper):
    def __init__(self, env, stream, step_dt):
        super().__init__(env)
        self.stream = stream
        self.step_dt = float(step_dt)
        self.step_index = 0
        self.done = False
        self.episodes_completed = 0
        self.episode_step = 0
        self.episode_results = []
        self.baseline_z = None
        self.max_z = float("-inf")
        self.max_contact_n = [0.0, 0.0]
        self.sustained_steps = 0
        self.best_sustained_steps = 0

    def step(self, action):
        unwrapped = self.env.unwrapped
        diagnostic = {}
        if self.step_index % 30 == 0:
            robot_before = unwrapped.scene["robot"]
            diagnostic = {
                "pre_action_observation_terms": dict(unwrapped.observation_manager.get_active_iterable_terms(0)),
                "pre_action_root_pose_w": _tensor(robot_before.data.root_pose_w)[0].detach().cpu().tolist(),
                "pre_action_cube_pose_w": _tensor(unwrapped.scene["object"].data.root_pose_w)[0].detach().cpu().tolist(),
                "observation_note": "Environment observations before this action; before actor normalization; includes configured history",
            }
        result = self.env.step(action)
        if len(result) == 5:
            observations, rewards, terminated, truncated, infos = result
            dones = torch.logical_or(terminated, truncated)
        elif len(result) == 4:
            observations, rewards, dones, infos = result
        else:
            raise RuntimeError(f"Unexpected environment step result with {len(result)} values")
        self.step_index += 1
        unwrapped = self.env.unwrapped
        cube = unwrapped.scene["object"]
        robot = unwrapped.scene["robot"]
        cube_pose = _tensor(cube.data.root_pose_w)[0].detach().cpu().tolist()
        joint_pos = _tensor(robot.data.joint_pos)[0].detach().cpu().tolist()
        hand_id = robot.find_bodies("panda_hand")[0][0]
        hand_pose = _tensor(robot.data.body_pose_w)[0, hand_id].detach().cpu().tolist()
        actions = _tensor(action)[0].detach().cpu().tolist()
        contact_n = []
        for name in CONTACT_SENSORS:
            matrix = _tensor(unwrapped.scene.sensors[name].data.normal_force_matrix_w)[0]
            contact_n.append(float(torch.linalg.vector_norm(matrix).item()))

        cube_z = float(cube_pose[2])
        self.done = bool(torch.as_tensor(dones).reshape(-1)[0].item())
        if self.baseline_z is None and not self.done:
            self.baseline_z = cube_z
        if not self.done and self.baseline_z is not None:
            self.max_z = max(self.max_z, cube_z)
            self.max_contact_n = [max(previous, value)
                                  for previous, value in zip(self.max_contact_n, contact_n)]
            if cube_z - self.baseline_z >= LIFT_THRESHOLD_M and min(contact_n) >= CONTACT_THRESHOLD_N:
                self.sustained_steps += 1
            else:
                self.sustained_steps = 0
            self.best_sustained_steps = max(self.best_sustained_steps, self.sustained_steps)
        self.episode_step += 1
        episode = self.episodes_completed + 1
        post_reset = self.done

        self.stream.write(json.dumps({
            **diagnostic,
            "weighted_reward_terms_before_dt": dict(unwrapped.reward_manager.get_active_iterable_terms(0)),
            "step": self.step_index,
            "episode": episode,
            "episode_time_s": round(self.episode_step * self.step_dt, 5),
            "time_s": round(self.step_index * self.step_dt, 5),
            "cube_pose_w": cube_pose,
            "cube_height_m": cube_z,
            "cube_height_delta_m": cube_z - (self.baseline_z if self.baseline_z is not None else cube_z),
            "cube_to_hand_distance_m": float(torch.linalg.vector_norm(
                torch.as_tensor(cube_pose[:3]) - torch.as_tensor(hand_pose[:3])
            ).item()),
            "hand_pose_w": hand_pose,
            "robot_joint_pos": joint_pos,
            "finger_contact_n": contact_n,
            "finger_joint_pos": joint_pos[-2:],
            "policy_action": actions,
            "lift_and_bilateral_contact": (
                self.baseline_z is not None
                and cube_z - self.baseline_z >= LIFT_THRESHOLD_M
                and min(contact_n) >= CONTACT_THRESHOLD_N
            ),
            "done": self.done,
            "state_after_automatic_reset": post_reset,
        }) + "\n")
        self.stream.flush()

        # Isaac Lab returns the reset observation on a terminal step. Exclude
        # that post-reset state from the episode's physical success metrics.
        if self.done:
            self.episode_results.append({
                "episode": episode,
                "steps": self.episode_step,
                "duration_s": round(self.episode_step * self.step_dt, 5),
                "baseline_cube_height_m": self.baseline_z,
                "max_cube_height_m": self.max_z if self.max_z != float("-inf") else None,
                "max_cube_lift_m": (self.max_z - self.baseline_z)
                if self.baseline_z is not None and self.max_z != float("-inf") else None,
                "max_finger_contact_n": self.max_contact_n,
                "best_lift_with_bilateral_contact_s": round(
                    self.best_sustained_steps * self.step_dt, 5
                ),
                "physical_lift_pass": self.best_sustained_steps * self.step_dt >= SUSTAINED_CONTACT_S,
            })
            self.episodes_completed += 1
            self.episode_step = 0
            self.baseline_z = None
            self.max_z = float("-inf")
            self.max_contact_n = [0.0, 0.0]
            self.sustained_steps = 0
            self.best_sustained_steps = 0
        return result

    def summary(self):
        return {
            "steps": self.step_index,
            "duration_s": round(self.step_index * self.step_dt, 5),
            "episodes_completed": self.episodes_completed,
            "episode_results": self.episode_results,
            "physical_lift_successes": sum(x["physical_lift_pass"] for x in self.episode_results),
            "terminated_or_truncated": self.episodes_completed > 0,
            "contact_threshold_n": CONTACT_THRESHOLD_N,
            "lift_threshold_m": LIFT_THRESHOLD_M,
            "sustained_contact_threshold_s": SUSTAINED_CONTACT_S,
        }


def register():
    """Install environment telemetry and a finite playback loop before task setup."""
    import isaaclab_rl.entrypoints.backends.play_rsl_rl as play

    output = Path(os.environ.get("FRI_PPO_TELEMETRY", "ppo-playback.jsonl"))
    max_steps = int(os.environ.get("FRI_PPO_MAX_STEPS", "360"))
    episodes = int(os.environ.get("FRI_PPO_EPISODES", "1"))
    if max_steps < 1 or episodes < 1:
        raise ValueError("FRI_PPO_MAX_STEPS must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("w", encoding="utf-8")
    capture = {"wrapper": None}
    create_env = play.create_isaaclab_env

    def create_measured_env(*args, **kwargs):
        env = create_env(*args, **kwargs)
        if int(env.unwrapped.num_envs) != 1:
            env.close()
            raise ValueError("Measured pre-grasp playback requires exactly one environment")
        step_dt = float(env.unwrapped.step_dt)
        measured = _TelemetryWrapper(env, stream, step_dt)
        capture["wrapper"] = measured
        return measured

    def bounded_playback(step, *, dt, args_cli, env_cfg):
        measured = capture["wrapper"]
        if measured is None:
            raise RuntimeError("Telemetry environment was not constructed")
        try:
            for _ in range(max_steps * episodes):
                with torch.inference_mode():
                    step()
                if measured.episodes_completed >= episodes:
                    break
            summary = measured.summary()
            summary["telemetry_path"] = str(output)
            summary["checkpoint"] = str(args_cli.checkpoint)
            summary["episodes_requested"] = episodes
            print(json.dumps({"ppo_physical_evaluation": summary}), flush=True)
        finally:
            stream.close()

    play.create_isaaclab_env = create_measured_env
    play.run_playback = bounded_playback
    return []
