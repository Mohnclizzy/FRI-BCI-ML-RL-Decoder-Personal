"""Physics-step telemetry for bounded RSL-RL policy playback.

Loaded by Isaac Lab through ``--external_callback=fri_bci_sim.ppo_telemetry.register``.
Task and reset configuration remain owned by the registered environment. The
callback wraps reset/step observation points and temporarily asks Newton to expose
each physics step so physical state and contact are sampled at the simulator rate.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import gymnasium as gym
import torch


CONTACT_SENSORS = ("panda_leftfinger_object_s", "panda_rightfinger_object_s")
FINGERTIP_BODIES = ("panda_leftfinger", "panda_rightfinger")
CONTACT_THRESHOLD_N = 0.1
LIFT_THRESHOLD_M = 0.08
SUSTAINED_CONTACT_S = 0.5
DROP_HEIGHT_M = 0.04
SUSTAINED_DROP_S = 0.25
MAX_SAMPLE_GAP_FACTOR = 1.5
EARLY_ARM_FILTER = "joint1_joint3_first_two"


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _tolist(value):
    return _tensor(value).detach().cpu().tolist()


def _split_joint_targets(joint_names: list[str], target: list[float]):
    targets = {name: float(value) for name, value in zip(joint_names, target)}
    arm = {name: value for name, value in targets.items() if name.startswith("panda_joint")}
    fingers = {name: value for name, value in targets.items() if "finger" in name}
    return targets, arm, fingers


def _fingertip_body_indices(robot):
    """Resolve the two sensor-bearing links once, failing on an ambiguous match."""
    indices = {}
    for name in FINGERTIP_BODIES:
        body_ids, body_names = robot.find_bodies(name)
        if len(body_ids) != 1 or body_names != [name]:
            raise RuntimeError(f"expected exactly one {name} body, got {body_names!r}")
        indices[name] = body_ids[0]
    if len(set(indices.values())) != len(FINGERTIP_BODIES):
        raise RuntimeError("left and right fingertips resolve to the same body")
    return indices


def _early_arm_action(action: torch.Tensor, policy_step: int) -> torch.Tensor:
    """Return an inference-only first-two-step ablation without mutating policy output."""
    if action.ndim != 2 or action.shape[1] != 9:
        raise ValueError("early arm filter requires a [num_envs,9] action tensor")
    if policy_step not in (1, 2):
        return action
    applied = action.clone()
    applied[:, (0, 2)] = 0.0
    return applied


class EpisodeAccumulator:
    """Score timestamped physical samples using elapsed simulation time."""

    def __init__(self, episode: int, baseline_z: float, physics_dt: float):
        self.episode = int(episode)
        self.baseline_z = float(baseline_z)
        self.physics_dt = float(physics_dt)
        self.elapsed_s = 0.0
        self.samples = 0
        self.contact_samples = 0
        self.contact_missing_samples = 0
        self.max_z = self.baseline_z
        self.max_contact_n = [0.0, 0.0]
        self.first_contact_time_s = None
        self.first_lift_time_s = None
        self.first_bilateral_lift_time_s = None
        self.acquired_time_s = None
        self.drops = []
        self._lift_contact_streak_s = 0.0
        self.best_lift_contact_s = 0.0
        self._drop_streak_s = 0.0
        self._in_drop = False
        self._last_timestamp_s = None
        self.gaps = 0

    def add_sample(self, timestamp_s: float, cube_z: float, contact_n: list[float | None]):
        timestamp_s = float(timestamp_s)
        cube_z = float(cube_z)
        if self._last_timestamp_s is None:
            dt = self.physics_dt
            dwell_dt = dt
        else:
            dt = timestamp_s - self._last_timestamp_s
            dwell_dt = dt
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("physics sample timestamps must increase")
            if dt > self.physics_dt * MAX_SAMPLE_GAP_FACTOR:
                self.gaps += 1
                self._lift_contact_streak_s = 0.0
                self._drop_streak_s = 0.0
                # The state between samples is unknown. Count only one current
                # physics interval toward either dwell predicate after a gap.
                dwell_dt = self.physics_dt
        self._last_timestamp_s = timestamp_s
        self.elapsed_s += dt
        self.samples += 1
        self.max_z = max(self.max_z, cube_z)

        valid = len(contact_n) == 2 and all(
            value is not None and math.isfinite(float(value)) for value in contact_n
        )
        if not valid:
            self.contact_missing_samples += 1
            self._lift_contact_streak_s = 0.0
            self._drop_streak_s = 0.0
            return {
                "contact_valid": False,
                "lift_contact_streak_s": 0.0,
                "acquired": self.acquired_time_s is not None and not self._in_drop,
                "drop_count": len(self.drops),
            }

        self.contact_samples += 1
        values = [float(value) for value in contact_n]
        self.max_contact_n = [max(prev, value) for prev, value in zip(self.max_contact_n, values)]
        has_contact = min(values) >= CONTACT_THRESHOLD_N
        has_lift = cube_z - self.baseline_z + 1e-9 >= LIFT_THRESHOLD_M
        if has_contact and self.first_contact_time_s is None:
            self.first_contact_time_s = round(timestamp_s, 6)
        if has_lift and self.first_lift_time_s is None:
            self.first_lift_time_s = round(timestamp_s, 6)
        if has_lift and has_contact and self.first_bilateral_lift_time_s is None:
            self.first_bilateral_lift_time_s = round(timestamp_s, 6)

        if has_lift and has_contact:
            self._lift_contact_streak_s += dwell_dt
            self.best_lift_contact_s = max(self.best_lift_contact_s, self._lift_contact_streak_s)
            self._drop_streak_s = 0.0
            if self._in_drop:
                self._in_drop = False
            if self.acquired_time_s is None and self._lift_contact_streak_s + 1e-9 >= SUSTAINED_CONTACT_S:
                self.acquired_time_s = round(timestamp_s, 6)
        else:
            self._lift_contact_streak_s = 0.0

        is_below_drop_height = cube_z < self.baseline_z + DROP_HEIGHT_M - 1e-9
        if self.acquired_time_s is not None and not self._in_drop and is_below_drop_height:
            self._drop_streak_s += dwell_dt
            if self._drop_streak_s + 1e-9 >= SUSTAINED_DROP_S:
                self.drops.append(round(timestamp_s, 6))
                self._in_drop = True
        else:
            self._drop_streak_s = 0.0

        return {
            "contact_valid": True,
            "lift_contact_streak_s": round(self._lift_contact_streak_s, 6),
            "acquired": self.acquired_time_s is not None and not self._in_drop,
            "drop_count": len(self.drops),
        }

    def finish(
        self,
        *,
        status: str,
        reason: str | None = None,
        termination: dict | None = None,
        error: str | None = None,
        policy_steps: int | None = None,
    ):
        coverage = self.contact_samples / self.samples if self.samples else 0.0
        return {
            "episode": self.episode,
            "status": status,
            "reason": reason,
            "error": error,
            "policy_steps": policy_steps,
            "termination": termination or {},
            "physics_samples": self.samples,
            "contact_samples": self.contact_samples,
            "contact_missing_samples": self.contact_missing_samples,
            "contact_coverage": round(coverage, 6),
            "physics_step_gaps": self.gaps,
            "steps": self.samples,
            "duration_s": round(self.elapsed_s, 6),
            "baseline_cube_height_m": self.baseline_z,
            "max_cube_height_m": self.max_z if self.samples else None,
            "max_cube_lift_m": self.max_z - self.baseline_z if self.samples else None,
            "max_finger_contact_n": self.max_contact_n,
            "first_contact_time_s": self.first_contact_time_s,
            "first_lift_time_s": self.first_lift_time_s,
            "first_bilateral_lift_time_s": self.first_bilateral_lift_time_s,
            "acquisition_time_s": self.acquired_time_s,
            "best_sustained_lift_contact_s": round(self.best_lift_contact_s, 6),
            "drop_count": len(self.drops),
            "drop_times_s": self.drops,
            "physical_lift_pass": self.acquired_time_s is not None,
            "contact_threshold_n": CONTACT_THRESHOLD_N,
            "lift_threshold_m": LIFT_THRESHOLD_M,
            "sustained_contact_threshold_s": SUSTAINED_CONTACT_S,
            "drop_height_m_above_baseline": DROP_HEIGHT_M,
            "sustained_drop_threshold_s": SUSTAINED_DROP_S,
        }


class _TelemetryWrapper(gym.Wrapper):
    def __init__(self, env, stream, step_dt, output_path, episodes_limit):
        super().__init__(env)
        self.stream = stream
        self.step_dt = float(step_dt)
        self.physics_dt = float(env.unwrapped.physics_dt)
        if self.physics_dt <= 0 or self.step_dt <= 0:
            raise ValueError("environment physics_dt and step_dt must be positive")
        self.output_path = Path(output_path)
        self.summary_path = self.output_path.with_suffix(".summary.json")
        self.episodes_limit = int(episodes_limit)
        self.physics_step_index = 0
        self.policy_step_index = 0
        self.episodes_completed = 0
        self.episodes_started = 0
        self.episode_policy_steps = 0
        self.accumulator: EpisodeAccumulator | None = None
        self.episode_results = []
        self._last_state = None
        self._stepping = False
        self._in_reset = False
        self._physics_samples_this_policy_step = 0
        self._error = None
        self._original_physics_flag = None
        self._original_handles_decimation = bool(env.unwrapped._physics_handles_decimation)
        self._original_scene_update = env.unwrapped.scene.update
        self._original_apply_action = env.unwrapped.action_manager.apply_action
        self._original_reset_idx = env.unwrapped._reset_idx
        fingertip_option = os.environ.get("FRI_PPO_RECORD_FINGERTIP_BODY_POSES", "0")
        if fingertip_option not in ("0", "1"):
            raise ValueError("FRI_PPO_RECORD_FINGERTIP_BODY_POSES must be 0 or 1")
        self.record_fingertip_body_poses = fingertip_option == "1"
        self.early_arm_filter = os.environ.get("FRI_PPO_EARLY_ARM_FILTER", "")
        if self.early_arm_filter not in ("", EARLY_ARM_FILTER):
            raise ValueError(f"FRI_PPO_EARLY_ARM_FILTER must be empty or {EARLY_ARM_FILTER}")
        self._fingertip_body_indices = (
            _fingertip_body_indices(env.unwrapped.scene["robot"])
            if self.record_fingertip_body_poses else {}
        )
        self._install_physics_step_sampling()
        self._install_state_hooks()

    def _install_physics_step_sampling(self):
        base = self.env.unwrapped
        manager = base.sim.physics_manager
        handles_decimation = bool(manager.handles_decimation())
        if handles_decimation:
            # Pinned Newton can execute every decimation substep inside a single
            # CUDA graph. Temporarily use its eager-actuator path so Isaac Lab's
            # unchanged environment loop updates the scene/contact sensors once
            # after each physics step. Policy rate, task config, resets, actions,
            # and solver dt are unchanged.
            if not hasattr(manager, "_use_newton_actuators_active"):
                raise RuntimeError("physics backend folds decimation but exposes no Newton eager-step switch")
            self._original_physics_flag = bool(manager._use_newton_actuators_active)
            manager._use_newton_actuators_active = False
            base._physics_handles_decimation = False
            def apply_policy_targets_once(*args, **kwargs):
                # The graphable Newton path applies the relative policy action
                # once per policy step and then keeps those absolute targets
                # through the internal decimation loop. Preserve that behavior
                # while the eager loop exposes intermediate physics states.
                if not self._stepping or self._physics_samples_this_policy_step == 0:
                    return self._original_apply_action(*args, **kwargs)
                return None

            base.action_manager.apply_action = apply_policy_targets_once
        self.physics_sampling_mode = (
            "eager_physics_step_fixed_policy_targets" if handles_decimation else "environment_physics_step"
        )

        def update_and_record(*args, **kwargs):
            result = self._original_scene_update(*args, **kwargs)
            if self._stepping:
                update_dt = kwargs.get("dt", args[0] if args else self.physics_dt)
                if not math.isclose(float(update_dt), self.physics_dt, rel_tol=1e-5, abs_tol=1e-8):
                    raise RuntimeError(
                        f"physics telemetry expected one {self.physics_dt}s update, got {update_dt}s"
                    )
                self._record_physics_sample()
            return result

        base.scene.update = update_and_record

    def _install_state_hooks(self):
        base = self.env.unwrapped

        def reset_idx_with_terminal(env_ids):
            ids = torch.as_tensor(env_ids, device=base.device).reshape(-1) if not isinstance(env_ids, slice) else None
            includes_env_zero = ids is None or bool(torch.any(ids == 0).item())
            if includes_env_zero and self.accumulator is not None and not self._in_reset:
                self._write_terminal_state()
            result = self._original_reset_idx(env_ids)
            if includes_env_zero and not self._in_reset:
                if self.episodes_completed < self.episodes_limit:
                    self._begin_episode("automatic_reset")
            return result

        base._reset_idx = reset_idx_with_terminal

    def _capture_state(self):
        base = self.env.unwrapped
        cube = base.scene["object"]
        robot = base.scene["robot"]
        cube_pose = _tolist(cube.data.root_pose_w[0])
        joint_pos = _tolist(robot.data.joint_pos[0])
        joint_names = list(robot.data.joint_names)
        target_proxy = getattr(robot.data, "joint_pos_target", None)
        target = None if target_proxy is None else _tolist(_tensor(target_proxy)[0])
        hand_id = robot.find_bodies("panda_hand")[0][0]
        hand_pose = _tolist(robot.data.body_pose_w[0, hand_id])
        contact_n = []
        for sensor_name in CONTACT_SENSORS:
            sensor = base.scene.sensors.get(sensor_name)
            if sensor is None:
                contact_n.append(None)
                continue
            matrix = getattr(sensor.data, "normal_force_matrix_w", None)
            if matrix is None:
                contact_n.append(None)
                continue
            contact_n.append(float(torch.linalg.vector_norm(_tensor(matrix)[0]).item()))
        targets, arm_targets, finger_targets = _split_joint_targets(joint_names, target) if target is not None else ({}, {}, {})
        action_terms = {}
        if hasattr(base, "action_manager"):
            for name in base.action_manager.active_terms:
                term = base.action_manager.get_term(name)
                processed = getattr(term, "processed_actions", None)
                if processed is not None:
                    action_terms[name] = _tolist(processed[0])
        state = {
            "cube_pose_w": cube_pose,
            "cube_height_m": float(cube_pose[2]),
            "robot_joint_pos": joint_pos,
            "joint_names": joint_names,
            "joint_pos_target": targets,
            "processed_arm_targets": arm_targets,
            "processed_finger_targets": finger_targets,
            "processed_action_terms": action_terms,
            "hand_pose_w": hand_pose,
            "finger_contact_n": contact_n,
            "cube_to_hand_distance_m": float(torch.linalg.vector_norm(
                torch.as_tensor(cube_pose[:3]) - torch.as_tensor(hand_pose[:3])
            ).item()),
        }
        if self.record_fingertip_body_poses:
            state["fingertip_body_pose_w"] = {
                name: _tolist(robot.data.body_pose_w[0, body_id])
                for name, body_id in self._fingertip_body_indices.items()
            }
        return state

    def _begin_episode(self, reason: str):
        state = self._capture_state()
        self.episodes_started += 1
        self.episode_policy_steps = 0
        self.accumulator = EpisodeAccumulator(
            episode=self.episodes_started,
            baseline_z=state["cube_height_m"],
            physics_dt=self.physics_dt,
        )
        self._last_state = state
        self._write({
            "record_type": "episode_start",
            "episode": self.episodes_started,
            "reason": reason,
            "physics_time_s": self._physics_time_s(),
            "episode_time_s": 0.0,
            "baseline_cube_height_m": state["cube_height_m"],
            "physics_dt_s": self.physics_dt,
            "policy_dt_s": self.step_dt,
            "physics_sampling_mode": self.physics_sampling_mode,
            **state,
        })

    def reset(self, **kwargs):
        self._in_reset = True
        try:
            result = self.env.reset(**kwargs)
        finally:
            self._in_reset = False
        self._begin_episode("explicit_reset")
        return result

    def step(self, action):
        if self.accumulator is None:
            raise RuntimeError("telemetry baseline was not captured before the first action")
        self.policy_step_index += 1
        self.episode_policy_steps += 1
        self._physics_samples_this_policy_step = 0
        raw_action = _tolist(action[0])
        applied_action = (
            _early_arm_action(action, self.episode_policy_steps)
            if self.early_arm_filter else action
        )
        diagnostic = {}
        if self.episode_policy_steps == 1 or self.episode_policy_steps % 30 == 0:
            base = self.env.unwrapped
            diagnostic = {
                "pre_action_observation_terms": dict(base.observation_manager.get_active_iterable_terms(0)),
                "observation_note": "pre-action values before actor normalization; observation history is configured by the task",
            }
        self._write({
            "record_type": "policy_step_start",
            "episode": self.accumulator.episode,
            "policy_step": self.episode_policy_steps,
            "global_policy_step": self.policy_step_index,
            "physics_time_s": self._physics_time_s(),
            "episode_time_s": round(self.accumulator.elapsed_s, 6),
            "policy_action": raw_action,
            **({
                "applied_action": _tolist(applied_action[0]),
                "early_arm_filter": self.early_arm_filter,
            } if self.early_arm_filter else {}),
            **diagnostic,
        })
        self._stepping = True
        try:
            result = self.env.step(applied_action)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._finish_incomplete("step_exception", self._error)
            raise
        finally:
            self._stepping = False
            self._physics_samples_this_policy_step = 0
        return result

    def _record_physics_sample(self):
        if self.accumulator is None:
            raise RuntimeError("physics step occurred without an active reset baseline")
        state = self._capture_state()
        physics_time = self._physics_time_s()
        episode_time = self.accumulator.elapsed_s + self.physics_dt
        scores = self.accumulator.add_sample(episode_time, state["cube_height_m"], state["finger_contact_n"])
        self._last_state = state
        self._write({
            "record_type": "physics_step",
            "episode": self.accumulator.episode,
            "policy_step": self.episode_policy_steps,
            "physics_step": self.physics_step_index + 1,
            "physics_time_s": physics_time,
            "episode_time_s": round(episode_time, 6),
            "physics_dt_s": self.physics_dt,
            "policy_dt_s": self.step_dt,
            "cube_height_delta_m": state["cube_height_m"] - self.accumulator.baseline_z,
            "lift_and_bilateral_contact": (
                state["cube_height_m"] - self.accumulator.baseline_z >= LIFT_THRESHOLD_M
                and len(state["finger_contact_n"]) == 2
                and all(x is not None and x >= CONTACT_THRESHOLD_N for x in state["finger_contact_n"])
            ),
            **scores,
            **state,
        })
        self.physics_step_index += 1
        self._physics_samples_this_policy_step += 1

    def _physics_time_s(self):
        manager = self.env.unwrapped.sim.physics_manager
        try:
            return round(float(manager.get_simulation_time()), 6)
        except Exception:
            return round(self.physics_step_index * self.physics_dt, 6)

    def _write_terminal_state(self):
        state = self._capture_state()
        base = self.env.unwrapped
        termination = {
            "terminated": bool(torch.as_tensor(base.termination_manager.terminated).reshape(-1)[0].item()),
            "time_out": bool(torch.as_tensor(base.termination_manager.time_outs).reshape(-1)[0].item()),
        }
        self._write({
            "record_type": "terminal_pre_reset",
            "episode": self.accumulator.episode,
            "physics_step": self.physics_step_index,
            "physics_time_s": self._physics_time_s(),
            "episode_time_s": round(self.accumulator.elapsed_s, 6),
            "termination": termination,
            **state,
        })
        self.episode_results.append(self.accumulator.finish(status="completed", termination=termination))
        self.episodes_completed += 1
        self.accumulator = None
        self._last_state = None

    def _finish_incomplete(self, reason: str, error: str | None = None):
        if self.accumulator is None:
            return
        state = self._capture_state()
        base = self.env.unwrapped
        termination = {
            "terminated": bool(torch.as_tensor(base.termination_manager.terminated).reshape(-1)[0].item()),
            "time_out": bool(torch.as_tensor(base.termination_manager.time_outs).reshape(-1)[0].item()),
        }
        self._write({
            "record_type": "episode_incomplete_end",
            "episode": self.accumulator.episode,
            "policy_step": self.episode_policy_steps,
            "physics_step": self.physics_step_index,
            "physics_time_s": self._physics_time_s(),
            "episode_time_s": round(self.accumulator.elapsed_s, 6),
            "end_reason": reason,
            "termination": termination,
            **state,
        })
        result = self.accumulator.finish(
            status="incomplete", reason=reason, error=error, policy_steps=self.episode_policy_steps,
            termination=termination,
        )
        self.episode_results.append(result)
        self.accumulator = None

    def summary(self, *, episodes_requested: int, max_steps_per_episode: int):
        if self.accumulator is not None:
            self._finish_incomplete("max_steps_exhausted" if not self._error else "playback_error", self._error)
        return {
            "physics_steps": self.physics_step_index,
            "policy_steps": self.policy_step_index,
            "duration_s": round(self.physics_step_index * self.physics_dt, 6),
            "physics_dt_s": self.physics_dt,
            "policy_dt_s": self.step_dt,
            "physics_sampling_mode": self.physics_sampling_mode,
            "record_fingertip_body_poses": self.record_fingertip_body_poses,
            "early_arm_filter": self.early_arm_filter or None,
            "episodes_requested": episodes_requested,
            "max_policy_steps_per_episode": max_steps_per_episode,
            "episodes_started": self.episodes_started,
            "episodes_completed": self.episodes_completed,
            "episode_results": self.episode_results,
            "physical_lift_successes": sum(bool(x["physical_lift_pass"]) for x in self.episode_results),
            "incomplete_episodes": sum(x["status"] == "incomplete" for x in self.episode_results),
            "contact_threshold_n": CONTACT_THRESHOLD_N,
            "lift_threshold_m": LIFT_THRESHOLD_M,
            "sustained_contact_threshold_s": SUSTAINED_CONTACT_S,
            "drop_height_m_above_baseline": DROP_HEIGHT_M,
            "sustained_drop_threshold_s": SUSTAINED_DROP_S,
            "telemetry_path": str(self.output_path),
            "summary_path": str(self.summary_path),
            "error": self._error,
        }

    def _write(self, record):
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self):
        base = self.env.unwrapped
        base.scene.update = self._original_scene_update
        base.action_manager.apply_action = self._original_apply_action
        base._reset_idx = self._original_reset_idx
        base._physics_handles_decimation = self._original_handles_decimation
        if self._original_physics_flag is not None:
            base.sim.physics_manager._use_newton_actuators_active = self._original_physics_flag
        super().close()


def register():
    """Install physical telemetry and a finite playback loop before task setup."""
    import isaaclab_rl.entrypoints.backends.play_rsl_rl as play

    output = Path(os.environ.get("FRI_PPO_TELEMETRY", "ppo-playback.jsonl"))
    max_steps = int(os.environ.get("FRI_PPO_MAX_STEPS", "360"))
    episodes = int(os.environ.get("FRI_PPO_EPISODES", "1"))
    if max_steps < 1 or episodes < 1:
        raise ValueError("FRI_PPO_MAX_STEPS and FRI_PPO_EPISODES must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("w", encoding="utf-8")
    capture = {"wrapper": None}
    create_env = play.create_isaaclab_env

    def create_measured_env(*args, **kwargs):
        env = create_env(*args, **kwargs)
        if int(env.unwrapped.num_envs) != 1:
            env.close()
            raise ValueError("Measured PPO playback requires exactly one environment")
        measured = _TelemetryWrapper(env, stream, env.unwrapped.step_dt, output, episodes)
        capture["wrapper"] = measured
        return measured

    def bounded_playback(step, *, dt, args_cli, env_cfg):
        measured = capture["wrapper"]
        if measured is None:
            raise RuntimeError("Telemetry environment was not constructed")
        error = None
        try:
            for _ in range(max_steps * episodes):
                with torch.inference_mode():
                    step()
                if measured.episodes_completed >= episodes:
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            measured._error = error
            measured._finish_incomplete("playback_exception", error)
            raise
        finally:
            summary = measured.summary(episodes_requested=episodes, max_steps_per_episode=max_steps)
            summary["checkpoint"] = str(args_cli.checkpoint)
            summary["playback_error"] = error
            measured.summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"ppo_physical_evaluation": summary}), flush=True)
            stream.close()

    play.create_isaaclab_env = create_measured_env
    play.run_playback = bounded_playback
    return []
