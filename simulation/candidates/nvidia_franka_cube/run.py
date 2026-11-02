"""Finite pretrained-only playback in the candidate's own scene; no training API."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

from prepare import CHECKPOINT_SHA256, ISAAC_COMMIT, sha256
from scoring import score, reset_continuity


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def values(value):
    return tensor(value).detach().cpu().tolist()


class Recorder:
    def __init__(self, env, stream):
        self.env, self.stream = env, stream
        self.active = None
        self.results = []
        self.stepping = False
        self.elapsed = 0.0
        self.interventions = {"finger_clamps": 0, "arm_velocity_clamps": 0,
                              "max_finger_correction_m": 0.0, "max_velocity_correction_rad_s": 0.0}
        original = env.scene.update

        def observed_update(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.stepping and self.active is not None:
                dt = float(kwargs.get("dt", args[0] if args else env.physics_dt))
                self.elapsed += dt
                row = {"record_type": "sample", "time_s": self.elapsed, "sample_interval_s": dt,
                       **self.snapshot()}
                if not self.active["samples"]:
                    self.active["reset_continuity"] = reset_continuity(self.active["initial"], row)
                self.active["samples"].append(row)
                self.write(row)
            return result

        # Observe existing updates. Do not change decimation, graph, or actuator flags.
        env.scene.update = observed_update

    def write(self, row):
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()

    def snapshot(self):
        e = self.env
        cube = values(e.cube.data.root_pose_w)[0]
        hand = values(e.robot.data.body_pose_w)[0][e.ee_body_idx]
        forces = []
        for name in ("candidate_left", "candidate_right"):
            matrix = e.scene.sensors[name].data.normal_force_matrix_w
            forces.append(None if matrix is None else float(tensor(matrix)[0].norm().item()))
        return {"episode": len(self.results) + 1, "cube_pose_w": cube, "cube_height_m": cube[2],
                "hand_pose_w": hand, "finger_contact_n": forces,
                "joint_pos": values(e.robot.data.joint_pos)[0],
                "joint_names": list(e.robot.joint_names), "joint_targets": values(e.robot_dof_targets)[0],
                "gripper_close_enabled": bool(e.gripper_close_enabled[0].item()),
                "interventions": dict(self.interventions)}

    def begin(self):
        initial = self.snapshot()
        local_height = initial["cube_height_m"] - float(self.env.scene.env_origins[0, 2].item())
        fingers = [q for name, q in zip(initial["joint_names"], initial["joint_pos"])
                   if "finger_joint" in name]
        distance = math.dist(initial["cube_pose_w"][:3], initial["hand_pose_w"][:3])
        if abs(local_height - self.env.cfg.cube_size / 2) > 0.005 or not fingers or min(fingers) < 0.03 or distance < 0.05:
            raise RuntimeError("Invalid native table/open-gripper initial state")
        self.elapsed = 0.0
        self.active = {"baseline": initial["cube_height_m"], "initial": initial,
                       "reset_continuity": {"reset_continuity_pass": False}, "samples": [],
                       "interventions_at_start": dict(self.interventions)}
        self.write({"record_type": "initial_pre_action", "time_s": 0.0, **initial})

    def finish(self, complete, termination=None):
        if self.active is None:
            return
        self.write({"record_type": "terminal_pre_reset" if complete else "incomplete",
                    "time_s": self.elapsed, "termination": termination, **self.snapshot()})
        result = score(self.active["samples"], self.active["baseline"],
                       max_gap_s=float(self.env.step_dt) * 1.01, complete=complete)
        result.update(episode=len(self.results) + 1, duration_s=self.elapsed, termination=termination,
                      interventions_at_start=self.active["interventions_at_start"],
                      interventions_at_end=dict(self.interventions),
                      observed_update_intervals_s=sorted(set(r["sample_interval_s"] for r in self.active["samples"])))
        result.update(self.active["reset_continuity"])
        self.results.append(result)
        self.active = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--isaaclab-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, choices=(1, 5), default=1)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--probe", action="store_true", help="CPU checkpoint/config validation; no simulation")
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.candidate_root / "checkpoint/model_399.pt"
    manifest = {"argv": sys.argv, "seed": args.seed, "episodes": args.episodes,
                "candidate_checkpoint_sha256": sha256(checkpoint), "training": False,
                "integration_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
    env = rec = None
    error = None
    try:
        commit = subprocess.check_output(["git", "-C", str(args.isaaclab_root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(args.isaaclab_root), "diff", "--name-only", "HEAD"], text=True).strip()
        if commit != ISAAC_COMMIT or dirty or sha256(checkpoint) != CHECKPOINT_SHA256:
            raise RuntimeError("Runtime pin/cleanliness or checkpoint hash verification failed")
        import torch
        manifest.update(isaac_commit=commit, torch=torch.__version__,
                        rsl_rl=importlib.metadata.version("rsl-rl-lib"))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actor = saved["actor_state_dict"]
        if tuple(actor["mlp.0.weight"].shape) != (256, 41) or tuple(actor["mlp.6.weight"].shape) != (8, 64):
            raise RuntimeError("Unexpected pretrained actor interface")
        manifest["checkpoint_iteration"] = saved.get("iter")
        sys.path.insert(0, str(args.candidate_root / "source/franka_cube"))
        from franka_cube.tasks.direct.franka_cube.franka_cube_env_cfg import FrankaCubeEnvCfg
        from franka_cube.tasks.direct.franka_cube.franka_cube_env import FrankaCubeEnv
        from franka_cube.tasks.direct.franka_cube.agents.rsl_rl_ppo_cfg import PPORunnerCfg
        cfg = FrankaCubeEnvCfg()
        cfg.scene.num_envs = 1
        cfg.seed = args.seed
        if cfg.action_space != 8 or cfg.observation_space != 41 or cfg.episode_length_s != 3.0:
            raise RuntimeError("Candidate interface or horizon changed")
        manifest.update(actor_input=41, actor_output=8, horizon_s=cfg.episode_length_s,
                        physics_dt=cfg.sim.dt, decimation=cfg.decimation,
                        actuators=list(cfg.robot_cfg.actuators),
                        candidate_config_path=sys.modules[FrankaCubeEnvCfg.__module__].__file__)
        if args.probe:
            manifest["probe_pass"] = True
            return
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.assets import AssetBaseCfg
        import isaaclab.sim as sim_utils
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        # The pinned runtime imports cfg-owned clone plans; the old manual scene
        # no longer registers its assets with that importer. Preserve native
        # asset configs/transforms while declaring them to InteractiveScene.
        cfg.scene.robot = cfg.robot_cfg
        cfg.scene.cube = cfg.cube
        cfg.scene.table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Table",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.70711, 0.70711)),
        )
        cfg.scene.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                                      init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)))
        cfg.scene.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)))
        for side in ("left", "right"):
            setattr(cfg.scene, f"candidate_{side}", ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/panda_{side}finger", update_period=0.0,
                filter_prim_paths_expr=["/World/envs/env_.*/object"]))

        class MeasuredEnv(FrankaCubeEnv):
            recorder = None

            def _setup_scene(self):
                self._register_newton_contact_callback()
                self.robot = self.scene["robot"]
                self.cube = self.scene["cube"]

            def _reset_idx(self, env_ids):
                if self.recorder is not None and self.recorder.active is not None:
                    self.recorder.finish(True, {"terminated": bool(self.reset_terminated[0].item()),
                                                "timeout": bool(self.reset_time_outs[0].item())})
                super()._reset_idx(env_ids)

            def _enforce_finger_joint_limits(self):
                if self.recorder is not None:
                    q = tensor(self.robot.data.joint_pos)[:, self.finger_joint_ids]
                    correction = float((q - q.clamp(self.finger_dof_lower_limits, self.finger_dof_upper_limits)).abs().max().item())
                    if correction > 1e-6:
                        r = self.recorder.interventions
                        r["finger_clamps"] += 1
                        r["max_finger_correction_m"] = max(r["max_finger_correction_m"], correction)
                super()._enforce_finger_joint_limits()

            def _enforce_arm_joint_velocity_limits(self):
                if self.recorder is not None:
                    v = tensor(self.robot.data.joint_vel)[:, self.arm_joint_indices]
                    correction = float((v - v.clamp(-self.arm_dof_velocity_limits, self.arm_dof_velocity_limits)).abs().max().item())
                    if correction > 1e-6:
                        r = self.recorder.interventions
                        r["arm_velocity_clamps"] += 1
                        r["max_velocity_correction_rad_s"] = max(r["max_velocity_correction_rad_s"], correction)
                super()._enforce_arm_joint_velocity_limits()

        # Sensor reporting is an explicit candidate instrumentation addition.
        cfg.robot_cfg.spawn.activate_contact_sensors = True
        with launch_simulation(cfg, args), (args.output / "telemetry.jsonl").open("x") as stream:
            try:
                env = MeasuredEnv(cfg)
                wrapped = RslRlVecEnvWrapper(env)
                agent = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), manifest["rsl_rl"])
                runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=str(env.device))
                runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False,
                            "optimizer": False, "iteration": False, "rnd": False}, map_location=str(env.device))
                policy = runner.get_inference_policy(device=str(env.device))
                rec = Recorder(env, stream)
                env.recorder = rec
                obs = wrapped.get_observations()
                manifest["native_physics_handles_decimation"] = bool(env._physics_handles_decimation)
                manifest["policy_dt_s"] = float(env.step_dt)
                for _ in range((math.ceil(3.0 / env.step_dt) + 2) * args.episodes):
                    if rec.active is None:
                        rec.begin()
                    rec.stepping = True
                    try:
                        with torch.inference_mode():
                            action = policy(obs)
                            rec.write({"record_type": "policy_action", "episode": len(rec.results) + 1,
                                       "time_s": rec.elapsed, "action": values(action)[0]})
                            obs, _, dones, _ = wrapped.step(action)
                            policy.reset(dones)
                    finally:
                        rec.stepping = False
                    if len(rec.results) >= args.episodes:
                        break
                rec.finish(False, {"reason": "step_budget_exhausted"})
                if len(rec.results) != args.episodes or any(
                    r["status"] != "completed" or r["contact_coverage"] != 1.0
                    or not r["reset_continuity_pass"] for r in rec.results
                ):
                    raise RuntimeError("Incomplete evaluation, missing contact coverage, or invalid reset continuity")
            finally:
                if rec is not None:
                    rec.finish(False, {"reason": "shutdown_or_error"})
                if env is not None:
                    env.close()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        (args.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        summary = {"error": error, "episodes_requested": args.episodes,
                   "episodes": [] if rec is None else rec.results,
                   "probe_pass": manifest.get("probe_pass", False), "training": False}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
